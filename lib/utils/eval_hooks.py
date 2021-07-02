import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper
from torch.utils.data import Dataset
from lib.dataset import build_dataloader

from mmcv.runner import get_dist_info
import tempfile
import shutil
from yxy.debug import dprint


def collect_results(result_part, size, tmpdir=None):
    # result_part should be unordered list
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        # rand = ''.join(map(str, np.random.randint(10, size=3)))
        # tmpdir = tmpdir + '_' + rand
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    # collect all parts
    # load results of all parts from tmp dir

    dist.barrier()
    total_result_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
        total_result_list.append(mmcv.load(part_file))
    # sort the results
    # ordered_results = []
    # for res in zip(*part_list):
    #     ordered_results.extend(list(res))
    # # the dataloader may pad some samples
    # ordered_results = ordered_results[:size]
    # remove tmp dir
    dist.barrier()
    if rank == 0:
        shutil.rmtree(tmpdir)

    return total_result_list


class DistEvalHook(Hook):
    def __init__(self, dataset, cfg, interval=1, eval_bs=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError('dataset must be a Dataset object or a dict, not {}'.format(
                type(dataset)))
        self.interval = interval
        self.cfg = cfg
        self.eval_bs = eval_bs

    def after_train_epoch(self, runner):  # fast version of eval
        if not self.every_n_epochs(runner, self.interval):
            return
        print('evaluation')
        dataloader = build_dataloader(dataset=self.dataset,
                                      workers_per_gpu=self.cfg.workers_per_gpu,
                                      batch_size=self.eval_bs,
                                      sampler=torch.utils.data.DistributedSampler(self.dataset,
                                                                                  shuffle=False),
                                      dist=True)
        model = runner.model
        model = model.eval()

        results = []
        rank = runner.rank
        world_size = runner.world_size

        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for i, data in enumerate(dataloader):
            meta = data[1]
            assert len(meta) == self.eval_bs
            with torch.no_grad():
                result = model.val_step(data, None)
                if isinstance(result, torch.Tensor):
                    result = result.detach().cpu()
            results.append({'result': result, 'meta': meta})

            if rank == 0:
                batch_size = self.eval_bs
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        # collect results from all ranks
        dist.barrier()

        # list of list [ [results1], [results2]]
        all_results = collect_results(results, len(self.dataset),
                                      os.path.join(runner.work_dir, f'temp/cycle_eval'))
        self.evaluate(runner, all_results, results)
        dist.barrier()
        model.train()

    def evaluate(self):
        raise NotImplementedError


class DistEvalAccuracy(DistEvalHook):
    # results in format of list of dict {total:N,top1:a,top3:b}
    def evaluate(self, runner, all_results, results):
        rank = runner.rank
        world_size = runner.world_size
        if rank == 0:
            all_sample = 0
            correct_sample = 0
            for part in all_results:
                all_sample += len(part) * self.eval_bs
                for each in part:
                    #result either 0 or 1
                    correct_sample += each['result']

            runner.log_buffer.output['accuracy'] = correct_sample / all_sample
            runner.log_buffer.ready = True


class DistEvalTopKRecallHook(DistEvalHook):
    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def evaluate(self, runner, all_results, results):
        rank = runner.rank
        world_size = runner.world_size
        target_tensors = [i['result'] for i in results]
        target_tensors = torch.cat(target_tensors, dim=0)
        target_meta = []
        for i in results:
            target_meta += i['meta']

        assert len(target_meta) == len(target_tensors)

        tensorlen = [len(r) * self.eval_bs for r in all_results]
        tensorlen = [0] + np.cumsum(tensorlen).tolist()

        all_tensors = []
        all_meta = []
        for rank_results in all_results:
            for i in rank_results:
                all_tensors.append(i['result'])
                all_meta += i['meta']

        all_tensors = torch.cat(all_tensors, dim=0)
        assert len(all_tensors) == tensorlen[-1]

        assert len(all_tensors) == len(all_meta)

        sim_matrix = self.sim_matrix(all_tensors, target_tensors)
        for i in range(len(target_tensors)):
            sim_matrix[tensorlen[rank] + i, i] = -2
        value, indice = torch.topk(sim_matrix, 30, dim=0)
        correct_10 = 0
        correct_30 = 0
        correct_5 = 0
        correct_3 = 0
        for i in range(len(target_tensors)):
            gt = target_meta[i]['class']
            choose_10 = indice[:10, i]
            choose_30 = indice[:30, i]
            choose_5 = indice[:5, i]
            choose_3 = indice[:3, i]

            choose_label_30 = [all_meta[j]['class'] for j in choose_30]
            choose_label_10 = [all_meta[j]['class'] for j in choose_10]
            choose_label_5 = [all_meta[j]['class'] for j in choose_5]
            choose_label_3 = [all_meta[j]['class'] for j in choose_3]

            if gt in choose_label_30:
                correct_30 += 1
            if gt in choose_label_3:
                correct_3 += 1
            if gt in choose_label_5:
                correct_5 += 1
            if gt in choose_label_10:
                correct_10 += 1

        dist.barrier()
        correct = torch.tensor([correct_3, correct_5, correct_10, correct_30]).cuda()
        all_correct = [torch.ones_like(correct)] * world_size

        torch.distributed.all_gather(all_correct, correct)
        if rank == 0:
            runner.log_buffer.output['recall_rate_3'] = sum(all_correct)[0].item() / len(
                all_tensors)
            runner.log_buffer.output['recall_rate_5'] = sum(all_correct)[1].item() / len(
                all_tensors)
            runner.log_buffer.output['recall_rate_10'] = sum(all_correct)[2].item() / len(
                all_tensors)
            runner.log_buffer.output['recall_rate_30'] = sum(all_correct)[3].item() / len(
                all_tensors)
            runner.log_buffer.ready = True
