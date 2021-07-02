import torch
import os
import torchvision
from mmcv.runner import get_dist_info, obj_from_dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from pathlib import Path
import time

from lib.dataset import DataLoaderFactory
from lib.utils.average_meter import AverageMeter
from lib.utils.checkpoint import CheckpointManager
from lib.utils.env import scale_learning_rate
from lib.utils.utils import get_lr
from .contrastive_builder import ModelFactory
from lib.model import Loss
from lib.utils.metrics import multiInstacneaccuracy, accuracy


class Engine:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        rank, world_size = get_dist_info()
        self.local_rank = rank
        self.logger = logger
        self.model_factory = ModelFactory(cfg, logger=self.logger)
        self.data_loader_factory = DataLoaderFactory(cfg=cfg.pretrain_data, logger=self.logger)

        self.device = torch.device(
            f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        self.loss_lambda = self.cfg.loss_lambda

        self.model = self.model_factory.build_cep_with_contrastive_loss()

        lambda_cfg = cfg.loss_lambda
        self.criterion = Loss(**lambda_cfg)

        self.train_loader = self.data_loader_factory.build(pretrain=True, device=self.device)

        # rescale lr according to the lr that is refer to bz of 64
        self.learning_rate = self.cfg.optimizer.lr
        self.batch_size = self.cfg.gpu_batch
        if not self.cfg.no_scale_lr:
            self.learning_rate, old_lr = scale_learning_rate(
                self.learning_rate,
                world_size,
                self.batch_size,
            )
            self.logger.warning(
                f'adjust lr according to the number of GPU and batch size：{old_lr} -> {self.learning_rate}'
            )

        # update new lr
        optimizer_cfg = self.cfg.optimizer
        optimizer_cfg.lr = self.learning_rate
        # buiding optimizer from cfg
        self.optimizer = obj_from_dict(optimizer_cfg,
                                       torch.optim,
                                       default_args=dict(params=self.model.parameters()))

        self.num_epochs = cfg.total_epochs

        # lr scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=self.num_epochs,
                                                                    eta_min=self.learning_rate /
                                                                    1000)
        self.arch = cfg.arch

        if self.local_rank == 0:
            if cfg.get('usetb', True):
                self.summary_writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tblogs'))
            else:
                self.summary_writer = None
            self.checkpoint = CheckpointManager(Path(cfg.work_dir),
                                                logger=self.logger,
                                                **cfg.checkpoint_config)
        else:
            self.summary_writer = None
            self.checkpoint = None

        self.log_interval = cfg.log_interval

        self.loss_meter = AverageMeter('Loss', device=self.device)
        self.loss_meter_A = AverageMeter('Loss_A', device=self.device)  # Scale of this is large
        self.loss_meter_CL = AverageMeter('Loss_CL', device=self.device)  # Scale of this is large
        self.loss_meter_CC = AverageMeter('Loss_CC', device=self.device)  # Scale of this is large

        self.top1_meter_A = AverageMeter('Acc@1_A', fmt=':6.2f', device=self.device)
        self.top5_meter_A = AverageMeter('Acc@5_A', fmt=':6.2f', device=self.device)
        # self.top1_meter_CL_1 = AverageMeter('Acc@1_CL_1', fmt=':6.2f', device=self.device)
        # self.top1_meter_CL_2 = AverageMeter('Acc@1_CL_2', fmt=':6.2f', device=self.device)

        self.current_epoch = 0

        self.overall_progress = None  # Place Holder

    def _load_ckpt_file(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=self.device)
        if states['arch'] != self.arch:
            raise ValueError(
                f'Loading checkpoint arch {states["arch"]} does not match current arch {self.arch}')
        return states

    def load_modelandstatus(self, checkpoint_path):
        states = self._load_ckpt_file(checkpoint_path)
        self.logger.info('Loading checkpoint from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])

        self.optimizer.load_state_dict(states['optimizer'])
        self.scheduler.load_state_dict(states['scheduler'])

        self.current_epoch = states['epoch']
        self.best_loss = states['best_loss']

    def load_model(self, checkpoint_path):
        states = self._load_ckpt_file(checkpoint_path)
        self.logger.info('Loading model from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])

    def reset_meters(self):

        self.loss_meter.reset()
        self.loss_meter_A.reset()
        self.loss_meter_CL.reset()
        self.loss_meter_CC.reset()

        self.top1_meter_A.reset()
        self.top5_meter_A.reset()
        # self.top1_meter_CL_1.reset()
        # self.top1_meter_CL_2.reset()
        self.loss_meter.reset()

    def train_epoch(self):
        epoch = self.current_epoch
        self.train_loader.set_epoch(epoch)
        num_iters = len(self.train_loader)
        self.reset_meters()
        epoch_lr = get_lr(self.optimizer)

        iter_data = tqdm(self.train_loader,
                         desc='Current Epoch',
                         disable=self.local_rank != 0,
                         dynamic_ncols=True)

        for i, ((clip_q, clip_k), *_) in enumerate(iter_data):

            # [B,3,F,size,size]
            # debug print data
            # print(i)
            # print(clip_q.shape)
            # for index, each in enumerate(clip_q):
            #     each = each.permute(1, 0, 2, 3)
            #     q_each = torchvision.utils.make_grid(each, nrow=4)
            #     each_k = clip_k[index].permute(1, 0, 2, 3)
            #     k_each = torchvision.utils.make_grid(each_k, nrow=4)
            #     out = torch.cat([q_each, k_each], dim=-1)
            #     out = out.cpu()
            #     out = out * 255
            #     out = out.type(torch.uint8)
            #     torchvision.io.write_jpeg(out, f'/home/youshyee/temp/tempwork1/{i}_{index}.jpg')

            logits_A, mask_A, logits_C_1, logits_C_2, CC_1, CC_2, std = self.model(clip_q, clip_k)
            loss, Aloss, CLRankLoss, CC_dis = self.criterion(
                Alogits=logits_A,
                mask=mask_A,
                CL_1=logits_C_1,
                CL_2=logits_C_2,
                CC_1=CC_1,
                CC_2=CC_2,
            )

            if not self.cfg.validate:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_size = len(clip_q)
            # top 4 are correct
            poslist = [0, 1]
            acc1_A, acc5_A = multiInstacneaccuracy(logits_A, poslist, topk=(1, 5))
            self.top1_meter_A.update(acc1_A, batch_size)
            self.top5_meter_A.update(acc5_A, batch_size)
            self.loss_meter_A.update(Aloss, batch_size)

            #CL
            target = torch.zeros(logits_C_1[0].size(0)).to(logits_C_1[0].device)
            # acc1_CL_1, = accuracy(torch.cat(logits_C_1, dim=1), target, topk=(1, ))
            # acc1_CL_2, = accuracy(torch.cat(logits_C_2, dim=1), target, topk=(1, ))
            # self.top1_meter_CL_1.update(acc1_CL_1, batch_size)
            # self.top1_meter_CL_2.update(acc1_CL_2, batch_size)
            self.loss_meter_CL.update(CLRankLoss, batch_size)

            # cc dist here
            self.loss_meter_CC.update(CC_dis, batch_size)

            # overall
            self.loss_meter.update(loss, batch_size)

            if i > 0 and i % self.log_interval == 0:
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                self.logger.info(
                    f'Train [{epoch}/{self.num_epochs}][{i - 1}/{num_iters}] || {self.loss_meter}\n'
                    f'{self.loss_meter_A} || {self.top1_meter_A} || {self.top5_meter_A}\n'
                    f'{self.loss_meter_CL}||{self.loss_meter_CC}||std:{round(std.cpu().item(),3)}|| lr :{epoch_lr}'
                )

                # write to tb loss write to current value, accuracy write to mvavg
                if self.summary_writer is not None:
                    # loss
                    self.summary_writer.add_scalar('train/iter/Loss', self.loss_meter.val.item(),
                                                   self.overall_progress.n)
                    self.summary_writer.add_scalar('train/iter/ALoss', self.loss_meter_A.val.item(),
                                                   self.overall_progress.n)
                    self.summary_writer.add_scalar('train/iter/CLloss',
                                                   self.loss_meter_CL.val.item(),
                                                   self.overall_progress.n)
                    self.summary_writer.add_scalar('train/iter/CCloss',
                                                   self.loss_meter_CC.val.item(),
                                                   self.overall_progress.n)
                    # acc
                    self.summary_writer.add_scalar('train/iter/ATop1', self.top1_meter_A.avg.item(),
                                                   self.overall_progress.n)
                    self.summary_writer.add_scalar('train/iter/ATop5', self.top5_meter_A.avg.item(),
                                                   self.overall_progress.n)

                    # self.summary_writer.add_scalar('train/iter/CL_1Top1',
                    #                                self.top1_meter_CL_1.avg.item(),
                    #                                self.overall_progress.n)
                    # self.summary_writer.add_scalar('train/iter/CL_2Top1',
                    #                                self.top1_meter_CL_2.avg.item(),
                    #                                self.overall_progress.n)
            self.overall_progress.update()

    def run(self):
        num_epochs = self.num_epochs
        best_loss = float('inf')

        self.model.train()

        num_iters = len(self.train_loader)

        with tqdm(total=num_epochs * num_iters,
                  disable=self.local_rank != 0,
                  smoothing=0.1,
                  desc='Overall',
                  dynamic_ncols=True,
                  initial=self.current_epoch * num_iters) as self.overall_progress:
            while self.current_epoch < num_epochs:
                self.train_epoch()

                self.scheduler.step()

                # summary per epoch level
                if self.summary_writer is not None:
                    loss = self.loss_meter.avg.item()
                    self.summary_writer.add_scalar('train/epoch/lr', get_lr(self.optimizer),
                                                   self.current_epoch)
                    self.summary_writer.add_scalar('train/epoch/loss', loss, self.current_epoch)

                self.current_epoch += 1

                if self.local_rank == 0:
                    loss = self.loss_meter.avg.item()
                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)

                    self.checkpoint.save(
                        {
                            'epoch': self.current_epoch,
                            'arch': self.arch,
                            'model': self.model.module.state_dict(),
                            'best_loss': best_loss,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                        },
                        is_best,
                        self.current_epoch,
                    )
                # sleep 10m for next round
                time.sleep(600)
