from __future__ import division

import os
import re

import mmcv
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Hook, get_dist_info, obj_from_dict, OptimizerHook
from yxy.debug import dprint

# from lib.api.dist_utils import DistOptimizerHook
from lib.api.runner import Ck4resumeHook, Runner
from lib.dataset import build_dataloader

from .env import get_root_logger
from .eval_hooks import DistEvalAccuracy, DistEvalTopKRecallHook


def train_model(model, dataset, cfg, validate=True, distributed=True, logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, validate, cfg, logger)
    else:
        raise NotImplementedError


def build_optimizer(model, optimizer_cfg):

    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim, dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_group = {'params': [param]}
            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, validate, cfg, logger):
    # prepare data loaders
    if validate:
        train_dataset = dataset[0]
        val_dataset = dataset[1]
    else:
        train_dataset = dataset
    batch_size = cfg.get('gpu_batch', 1)

    data_loaders = [
        build_dataloader(dataset=train_dataset,
                         workers_per_gpu=cfg.workers_per_gpu,
                         batch_size=batch_size,
                         dist=True,
                         sampler=torch.utils.data.DistributedSampler(train_dataset))
    ]
    # put model on gpus
    rank, _ = get_dist_info()
    num_gpus = torch.cuda.device_count()
    # syn batchnorm warp
    model = model.cuda(rank % num_gpus)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = MMDistributedDataParallel(model,
                                      device_ids=[rank % num_gpus],
                                      find_unused_parameters=True)
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model,
                    batch_processor=None,
                    optimizer=optimizer,
                    work_dir=cfg.work_dir,
                    logger=logger,
                    extra=None)
    # register hooks
    optimizer_config = OptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config,
                                   cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    check4resume = cfg.get('check4resume', None)
    if check4resume:
        runner.register_hook(Ck4resumeHook(**check4resume))
    if validate:
        if cfg.dataset.dataname.lower() in 'kinetics':
            print('kinetics regist eval hook')
            interval = cfg.get('eval_interval', 1)
            runner.register_hook(
                DistEvalTopKRecallHook(val_dataset, cfg, interval=interval, eval_bs=batch_size))
        else:
            print('downstream task regist eval hook')
            interval = cfg.get('eval_interval', 1)
            runner.register_hook(
                DistEvalAccuracy(val_dataset, cfg, interval=interval, eval_bs=batch_size))
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        part = cfg.get("part", None)
        runner.load_checkpoint(cfg.load_from, part=part)
    elif cfg.get('autoresume', False):
        ckpath = os.path.join(cfg.work_dir, 'resume_latest.pth')
        if os.path.exists(ckpath):
            runner.resume(ckpath)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
