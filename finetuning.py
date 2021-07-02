from __future__ import division

import argparse
import os

import torch
import torch.distributed as dist
from mmcv import Config
from yxy.notification import slack_sender

from lib.utils import (get_root_logger, init_dist, find_free_port, set_random_seed, handleworkspace)
from lib.finetune_engine import Engine
import sys
from tqdm import tqdm, trange


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from',
                        help='the git push -u origin mastercheckpoint file to resume from')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--validate', action='store_true', help='validate during training')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    if args.validate:
        cfg.validate = args.validate
    # cudnn default true

    # override by args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.resume_from is not None:
        cfg.load_checkpoint = args.resume_from

    # init dist training
    if args.launcher == 'none':
        raise NotImplementedError
    else:
        # free_port = find_free_port()
        # dist_url = f'tcp://127.0.0.1:{free_port}'
        # only be used in pytorch dist mode
        init_dist(args.launcher)

    # init logger before other steps
    logger = get_root_logger(cfg.work_dir, cfg.log_level)
    logger.info(f'Distributed training with {args.world_size}')

    # set random seeds
    rank = args.local_rank
    if args.seed is not None:
        set_random_seed(args.seed + rank)

    if cfg.validate:  # validate means validate only
        # if you dont have ckpt
        final_validate_checkpoint = cfg.load_checkpoint
        assert final_validate_checkpoint is not None
        engine = Engine(cfg, logger, only_final_validate=True)
    else:
        engine = Engine(cfg, logger, only_final_validate=False)

        if cfg.load_model is not None:
            # pretrained finetuning
            engine.load_pretrained(cfg.load_model)

        if cfg.load_checkpoint is not None:
            engine.load_modelandstatus(cfg.load_checkpoint)

        engine.run()
        final_validate_checkpoint = os.path.join(cfg.work_dir, 'model_best.pth.tar')

    logger.info('doing the final validation')
    engine.load_modelandstatus(final_validate_checkpoint)
    engine.validate_epoch(final_validate=True)


if __name__ == '__main__':
    main()
