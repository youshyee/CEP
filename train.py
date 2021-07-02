from __future__ import division

import argparse
import sys

import torch
from mmcv import Config
from tqdm import tqdm, trange

from lib.train_engine import Engine
from lib.utils import (find_free_port, get_root_logger, handleworkspace, init_dist, set_random_seed)


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
        init_dist(args.launcher)

    # init logger before other steps
    logger = get_root_logger(cfg.work_dir, cfg.log_level)
    logger.info(f'Distributed training with {args.world_size}')

    # set random seeds
    rank = args.local_rank
    if args.seed is not None:
        set_random_seed(args.seed + rank)

    engine = Engine(cfg, logger)

    if cfg.load_model is not None:
        engine.load_model(cfg.load_model)
    if cfg.load_checkpoint is not None:
        engine.load_modelandstatus(cfg.load_checkpoint)

    if cfg.validate:
        # Only used to retrieve statistical results
        with torch.no_grad():
            with trange(1) as engine.overall_progress:
                engine.train_epoch()
    else:
        engine.run()


if __name__ == '__main__':
    main()
