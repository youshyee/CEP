import logging
import os
import random
import socket
import resource
import subprocess
from contextlib import closing
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info, get_time_str


def handleworkspace(work_dir):
    if os.path.isdir(work_dir):
        warnings.warn(f'workdir {work_dir} exsits, forcing rename it')
        endname = get_time_str()[4:]
        work_dir = work_dir + endname
        os.mkdir(work_dir)
    else:
        os.mkdir(work_dir)
    return work_dir


def find_free_port():
    """
    Find a free port for dist url
    :return:
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    return port


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        raise NotImplementedError
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')


def get_root_logger(workdir, log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        endname = get_time_str() + '.log'
        filename = os.path.join(workdir, endname)
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
        # logging.basicConfig(filename=filename,
        #                     filemode='a',
        #                     format='%(asctime)s - %(levelname)s - %(message)s',
        #                     level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger


def ulimit_n_max():
    _soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))


def scale_learning_rate(lr, world_size, batch_size, base_batch_size=64):
    new_lr = lr * world_size * batch_size / base_batch_size
    return new_lr, lr
