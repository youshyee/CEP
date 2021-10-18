import os
import shutil
from pathlib import Path
import logging
from torch import Tensor
from collections import OrderedDict

import torch


class CheckpointManager:
    ''' Manage checkpoints

    * keep_interval (int): Keep a checkpoint every `keep_interval` epochs. The latest checkpoint is always saved.
        None means only keep the latest one.

    add feature: change the saving frequency for the last 1/4 epochs
    '''
    def __init__(self,
                 experiment_dir,
                 logger,
                 keep_interval=None,
                 filename='checkpoint.pth',
                 milestone=0):
        """

        :param experiment_dir:
        :param keep_interval:
        :param filename:
        :param milestone: used to control save from whihc epoch
        """
        self.experiment_dir = experiment_dir
        self.filename = filename
        self.keep_interval = keep_interval
        self.milestone = milestone
        self.logger = logger

    def save(self, state: dict, is_best: bool, epoch: int):
        checkpoint_path = self.experiment_dir / self.filename
        temp_checkpoint_path = self.experiment_dir / f'.next.{self.filename}'

        self.logger.info('Saving checkpoint to "%s"', checkpoint_path)
        try:
            # Save to temp file first.
            # In case of error, previous checkpoint should be kept intact.
            torch.save(state, temp_checkpoint_path)
        except:
            if temp_checkpoint_path.exists():
                temp_checkpoint_path.unlink()
            raise
        temp_checkpoint_path.rename(checkpoint_path)
        self.logger.info('Checkpoint saved')

        if is_best:
            model_best_path = self.experiment_dir / 'model_best.pth.tar'
            self.logger.info('Saving best model to "%s"', model_best_path)
            if model_best_path.exists():
                model_best_path.unlink()
            os.link(checkpoint_path, model_best_path)

        if self.keep_interval is not None and epoch % self.keep_interval == 0 and epoch > self.milestone:
            keep_path = self.experiment_dir / f'checkpoint_epoch_{epoch}.pth.tar'
            self.logger.info('Keep checkpoint "%s"', keep_path)
            os.link(checkpoint_path, keep_path)
