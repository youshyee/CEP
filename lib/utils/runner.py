from mmcv.runner import EpochBasedRunner
import time
import torch
import os.path as osp
import os
import mmcv

from mmcv.runner import load_state_dict, save_checkpoint, master_only, Hook


class Runner(EpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""
    def __init__(self, *args, extra=None, **kwargs):
        self.extra = extra
        super().__init__(*args, **kwargs)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def load_checkpoint(self, filename, map_location='cpu', strict=False, part=None):
        self.logger.info('load checkpoint from %s', filename)

        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)

        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        # get state_dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        if part is not None:
            state_dict = {k: v for k, v in state_dict.items() if part in k}
        # load state_dict
        load_state_dict(self.model, state_dict, strict, self.logger)
        return checkpoint

    def ck4resume(self, out_dir, save_optimizer=True, meta=None, create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = f'ck4resume_{self.iter+1}.pth'
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'resume_latest.pth')
            mmcv.symlink(filename, dst_file)


class Ck4resumeHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
    """
    def __init__(self, interval=6000, save_optimizer=True, out_dir=None, **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        runner.logger.info(f'Saving ck4resume at {runner.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.ck4resume(self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        current_iter = runner.iter + 1
        ckpt_l = [ckpt for ckpt in os.listdir(self.out_dir) if 'ck4resume' in ckpt]
        ckpt_l = [ckpt for ckpt in ckpt_l if ckpt.split('_')[1] != f'{runner.iter+1}.pth']
        for ckpt in ckpt_l:
            ckpt_path = os.path.join(self.out_dir, ckpt)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
