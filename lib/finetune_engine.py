import torch
import os
import torchvision
from mmcv.runner import get_dist_info, obj_from_dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from pathlib import Path

from lib.dataset import DataLoaderFactory
from lib.utils.average_meter import AverageMeter
from lib.utils.checkpoint import CheckpointManager
from lib.utils.env import scale_learning_rate
from lib.utils.utils import get_lr

from .contrastive_builder import ModelFactory
from lib.utils.metrics import accuracy
import torch
import torch.nn as nn


class Engine:
    def __init__(self,
                 cfg,
                 logger,
                 only_final_validate=False):  # only final_validate no traninng just validate
        self.cfg = cfg
        rank, world_size = get_dist_info()
        self.world_size = world_size
        self.local_rank = rank
        self.logger = logger

        self.model_factory = ModelFactory(cfg, logger=self.logger)

        self.data_loader_factory_final = DataLoaderFactory(cfg=cfg.finetune_data,
                                                           logger=self.logger,
                                                           final_validate=True)

        self.data_loader_factory = DataLoaderFactory(cfg=cfg.finetune_data,
                                                     logger=self.logger,
                                                     final_validate=False)

        self.device = torch.device(
            f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        self.model = self.model_factory.build_normal()

        if not only_final_validate:
            self.train_loader = self.data_loader_factory.build(
                pretrain=False,
                split='train',
                device=self.device,
            )
            self.validate_loader = self.data_loader_factory.build(
                pretrain=False,
                split='val',
                device=self.device,
            )

        self.criterion = nn.CrossEntropyLoss()

        # rescale lr
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

        self.schedule_type = self.cfg.scheduler.type
        if self.schedule_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                patience=self.cfg.scheduler.patience,
                verbose=True)
        elif self.schedule_type == "multi_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.cfg.scheduler.milestones,
            )
        elif self.schedule_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        T_max=self.num_epochs,
                                                                        eta_min=self.learning_rate /
                                                                        1000)
        elif self.schedule_type == 'none':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: 1,
            )
        else:
            raise ValueError("Unknow schedule type")

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

        self.best_acc1 = 0.
        self.current_epoch = 0
        self.next_epoch = None

        self.log_interval = cfg.log_interval

        self.loss_meter = AverageMeter(
            'Loss', device=self.device
        )  # This place displays decimals directly because the loss is relatively large
        self.top1_meter = AverageMeter('Acc@1', fmt=':6.2f', device=self.device)
        self.top5_meter = AverageMeter('Acc@5', fmt=':6.2f', device=self.device)

        self.overall_progress = None  # Place Holder

    def build_final_loader(self):
        validate_loader_final = self.data_loader_factory_final.build(pretrain=False,
                                                                     split='val',
                                                                     device=self.device)
        return validate_loader_final

    def _load_ckpt_file(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=self.device)
        if states['arch'] != self.arch:
            raise ValueError(
                f'Loading checkpoint arch {states["arch"]} does not match current arch {self.arch}')
        return states

    def load_modelandstatus(self, checkpoint_path):
        # resume training
        states = self._load_ckpt_file(checkpoint_path)

        self.logger.info('Loading checkpoint from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])

        self.optimizer.load_state_dict(states['optimizer'])
        self.scheduler.load_state_dict(states['scheduler'])

        self.current_epoch = states['epoch']
        self.best_acc1 = states['best_acc1']

    def load_pretrained(self, checkpoint_path):
        # transfer learning
        cp = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in cp and 'arch' in cp:
            self.logger.info('Loading MoCo checkpoint from %s (epoch %d)', checkpoint_path,
                             cp['epoch'])
            moco_state = cp['model']
            prefix = 'encoder_q.'
        else:
            # This checkpoint is from third-party
            self.logger.info('Loading third-party model from %s', checkpoint_path)
            if 'state_dict' in cp:
                moco_state = cp['state_dict']
            else:
                # For c3d
                moco_state = cp
                self.logger.warning(
                    'if you are not using c3d sport1m, maybe you use wrong checkpoint')
            if next(iter(moco_state.keys())).startswith('module'):
                prefix = 'module.'
            else:
                prefix = ''
        """
        fc -> fc. for c3d sport1m. Beacuse fc6 and fc7 is in use.
        """
        blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8']
        blacklist += ['encoder_fuse']

        def filter(k):
            return k.startswith(prefix) and not any(
                k.startswith(f'{prefix}{fc}') for fc in blacklist)

        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        msg = self.model.module.load_state_dict(model_state, strict=False)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or \
        #        set(msg.missing_keys) == {"linear.weight", "linear.bias"} or \
        #        set(msg.missing_keys) == {'head.projection.weight', 'head.projection.bias'} or \
        #        set(msg.missing_keys) == {'new_fc.weight', 'new_fc.bias'},\
        #     msg

        self.logger.warning(
            f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')

    def reset_meters(self):
        self.loss_meter.reset()

        self.top1_meter.reset()
        self.top5_meter.reset()

    def reshape_clip(self, clip, n_crop):
        if n_crop == 1:
            return clip
        clip = clip.refine_names('batch', 'channel', 'time', 'height', 'width')
        crop_len = clip.size(2) // n_crop
        clip = clip.unflatten('time', [('crop', n_crop), ('time', crop_len)])
        clip = clip.align_to('batch', 'crop', ...)
        clip = clip.flatten(['batch', 'crop'], 'batch')
        return clip.rename(None)

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

        for i, ((clip, ), target, *others) in enumerate(iter_data):
            clip = self.reshape_clip(clip, n_crop=1)  # training n_crop=1
            output = self.model(clip)

            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = target.size(0)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            self.top1_meter.update(acc1, batch_size)
            self.top5_meter.update(acc5, batch_size)

            self.loss_meter.update(loss, batch_size)

            if i > 0 and i % self.log_interval == 0:
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                self.logger.info(
                    f'Train [{epoch}/{self.num_epochs}][{i - 1}/{num_iters}] || {self.loss_meter}\n'
                    f'{self.top1_meter} || {self.top5_meter} || lr :{epoch_lr}')

                # write to tb loss write to current value, accuracy write to mvavg
                if self.summary_writer is not None:
                    # loss
                    self.summary_writer.add_scalar('train/iter/Loss', self.loss_meter.val.item(),
                                                   self.overall_progress.n)
                    # acc
                    self.summary_writer.add_scalar('train/iter/Top1', self.top1_meter.avg.item(),
                                                   self.overall_progress.n)
                    self.summary_writer.add_scalar('train/iter/Top5', self.top5_meter.avg.item(),
                                                   self.overall_progress.n)

                    # self.summary_writer.add_scalar('train/iter/CL_1Top1',
                    #                                self.top1_meter_CL_1.avg.item(),
                    #                                self.overall_progress.n)
                    # self.summary_writer.add_scalar('train/iter/CL_2Top1',
                    #                                self.top1_meter_CL_2.avg.item(),
                    #                                self.overall_progress.n)
            self.overall_progress.update()

    @torch.no_grad()
    def validate_epoch(self, final_validate=False):

        if final_validate:
            validate_loader = self.build_final_loader()
        else:
            validate_loader = self.validate_loader

        self.model.eval()
        if final_validate:
            all_logits = torch.empty(0, device=next(self.model.parameters()).device)
            all_target = torch.empty(0, device=next(self.model.parameters()).device)
        self.reset_meters()

        iter_data = tqdm(validate_loader,
                         desc='Current Epoch',
                         disable=self.local_rank != 0,
                         dynamic_ncols=True)

        num_iters = len(validate_loader)
        remaining_valid_samples = validate_loader.num_valid_samples()

        if final_validate:
            n_crop = self.cfg.finetune_data.temporal_transforms.validate.final_n_crop
        else:
            n_crop = self.cfg.finetune_data.temporal_transforms.validate.n_crop

        for i, ((clip, ), target, *others) in enumerate(iter_data):

            clip = self.reshape_clip(clip, n_crop=n_crop)  # training n_crop=1
            output = self.model(clip)
            loss = self.criterion(output, target)

            batch_size = target.size(0)
            if batch_size > remaining_valid_samples:
                # Distributed sampler will add some repeated samples. cut them off.
                output = output[:remaining_valid_samples]
                target = target[:remaining_valid_samples]
                others = [o[:remaining_valid_samples] for o in others]
                batch_size = remaining_valid_samples

            remaining_valid_samples -= batch_size

            if batch_size == 0:
                continue

            batch_size = target.size(0)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            self.top1_meter.update(acc1, batch_size)
            self.top5_meter.update(acc5, batch_size)

            self.loss_meter.update(loss, batch_size)

            if final_validate:
                all_logits = torch.cat((all_logits, output), dim=0)
                all_target = torch.cat((all_target, target), dim=0)

        self.logger.info(
            'Validation finished.\n\tLoss = %f\n\tAcc@1 = %.2f%% (%d/%d)\n\tAcc@5 = %.2f%% (%d/%d)',
            self.loss_meter.avg.item(),
            self.top1_meter.avg.item(),
            self.top1_meter.sum.item() / 100,
            self.top1_meter.count.item(),
            self.top5_meter.avg.item(),
            self.top5_meter.sum.item() / 100,
            self.top5_meter.count.item(),
        )

        # write to tb loss write to current value, accuracy write to mvavg
        if self.summary_writer is not None:
            # loss
            self.summary_writer.add_scalar('val/epoch/Loss', self.loss_meter.avg.item(),
                                           self.current_epoch)
            # acc
            self.summary_writer.add_scalar('val/epoch/Top1', self.top1_meter.avg.item(),
                                           self.current_epoch)
            self.summary_writer.add_scalar('val/epoch/Top5', self.top5_meter.avg.item(),
                                           self.current_epoch)
        if final_validate:
            self.logger.info('final validation caculating')
            basename = self.cfg.work_dir
            torch.save(all_logits.cpu(), os.path.join(basename, f'final_loggits_{self.local_rank}'))
            torch.save(all_target.cpu(), os.path.join(basename, f'final_target_{self.local_rank}'))
            torch.distributed.barrier()

            if self.local_rank == 0:
                logits = torch.empty(0)
                targets = torch.empty(0)
                for i in range(self.world_size):
                    logit = torch.load(os.path.join(basename, f'final_loggits_{self.local_rank}'))
                    target = torch.load(os.path.join(basename, f'final_target_{self.local_rank}'))
                    logits = torch.cat((logits, logit), dim=0)
                    targets = torch.cat((targets, target), dim=0)
                finallacc1, finalacc5 = accuracy(logits, targets)
                self.logger.info(f'final validation accuray top1:{finallacc1}, top5:{finalacc5}')

        return self.top1_meter.avg.item()

    def run(self):
        num_epochs = self.num_epochs

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
                if self.schedule_type == "plateau":
                    self.scheduler.step(self.loss_meter.val.item())
                else:
                    self.scheduler.step()

                acc1 = self.validate_epoch()
                self.model.train(True)

                # summary per epoch level
                if self.summary_writer is not None:
                    loss = self.loss_meter.avg.item()
                    self.summary_writer.add_scalar('train/epoch/lr', get_lr(self.optimizer),
                                                   self.current_epoch)
                    self.summary_writer.add_scalar('train/epoch/top1', acc1, self.current_epoch)
                    self.summary_writer.add_scalar('train/epoch/loss', loss, self.current_epoch)

                self.current_epoch += 1

                if self.local_rank == 0:
                    is_best = acc1 > self.best_acc1
                    self.best_acc1 = max(acc1, self.best_acc1)

                    self.checkpoint.save(
                        {
                            'epoch': self.current_epoch,
                            'arch': self.arch,
                            'model': self.model.module.state_dict(),
                            'best_acc1': self.best_acc1,
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict(),
                        },
                        is_best,
                        self.current_epoch,
                    )
