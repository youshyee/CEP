import logging
import multiprocessing as mp
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from .transforms_video import (
    transforms_spatial, transforms_temporal, transforms_tensor)
from .video import VideoDataset


def num_valid_samples(num_samples, rank, num_replicas):
    '''Note: depends on the implementation detail of `DistributedSampler`
    '''
    return (num_samples - rank - 1) // num_replicas + 1


class MainProcessCollateWrapper:
    def __init__(self, dataloader: DataLoader, collate_fn, epoch_callback=None):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
        self.epoch_callback = epoch_callback

    @torch.no_grad()
    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)
        if self.epoch_callback is not None:
            self.epoch_callback(epoch)

    def num_valid_samples(self):
        sampler = self.dataloader.sampler
        return num_valid_samples(len(sampler.dataset), sampler.rank, sampler.num_replicas)


def identity(x):
    return x


class DataLoaderFactory:
    def __init__(self, cfg, logger, final_validate=False):
        self.cfg = cfg
        self.final_validate = final_validate
        self.logger = logger

    def build_pretrain(self, ds_name, split='train'):
        assert 'kinetics' in ds_name
        from .kinetics import Kinetics
        ds = Kinetics(
            self.cfg.dataset.root,
            split=split,
            blacklist=self.cfg.dataset.blacklist,
        )
        # transform pipline
        cpu_transform, gpu_transform = self.get_transform_pretrain()

        size = self.cfg.temporal_transforms.size
        strides = self.cfg.temporal_transforms.strides
        self.logger.info(f'Using frames: {size}, with strides: {strides}')
        temporal_transform = transforms_temporal.RandomStrideCrop(
            size=size,
            strides=strides,
        )
        return ds, cpu_transform, gpu_transform, temporal_transform

    def build_finetune(self, ds_name, split):
        if ds_name == 'ucf101':
            from .ucf101 import UCF101
            ds = UCF101(
                self.cfg.dataset.root,
                self.cfg.dataset.annotation_path,
                fold=self.cfg.dataset.fold,
                split=split,
            )
        elif ds_name.startswith('hmdb51'):
            from .hmdb51 import HMDB51
            ds = HMDB51(
                self.cfg.dataset.root,
                self.cfg.dataset.annotation_path,
                fold=self.cfg.dataset.fold,
                split=split,
            )
        else:
            raise ValueError(f'Unknown dataset "{ds_name}"')
        # transform pipline
        temporal_transform = self.get_temporal_transform(split)
        cpu_transform, gpu_transform = self.get_transform(split)

        return ds, cpu_transform, gpu_transform, temporal_transform

    def build(self, split='train', device=None, pretrain=True):
        stat = 'pretrain' if pretrain else 'finetune'
        ds_name = self.cfg.dataset.name
        self.logger.info(
            f'Building Dataset:{stat} on {ds_name}, Split={split}')

        if pretrain:
            ds, cpu_transform, gpu_transform, temporal_transform = self.build_pretrain(
                ds_name)
        else:
            ds, cpu_transform, gpu_transform, temporal_transform = self.build_finetune(
                ds_name, split=split)

        gpu_collate_fn = transforms_tensor.SequentialGPUCollateFn(
            transform=gpu_transform,
            target_transform=(not pretrain),
            device=device,
        )

        video_dataset = VideoDataset(
            samples=ds,
            logger=self.logger,
            temporal_transform=temporal_transform,
            spatial_transform=cpu_transform,
            num_clips_per_sample=2 if pretrain else 1,
            frame_rate=self.cfg.temporal_transforms.frame_rate,
        )

        sampler = DistributedSampler(ds, shuffle=(split == 'train'))

        if split == 'train':
            batch_size = self.cfg.batch_size
        elif self.final_validate:
            batch_size = self.cfg.final_validate.batch_size
        else:
            batch_size = self.cfg.validate.batch_size

        dl = DataLoader(
            video_dataset,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            sampler=sampler,
            drop_last=(split == 'train'),
            collate_fn=identity,
            multiprocessing_context=mp.get_context('fork'),
        )

        return MainProcessCollateWrapper(dl, gpu_collate_fn)

    def get_transform_pretrain(self):
        aug_plus = self.cfg.aug_plus

        st_cfg = self.cfg.spatial_transforms
        size = st_cfg.size

        cpu_transform = transforms_tensor.Compose([
            transforms_spatial.RawVideoRandomCrop(scale=(0.4, 1.0)),
        ])

        normalize = transforms_spatial.Normalize(
            self.cfg.dataset.mean,
            self.cfg.dataset.std,
            inplace=True,
        )

        if not aug_plus:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.4,
                ),
                transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])
        else:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomApply(
                    [transforms_spatial.ColorJitter(
                        0.4,
                        0.4,
                        0.4,
                        0.1,
                    )], p=0.8),
                transforms_spatial.RandomGrayScale(p=0.2),
                transforms_spatial.RandomApply(
                    [transforms_tensor.GaussianBlur((3, 3), (1.5, 1.5)).cuda()], p=0.5),
                transforms_spatial.RandomHorizontalFlip(),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_transform(self, split='train'):
        normalize = transforms_spatial.Normalize(
            self.cfg.dataset.mean,
            self.cfg.dataset.std,
            inplace=True,
        )

        st_cfg = self.cfg.spatial_transforms
        size = st_cfg.size
        if split == 'train':
            cpu_transform = transforms_tensor.Compose([
                transforms_spatial.RawVideoRandomCrop(scale=(
                    st_cfg.crop_area.min,
                    st_cfg.crop_area.max,
                )),
            ])
        else:
            cpu_transform = transforms_tensor.Compose(
                [transforms_spatial.RawVideoCenterMaxCrop()])

        if split == 'train':
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                transforms_spatial.RandomGrayScale(p=st_cfg.gray_scale),
                transforms_spatial.ColorJitter(
                    brightness=st_cfg.color_jitter.brightness,
                    contrast=st_cfg.color_jitter.contrast,
                    saturation=st_cfg.color_jitter.saturation,
                    hue=st_cfg.color_jitter.hue,
                ),
                transforms_spatial.RandomHorizontalFlip(st_cfg.h_flip),
                normalize,
            ])
        else:
            gpu_transform = transforms_tensor.Compose([
                transforms_spatial.ToTensor(),
                transforms_spatial.Resize(size),
                normalize,
            ])

        return cpu_transform, gpu_transform

    def get_temporal_transform(self, split):
        tt_cfg = self.cfg.temporal_transforms
        size = tt_cfg.size
        tt_type = tt_cfg.get('type', 'clip')
        self.logger.info('Temporal transform type: %s', tt_type)

        if split == 'train':
            if tt_type == 'clip':
                crop = transforms_temporal.RandomStrideCrop(
                    size=size,
                    strides=tt_cfg.strides,
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(size=size)
            else:
                raise ValueError(
                    f'Unknown temporal_transforms.type "{tt_type}"')
        elif split in ['val', 'test']:
            if self.final_validate:
                n = tt_cfg.validate.final_n_crop
            else:
                n = tt_cfg.validate.n_crop

            if tt_type == 'clip':
                crop = transforms_temporal.EvenNCrop(
                    size=size,
                    stride=tt_cfg.validate.stride,
                    n=n,
                )
            elif tt_type == 'cover':
                crop = transforms_temporal.Cover(
                    size=size,
                    n_crop=n,
                )
            else:
                raise ValueError(
                    f'Unknown temporal_transforms.type "{tt_type}"')
        else:
            raise ValueError(f'Unknown split "{split}"')
        return crop
