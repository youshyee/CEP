from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import decord
import numpy as np
import torch
from .transforms_video.transforms_temporal import Resample
from torch.utils.data import Dataset

decord.bridge.set_bridge('torch')


@dataclass
class Sample:
    video_path: str
    class_index: int


class VideoDataset(Dataset):
    temporal_transform: Callable
    spatial_transform: Callable
    video_width: int = -1
    video_height: int = -1
    frame_rate: Optional[float] = None

    def __init__(self,
                 samples,
                 logger,
                 temporal_transform=None,
                 spatial_transform=None,
                 video_width=-1,
                 video_height=-1,
                 num_clips_per_sample=2,
                 frame_rate=None):
        """
        For CEP task: num_clips_per_sample = 2
        For finetune 10 crops validation，num_clips_per_sample = 1，but "temporal_transform" will output 10 times longer video
        """
        self.samples = samples
        self.num_clips_per_sample = num_clips_per_sample
        self.video_width = video_width
        self.video_height = video_height
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_rate = frame_rate
        self.logger = logger
        self.resample_fps = Resample(target_fps=frame_rate, logger=self.logger)

        self.logger.info(f'You are using VideoDataset: {self.__class__}')
        self.prev = None

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            vr = decord.VideoReader(
                str(sample.video_path),
                width=self.video_width,
                height=self.video_height,
                num_threads=1,
            )
        except:
            print('no validate video ')
            print(f'{sample.video_path}')
            print('fallback')
            return self.prev

        num_frames = len(vr)
        if num_frames == 0:
            raise Exception(f'Empty video: {sample.video_path}')
        frame_indices = np.arange(num_frames)  # [0, 1, 2, ..., N - 1]

        if self.frame_rate is not None:
            frame_indices = self.resample_fps(frame_indices, vr.get_avg_fps())

        # same temporal frame but different spatial transform
        clip_frame_indices_list = [self.temporal_transform(frame_indices)]
        clip_frame_indices_list = clip_frame_indices_list * self.num_clips_per_sample

        # print('#' * 20)
        # print(clip_frame_indices_list[0])
        # print(clip_frame_indices_list[1])
        # print('#' * 20)

        # Fetch all frames in one `vr.get_batch` call
        clip_frame_indices = np.concatenate(
            clip_frame_indices_list)  # [a1, a2, ..., an, b1, b2, ...,bn]
        clips: torch.Tensor = vr.get_batch(clip_frame_indices)  # [N*T, H, W, C]
        clip_list = clips.chunk(len(clip_frame_indices_list), dim=0)  # List[Tensor[T, H, W, C]]

        clip_list = [self.spatial_transform(clip) for clip in clip_list]
        for c in clip_list:
            assert isinstance(c, torch.Tensor)

        self.prev = (clip_list, sample.class_index)
        return clip_list, sample.class_index

    def __len__(self):
        return len(self.samples)
