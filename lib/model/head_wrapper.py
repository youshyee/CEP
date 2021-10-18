import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import *


class Flatten(nn.Module):
    def forward(self, x: Tensor):
        return x.flatten(1)


class ConvFc(nn.Module):
    """
    conv->relu->conv->downsample->linear

    """

    def __init__(self, feat_dim: int, opdim: int, kernel_size: Tuple[int, int, int],
                 padding: Tuple[int, int, int]):
        super().__init__()
        self.conv1 = nn.Conv3d(
            feat_dim, feat_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            feat_dim, feat_dim, kernel_size, padding=padding)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(feat_dim, opdim)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.linear(out)
        return out


class ConvBnFc(nn.Module):
    """
    conv->relu->conv->downsample->linear

    """

    def __init__(self, feat_dim: int, opdim: int, kernel_size: Tuple[int, int, int],
                 padding: Tuple[int, int, int]):
        super().__init__()
        self.conv1 = nn.Conv3d(
            feat_dim, feat_dim, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(feat_dim, opdim)

    def forward(self, x: Tensor):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.flatten(1)
        out = self.linear(out)
        return out


class HeadWrapper(nn.Module):
    """
    The projection layer type can be linear layer and mlp (as indicated in SimCLR).
    """

    def __init__(
            self,
            base_encoder,
            logger,
            num_classes: int = 128,
            finetune: bool = False,
            fc_type: str = 'linear',
            groups: int = 1,
            projectorbn=[False, False],  # usebn for each projector
        pool=True,
    ):
        super().__init__()

        self.logger = logger
        self.logger.info('Using MultiTask Wrapper')
        self.finetune = finetune
        self.op_dim = num_classes
        self.num_classes = num_classes
        self.groups = groups
        self.fc_type = fc_type

        self.logger.warning(f'{self.__class__} using groups: {groups}')

        self.encoder = base_encoder(num_classes=1)
        self.projectorbn = projectorbn

        feat_dim = self._get_feat_dim(self.encoder)
        feat_dim //= groups

        if self.finetune:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(feat_dim, num_classes)
        else:
            if fc_type == 'linear':
                self.fc1 = self._get_linear_fc(
                    feat_dim, self.op_dim, bn=projectorbn[0], pool=pool)
                self.fc2 = self._get_linear_fc(
                    feat_dim, self.op_dim, bn=projectorbn[1], pool=pool)
            elif fc_type == 'mlp':
                self.fc1 = self._get_mlp_fc(
                    feat_dim, self.op_dim, bn=projectorbn[0], pool=pool)
                self.fc2 = self._get_mlp_fc(
                    feat_dim, self.op_dim, bn=projectorbn[1], pool=pool)
            elif fc_type == 'conv':
                if projectorbn[0]:
                    self.fc1 = ConvBnFc(
                        feat_dim, self.op_dim, (3, 3, 3), (1, 1, 1))
                else:
                    self.fc1 = ConvFc(feat_dim, self.op_dim,
                                      (3, 3, 3), (1, 1, 1))
                if projectorbn[1]:
                    self.fc2 = ConvBnFc(
                        feat_dim, self.op_dim, (3, 3, 3), (1, 1, 1))
                else:
                    self.fc2 = ConvFc(feat_dim, self.op_dim,
                                      (3, 3, 3), (1, 1, 1))

    @torch.no_grad()
    def getstd(self, feat):
        out = F.adaptive_avg_pool3d(feat, (1, 1, 1))
        std = out.std(dim=0)
        return std.mean()

    def forward(self, x: Tensor, getstd=False):
        feat: Tensor = self.encoder.get_feature(x)
        if getstd:
            std = self.getstd(feat)

        if self.finetune:
            x3 = self.avg_pool(feat)
            x3 = x3.flatten(1)
            x3 = self.fc(x3)
            return x3
        else:
            if self.groups == 1:
                x1 = self.fc1(feat)
                x2 = self.fc2(feat)
            elif self.groups == 2:
                feat1, feat2 = feat.chunk(2, 1)
                x1 = self.fc1(feat1)
                x2 = self.fc2(feat2)
            else:
                raise Exception
            x1 = F.normalize(x1, dim=1)
            x2 = F.normalize(x2, dim=1)

            if not getstd:
                return x1, x2
            else:
                return x1, x2, std

    def _get_feat_dim(self, encoder):
        fc_names = ['fc', 'new_fc', 'classifier']
        feat_dim = 512
        for fc_name in fc_names:
            if hasattr(encoder, fc_name):
                feat_dim = getattr(encoder, fc_name).in_features
                self.logger.info(
                    f'Found fc: {fc_name} with in_features: {feat_dim}')
                break
        return feat_dim

    @staticmethod
    def _get_linear_fc(feat_dim: int, opdim: int, bn=False, pool=True):
        seq = []
        if pool:
            seq += [
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                Flatten()]
        if bn:
            seq += [nn.LayerNorm(feat_dim)]

        seq += [nn.Linear(feat_dim, opdim)]

        return nn.Sequential(*seq)

    @staticmethod
    def _get_mlp_fc(feat_dim: int, opdim: int, bn=False, pool=True):
        seq = []
        if pool:
            seq += [
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                Flatten()]
        seq += [nn.Linear(feat_dim, feat_dim)]

        if bn:
            seq += [nn.LayerNorm(feat_dim)]

        seq += [nn.ReLU(inplace=True), nn.Linear(feat_dim, opdim)]

        return nn.Sequential(*seq)
