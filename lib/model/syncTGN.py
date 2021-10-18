import torch
import torch.nn as nn


class SyncTGN(torch.nn.Module):
    # use torch.nn.SyncBatchNorm.convert_sync_batchnorm in the module

    def __init__(self, segment, *args, **kwargs):
        super(SyncTGN, self).__init__()
        self.segment = segment
        for i in range(self.segment):
            self.add_module(f'seg_bn_{i}', nn.BatchNorm3d(*args, **kwargs))

    def forward(self, x):
        # bcthw
        B, C, T, H, W = x.shape

        assert T % self.segment == 0, f'temporal size is {T} should be divided by {self.segment}'
        dis = T//self.segment
        output = torch.tensor([], device=x.device)

        for i in range(self.segment):
            a = x[:, :, dis*i:dis*(i+1), ...]
            a = self.__getattr__(f'seg_bn_{i}')(a)
            output = torch.cat([output, a], dim=2)
        return output


# if __name__ == "__main__":
#     i=torch.randn(5,15,48,24,14)
#     bn=torch.nn.BatchNorm3d(15)
#     tgn=SyncTGN(3,15)
#     bnout=bn(i)
#     tgnout=tgn(i)
#     print(bnout.shape,tgnout.shape)
#
#     print(dir(tgn))
