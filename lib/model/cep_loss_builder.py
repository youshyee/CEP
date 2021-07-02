import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import *
import random


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# @torch.jit.script
# def MultiInfoNCELoss(logits, mask):
#     loss = -torch.log((F.softmax(logits, dim=1) * mask).sum(1))
#     return loss.mean()


@torch.jit.script
def MultiInfoNCELoss2(logits, mask):
    loss = torch.sum(-logits.log_softmax(dim=1) * mask, dim=1)
    return loss.mean()


def cosloss(input, target):
    # assume input and target are normalize tensors
    loss = torch.einsum('nc,nc->n', [input, target])
    loss = loss.mean()
    return loss


class Loss(nn.Module):
    def __init__(self, A=1.0, CL=1.0, CC=1.0, CC_type='l1'):
        super().__init__()
        assert CC_type in ['l1', 'l2', 'cos']
        self.A = A

        self.CL = CL
        self.CC = CC
        self.BYOLloss = nn.MSELoss()
        if CC_type == 'l1':
            self.dis = nn.L1Loss()
        elif CC_type == 'l2':
            self.dis = nn.MSELoss()
        else:
            self.dis = cosloss

    def forward(
        self,
        Alogits,
        mask,
        CL_1,
        CL_2,
        CC_1,  # tuple
        CC_2,
    ):
        #A task
        # Aloss = Alogits.mean()
        Aloss = MultiInfoNCELoss2(Alogits, mask)
        #CL
        # ranking_target = torch.ones(ranking_logits1[0].size(0),
        #                             dtype=torch.long,
        #                             device=ranking_logits1[0].device)
        # CLRankLoss1 = self._margin_ranking_loss(ranking_logits1[0], ranking_logits1[1],
        #                                         ranking_target)
        # CLRankLoss2 = self._margin_ranking_loss(ranking_logits2[0], ranking_logits2[1],
        #                                         ranking_target)

        CLLoss1 = self.BYOLloss(CL_1[0], CL_1[1])
        CLLoss2 = self.BYOLloss(CL_2[0], CL_2[1])
        CLLoss = 0.5 * (CLLoss1 + CLLoss2)
        #CC loss
        dis1 = self.dis(CC_1[0], CC_1[1])
        dis2 = self.dis(CC_2[0], CC_2[1])
        CC_dis = 0.5 * (dis1 + dis2)

        Aloss = self.A * Aloss
        CLLoss = self.CL * CLLoss
        CC_dis = self.CC * CC_dis
        loss = Aloss + CLLoss + CC_dis
        return loss, Aloss, CLLoss, CC_dis


class Predictor(torch.nn.Module):
    def __init__(self, opdim, layer=4, meddim=None, usebn=True, lastusebn=False):
        super(Predictor, self).__init__()
        assert layer >= 2
        if meddim is None:
            meddim = opdim

        models = []
        for _ in range(layer - 1):
            models.append(nn.Linear(opdim, meddim))
            if usebn:
                models.append(nn.BatchNorm1d(meddim))
            models.append(nn.ReLU(inplace=True))
        # last layer
        models.append(nn.Linear(meddim, opdim))

        if lastusebn:
            models.append(nn.BatchNorm1d(opdim))

        self.model = nn.Sequential(*models)

    def forward(self, x):
        x = self.model(x)
        return x


class CepLossBuilder(nn.Module):
    def __init__(
            self,
            base_encoder,
            logger,
            dim=128,
            K=65536,
            m=0.999,
            T=0.07,
            mlp=False,
            predictor_layer=[4, 4],  # first for forward, second for backward
            predictor_meddim=None,  # when None meddim==dim
            predictor_usebn=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        self.logger = logger

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)  # no grad for this branch

        # predictors
        self.forward_predictor = Predictor(opdim=dim,
                                           layer=predictor_layer[0],
                                           meddim=predictor_meddim,
                                           usebn=predictor_usebn)
        self.backward_predictor = Predictor(opdim=dim,
                                            layer=predictor_layer[1],
                                            meddim=predictor_meddim,
                                            usebn=predictor_usebn)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                              self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                              self.encoder_k.fc)

        # init k same with q
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # init pointer for queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _forward_encoder_k(self, im_k):
        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k_A, k_C = self.encoder_k(im_k)  # keys: NxC # selected only for A task

        # undo shuffle
        k_A = self._batch_unshuffle_ddp(k_A, idx_unshuffle)
        k_C = self._batch_unshuffle_ddp(k_C, idx_unshuffle)

        return k_A, k_C

    @torch.no_grad()
    def selecting_k_op(self, im_k):
        B, C, T, H, W = im_k.shape
        random_indices = torch.randperm(T, device='cpu')
        selected = random_indices[:int(T // 2)].sort()[0]  # half using first
        selected = selected.to(im_k.device)

        im_k_selected = torch.empty(B, C, T // 2, H, W, device=im_k.device)

        im_k_selected = im_k.index_select(2, selected)

        return im_k_selected

    @torch.no_grad()
    def half_selecting_op(self, im):
        B, C, T, H, W = im.shape
        im_q_1 = torch.empty(B, C, T // 2, H, W, device=im.device)
        im_q_2 = torch.empty(B, C, T // 2, H, W, device=im.device)
        im_q_1 = im[:, :, :T // 2, ...].clone()
        im_q_2 = im[:, :, T // 2:, ...].clone()
        return im_q_1, im_q_2

    def forward(self, im_q, im_k):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k_selected = self.selecting_k_op(im_k)

            im_q_1, im_q_2 = self.half_selecting_op(im_q)
            im_k_1, im_k_2 = self.half_selecting_op(im_k)

            A_k, _ = self._forward_encoder_k(im_k_selected)
            _, C_k_1 = self._forward_encoder_k(im_k_1)
            _, C_k_2 = self._forward_encoder_k(im_k_2)

        # compute query features
        A_q_1, C_q_1, std = self.encoder_q(im_q_1, getstd=True)
        A_q_2, C_q_2 = self.encoder_q(im_q_2)

        # A_q_2_ = self.forward_predictor(A_q_1)
        C_q_2_ = self.forward_predictor(C_q_1)

        # A_q_1_ = self.backward_predictor(A_q_2)
        C_q_1_ = self.backward_predictor(C_q_2)

        # A task apperance discrimination task
        A_q = torch.stack([A_q_1, A_q_2], dim=-1)  # B,C,2
        batch_size = A_q.size(0)
        pos_A = torch.einsum('ncm,nc->nm', [A_q, A_k])
        neg_A = torch.einsum('ncm,ck->nkm', [A_q, self.queue.clone().detach()])

        pos_A /= self.T
        neg_A /= self.T

        # logits: Nx(m+km) # first m is positive and the other negs
        logits_A = torch.cat([pos_A, neg_A.reshape(batch_size, -1)], dim=1)
        mask_A = torch.zeros(batch_size, logits_A.size(1), dtype=torch.long, device=logits_A.device)
        mask_A[:, :2] = 1

        # C task cycle task: 1. cycle consistency 2. BYOL loss

        #  BYOL
        CL_1 = (C_q_1_, C_k_1)
        CL_2 = (C_q_2_, C_k_2)

        # cycle consistency
        C_q_1__ = self.backward_predictor(C_q_2_)
        C_q_2__ = self.forward_predictor(C_q_1_)
        CC_1 = (C_q_1, C_q_1__)
        CC_2 = (C_q_2, C_q_2__)

        # dequeue and enqueue
        self._dequeue_and_enqueue(A_k)

        return logits_A, mask_A, CL_1, CL_2, CC_1, CC_2, std
