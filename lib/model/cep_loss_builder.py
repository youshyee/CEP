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
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
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


def norm_dis(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1)
    return loss.mean()


class Lossv2(nn.Module):
    '''
    CL loss is norm dis loss like byol
    '''

    def __init__(self, A=1.0, CL=1.0, CC=1.0, **kwargs):
        super().__init__()
        self.A = A
        self.CL = CL
        self.CC = CC

        self.CLloss = norm_dis
        self.CCloss = norm_dis

    def forward(
        self,
        Alogits,
        mask,
        CL,
        CC,  # tuple
    ):
        # A task
        # Aloss = Alogits.mean()
        Aloss = MultiInfoNCELoss2(Alogits, mask)
        # CL

        CLLoss = 0
        for cl in CL:
            CLLoss += self.CLloss(cl[0], cl[1])

        CLLoss /= 6

        # CC loss

        CCLoss = 0
        for cc in CC:
            CCLoss += self.CCloss(cc[0], cc[1])

        CCLoss /= 6

        Aloss = self.A * Aloss
        CLLoss = self.CL * CLLoss

        CLLoss_all = Aloss+CLLoss

        CCLoss = self.CC * CCLoss

        loss = CLLoss_all + CCLoss

        return loss, CLLoss_all, CCLoss


class Loss(nn.Module):
    '''
    CL loss is contrast loss version
    '''

    def __init__(self, A=1.0, CL=1.0, CC=1.0, CC_type='l1', contrastive=False, cllambda=0.5):
        super().__init__()
        assert CC_type in ['l1', 'l2', 'cos']
        self.A = A
        self.cllambda = cllambda

        self.CL = CL
        self.CC = CC
        self.contrastive = contrastive
        self.CLloss = nn.MSELoss()
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
        CL_1,  # second is k
        CL_2,
        CC_1,  # tuple
        CC_2,
    ):
        # A task
        # Aloss = Alogits.mean()
        Aloss = MultiInfoNCELoss2(Alogits, mask)
        # CL
        # ranking_target = torch.ones(ranking_logits1[0].size(0),
        #                             dtype=torch.long,
        #                             device=ranking_logits1[0].device)
        # CLRankLoss1 = self._margin_ranking_loss(ranking_logits1[0], ranking_logits1[1],
        #                                         ranking_target)
        # CLRankLoss2 = self._margin_ranking_loss(ranking_logits2[0], ranking_logits2[1],
        #                                         ranking_target)

        CLLoss1 = self.CLloss(CL_1[0], CL_1[1])
        CLLoss2 = self.CLloss(CL_2[0], CL_2[1])
        if self.contrastive:
            CLLoss1 = CLLoss1 - self.cllambda * self.CLloss(CL_1[0], CL_2[1])
            CLLoss2 = CLLoss2 - self.cllambda * self.CLloss(CL_2[0], CL_1[1])

        CLLoss = 0.5 * (CLLoss1 + CLLoss2)
        # CC loss
        dis1 = self.dis(CC_1[0], CC_1[1])
        dis2 = self.dis(CC_2[0], CC_2[1])
        CC_dis = 0.5 * (dis1 + dis2)

        Aloss = self.A * Aloss
        CLLoss = self.CL * CLLoss
        CC_dis = self.CC * CC_dis
        loss = Aloss + CLLoss + CC_dis
        return loss, Aloss, CLLoss, CC_dis


class Predictor(torch.nn.Module):
    def __init__(self, opdim, layer=4, meddim=None, usebn=True, lastusebn=False, usenoise=False):
        super(Predictor, self).__init__()
        self.usenoise = usenoise
        if usenoise:
            opdim = opdim+opdim//2
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
        if self.usenoise:
            x = torch.cat(
                [x, torch.randn(x.size(0), x.size(1)//2).to(x.device)], dim=1)
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
        self.encoder_k = base_encoder(
            num_classes=dim)  # no grad for this branch

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
    def _forward_encoder_k(self, im_k):
        # shuffle for making use of BN

        k_A, k_C = self.encoder_k(im_k)  # keys: NxC # selected only for A task

        return k_A, k_C

    @torch.no_grad()
    def selecting_k_op(self, im_k):
        B, C, T, H, W = im_k.shape
        random_indices = torch.randperm(T, device='cpu')
        selected = random_indices[:int(T // 3)].sort()[0]  #
        selected = selected.to(im_k.device)

        im_k_selected = torch.empty(B, C, T // 3, H, W, device=im_k.device)

        im_k_selected = im_k.index_select(2, selected)

        return im_k_selected

    @torch.no_grad()
    def split_op(self, im):
        B, C, T, H, W = im.shape
        assert T % 3 == 0
        temporal_stride = T//3
        im_q_p = torch.empty(B, C, temporal_stride, H, W, device=im.device)
        im_q_c = torch.empty(B, C, temporal_stride, H, W, device=im.device)
        im_q_f = torch.empty(B, C, temporal_stride, H, W, device=im.device)

        im_q_p = im[:, :, :temporal_stride, ...].clone()
        im_q_c = im[:, :, temporal_stride:2*temporal_stride, ...].clone()
        im_q_f = im[:, :, 2*temporal_stride:, ...].clone()
        return im_q_p, im_q_c, im_q_f

    def forward(self, im_q, im_k):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k_selected = self.selecting_k_op(im_k)

            # split to get past current future frames
            im_q_p, im_q_c, im_q_f = self.split_op(im_q)

            A_k, _ = self._forward_encoder_k(im_k_selected)

            # using k img
            _, C_k_p = self._forward_encoder_k(im_q_p)
            _, C_k_c = self._forward_encoder_k(im_q_c)
            _, C_k_f = self._forward_encoder_k(im_q_f)

        # compute query features
        A_q_p, C_q_p = self.encoder_q(im_q_p)
        A_q_c, C_q_c = self.encoder_q(im_q_c)
        A_q_f, C_q_f = self.encoder_q(im_q_f)

        # forward prediction
        C_q_c_ = self.forward_predictor(C_q_p)
        C_q_f_ = self.forward_predictor(C_q_c)
        C_q_f__ = self.forward_predictor(self.forward_predictor(C_q_p))

        # backward prediction
        C_q_c__ = self.backward_predictor(C_q_f)
        C_q_p_ = self.backward_predictor(C_q_c)
        C_q_p__ = self.backward_predictor(self.backward_predictor(C_q_f))

        # A task apperance discrimination task
        A_q = torch.stack([A_q_p, A_q_c, A_q_f], dim=-1)  # B,C,3
        batch_size = A_q.size(0)
        pos_A = torch.einsum('ncm,nc->nm', [A_q, A_k])
        neg_A = torch.einsum('ncm,ck->nkm', [A_q, self.queue.clone().detach()])

        pos_A /= self.T
        neg_A /= self.T

        # logits: Nx(m+km) # first m is positive and the other negs
        logits_A = torch.cat([pos_A, neg_A.reshape(batch_size, -1)], dim=1)
        # B x (3+3K)
        mask_A = torch.zeros(batch_size, logits_A.size(
            1), dtype=torch.long, device=logits_A.device)
        mask_A[:, :3] = 1
        print('logitA shape:', logits_A.shape)

        # C task cycle task

        CL_p = (C_q_p_, C_k_p.detach())
        CL_p_ = (C_q_p__, C_k_p.detach())

        CL_c = (C_q_c_, C_k_c.detach())
        CL_c_ = (C_q_c__, C_k_c.detach())

        CL_f = (C_q_f_, C_k_f.detach())
        CL_f_ = (C_q_f__, C_k_f.detach())

        CL = [CL_p, CL_p_, CL_c, CL_c_, CL_f, CL_f_]

        # cycle consistency of 6 possible cycles
        C_q_b_p = self.backward_predictor(C_q_c_)
        C_q_b_c = self.backward_predictor(C_q_f_)
        C_q_b_p_ = self.backward_predictor(self.backward_predictor(C_q_f__))

        C_q_b_f = self.forward_predictor(C_q_c__)
        C_q_b_c_ = self.forward_predictor(C_q_p_)
        C_q_b_f_ = self.forward_predictor(self.forward_predictor(C_q_p__))

        CC_p = (C_k_p.detach(), C_q_b_p)
        CC_p_ = (C_k_p.detach(), C_q_b_p_)

        CC_c = (C_k_c.detach(), C_q_b_c)
        CC_c_ = (C_k_c.detach(), C_q_b_c_)

        CC_f = (C_k_f.detach(), C_q_b_f)
        CC_f_ = (C_k_f.detach(), C_q_b_f_)

        CC = [CC_p, CC_p_, CC_c, CC_c_, CC_f, CC_f_]
        # dequeue and enqueue
        self._dequeue_and_enqueue(A_k)

        return logits_A, mask_A, CL, CC
