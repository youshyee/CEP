import torch
from torch import Tensor
from typing import *
from functools import partial


def objectinlist(element, l):
    return True if element in l else False


@torch.no_grad()
def accuracy(output: Tensor, target: Tensor, topk=(1, )) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float)
        res.append(correct_k * (100.0 / batch_size))
    return res


@torch.no_grad()
def multiInstacneaccuracy(output: Tensor, target, topk=(1, )) -> List[Tensor]:
    # target: a list of value in valid

    maxk = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    func = partial(objectinlist, l=target)
    device = pred.device

    predcp = pred.cpu()
    predcp.apply_(func)
    predcp.to(device)
    correct = predcp

    res = []
    for k in topk:
        correct_k = correct[:k].bool()
        correct_k = correct_k.sum(dim=0).bool()
        correct_k = correct_k.long().sum()

        res.append(correct_k * (100.0 / batch_size))
    return res


@torch.jit.script
def top5_accuracy(output: Tensor, target: Tensor) -> List[Tensor]:
    topk = (1, 5)
    maxk = 5
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float)
        res.append(correct_k * (100.0 / batch_size))
    return res


def binary_accuracy(output: Tensor, target: Tensor) -> Tensor:
    batch_size = target.shape[0]
    pred = output > 0.5
    correct = pred.eq(target).sum()
    return correct * (100.0 / batch_size)


if __name__ == "__main__":
    # print(multiInstacneaccuracy(torch.randn(10, 10), target=[0, 1], topk=(1, 2, 3)))

    accuracy()
