from collections import OrderedDict
import numpy as np
import random
import torch


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


class ReplayBuffer():
    def __init__(self, max_size=10):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push(self, data_in):
        assert data_in.device == torch.device('cpu')
        data_in = data_in.detach()
        if len(self.data) < self.max_size:
            self.data.append(data_in)
        else:
            self.data.pop(0)
            self.data.append(data_in)
            assert len(self.data) <= self.max_size

    def pop(self):
        if len(self.data) == 0:  # no data
            return False, None
        else:
            index = random.randint(0, len(self.data) - 1)
            return True, self.data[index].clone()


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs -
                                                                             self.decay_start_epoch)


class LogBuffer(object):
    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
