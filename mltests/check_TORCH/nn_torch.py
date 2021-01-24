#!/usr/bin/env python3

import time
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(156)
tf.random.set_seed(156)
torch.random.manual_seed(156)


def convert_to_torch_tensor(var, requires_grad=True):
    return torch.autograd.Variable(torch.FloatTensor(var), requires_grad=requires_grad)


class TorchDenseLayer(nn.Module):
    def __init__(self, n_in, n_hid, act_fn, params, **kwargs):
        super(TorchDenseLayer, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hid)

        self.layer1.weight = nn.Parameter(torch.FloatTensor(params["W"].T))
        self.layer1.bias = nn.Parameter(torch.FloatTensor(params["b"]))

        self.act_fn = act_fn
        self.model = nn.Sequential(self.layer1, self.act_fn)

    def forward(self, X):
        self.X = X
        if not isinstance(X, torch.Tensor):
            self.X = convert_to_torch_tensor(X)

        self.z1 = self.layer1(self.X)
        self.z1.retain_grad()

        self.out1 = self.act_fn(self.z1)
        self.out1.retain_grad()

    def extract_grads(self, X):
        self.forward(X)
        self.loss1 = self.out1.sum()
        self.loss1.backward()
        grads = {
            "X": self.X.detach().numpy(),
            "b": self.layer1.bias.detach().numpy(),
            "W": self.layer1.weight.detach().numpy(),
            "y": self.out1.detach().numpy(),
            "dLdy": self.out1.grad.numpy(),
            "dLdZ": self.z1.grad.numpy(),
            "dLdB": self.layer1.bias.grad.numpy(),
            "dLdW": self.layer1.weight.grad.numpy(),
            "dLdX": self.X.grad.numpy(),
        }
        return grads
