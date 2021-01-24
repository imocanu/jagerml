#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.layers.base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, keep_dim="first", optimizer=None):

        super().__init__(optimizer)
        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {"in_dims": []}

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "Flatten",
            "keep_dim": self.keep_dim,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):

        if retain_derived:
            self.derived_variables["in_dims"].append(X.shape)
        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)
        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdy, retain_grads=True):

        if not isinstance(dLdy, list):
            dLdy = [dLdy]
        in_dims = self.derived_variables["in_dims"]
        out = [dy.reshape(*dims) for dy, dims in zip(dLdy, in_dims)]
        return out[0] if len(dLdy) == 1 else out
