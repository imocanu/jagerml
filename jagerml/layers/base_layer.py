#!/usr/bin/env python3

from jagerml.helper import *
from abc import ABC, abstractmethod
from jagerml.initializers import ActivationInitializer, \
    WeightInitializer, \
    OptimizerInitializer


class BaseLayer(ABC):
    def __init__(self, optimizer=None):
        """An abstract base class inherited by all neural network layers"""
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.optimizer = OptimizerInitializer(optimizer)()

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """Perform a forward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """Perform a backward pass through the layer"""
        raise NotImplementedError

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, cur_loss=None):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        self.flush_gradients()

    def set_params(self, summary_dict):
        """
        Set the layer parameters from a dictionary of values.

        Parameters
        ----------
        summary_dict : dict
            A dictionary of layer parameters and hyperparameters. If a required
            parameter or hyperparameter is not included within `summary_dict`,
            this method will use the value in the current layer's
            :meth:`summary` method.

        Returns
        -------
        layer : :doc:`Layer <numpy_ml.neural_nets.layers>` object
            The newly-initialized layer.
        """
        layer, sd = self, summary_dict

        # # collapse `parameters` and `hyperparameters` nested dicts into a single
        # # merged dictionary
        # flatten_keys = ["parameters", "hyperparameters"]
        # for k in flatten_keys:
        #     if k in sd:
        #         entry = sd[k]
        #         sd.update(entry)
        #         del sd[k]
        #
        # for k, v in sd.items():
        #     if k in self.parameters:
        #         layer.parameters[k] = v
        #     if k in self.hyperparameters:
        #         if k == "act_fn":
        #             layer.act_fn = ActivationInitializer(v)()
        #         if k == "optimizer":
        #             layer.optimizer = OptimizerInitializer(sd[k])()
        #         if k not in ["wrappers", "optimizer"]:
        #             setattr(layer, k, v)
        #         if k == "wrappers":
        #             layer = init_wrappers(layer, sd[k])
        return layer

    def summary(self):
        """Return a dict of the layer parameters, hyperparameters, and ID."""
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }


# class Conv2D(BaseLayer):
#     def __init__(self,
#                  out_ch,
#                  kernel_shape,
#                  pad=0,
#                  stride=1,
#                  dilation=0,
#                  act_fn=None,
#                  optimizer=None,
#                  init="glorot_uniform",
#                  ):
#         super().__init__(optimizer)


class DenseLayer(BaseLayer):
    def __init__(self,
                 n_out,
                 act_fn=None,
                 init="glorot_uniform",
                 optimizer=None):
        super().__init__(optimizer)

        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.act_fn = ActivationInitializer(act_fn)()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        b = np.zeros((1, self.n_out))
        W = init_weights((self.n_in, self.n_out))

        self.parameters = {"W": W, "b": b}
        self.derived_variables = {"Z": []}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """Return a dictionary containing the layer hyperparameters."""
        return {
            "layer": "DenseLayer",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        print("[debug] forward")
        if not self.is_initialized:
            self.n_in = X.shape[1]
            print("[debug] start init", self.n_in)
            self._init_params()

        Y, Z = self._fwd(X)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)

        return Y

    def _fwd(self, X):
        """Actual computation of forward pass"""
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = X @ W + b
        Y = self.act_fn(Z)
        return Y, Z

    def backward(self, dLdy, retain_grads=True):
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dw, db = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        """Actual computation of gradient of the loss wrt. X, W, and b"""
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = X @ W + b
        dZ = dLdy * self.act_fn.grad(Z)

        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)
        return dX, dW, dB
