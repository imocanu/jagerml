#!/usr/bin/env python3

from jagerml.helper import *
from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, optimizer=None):
        """An abstract base class inherited by all neural network layers"""
        self.X = []
        self.act_fn = None
        self.trainable = True
        # self.optimizer = OptimizerInitializer(optimizer)()

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


