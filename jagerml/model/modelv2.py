#!/usr/bin/env python3


from time import time
from collections import OrderedDict
from jagerml.helper import *
from jagerml.layers import Input, Flatten, DenseLayer
from jagerml.activations import Softmax, SoftmaxLossCrossentropy, ReLUbase
from jagerml.evaluate import LossCategoricalCrossentropy


class ModelV2:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def _init_params(self):
        pass

    def _build_encoder(self):
        """
        CNN encoder
        """
        self.encoder = OrderedDict()
        self.encoder["Flatten"] = Flatten(optimizer=self.optimizer)
        self.encoder["Dense1"] = DenseLayer(
            n_out=self.latent_dim,
            act_fn=ReLUbase(),
            optimizer=self.optimizer
        )
        self.encoder["Dense2"] = DenseLayer(
            n_out=self.T * 2,
            optimizer=self.optimizer,
            act_fn=ReLUbase(),
            init=self.init,
        )
        # self.encoder["Conv1"] = Conv2D(
        #     act_fn=ReLUbase(),
        #     init=self.init,
        #     pad=self.enc_conv1_pad,
        #     optimizer=self.optimizer,
        #     out_ch=self.enc_conv1_out_ch,
        #     stride=self.enc_conv1_stride,
        #     kernel_shape=self.enc_conv1_kernel_shape,
        # )

    def _build_decoder(self):
        """
        CNN decoder
        """
        self.decoder = OrderedDict()
        self.decoder["FC1"] = DenseLayer(
            act_fn=ReLUbase(),
            init=self.init,
            n_out=self.latent_dim,
            optimizer=self.optimizer,
        )
        # NB. `n_out` is dependent on the dimensionality of X. we use a
        # placeholder for now, and update it within the `forward` method
        self.decoder["FC2"] = DenseLayer(
            n_out=None,
            act_fn=ReLUbase(),
            optimizer=self.optimizer,
            init=self.init
        )

    @property
    def parameters(self):
        return {}

    @property
    def hyperparameters(self):
        return {}

    @property
    def derived_variabled(self):
        return {}

    @property
    def gradients(self):
        return {}

    def _sample(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def update(self):
        pass

    def flush_gradients(self):
        pass

    def fit(self):
        pass
