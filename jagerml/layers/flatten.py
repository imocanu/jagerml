#!/usr/bin/env python3

from jagerml.helper import *
from jagerml.layers.base import Base


class Flatten(Base):
    def __init__(self, keep_dim="first", optimizer=None):
        """
        Flatten a multidimensional input into a 2D matrix.

        Parameters
        ----------
        keep_dim : {'first', 'last', -1}
            The dimension of the original input to retain. Typically used for
            retaining the minibatch dimension.. If -1, flatten all dimensions.
            Default is 'first'.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.
        """  # noqa: E501
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
        r"""
        Compute the layer output on a single minibatch.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>`
            Input volume to flatten.
        retain_derived : bool
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through wrt. this input. Default
            is True.

        Returns
        -------
        Y : :py:class:`ndarray <numpy.ndarray>` of shape `(*out_dims)`
            Flattened output. If `keep_dim` is `'first'`, `X` is reshaped to
            ``(X.shape[0], -1)``, otherwise ``(-1, X.shape[0])``.
        """
        if retain_derived:
            self.derived_variables["in_dims"].append(X.shape)
        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)
        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdy, retain_grads=True):
        r"""
        Backprop from layer outputs to inputs.

        Parameters
        ----------
        dLdY : :py:class:`ndarray <numpy.ndarray>` of shape `(*out_dims)`
            The gradient of the loss wrt. the layer output `Y`.
        retain_grads : bool
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is
            True.

        Returns
        -------
        dX : :py:class:`ndarray <numpy.ndarray>` of shape `(*in_dims)` or list of arrays
            The gradient of the loss wrt. the layer input(s) `X`.
        """  # noqa: E501
        if not isinstance(dLdy, list):
            dLdy = [dLdy]
        in_dims = self.derived_variables["in_dims"]
        out = [dy.reshape(*dims) for dy, dims in zip(dLdy, in_dims)]
        return out[0] if len(dLdy) == 1 else out
