#!/usr/bin/env python3

from jagerml.imports import *

class ReLU:

    def __init__(self):
        pass

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)