#!/usr/bin/env python3

from jagerml.helper import *


class Input:

    def forward(self, inputs, training):
        self.output = inputs
