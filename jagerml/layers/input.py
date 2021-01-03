#!/usr/bin/env python3

from jagerml.helper import *


class Input:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = inputs
