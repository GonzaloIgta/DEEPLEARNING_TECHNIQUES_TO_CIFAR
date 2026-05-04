"""
CNN Components Package - Assignment 2
Deep Learning Course, University of Deusto

Implements core CNN components from scratch using PyTorch.
You are NOT allowed to use nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d.
"""

from .conv2d import Conv2d
from .pooling import MaxPool2d, AvgPool2d
from .flatten import Flatten
from .model import SimpleCNN
