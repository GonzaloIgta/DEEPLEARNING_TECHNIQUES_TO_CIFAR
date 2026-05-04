"""
SimpleCNN - Complete CNN Architecture

Assignment 2 - Deep Learning Course, University of Deusto

Build a CNN for CIFAR-10 classification using YOUR custom layers
(Conv2d, MaxPool2d, Flatten) plus PyTorch's nn.Linear and nn.ReLU.
"""

import torch
import torch.nn as nn

from .conv2d import Conv2d
from .pooling import MaxPool2d
from .flatten import Flatten


class SimpleCNN(nn.Module):
    """
    A simple CNN for image classification on CIFAR-10.

    Required architecture:
        Conv2d(3, 16, kernel_size=3, padding=1) -> ReLU
        MaxPool2d(kernel_size=2, stride=2)
        Conv2d(16, 32, kernel_size=3, padding=1) -> ReLU
        MaxPool2d(kernel_size=2, stride=2)
        Flatten
        Linear(32 * 8 * 8, 128) -> ReLU
        Linear(128, num_classes)

    Note: After two MaxPool2d(2,2) on a 32x32 input:
        32 -> 16 -> 8, so the flattened size is 32 * 8 * 8 = 2048.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # TODO: Define the layers following the architecture above.
        # Use YOUR Conv2d, MaxPool2d, and Flatten (imported above).
        # Use nn.Linear and nn.ReLU from PyTorch.
        
        self.features = nn.Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            MaxPool2d(2, 2),

            Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32)

        Returns:
            Logits tensor of shape (N, num_classes)
        """
        # TODO: Implement the forward pass through all layers
        x = self.features(x)
        x= self.classifier(x)
        return x  
        #Como hemos usado el nn.Sequential hace el proceso de iterar sobre toda las layers activandolas de una en una

        



