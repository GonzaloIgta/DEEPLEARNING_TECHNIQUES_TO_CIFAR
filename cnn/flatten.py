"""
Flatten Layer

Assignment 2 - Deep Learning Course, University of Deusto

Converts multi-dimensional feature maps into a 1D vector per sample,
bridging convolutional layers and fully connected layers.
"""

import torch
import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten layer: reshapes (N, C, H, W) to (N, C*H*W).

    This is the bridge between convolutional feature maps and
    fully connected (Linear) layers.
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, H, W) or (N, ...)

        Returns:
            Flattened tensor of shape (N, C*H*W)

        Hint: Use x.view() or x.reshape(). Keep the batch dimension (N) intact.
        """
        # TODO: Implement flatten forward pass

        N = x.shape[0]
        
        return x.view(N, -1)

       
