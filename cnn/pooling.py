"""
Pooling Layers - MaxPool2d and AvgPool2d (from scratch)

Assignment 2 - Deep Learning Course, University of Deusto

You must implement the forward pass WITHOUT using nn.MaxPool2d,
nn.AvgPool2d, or nn.functional.max_pool2d / avg_pool2d.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MaxPool2d(nn.Module):
    """
    2D Max Pooling layer implemented from scratch.

    Takes the MAXIMUM value within each pooling window.

    Args:
        kernel_size: Size of the pooling window (int or tuple)
        stride:      Step size (default: same as kernel_size)
        padding:     Zero-padding (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.padding = padding

    def forward(self, x):
        """
        Forward pass of max pooling.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, C, H_out, W_out)

        Hint: For each window position, extract the patch and take torch.max().
              You can use nested loops or F.unfold (applied per-channel).
        """
        # TODO: Implement max pooling forward pass
        N, C, H, W = x.shape
        H_out = math.floor((H + 2*self.padding - self.kernel_size[0]) / self.stride[0]) + 1
        W_out = math.floor((W + 2*self.padding - self.kernel_size[1]) / self.stride[1]) + 1
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unf = x_unf.view(N, C, self.kernel_size[0] * self.kernel_size[1], H_out * W_out)
        out = x_unf.max(dim=2).values  
        return out.view(N, C, H_out, W_out)       

        
        
        


class AvgPool2d(nn.Module):
    """
    2D Average Pooling layer implemented from scratch.

    Takes the MEAN value within each pooling window.

    Args:
        kernel_size: Size of the pooling window (int or tuple)
        stride:      Step size (default: same as kernel_size)
        padding:     Zero-padding (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if stride is None:
            stride = kernel_size
            
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        self.padding = padding

    def forward(self, x):
        """
        Forward pass of average pooling.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Output tensor of shape (N, C, H_out, W_out)

        Hint: Same structure as MaxPool2d, but use torch.mean() instead of max.
        """
        # TODO: Implement average pooling forward pass
        N, C, H, W = x.shape

        H_out = math.floor((H + 2*self.padding - self.kernel_size[0]) / self.stride[0]) + 1
        W_out = math.floor((W + 2*self.padding - self.kernel_size[1]) / self.stride[1]) + 1
        
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        x_unf = x_unf.view(N, C, self.kernel_size * self.kernel_size, H_out * W_out)
        out = x_unf.mean(dim=2)
        return out.view(N, C, H_out, W_out)                

        


