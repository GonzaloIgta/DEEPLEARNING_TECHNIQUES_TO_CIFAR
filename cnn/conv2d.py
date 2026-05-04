"""
Conv2d - 2D Convolutional Layer (from scratch)

Assignment 2 - Deep Learning Course, University of Deusto

You must implement the forward pass of 2D convolution WITHOUT using
nn.Conv2d or nn.functional.conv2d. You CAN use:
  - torch.nn.functional.pad (for padding)
  - torch.nn.functional.unfold (for extracting patches)
  - Standard tensor operations (matmul, sum, view, reshape, etc.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    """
    2D Convolutional layer implemented from scratch.

    Args:
        in_channels:  Number of input channels (e.g. 3 for RGB)
        out_channels: Number of output channels (number of filters)
        kernel_size:  Size of each filter (int or tuple)
        stride:       Step size when sliding the filter (default: 1)
        padding:      Zero-padding added to input borders (default: 0)
        bias:         Whether to include a bias term (default: True)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()

        # Normalize kernel_size to tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable parameters ------------------------------------------------
        # Weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels,
                        kernel_size[0], kernel_size[1])
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Kaiming (He) initialization -----------------------------------------
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass of the 2D convolution.

        Args:
            x: Input tensor of shape (N, C_in, H, W)
            Cin = canales de entrada

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)

        Where:
            H_out = floor((H + 2*padding - K_h) / stride) + 1
            W_out = floor((W + 2*padding - K_w) / stride) + 1

        Hints:
            - Apply padding first using F.pad
            - Two approaches:
              (a) F.unfold to extract patches, then matrix multiply with
                  reshaped weights  (efficient, recommended)
              (b) Nested loops over output positions  (simpler, slower)
        """
                
        N, Cin, H, W = x.shape

        H_out = math.floor((H + 2*self.padding - self.kernel_size[0]) / self.stride) + 1
        W_out = math.floor((W + 2*self.padding - self.kernel_size[1]) / self.stride) + 1

        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        w = self.weight.view(self.out_channels, -1)

        out = w @ x_unfold

        if self.bias is not None:
            out += self.bias.view(1, -1, 1)

        return out.view(N, self.out_channels, H_out, W_out)


         

