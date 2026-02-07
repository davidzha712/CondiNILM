"""Dilated convolution embedding layers -- CondiNILM.

Author: Siyi Li
"""
import torch
import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 8,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=k,
                dilation=dilation,
                stride=stride,
                bias=bias,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm1d(c_out),
        )
        if c_in > 1 and c_in != c_out:
            self.match_residual = True
            self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=1)
        else:
            self.match_residual = False

    def forward(self, x) -> torch.Tensor:
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)

            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    def __init__(
        self,
        c_in: int = 32,
        c_out: int = 32,
        kernel_size: int = 8,
        dilation_list: list = [1, 2, 4, 8],
        bias: bool = True,
    ):
        super().__init__()

        layers = []
        for i, dilation in enumerate(dilation_list):
            if i == 0:
                layers.append(
                    ResUnit(c_in, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
            else:
                layers.append(
                    ResUnit(c_out, c_out, k=kernel_size, dilation=dilation, bias=bias)
                )
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.network(x)
