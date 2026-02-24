"""Dilated convolution embedding layers for CondiNILM.

Author: Siyi Li
"""
import torch
import torch.nn as nn


class ResUnit(nn.Module):
    """Residual 1-D convolution unit: Conv1d -> GELU -> BatchNorm1d.

    Uses a 1x1 convolution to match the residual dimension when c_in > 1 and
    c_in != c_out. When c_in == 1 or c_in == c_out, the residual is added
    directly (identity shortcut).

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        k: Kernel size for the main convolution.
        dilation: Dilation factor for the main convolution.
        stride: Stride for the main convolution.
        bias: Whether to use bias in the convolution.
    """

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
        """Apply convolution block and add residual.

        Args:
            x: (B, c_in, L) input tensor.

        Returns:
            (B, c_out, L) output tensor.
        """
        if self.match_residual:
            x_bottleneck = self.conv(x)
            x = self.layers(x)

            return torch.add(x_bottleneck, x)
        else:
            return torch.add(x, self.layers(x))


class DilatedBlock(nn.Module):
    """Sequential stack of ResUnit layers with increasing dilation rates.

    The first ResUnit handles the channel change from c_in to c_out.
    Subsequent ResUnits maintain c_out channels with progressively larger
    dilation factors from dilation_list.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels (used for all layers after the first).
        kernel_size: Kernel size for all convolutions.
        dilation_list: List of dilation factors, one per ResUnit.
        bias: Whether to use bias in convolutions.
    """

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
        """Pass input through the sequential dilated residual stack.

        Args:
            x: (B, c_in, L) input tensor.

        Returns:
            (B, c_out, L) output tensor.
        """
        return self.network(x)
