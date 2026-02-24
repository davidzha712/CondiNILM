"""TSILNet: Temporal Self-attention Improved LSTM Network for NILM.

Combines a Temporal Self-attention TCN (TSTCN) block with an IECA-LSTM block
and a fully connected head. The TSTCN stacks temporal blocks with dilated
causal convolutions and residual self-attention. The IECA-LSTM applies
channel attention via dilated causal convolution before two LSTM layers.
Supports seq2seq and seq2point output modes.
"""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class ResidualSelfAttention(nn.Module):
    def __init__(self, dim):
        super(ResidualSelfAttention, self).__init__()
        self.SA = torch.nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, input):
        x, att_weights = self.SA(
            input.permute(0, 2, 1), input.permute(0, 2, 1), input.permute(0, 2, 1)
        )
        return input + x.permute(0, 2, 1), att_weights


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.rsa1 = ResidualSelfAttention(n_outputs)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.rsa2 = ResidualSelfAttention(n_outputs)

        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.act1, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.act2, self.dropout2)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.gelu = nn.GELU()

    def forward(self, input):
        x = self.net1(input)
        x, att1 = self.rsa1(x)

        x = self.net2(x)
        x, att2 = self.rsa2(x)

        # Calculate the enhanced residual
        res = input if self.downsample is None else self.downsample(input)
        # Add enhanced residual and apply activation
        x = x + res
        return self.gelu(x)


class TSTCN_Block(nn.Module):
    def __init__(
        self, in_channels, num_channels=[4, 16, 64], kernel_size=5, dropout=0.2
    ):
        super(TSTCN_Block, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels_block = in_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels_block,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalConv1D(nn.Module):
    """1D causal convolution: output at time t depends only on inputs at time <= t."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        """Apply causal conv. Input/output shape: (batch, channels, seq_len)."""
        output = self.conv(x)
        if self.padding != 0:
            output = output[:, :, : -self.padding]

        return output


class IECA(nn.Module):
    """Improved Efficient Channel Attention using dilated causal convolution."""

    def __init__(self, kernel_size=3, dilation=8):
        super(IECA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dilated_conv = CausalConv1D(
            1, 1, kernel_size=kernel_size, dilation=dilation
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Compute channel attention weights and scale input. Shape: (B, C, T)."""
        avg_pooled = self.global_avg_pool(x)
        conv_out = self.dilated_conv(avg_pooled.transpose(1, 2))
        attention_weights = self.sigmoid(conv_out.transpose(1, 2))
        return x * attention_weights


class IECA_LSTM(nn.Module):
    """IECA-LSTM block: channel attention via IECA with skip connection, then two LSTM layers."""

    def __init__(self, input_channels, dilation=8, hidden_size=[128, 256], dropout=0.2):
        super(IECA_LSTM, self).__init__()
        self.ieca = IECA(dilation=dilation)
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size[0],
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size[0],
            hidden_size=hidden_size[1],
            num_layers=1,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        """Input/output shape: (B, C, T)."""
        out = self.ieca(x)
        x = x + out
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x.permute(0, 2, 1)


class TSILNet(nn.Module):
    """TSILNet: TSTCN + IECA-LSTM with a fully connected head for NILM."""

    def __init__(
        self,
        c_in,
        window_size=480,
        downstreamtask="seq2seq",
        tcn_channels=[4, 16, 64],
        tcn_kernel_size=5,
        tcn_dropout=0.2,
        lstm_hidden_sizes=[128, 256],
        lstm_dropout=0.2,
        dilation=8,
        head_ffn_dim=512,
        head_dropout=0.2,
    ):
        super(TSILNet, self).__init__()

        self.downstreamtask = downstreamtask

        # TSTCN Block
        self.tstcn = TSTCN_Block(
            in_channels=c_in,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
        )

        # IECA-LSTM Block
        self.ieca_lstm = IECA_LSTM(
            input_channels=tcn_channels[-1],
            dilation=dilation,
            hidden_size=lstm_hidden_sizes,
            dropout=lstm_dropout,
        )

        # Fully connected layers for regression
        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(lstm_hidden_sizes[-1] * window_size, head_ffn_dim),
            nn.Tanh(),
            nn.Dropout(head_dropout),
        )

        if self.downstreamtask == "seq2seq":
            self.fc2 = nn.Linear(head_ffn_dim, window_size)  # Seq2Seq
        else:
            self.fc2 = nn.Linear(head_ffn_dim, 1)  # Seq2Point

    def forward(self, x):
        """Forward pass. Input: (B, C, T). Output: (B, 1, T) for seq2seq or (B, 1) for seq2point."""
        x = self.tstcn(x)
        x = self.ieca_lstm(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.downstreamtask == "seq2seq":
            return x.unsqueeze(1)
        else:
            return x
