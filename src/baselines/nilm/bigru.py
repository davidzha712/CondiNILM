"""BiGRU: Bidirectional GRU model for non-intrusive load monitoring.

Two 1D conv layers followed by two stacked BiGRU layers, a dense layer,
and separate 1x1 conv heads for power regression and state classification.

Reference: Bonfigli et al., "Thresholding methods in non-intrusive load
monitoring", 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class BiGRU(nn.Module):
    def __init__(
        self,
        window_size,
        c_in=1,
        out_channels=1,
        dropout=0.1,
        return_values="power",
        verbose_loss=False,
    ):
        """Initialize BiGRU with conv front-end, bidirectional GRU layers, and dual output heads."""
        super(BiGRU, self).__init__()

        self.return_values = return_values
        self.verbose_loss = verbose_loss

        self.drop = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(
            in_channels=c_in,
            out_channels=16,
            kernel_size=5,
            dilation=1,
            stride=1,
            bias=True,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=8,
            kernel_size=5,
            dilation=1,
            stride=1,
            bias=True,
            padding="same",
        )

        self.gru1 = nn.GRU(8, 64, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(128, 128, batch_first=True, bidirectional=True)

        self.dense = Dense(256, 64)
        self.regressor = nn.Conv1d(64, out_channels, kernel_size=1, padding=0)
        self.activation = nn.Conv1d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.drop(x))

        x = self.gru1(self.drop(x.permute(0, 2, 1)))[0]
        x = self.gru2(self.drop(x))[0]

        x = self.drop(self.dense(self.drop(x)))

        power_logits = self.regressor(self.drop(x.permute(0, 2, 1)))
        states_logits = self.activation(self.drop(F.relu(x.permute(0, 2, 1))))

        if self.return_values == "power":
            return power_logits
        elif self.return_values == "states":
            return states_logits
        else:
            return power_logits, states_logits
