#!/usr/bin/env python3
"""
LSTM and ConvLSTM baseline models for streamflow forecasting.
"""

import torch
import torch.nn as nn


class StreamflowLSTM(nn.Module):
    """
    LSTM for daily streamflow prediction.
    Input: (batch, seq_len, n_forcing) dynamic forcing
           (batch, n_static) static basin attributes (optional)
    Output: (batch, horizon) predicted discharge
    """

    def __init__(self, n_forcing: int, n_static: int = 0, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2, horizon: int = 1):
        super().__init__()
        self.n_static = n_static
        self.hidden_size = hidden_size

        # If static attrs, project them and concatenate with forcing
        if n_static > 0:
            self.static_proj = nn.Linear(n_static, 32)
            lstm_input = n_forcing + 32
        else:
            self.static_proj = None
            lstm_input = n_forcing

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x_forcing, x_static=None):
        # x_forcing: (B, T, F)
        if self.static_proj is not None and x_static is not None:
            s = self.static_proj(x_static)  # (B, 32)
            s = s.unsqueeze(1).expand(-1, x_forcing.size(1), -1)  # (B, T, 32)
            x = torch.cat([x_forcing, s], dim=-1)
        else:
            x = x_forcing

        out, _ = self.lstm(x)  # (B, T, H)
        out = self.dropout(out[:, -1, :])  # last timestep
        return self.head(out)  # (B, horizon)


class StreamflowConvLSTM(nn.Module):
    """
    1D-Conv + LSTM hybrid for streamflow prediction.
    Conv layers extract local temporal patterns before LSTM.
    """

    def __init__(self, n_forcing: int, n_static: int = 0, hidden_size: int = 128,
                 num_layers: int = 2, kernel_size: int = 3, dropout: float = 0.2,
                 horizon: int = 1):
        super().__init__()
        self.n_static = n_static

        # 1D conv block: (B, T, F) -> (B, T, 64)
        self.conv = nn.Sequential(
            nn.Conv1d(n_forcing, 64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )

        if n_static > 0:
            self.static_proj = nn.Linear(n_static, 32)
            lstm_input = 64 + 32
        else:
            self.static_proj = None
            lstm_input = 64

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x_forcing, x_static=None):
        # Conv expects (B, C, T), input is (B, T, F)
        x = x_forcing.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # back to (B, T, 64)

        if self.static_proj is not None and x_static is not None:
            s = self.static_proj(x_static)
            s = s.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, s], dim=-1)

        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out)
