from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.25) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        return self.act(x + residual)


class LSETCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: int = 64,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.proj = weight_norm(nn.Conv1d(input_dim, channels, kernel_size=1))
        self.blocks = nn.Sequential(
            *[ResidualTCNBlock(channels, d, kernel_size=kernel_size, dropout=dropout) for d in dilations]
        )
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.blocks(x)
        x = x.mean(dim=-1)
        return self.head(x)
