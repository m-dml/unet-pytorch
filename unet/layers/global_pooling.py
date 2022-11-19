import torch
import torch.nn as nn


class GlobalMaxPool(nn.Module):
    def __init__(self, dim: int = -1):
        super(GlobalMaxPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim).values


class GlobalAvgPool(nn.Module):
    def __init__(self, dim: int = -1):
        super(GlobalAvgPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)
