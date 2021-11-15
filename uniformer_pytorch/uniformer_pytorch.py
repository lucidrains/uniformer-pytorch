import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

class Uniformer(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
