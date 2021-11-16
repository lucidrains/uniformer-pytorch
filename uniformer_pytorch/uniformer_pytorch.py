import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce
import torch.nn.functional as F

# helpers

def exists(val):
    return val is not None

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim * mult, dim, 1)
    )

# main class

class Uniformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dims = (64, 128, 256, 512),
        depths = (3, 4, 8, 3),
        attn_types = ('l', 'l', 'g', 'g'),
        channels = 3,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv3d(channels, init_dim, (3, 4, 4), stride = (2, 4, 4), padding = (1, 0, 0))

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))
        self.stages = nn.ModuleList([])

        for ind, depth in enumerate(depths):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]

            self.stages.append(nn.ModuleList([
                nn.Conv3d(stage_dim, stage_dim, 3, padding = 1),
                FeedForward(stage_dim, mult = ff_mult, dropout = ff_dropout),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv3d(stage_dim, dims[ind + 1], (1, 2, 2), stride = (1, 2, 2)),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b c t h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, video):
        tokens = self.to_tokens(video)

        for dpe, ff, conv in self.stages:
            tokens = dpe(tokens) + tokens
            tokens = ff(tokens) + tokens

            if exists(conv):
                tokens = conv(tokens)

        return self.to_logits(tokens)
