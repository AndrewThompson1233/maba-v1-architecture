import torch
import torch.nn as nn
from .embeddings import EmbHead

class MTPHead(nn.Module):
    def __init__(self, dim: int = 640, d_emb: int = 128):
        super().__init__()
        self.dim = dim
        self.d_emb = d_emb
        self.proj = nn.Linear(dim + d_emb, dim, bias=True)

    def forward(self, h: torch.Tensor, f_emb: torch.Tensor, emb: EmbHead) -> torch.Tensor:
        cat = torch.cat([h, f_emb], dim=-1)
        h_mtp = self.proj(cat)
        return emb.forward_out(h_mtp)

MultiTokenPredictionHead = MTPHead
