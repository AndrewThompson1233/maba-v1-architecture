import torch
import torch.nn as nn
from typing import Tuple, Optional

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int = 64, max_len: int = 4096, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, n: int):
        self.max_len = n
        t = torch.arange(n, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        tot = pos + seq_len
        if tot > self.max_len or self.cos_cached.device != x.device:
            self._build_cache(max(tot, self.max_len))
        cos = self.cos_cached[pos:tot].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[pos:tot].to(dtype=x.dtype, device=x.device)
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pos_ids is not None:
        c = cos[pos_ids].unsqueeze(1)
        s = sin[pos_ids].unsqueeze(1)
    else:
        c = cos.unsqueeze(0).unsqueeze(1)
        s = sin.unsqueeze(0).unsqueeze(1)
    q_out = (q * c) + (rotate_half(q) * s)
    k_out = (k * c) + (rotate_half(k) * s)
    return q_out, k_out

apply_rotary_emb = apply_rope
