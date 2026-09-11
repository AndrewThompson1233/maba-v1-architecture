import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .rms_norm import RMSNorm
from .rope import apply_rope

class GQA(nn.Module):
    def __init__(
        self,
        dim: int = 640,
        n_heads: int = 10,
        n_kv_heads: int = 2,
        d_head: int = 64,
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * d_head, dim, bias=False)

        self.q_norm = RMSNorm(dim, eps=eps)
        self.k_norm = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q)
        k_exp = k.view(B, L, self.n_kv_heads, 1, self.d_head).expand(
            B, L, self.n_kv_heads, self.n_rep, self.d_head
        ).reshape(B, L, self.dim)
        k_normed = self.k_norm(k_exp)

        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k_normed.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.repeat_interleave(self.n_rep, dim=1)

        q, k = apply_rope(q, k, cos, sin)

        if kv is not None:
            pk, pv = kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)

        scale = 1.0 / math.sqrt(self.d_head)
        if mask is None and kv is None and L > 1:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            elif L > 1:
                Lk = k.shape[-2]
                past_len = Lk - L
                qp = past_len + torch.arange(L, device=q.device).unsqueeze(1)
                kp = torch.arange(Lk, device=q.device).unsqueeze(0)
                scores = scores + torch.where(kp <= qp, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
            w = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn = torch.matmul(w, v)

        out = attn.transpose(1, 2).contiguous().view(B, L, self.dim)
        return self.o_proj(out), new_kv

GroupedQueryAttention = GQA
