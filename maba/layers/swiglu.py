import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim: int = 640, d_ffn: int = 1728):
        super().__init__()
        self.w_gate = nn.Linear(dim, d_ffn, bias=False)
        self.w_up = nn.Linear(dim, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

SwiGLUFFN = SwiGLU
