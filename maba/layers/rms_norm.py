import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        s = torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return (t * s * self.weight).type_as(x)
