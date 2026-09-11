from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    vocab_size: int = 32768
    d_emb: int = 128
    dim: int = 640
    n_layers: int = 20
    n_passes: int = 2
    layer_types: List[int] = field(default_factory=lambda: [0, 0, 0, 1] * 5)
    n_heads: int = 10
    d_head: int = 64
    n_kv_heads: int = 2
    kernel_size: int = 4
    d_ffn: int = 1728
    gate_bias: float = 2.0
    rope_theta: float = 500000.0
    max_len: int = 4096
    mtp_k: int = 2
    mtp_weight: float = 0.3
    eps: float = 1e-6
    init_std: float = 0.02

    @property
    def d_model(self) -> int: return self.dim
    @property
    def num_layers(self) -> int: return self.n_layers
    @property
    def passes_per_block(self) -> int: return self.n_passes
    @property
    def num_heads(self) -> int: return self.n_heads
    @property
    def num_kv_heads(self) -> int: return self.n_kv_heads
    @property
    def conv_kernel_size(self) -> int: return self.kernel_size
    @property
    def gated_res_bias_init(self) -> float: return self.gate_bias
    @property
    def max_position_embeddings(self) -> int: return self.max_len
    @property
    def mtp_loss_weight(self) -> float: return self.mtp_weight
    @property
    def rms_norm_eps(self) -> float: return self.eps
    @property
    def initializer_range(self) -> float: return self.init_std

Apex100MConfig = Config
MabaConfig = Config
