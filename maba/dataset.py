import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Dict
from .tokenizer import Tokenizer

class Dataset(TorchDataset):
    SAMPLE_TEXTS = [
        "template <typename T> class Tensor { public: size_t rows, cols; T* data; void matmul(const Tensor& other); };",
        "int main() { auto model = MabaModel(); model.forward(); return 0; }",
        "#include <iostream>\n#include <vector>\nusing namespace std; int fib(int n) { if (n <= 1) return n; return fib(n-1) + fib(n-2); }",
        "def decoupled_delta_update(S, k, v, b, w, alpha):\n    e = b * k\n    z = w * v\n    return (torch.eye(d) - torch.outer(k, e)) @ (alpha * S) + torch.outer(k, z)",
        "class SwiGLU(nn.Module):\n    def forward(self, x):\n        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))",
        "Theorem: For any orthogonal matrix Q, ||Q x|| = ||x|| for all vectors x. Proof: ||Q x||^2 = (Q x)^T (Q x) = x^T Q^T Q x = x^T I x = ||x||^2. QED.",
        "Solve 2x + 7 = 19. Step 1: subtract 7 from both sides: 2x = 12. Step 2: divide by 2: x = 6. Verification: 2(6) + 7 = 19.",
        "Calculate dot product: u = [1, 2, 3], v = [4, 5, 6]. u . v = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.",
        "Архитектура Maba объединяет линейное внимание GDN-2 и квадратичное GQA в пропорции 3:1.",
        "Факторизация эмбеддингов снижает налог на словарь до 4.3% параметров.",
        "Поблочное связывание весов удваивает эффективную глубину до 40 слоев.",
        "Оптимизатор Muon выполняет ортогонализацию градиентов по формуле Ньютона-Шульца."
    ]

    def __init__(self, tokenizer: Optional[Tokenizer] = None, seq_len: int = 64, repeat: int = 40):
        tok = tokenizer or Tokenizer()
        self.seq_len = seq_len
        self.samples = []

        for _ in range(repeat):
            for txt in self.SAMPLE_TEXTS:
                ids = tok.encode(txt, add_bos=True, add_eos=True)
                if len(ids) < seq_len:
                    ids = ids + [tok.PAD_ID] * (seq_len - len(ids))
                else:
                    ids = ids[:seq_len]
                self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.samples[idx]
        y = x.clone()
        y[y == Tokenizer.PAD_ID] = -100
        return {"input_ids": x, "labels": y}

MicroCurriculumDataset = Dataset
MabaDataset = Dataset
