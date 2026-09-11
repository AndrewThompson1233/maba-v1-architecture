import os
import time
import math
import torch
from torch.utils.data import DataLoader
from .config import Config
from .model import Model
from .optimizers import HybridOpt
from .dataset import Dataset
from .tokenizer import Tokenizer

def train(
    n_steps: int = 50,
    batch_size: int = 4,
    seq_len: int = 64,
    ckpt_path: str = "maba_checkpoint.pt",
    device: str = "cpu"
):
    print(f"Training Maba on {device.upper()}")
    cfg = Config()
    model = Model(cfg).to(device)

    counts = model.count_params()
    print(f"Params: {counts['total']:,} (emb: {counts['embedding_total']:,}, core: {counts['core_total']:,})")

    tok = Tokenizer()
    ds = Dataset(tokenizer=tok, seq_len=seq_len, repeat=20)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = HybridOpt(model, lr_muon=0.02, wd_muon=0.01, lr_adamw=1.5e-3, wd_adamw=0.1)

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(dl)
    init_loss = None
    last_loss = None

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        opt.zero_grad()
        out = model(x, labels=y)
        loss_dict = out["loss"]
        tot_loss = loss_dict["total_loss"]
        main_loss = loss_dict["main_loss"]
        mtp_loss = loss_dict["mtp_loss"]

        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        t_val = tot_loss.item()
        m_val = main_loss.item()
        mtp_val = mtp_loss.item()

        del out, loss_dict, tot_loss, main_loss, mtp_loss
        opt.zero_grad(set_to_none=True)

        if init_loss is None:
            init_loss = t_val
        last_loss = t_val

        step += 1
        if step % 10 == 0 or step == 1 or step == n_steps:
            el = time.time() - t0
            ppl = math.exp(min(m_val, 20.0))
            print(f"step {step:3d}/{n_steps:3d} | loss {t_val:.4f} | ntp {m_val:.4f} | mtp {mtp_val:.4f} | ppl {ppl:.2f} | {el:.1f}s")

    print(f"Loss: {init_loss:.4f} -> {last_loss:.4f}")

    os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else ".", exist_ok=True)
    torch.save({
        "config": cfg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "loss": last_loss
    }, ckpt_path)

    prompt = "class Tensor"
    p_ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=device)
    out_ids = model.generate(p_ids, max_new_tokens=15, temperature=0.7)
    print(f"Sample generation: {tok.decode(out_ids[0].tolist())}")

    return model, last_loss

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(n_steps=30, batch_size=2, seq_len=32, device=dev)
