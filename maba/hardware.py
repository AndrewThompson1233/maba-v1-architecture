import os
import torch

def get_device(dev_name: str = "auto") -> torch.device:
    if dev_name == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
            threads = min(8, os.cpu_count() or 4)
            torch.set_num_threads(threads)
    else:
        dev = torch.device(dev_name)
    return dev

def get_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        return torch.float16
    return torch.float32

configure_hardware = get_device
get_optimal_dtype = get_dtype
