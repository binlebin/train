# REPLACE the LoRALinear class in lora.py with this:

import re, torch
from safetensors.torch import save_file

TARGET_PATTERNS = [
    re.compile(r"^diffusion_model\.audio_proj\.(proj1|proj1_vf|proj2|proj3)\.weight$"),
    re.compile(r"^diffusion_model\.blocks\.\d+\.audio_cross_attn\.(q_linear|kv_linear|proj)\.weight$"),
]

def _matches(name):
    return any(pat.match(name) for pat in TARGET_PATTERNS)

def add_lora_adapters(model, rank=32, alpha=16, dropout=0.0):
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base_linear, r, alpha, dropout):
            super().__init__()
            self.base = base_linear
            in_f, out_f = base_linear.in_features, base_linear.out_features
            dev = base_linear.weight.device
            dt = base_linear.weight.dtype
            self.lora_A = nn.Linear(in_f, r, bias=False, device=dev, dtype=dt)
            self.lora_B = nn.Linear(r, out_f, bias=False, device=dev, dtype=dt)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        def forward(self, x):
            return self.base(x) + self.drop(self.lora_B(self.lora_A(x))) * self.scaling

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            full = name + ".weight"
            prefixed = "diffusion_model." + full if not full.startswith("diffusion_model.") else full
            if _matches(prefixed):
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapper = LoRALinear(module, rank, alpha, dropout)
                setattr(parent, attr_name, wrapper)
                replaced += 1
    print(f"Attached LoRA to {replaced} Linear layers.")

def export_lora_safetensors(model, path, prefix="diffusion_model."):
    tensors = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            base = prefix + name
            tensors[base + ".lora_down.weight"] = module.lora_A.weight.detach().cpu()
            tensors[base + ".lora_up.weight"]   = module.lora_B.weight.detach().cpu()
    save_file(tensors, path)
    print(f"Saved LoRA to {path}")
