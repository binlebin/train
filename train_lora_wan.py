import os
import sys
import json
import time
import math
import yaml
import torch
import random
import argparse
from torch.utils.data import DataLoader

# Ensure plugin and InfiniteTalk repo are importable
sys.path.append(os.path.expanduser("~/work/diffsynth-studio"))
sys.path.append(os.path.expanduser("~/work/InfiniteTalk"))

from plugins.wan_infinitetalk import (
    build_model,
    WanDataset,
    add_lora_adapters,
    export_lora_safetensors,
)
from plugins.wan_infinitetalk.trainer_bind import (
    prepare_conditioners,
    training_step,
)

def wrap_to_obj(d):
    if isinstance(d, dict):
        return type("N", (), {k: wrap_to_obj(v) for k, v in d.items()})
    return d

class NoiseScheduler:
    def __init__(self, num_steps=1000):
        self.num_steps = num_steps
    def add_noise(self, original_samples, noise, t):
        # Match repo’s flow-matching add_noise
        t = t.float() / self.num_steps
        t = t.view(t.shape + (1,) * (len(noise.shape) - 1))
        return (1 - t) * original_samples + t * noise

def set_seed(seed: int):
    if seed is None or seed < 0:
        seed = random.randint(0, 10_000_000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/wan_vi_lora.yaml")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg = wrap_to_obj(cfg)

    # Load Wan config.json (names for checkpoints, strides, patch sizes, dtypes)
    with open(os.path.join(cfg.model.ckpt_dir, "config.json"), "r") as f:
        cfg.config = wrap_to_obj(json.load(f))

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Seed (env SEED takes precedence if set)
    seed_env = os.getenv("SEED")
    seed = int(seed_env) if seed_env is not None else 42
    seed = set_seed(seed)
    print(f"[Info] Using seed: {seed}")

    # Build model (loads Wan + InfiniteTalk weights and exposes diffusion_model)
    model = build_model(cfg).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # Attach LoRA adapters to audio-specific modules
    add_lora_adapters(
        model.diffusion_model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )

    # Collect LoRA params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in lora_params)
    print(f"[Info] Trainable LoRA params: {total_trainable/1e6:.2f}M")

    # Optimizer
    opt = torch.optim.AdamW(lora_params, lr=cfg.train.lr, weight_decay=cfg.train.wd)

    # Simple warmup + cosine decay
    total_steps = cfg.train.steps
    warmup_steps = cfg.train.warmup_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Data
    def collate_keep(batch):
        # Keep samples as a list to avoid stacking variable-length tensors (audio_emb)
        return batch
    train_ds = WanDataset(cfg.data.manifest, split="train")
    # batch_size=1 at 480p is typical; if memory allows you can try 2
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_keep,
    )

    # Conditioners (VAE/CLIP/T5) using config.json names/dtypes
    conds = prepare_conditioners(model, cfg)

    # Noise scheduler aligned with repo
    noise_sched = NoiseScheduler(num_steps=1000)

    # AMP dtype
    amp_dtype = torch.bfloat16 if str(cfg.train.precision).lower() == "bf16" else torch.float16

    # Training loop
    os.makedirs(cfg.log.out_dir, exist_ok=True)
    global_step = 0
    accum = 0
    scaler = None  # bf16/fp16 mixed precision via autocast only; no GradScaler needed for bf16

    start_time = time.time()
    last_step_wall = start_time
    print(f"[Info] Start training for {cfg.train.steps} steps | bucket={cfg.train.bucket} | precision={cfg.train.precision}")

    while global_step < cfg.train.steps:
        for batch in train_dl:
            model.train()
            def compute_loss_batch(b):
                if isinstance(b, list):
                    total = 0.0
                    for sample in b:
                        total = total + training_step(model, sample, conds, cfg, noise_sched)
                    return total / len(b)
                return training_step(model, b, conds, cfg, noise_sched)

            with torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                loss = compute_loss_batch(batch) / cfg.train.grad_accum

            loss.backward()
            accum += 1

            if accum >= cfg.train.grad_accum:
                # gradient accumulation reached: do one optimization step
                # 1) clip and record grad-norm
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                # 2) optimizer step
                opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                global_step += 1

                # per-step wall time
                now = time.time()
                step_dur = now - last_step_wall
                last_step_wall = now
                micro_avg = step_dur / max(1, cfg.train.grad_accum)
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                mem_max = torch.cuda.max_memory_allocated() / 1e9
                print(
                    f"[Step {global_step:6d}] step_time={step_dur:.2f}s (~{micro_avg:.2f}s/micro) "
                    f"loss={loss.item()*cfg.train.grad_accum:.4f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"grad_norm={float(grad_norm):.3f} mem={mem_alloc:.1f}G alloc/{mem_reserved:.1f}G resv/{mem_max:.1f}G max "
                    f"eff_batch={cfg.train.batch_size*cfg.train.grad_accum}"
                )

                if global_step % 50 == 0:
                    elapsed = now - start_time
                    eta = elapsed / max(1, global_step) * (cfg.train.steps - global_step)
                    print(f"[Step {global_step:6d}] elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h")

                if global_step % max(1, cfg.log.save_every) == 0:
                    out = os.path.join(cfg.log.out_dir, f"step_{global_step}.safetensors")
                    export_lora_safetensors(model.diffusion_model, out)

                if global_step >= cfg.train.steps:
                    break

    # Final save
    final_path = os.path.join(cfg.log.out_dir, "final.safetensors")
    export_lora_safetensors(model.diffusion_model, final_path)
    print(f"[Info] Training complete. Saved final LoRA to {final_path}")

if __name__ == "__main__":
    main()
