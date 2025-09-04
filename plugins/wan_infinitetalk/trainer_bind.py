import os
import sys
import glob
import torch

# Ensure InfiniteTalk repo is importable
sys.path.append(os.path.expanduser("~/work/InfiniteTalk"))

from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _guess_file(ckpt_dir, patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(ckpt_dir, pat)))
        if hits:
            return hits[0]
    return None


def _find_tokenizer_dir(ckpt_dir, preferred_dirs=None):
    if preferred_dirs:
        for name in preferred_dirs:
            cand = os.path.join(ckpt_dir, name)
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "tokenizer_config.json")):
                return cand
            if os.path.isdir(cand):
                subs = [d for d in glob.glob(os.path.join(cand, "*")) if os.path.isdir(d)]
                for sd in subs:
                    if os.path.exists(os.path.join(sd, "tokenizer_config.json")):
                        return sd
    hits = glob.glob(os.path.join(ckpt_dir, "**", "tokenizer_config.json"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    return None


def prepare_conditioners(model_wrapper, cfg):
    ckpt_dir = cfg.model.ckpt_dir
    device = torch.device("cuda")
    param_dtype = next(model_wrapper.diffusion_model.parameters()).dtype
    cfgc = getattr(cfg, "config", None)

    vae_ckpt = None if cfgc is None else getattr(cfgc, "vae_checkpoint", None)
    clip_ckpt = None if cfgc is None else getattr(cfgc, "clip_checkpoint", None)
    clip_tok  = None if cfgc is None else getattr(cfgc, "clip_tokenizer", None)
    t5_ckpt   = None if cfgc is None else getattr(cfgc, "t5_checkpoint", None)
    t5_tok    = None if cfgc is None else getattr(cfgc, "t5_tokenizer", None)

    vae_path = _first_existing([os.path.join(ckpt_dir, vae_ckpt) if vae_ckpt else None]) \
        or _guess_file(ckpt_dir, ["*VAE*.pth", "*vae*.pth", "vae*.pth", "*.safetensors"])
    clip_path = _first_existing([os.path.join(ckpt_dir, clip_ckpt) if clip_ckpt else None]) \
        or _guess_file(ckpt_dir, ["models_clip*.pth", "*clip*.pth", "*.safetensors"])
    t5_path = _first_existing([os.path.join(ckpt_dir, t5_ckpt) if t5_ckpt else None]) \
        or _guess_file(ckpt_dir, ["models_t5*.pth", "*t5*.pth", "*.safetensors"])

    clip_tok_path = _first_existing([os.path.join(ckpt_dir, clip_tok) if clip_tok else None]) \
        or _find_tokenizer_dir(ckpt_dir, preferred_dirs=["xlm-roberta-large", "clip_tokenizer"])
    t5_tok_path = _first_existing([os.path.join(ckpt_dir, t5_tok) if t5_tok else None]) \
        or _find_tokenizer_dir(ckpt_dir, preferred_dirs=["google", "t5_tokenizer"])

    if vae_path is None: raise FileNotFoundError(f"VAE checkpoint not found in {ckpt_dir}")
    if clip_path is None: raise FileNotFoundError(f"CLIP checkpoint not found in {ckpt_dir}")
    if clip_tok_path is None: raise FileNotFoundError(f"CLIP tokenizer dir not found in {ckpt_dir}")
    if t5_path is None: raise FileNotFoundError(f"T5 checkpoint not found in {ckpt_dir}")
    if t5_tok_path is None: raise FileNotFoundError(f"T5 tokenizer dir not found in {ckpt_dir}")

    print(f"[Paths] VAE: {vae_path}")
    print(f"[Paths] CLIP: {clip_path}")
    print(f"[Paths] CLIP tokenizer: {clip_tok_path}")
    print(f"[Paths] T5: {t5_path}")
    print(f"[Paths] T5 tokenizer: {t5_tok_path}")

    vae = WanVAE(vae_pth=vae_path, device=device)
    clip = CLIPModel(dtype=param_dtype, device=device,
                     checkpoint_path=clip_path,
                     tokenizer_path=clip_tok_path)

    # Keep T5 on CPU to save VRAM
    t5 = T5EncoderModel(text_len=(getattr(cfgc, "text_len", 512) if cfgc else 512),
                        dtype=param_dtype,
                        device=torch.device("cpu"),
                        checkpoint_path=t5_path,
                        tokenizer_path=t5_tok_path)
    return vae, clip, t5


def encode_video_to_latent(vae, frames_bcthw):
    with torch.no_grad():
        enc = vae.encode(frames_bcthw)
    y0 = enc[0] if isinstance(enc, (list, tuple)) else enc
    if y0.dim() == 4:
        y0 = y0.unsqueeze(0)
    return y0  # [1, C_lat, T_lat, H_lat, W_lat]

def compute_seq_len(T, H, W, patch_size, vae_stride, sp_size=1):
    # VAE downsamples by vae_stride; 3D patch embedding further chunks H/W by patch_size[1:].
    lat_h = H // vae_stride[1]
    lat_w = W // vae_stride[2]
    t_lat = (T - 1) // vae_stride[0] + 1
    # tokens after 3D patchifying
    n_tokens = (t_lat * lat_h * lat_w) // (patch_size[1] * patch_size[2])
    # USP alignment if used (no-op for sp_size=1)
    n_tokens = int(((n_tokens + sp_size - 1) // sp_size) * sp_size)
    return n_tokens

def _normalize_audio_emb(full):
    """
    Coerce audio emb to [T_audio, 12, 768].
    Accepts shapes like:
      - [T, 12, 768]
      - [1, T, 12, 768]
      - [T, 1, 12, 768]
      - [T, 12, 1, 768]
      - [T, 12, 768, 1]
      - Any permuted/expanded variant; we detect axes sized 12 and 768 and keep all other dims folded into time.
    """
    t = full
    # Drop leading singleton batch if present
    while t.dim() > 0 and t.shape[0] == 1 and t.dim() >= 3:
        t = t.squeeze(0)

    # If already [T, 12, 768], done
    if t.dim() == 3 and t.shape[1] == 12 and t.shape[2] == 768:
        return t

    # Flatten all dims except keep the 12- and 768-sized axes
    dims = list(t.shape)
    # Find 12 and 768 axes anywhere
    idx_12 = next((i for i, d in enumerate(dims) if d == 12), None)
    idx_768 = next((i for i, d in enumerate(dims) if d == 768), None)

    if idx_12 is None or idx_768 is None:
        raise ValueError(f"Audio emb must contain axes sized 12 and 768. Got shape {t.shape}")

    # Build a permutation that moves: time_dims (all except 12/768) first (flattened), then 12, then 768
    axes = list(range(t.dim()))
    time_axes = [i for i in axes if i not in (idx_12, idx_768)]
    perm = time_axes + [idx_12, idx_768]
    t = t.permute(perm).contiguous()

    # Fold all time axes into one
    time_prod = 1
    for i in time_axes:
        time_prod *= dims[i]
    t = t.view(time_prod, 12, 768)
    return t

def build_audio_windows_adaptive(full_audio_emb, T, desired_window, model_window, base_idx=0):
    """
    Normalize to [T_audio, 12, 768], build per-frame audio windows of length 'desired_window',
    then pad/crop to 'model_window' so the feature size always matches AudioProjModel.

    Returns: [1, T, model_window, 12, 768]
    """
    full = _normalize_audio_emb(full_audio_emb)  # [T_audio, 12, 768]
    device = full.device

    # Build desired window (centered)
    half = desired_window // 2
    offsets = torch.arange(-half, desired_window - half, device=device)  # length=desired_window
    centers = base_idx + torch.arange(0, T, device=device).unsqueeze(1)  # [T,1]
    idx = torch.clamp(centers + offsets.unsqueeze(0), 0, full.shape[0] - 1)  # [T, desired_window]
    audio = full[idx]  # [T, desired_window, 12, 768]

    # Adapt to model_window by symmetric pad/crop (replicate edges)
    if desired_window < model_window:
        pad_total = model_window - desired_window
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        left = audio[:, :1].expand(-1, pad_left, -1, -1) if pad_left > 0 else None
        right = audio[:, -1:].expand(-1, pad_right, -1, -1) if pad_right > 0 else None
        audio = torch.cat([left, audio, right] if left is not None and right is not None
                          else [audio, right] if left is None
                          else [left, audio], dim=1)
    elif desired_window > model_window:
        start = (desired_window - model_window) // 2
        audio = audio[:, start:start + model_window]

    # Now audio is [T, model_window, 12, 768]
    audio = audio.unsqueeze(0)  # [1, T, model_window, 12, 768]
    return audio


def training_step(model_wrapper, batch, conditioners, cfg, noise_scheduler):
    device = torch.device("cuda")
    diffusion_model = model_wrapper.diffusion_model
    vae, clip, t5 = conditioners

    # Log current sample file
    vp = batch.get("video_path")
    if vp is not None:
        if isinstance(vp, (list, tuple)):
            vp = vp[0]
        print(f"[Info] Sample: {os.path.basename(vp)}")

    # Video: to [B, C, T, H, W] in [-1,1]
    v = batch["video"].to(device)
    if v.dim() == 4:
        v = v.unsqueeze(0)
    B, T_full, H, W, C = v.shape
    v = v.permute(0, 4, 1, 2, 3).float() / 255.0
    v = (v - 0.5) * 2.0

    # Temporal cropping
    window_frames = int(getattr(getattr(cfg, "train", None), "window_frames", 81))
    if T_full > window_frames:
        s = torch.randint(low=0, high=T_full - window_frames + 1, size=(1,)).item()
        v_win = v[:, :, s:s + window_frames]
        T = window_frames
    else:
        s = 0
        v_win = v
        T = T_full

    # Use precomputed latents if available; else VAE encode current window
    lat_path = batch.get("video_latent_path")
    if isinstance(lat_path, (list, tuple)):
        lat_path = lat_path[0]
    if isinstance(lat_path, str) and os.path.exists(lat_path):
        lat_all = torch.load(lat_path, map_location="cpu")  # [C_lat, T_lat, H_lat, W_lat]
        lat_all = lat_all.to(device, dtype=next(diffusion_model.parameters()).dtype)
        stride_t = getattr(getattr(cfg, "config", None), "vae_stride", (4,8,8))[0]
        lat_s = max(0, s // max(1, stride_t))
        lat_e = max(lat_s + 1, min(lat_all.shape[1], (s + T + stride_t - 1) // max(1, stride_t)))
        lat_4d = lat_all[:, lat_s:lat_e]
    else:
        lat_5d = encode_video_to_latent(vae, v_win).to(next(diffusion_model.parameters()).dtype)
        lat_4d = lat_5d[0]  # [C_lat, T_lat, H_lat, W_lat]
    if lat_4d.shape[0] >= 16:
        lat16 = lat_4d[:16]
    else:
        pad = 16 - lat_4d.shape[0]
        lat16 = torch.cat([lat_4d, torch.zeros(pad, *lat_4d.shape[1:], device=lat_4d.device, dtype=lat_4d.dtype)], dim=0)

    # Build y = 4 masks + 16 latents -> [20, T_lat, H_lat, W_lat]
    _, T_lat, H_lat, W_lat = lat16.shape
    msk4 = torch.zeros(4, T_lat, H_lat, W_lat, device=lat16.device, dtype=lat16.dtype)
    msk4[:, 0] = 1
    y_cond = torch.cat([msk4, lat16], dim=0)

    # x = noised 16-channel latent (4D)
    t = torch.randint(low=1, high=noise_scheduler.num_steps, size=(1,), device=device).float()
    noise = torch.randn_like(lat16)
    noisy16 = noise_scheduler.add_noise(lat16, noise, t)

    # Text context (CPU -> GPU)
    prompt = batch.get("prompt", "")
    prompts = [prompt] if isinstance(prompt, str) else prompt
    context_list_cpu = t5(prompts, device=torch.device("cpu"))
    context_list = [t_.to(device) for t_ in context_list_cpu]

    # CLIP feature (last frame)
    with torch.no_grad():
        clip_fea = clip.visual(v_win[:, :, -1:, :, :]).to(lat16.dtype)

    # Audio windows [1, T, window, blocks, dim]
    full_audio_emb = batch["audio_emb"].to(device)

    cfgc = getattr(cfg, "config", None)
    model_audio_window = int(getattr(cfgc, "audio_window", 5))
    desired_audio_window = int(getattr(getattr(cfg, "train", None), "audio_window", model_audio_window))
    audio_embs = build_audio_windows_adaptive(full_audio_emb, T=T,
                                             desired_window=desired_audio_window,
                                          model_window=model_audio_window,
                                          base_idx=s).to(lat16.dtype)
    print(f"[Info] window_frames={T}, audio_window(desired/model)={desired_audio_window}/{model_audio_window}")

    assert audio_embs.dim() == 5, f"audio_embs must be 5D, got {audio_embs.shape}"

    # seq_len derived from latent grid with per-dim flooring (matches Conv3d stride behavior)
    patch_size = getattr(cfgc, "patch_size", (1, 2, 2))
    h_p = H_lat // patch_size[1]
    w_p = W_lat // patch_size[2]
    seq_len = int(T_lat * h_p * w_p)
    print(f"[Info] tokens: seq_len={seq_len}, T_lat={T_lat}, H_lat={H_lat}, W_lat={W_lat}, patch={patch_size}, Hp={h_p}, Wp={w_p}")

    # Dummy spatial ref mask to satisfy model's internal conversion
    # Shape expected before interpolate: [C, H, W] (we use C=1)
    ref_mask = torch.ones(1, H, W, device=device, dtype=torch.float32)
   
    # Forward: x=[noisy16] 4D, y=[y_cond] 4D
    pred = diffusion_model([noisy16], t=t, context=context_list, seq_len=seq_len,
                           clip_fea=clip_fea, y=[y_cond], audio=audio_embs, ref_target_masks=ref_mask)[0]

    # Align noise target to model output shape (handles odd H_lat -> floor*2 after patching)
    if pred.shape != noise.shape:
        c, tt, hh, ww = pred.shape
        noise = noise[:, :tt, :hh, :ww]
    loss = torch.nn.functional.mse_loss(pred, noise)
    return loss
