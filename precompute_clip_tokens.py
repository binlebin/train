import os
import sys
import glob
import yaml
import torch
import argparse
from torchvision.io import read_video

# Make InfiniteTalk modules importable
sys.path.append(os.path.expanduser("~/work/InfiniteTalk"))
from wan.modules.clip import CLIPModel


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


def to_norm_5d(frames_thwc: torch.Tensor) -> torch.Tensor:
    """[N,H,W,C] uint8 -> [N,C,1,H,W] float in [-1,1]"""
    x = frames_thwc.float() / 255.0
    x = (x - 0.5) * 2.0
    x = x.permute(0, 3, 1, 2).contiguous()  # [N,C,H,W]
    return x.unsqueeze(2)  # [N,C,1,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = yaml.safe_load(open(args.manifest, "r"))

    # Resolve CLIP weights and tokenizer
    clip_path = _guess_file(args.ckpt_dir, [
        "models_clip*.pth",
        "*clip*.pth",
        "*.safetensors",
    ])
    clip_tok_path = _find_tokenizer_dir(args.ckpt_dir, preferred_dirs=["xlm-roberta-large", "clip_tokenizer"])
    if clip_path is None:
        raise FileNotFoundError(f"CLIP checkpoint not found in {args.ckpt_dir}")
    if clip_tok_path is None:
        raise FileNotFoundError(f"CLIP tokenizer dir not found in {args.ckpt_dir}")

    device = torch.device("cuda")
    clip = CLIPModel(dtype=torch.bfloat16, device=device,
                     checkpoint_path=clip_path, tokenizer_path=clip_tok_path)

    def process(items):
        for it in items:
            vid = it["video"]
            base = os.path.splitext(os.path.basename(vid))[0]
            out_pt = os.path.join(args.out_dir, f"{base}_clip.pt")
            if os.path.exists(out_pt):
                it["clip_tokens"] = out_pt
                print("skip:", out_pt)
                continue

            frames, _, _ = read_video(vid, pts_unit="sec")  # [T,H,W,C] uint8
            T = frames.shape[0]
            tokens = []
            with torch.no_grad():
                for s in range(0, T, args.batch):
                    chunk = frames[s : s + args.batch]
                    x5d = to_norm_5d(chunk).to(device=device, dtype=torch.bfloat16)  # [N,C,1,H,W]
                    tok = clip.visual(x5d).to("cpu")  # [N,257,1280]
                    tokens.append(tok)
            tok_all = torch.cat(tokens, dim=0) if tokens else torch.empty(0, 257, 1280)
            torch.save(tok_all, out_pt)
            it["clip_tokens"] = out_pt
            print("ok:", out_pt, tuple(tok_all.shape))

    if "train" in data:
        process(data["train"])
    if "val" in data:
        process(data["val"])

    out_manifest = args.manifest.replace(".yaml", "_clip.yaml")
    yaml.safe_dump(data, open(out_manifest, "w"))
    print("wrote:", out_manifest)


if __name__ == "__main__":
    main()


