import os
import yaml
import torch
import argparse
from torchvision.io import read_video

# Make InfiniteTalk modules importable
import sys
sys.path.append(os.path.expanduser("~/work/InfiniteTalk"))
from wan.modules.vae import WanVAE


def to_bcthw_uint8_to_norm(x_thwc: torch.Tensor) -> torch.Tensor:
    """[T,H,W,C] uint8 -> [1,C,T,H,W] in [-1,1] float32"""
    x = x_thwc.float() / 255.0
    x = (x - 0.5) * 2.0
    return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def downscale_thwc(v_thwc: torch.Tensor, target_h: int) -> torch.Tensor:
    """Return [T,H,W,C] uint8 resized to target_h (rounded to /8), preserving aspect."""
    T, H, W, C = v_thwc.shape
    if H <= target_h:
        return v_thwc
    new_h = max(8, (target_h // 8) * 8)
    new_w = max(8, int(round(W * (new_h / H))))
    new_w = (new_w // 8) * 8
    v_4d = v_thwc.permute(0, 3, 1, 2).float()  # [T,C,H,W]
    v_4d = torch.nn.functional.interpolate(
        v_4d, (new_h, new_w), mode="bilinear", align_corners=False
    )
    return v_4d.permute(0, 2, 3, 1).to(torch.uint8).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_h", type=int, default=360)
    ap.add_argument("--time_chunk", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = yaml.safe_load(open(args.manifest, "r"))

    vae = WanVAE(
        vae_pth=os.path.join(args.ckpt_dir, "Wan2.1_VAE.pth"),
        device=torch.device("cuda:0"),
    )

    def process(items):
        for it in items:
            vid = it["video"]
            base = os.path.splitext(os.path.basename(vid))[0]
            out_pt = os.path.join(args.out_dir, f"{base}_lat.pt")
            if os.path.exists(out_pt):
                it["video_latent"] = out_pt
                print("skip:", out_pt)
                continue

            frames, _, _ = read_video(vid, pts_unit="sec")  # [T,H,W,C] uint8
            frames = downscale_thwc(frames, args.target_h)
            T = frames.shape[0]

            latents = []
            for s in range(0, T, args.time_chunk):
                clip = frames[s : s + args.time_chunk].cuda(non_blocking=True)
                x = to_bcthw_uint8_to_norm(clip)  # [1,C,T,H,W]
                with torch.no_grad():
                    y = vae.encode(x)[0]  # [C_lat,T_lat,H_lat,W_lat]
                latents.append(y.cpu())
                torch.cuda.empty_cache()

            lat = torch.cat(latents, dim=1)
            torch.save(lat, out_pt)
            it["video_latent"] = out_pt
            print("ok:", out_pt, tuple(lat.shape))

    if "train" in data:
        process(data["train"])
    if "val" in data:
        process(data["val"])

    out_manifest = args.manifest.replace(".yaml", "_latents.yaml")
    yaml.safe_dump(data, open(out_manifest, "w"))
    print("wrote:", out_manifest)


if __name__ == "__main__":
    main()


