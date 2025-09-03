# REPLACE the file contents with this version:

import os, torch, yaml
from torch.utils.data import Dataset
from torchvision.io import read_video

class WanDataset(Dataset):
    def __init__(self, manifest_path, split="train"):
        data = yaml.safe_load(open(manifest_path))
        self.items = data[split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        video_path = it["video"]
        emb_path = it["audio_emb"]
        vframes, _, _ = read_video(video_path, pts_unit="sec")  # [T, H, W, C] uint8
        audio_emb = torch.load(emb_path)  # [T_a, 12, 768] (or normalized later)
        return {
            "video": vframes,
            "audio_emb": audio_emb,
            "prompt": it.get("prompt", ""),
            "video_path": video_path,
            "audio_path": emb_path,
        }
