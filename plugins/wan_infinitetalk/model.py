import json, os, torch
from safetensors.torch import load_file
import sys
sys.path.append(os.path.expanduser("~/work/InfiniteTalk"))  # Ensure importable
from wan.modules.multitalk_model import WanModel

import wan.utils.multitalk_utils as _mtu
def _skip_attn_map(*args, **kwargs):
    return None  # avoids heavy compiled path; not used when human_num==1
_mtu.get_attn_map_with_target = _skip_attn_map

import wan.modules.multitalk_model as _mm

def load_wan_infinite_talk(ckpt_dir, infinitetalk_path, dtype):
    wan_config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    model = WanModel(weight_init=False, **wan_config).to(dtype=dtype)
    shard_files = [f"{ckpt_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1,8)]
    merged = {}
    for wf in shard_files:
        sd = load_file(wf)
        merged.update(sd)
    it_sd = load_file(infinitetalk_path)
    merged.update(it_sd)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if len(missing) > 0: print(f"Missing keys: {len(missing)}")
    if len(unexpected) > 0: print(f"Unexpected keys: {len(unexpected)}")
    model.eval()
    try:
        # match pipeline behavior; ensures the flag exists
        model.disable_teacache()
    except AttributeError:
        # older builds: just set the flag
        setattr(model, "enable_teacache", False)

    _mm.get_attn_map_with_target = _skip_attn_map
    print("[Info] Patched get_attn_map_with_target in multitalk_model")
    
    # enable gradient checkpointing for all transformer blocks to save activation memory
    import types
    from torch.utils.checkpoint import checkpoint

    def _wrap_block_with_checkpoint(block):
        original_forward = block.forward
        def _ckpt_forward(self, x, **kwargs):
            def _fn(x_):
                return original_forward(x_, **kwargs)
            return checkpoint(_fn, x, use_reentrant=False)
        block.forward = types.MethodType(_ckpt_forward, block)

    for _blk in model.blocks:
        _wrap_block_with_checkpoint(_blk)
    print("[Info] Enabled gradient checkpointing for DiT blocks")

    return model
class WanModule(torch.nn.Module):
    def __init__(self, ckpt_dir, infinitetalk_path, param_dtype=torch.bfloat16):
        super().__init__()
        self.diffusion_model = load_wan_infinite_talk(ckpt_dir, infinitetalk_path, param_dtype)
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use training_step for forward.")
def build_model(cfg):
    dtype = torch.bfloat16 if str(cfg.train.precision).lower() == "bf16" else torch.float16
    m = WanModule(cfg.model.ckpt_dir, cfg.model.infinitetalk_dir, dtype)
    return m
