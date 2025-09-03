import sys, torch, os

path = sys.argv[1] if len(sys.argv) > 1 else None
if not path or not os.path.exists(path):
    print("Usage: python check_emb.py /path/to/emb.pt")
    raise SystemExit(1)

t = torch.load(path, map_location="cpu")
print("Loaded:", path)
print("type:", type(t))
print("dtype:", getattr(t, "dtype", None))
print("device:", getattr(t, "device", None))
print("shape:", getattr(t, "shape", None))

# Try common normalizations to [T, 12, 768]
def summarize(x):
    try:
        print("first slice stats:", float(x.float().mean()), float(x.float().std()))
    except Exception:
        pass

if hasattr(t, "dim"):
    if t.dim() == 3:
        print("3D tensor (expected [T, 12, 768])")
        summarize(t[:1])
    elif t.dim() == 4:
        print("4D tensor; possible leading batch or extra axis")
        print("as [1,T,12,768] squeezed:", t.squeeze(0).shape)
        print("permute guesses:")
        for p in [(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,0,2,3),(1,2,0,3),(1,3,0,2)]:
            print("perm", p, "->", t.permute(*p).shape)
    else:
        print("Tensor with dim", t.dim(), "- inspect manually")
else:
    print("Object is not a tensor; contains:", type(t))
