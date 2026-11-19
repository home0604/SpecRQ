# build a full cb file from pilot layer files; missing layers taken from the init codebook (only pilot layers are evaluated)
import sys, os, torch
tag, init = sys.argv[1], sys.argv[2]
st = torch.load(init, weights_only=False)
TR = "/data/lsh/ckpts/magicvq_v6/train/" + tag
n = 0
for l in range(32):
    p = f"{TR}/L{l}.pt"
    if os.path.exists(p):
        x = torch.load(p)
        for kv in range(2):
            st["cb"][l][kv], st["mu"][l][kv], st["sd"][l][kv] = x["kv"][kv]
        n += 1
torch.save(st, f"/data/lsh/temp/v9/{tag}_partial.pt"); print(tag, n, "layers")
