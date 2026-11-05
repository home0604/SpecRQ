"""Check that a decoder layer called the AQLM/DbD way (only attention_mask, position_ids kwargs captured from layer 0)
reproduces the full-model hidden state. Run: python check_layer_call.py <model_dir>"""
import sys, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
p = sys.argv[1]
tok = AutoTokenizer.from_pretrained(p)
m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32).cuda().eval()
ids = tok("The quick brown fox jumps over the lazy dog. " * 8, return_tensors="pt").input_ids.cuda()
cap = {}
class Catcher(torch.nn.Module):
    def __init__(s, l): super().__init__(); s.l = l
    def forward(s, x, **kw):
        cap["x"] = x; cap["kw"] = kw; raise ValueError
L = m.model.layers; orig = L[0]; L[0] = Catcher(orig)
try: m(ids, attention_mask=torch.ones_like(ids))
except ValueError: pass
L[0] = orig
print("transformers", transformers.__version__, "| kwargs captured:", sorted(cap["kw"].keys()))
fw = {k: cap["kw"][k] for k in ["attention_mask", "position_ids"]}  # what main.py forwards
h = cap["x"]
with torch.no_grad():
    for l in L: h = l(h, **fw)[0]
    ref = m(ids, output_hidden_states=True).hidden_states[-1]
    h = m.model.norm(h)
print("max|diff| vs full forward:", (h - ref).abs().max().item(), "rel:", ((h - ref).norm() / ref.norm()).item())
