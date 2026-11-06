"""Recompose config-sweep tau* with the TARGET fixed at q16 (draft-only q), and the tau needed to beat C1-specialized AR.
config_sweep applies q to the whole model, so its per-q tau* rows use a q-quantized target AR; use this instead."""
import json,glob
R='results/E4/l1_llama31_config_sweep_v1_'
cfgs=['C1','d16K256L10','d8K16L10','d8K4096L3','d8K4096L5','d16K4096L6','d16K4096L7','BF','BL']
spec=json.load(open(R+'C1_q16_specialized.json'))['metrics']
AR_C1s=spec['ar'][str(spec['L'])]['step_us']/1e3
print(f"C1-specialized AR = {AR_C1s:.2f} ms")
for c in cfgs:
    t=json.load(open(R+c+'_q16.json'))['metrics']; L=t['L']
    ovh=None
    m=t.get('model',{})
    for v in (m.values() if isinstance(m,dict) else []):
        if isinstance(v,dict) and 'ovh_ms' in v: ovh=v['ovh_ms'];break
        if isinstance(v,list) and v and 'ovh_ms' in v[0]: ovh=v[0]['ovh_ms'];break
    if ovh is None: ovh=0.0
    TAR=t['ar'][str(L)]['step_us']/1e3
    print(f"\n{c}: target AR(q16)={TAR:.2f} ms ({AR_C1s/TAR:.2f}x of C1spec AR speed) ovh={ovh:.3f}")
    for q in (16,8,4):
        d=json.load(open(R+f'{c}_q{q}.json'))['metrics']
        row=[]
        for k in range(1,L):
            td=d['ar'][str(k)]['step_us']/1e3
            out=[]
            for g in (2,4,7):
                Rr=g*td+t['fwd'][str(min(g+1,8))]/1e3+ovh
                out.append(f"g{g}:{Rr/TAR:.2f}")
            row.append(f"k{k}({d['draft_bpw_index'][str(k)]}b) td={td:.2f} "+" ".join(out))
        print(f"  q{q}: "+" | ".join(row))
print("\n=== tau needed to beat C1-specialized AR (8.45ms/token), g=4 [and min over g in 2,4,7 with tau<=g+1] ===")
for c in cfgs:
    t=json.load(open(R+c+'_q16.json'))['metrics']; L=t['L']
    best=[]
    for q in (16,8,4):
        d=json.load(open(R+c+f'_q{q}.json'))['metrics']
        for k in range(1,L):
            td=d['ar'][str(k)]['step_us']/1e3
            for g in (2,4,7):
                Rr=g*td+t['fwd'][str(min(g+1,8))]/1e3
                need=Rr/AR_C1s
                if need<=g+1: best.append((need/(g+1),need,g,k,q,d['draft_bpw_index'][str(k)]))
    best.sort()
    print(c, [f"q{q} k{k}({b}b) g{g}: tau>{n:.2f}" for _,n,g,k,q,b in best[:3]] or "impossible even at tau=g+1")
