"""Recompute E7 v5 measured decode speedups under two batch accountings.

static   : measured. Every sequence makes exactly `gen` tokens; finished rows idle (tau_eff = gen/rounds).
all-active (MagicDec official accounting, ~ continuous batching): every row keeps generating, so
           speedup = tau * T_AR_step / T_round, from the same measured tau and step times (derived).
Input: results/E7/llama8b_magicvq_v5_tables.md. Output: results/E7/llama8b_magicvq_v5_accounting.md
"""
import re, sys
src = 'results/E7/llama8b_magicvq_v5_tables.md'
out = 'results/E7/llama8b_magicvq_v5_accounting.md'
secs, sec = [], None
for l in open(src):
    if l.startswith('## '):
        sec = {'name': l[3:].split('(')[0].strip(), 'rows': []}; secs.append(sec); continue
    if sec is None or not l.startswith('| ') or l.startswith('| draft') or l.startswith('|---'): continue
    c = [x.strip() for x in l.strip().strip('|').split('|')]
    if c[0] == 'AR': sec['tar'] = float(re.findall(r'[\d.]+', c[11])[0]); continue
    try:
        ms = [float(x) for x in c[11].split('/')]
        sec['rows'].append(dict(draft=c[0], g=c[1], sp=float(c[3].strip('*')), tau=float(c[5].split()[0]),
                                te=float(c[7]), tr=sum(ms)))
    except ValueError: pass
L = ['# E7 v5: decode speedup vs AR under two batch accountings', '',
     '- **static (measured):** each sequence makes exactly gen tokens, finished rows idle until the batch ends (tau_eff = gen/rounds < tau).',
     '- **all-active (derived, MagicDec official accounting; close to continuous batching):** speedup = tau x T_AR / T_round from the same measured tau and step times. Our recomputation matched the official MagicDec numbers within 1% (results/E7/llama8b_magicdec_official_tables.md).',
     '- Headline for E7 from 2026-11-18: **all-active**; static is reported as a secondary result (eviction drafts lose more to stragglers).', '']
summ = []
for s in secs:
    if not s['rows']: continue
    L += [f"## {s['name']}", '', '| draft | gamma | tau | tau_eff/tau | static (measured) | all-active (derived) |', '|---|---|---|---|---|---|']
    for r in s['rows']:
        r['aa'] = r['tau'] * s['tar'] / r['tr']
        L.append(f"| {r['draft']} | {r['g']} | {r['tau']:.2f} | {r['te']/r['tau']:.2f} | {r['sp']:.3f} | {r['aa']:.3f} |")
    snap = [r for r in s['rows'] if r['draft'].startswith('SnapKV-')]
    vq = [r for r in s['rows'] if not r['draft'].startswith('SnapKV-')]
    bs, ba = max(r['sp'] for r in snap), max(r['aa'] for r in snap)
    vs, va = max(r['sp'] for r in vq), max(r['aa'] for r in vq)
    L += ['', f"best VQ/hybrid ÷ best SnapKV: static **{vs/bs:.3f}**, all-active **{va/ba:.3f}**", '']
    summ.append(f"| {s['name']} | {bs:.3f} | {vs:.3f} | {vs/bs:.3f} | {ba:.3f} | {va:.3f} | {va/ba:.3f} |")
L[6:6] = ['## Summary (best per family)', '', '| point | SnapKV static | VQ static | ratio | SnapKV all-active | VQ all-active | ratio |', '|---|---|---|---|---|---|---|'] + summ + ['']
open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[:6+4+len(summ)]))
