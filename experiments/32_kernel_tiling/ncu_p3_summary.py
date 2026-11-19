"""E10 phase 3: per-token instruction mix (warp instructions per token, by pipe) from the ncu_p3.sh reports, and a
simple per-SM throughput-bound model evaluated for the RTX 4090 (check against measured) and for the A100 [estimate].

Bounds per token and SM (cycles):  issue = inst / 4 SMSPs;  fp32 = (fma-pipe inst - fp16 inst) / (4 * r32);
fp16 = fp16 inst / (4 * r16); alu = alu inst / (4 * 0.5); smem = shared wavefronts (ld + st) / 1;
tensor = hmma / (4 * rT) ; dram = code bytes per token / (BW / (SMs * clock)).  r32 = warp FP32 instructions per
SMSP clock (Ada 1.0: fmaheavy + fmalite 2 x 16 lanes; A100 0.5: 16 lanes), r16 = warp HFMA2/HADD2 per SMSP clock
(both 1.0 [assumed]), rT = m16n8k16 per SMSP clock (A100 1/8: 256 dense f16 FMA/clk/SMSP; Ada GeForce with fp32
accumulate 1/16 [assumed]). Model time = max of the bounds (perfect overlap: a lower bound) and the sum of the
non-DRAM bounds (no overlap: a pessimistic bound).
Usage: python ncu_p3_summary.py <report dir> <out.md> [tokens=bh*n]"""
import csv
import io
import subprocess
import sys
from pathlib import Path

NCU = "/usr/local/cuda-12.1/bin/ncu"
DEV = {
    "4090": dict(sms=128, clk=2.52e9, r32=1.0, r16=1.0, rT=1 / 16, bw=0.94e12),  # bw = measured read BW of bench
    "A100-40GB": dict(sms=108, clk=1.41e9, r32=0.5, r16=1.0, rT=1 / 8, bw=1.3e12),   # ~85% of 1.555 TB/s
    "A100-80GB": dict(sms=108, clk=1.41e9, r32=0.5, r16=1.0, rT=1 / 8, bw=1.7e12),   # ~85% of 2.04 TB/s
}


def load(f):
    txt = subprocess.run([NCU, "-i", str(f), "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    r = list(csv.reader(io.StringIO(txt)))
    h, u, v = r[0], r[1], r[2]

    def g(name):
        i = h.index(name)
        x = float(v[i].replace(",", ""))
        if name == "gpu__time_duration.sum":
            return x * {"usecond": 1, "msecond": 1e3, "nsecond": 1e-3}[u[i]]
        if "bytes" in name:
            return x * {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}[u[i]]
        return x
    return g


def main():
    d = Path(sys.argv[1])
    out = Path(sys.argv[2])
    rows = []
    for f in sorted(d.glob("*.ncu-rep")):
        stem = f.stem
        name, rest = stem.rsplit("_n", 1)
        n, bh = int(rest.split("_")[0]), int(rest.split("_bh")[1].split("_")[0])
        kst = int(name.split("k")[1][0]) if not name.startswith("orig") else int(name[-1])
        tok = n * bh
        g = load(f)
        x = dict(name=name, kst=kst, tok=tok, t_us=g("gpu__time_duration.sum"),
                 inst=g("smsp__inst_executed.sum") / tok, fma=g("smsp__inst_executed_pipe_fma.sum") / tok,
                 fp16=g("smsp__inst_executed_pipe_fma_type_fp16.sum") / tok,
                 alu=g("smsp__inst_executed_pipe_alu.sum") / tok, lsu=g("smsp__inst_executed_pipe_lsu.sum") / tok,
                 xu=g("smsp__inst_executed_pipe_xu.sum") / tok, hmma=g("smsp__inst_executed_pipe_tensor_op_hmma.sum") / tok,
                 uni=g("smsp__inst_executed_pipe_uniform.sum") / tok, adu=g("smsp__inst_executed_pipe_adu.sum") / tok,
                 cbu=g("smsp__inst_executed_pipe_cbu.sum") / tok,
                 ffma=g("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") / tok / 32,
                 fadd=g("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") / tok / 32,
                 fmul=g("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") / tok / 32,
                 hfma=g("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum") / tok / 32,
                 hadd=g("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum") / tok / 32,
                 lds=g("smsp__sass_inst_executed_op_shared_ld.sum") / tok,
                 sts=g("smsp__sass_inst_executed_op_shared_st.sum") / tok,
                 wf_ld=g("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") / tok,
                 bc_ld=g("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum") / tok,
                 wf_st=g("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") / tok,
                 issue=g("smsp__issue_active.avg.pct_of_peak_sustained_active"),
                 fmapct=g("sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
                 dram=g("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
                 dram_rd=g("dram__bytes_read.sum"))
        rows.append(x)
    L = ["# E10 phase 3: instruction mix per token (Nsight Compute on RTX 4090) and throughput-bound model\n",
         f"- reports: `{d}` (ncu_p3.sh: one launch at the reference shape, KST1/2, n 32768, B*Hkv 32, chunk 8192 = "
         "splits 4; ncu serializes and flushes caches so its times differ from graph timing).",
         "- all counts are warp-level instructions per token (one token = one (b, kv-head, position): 4 q-heads x 128 "
         "dims); thread-level SASS op counts are divided by 32. fma = FMA pipe incl. fp16; fp32 = fma - fp16.\n",
         "| impl | t us (ncu) | inst | fp32 (fma-fp16) | fp16 | alu | lsu | xu | hmma | FFMA | FADD | FMUL | HFMA2 | "
         "HADD2 | LDS | STS | smem ld wavefr. | ld bank confl. | st wavefr. | issue % | FMA pipe % | DRAM % |",
         "|---|" + "---|" * 21]
    for x in rows:
        L.append(f"| {x['name']} | {x['t_us']:.1f} | {x['inst']:.1f} | {x['fma'] - x['fp16']:.1f} | {x['fp16']:.1f} | "
                 f"{x['alu']:.1f} | {x['lsu']:.1f} | {x['xu']:.2f} | {x['hmma']:.2f} | {x['ffma']:.1f} | "
                 f"{x['fadd']:.1f} | {x['fmul']:.1f} | {x['hfma']:.1f} | {x['hadd']:.1f} | {x['lds']:.1f} | "
                 f"{x['sts']:.1f} | {x['wf_ld']:.1f} | {x['bc_ld']:.1f} | {x['wf_st']:.1f} | {x['issue']:.0f} | "
                 f"{x['fmapct']:.0f} | {x['dram']:.0f} |")
    L.append("\n## Throughput-bound model (us at the reference shape: B*Hkv 32 x n 32768 = 1.05 M tokens)\n")
    L.append("Bounds in us for the whole call; `max` = perfect overlap (optimistic), `sum` = no overlap of the "
             "on-chip bounds, max'd with DRAM (pessimistic). A100 numbers are an [I] estimate from 4090 SASS counts "
             "(sm_80 SASS differs slightly), not a measurement.\n")
    L.append("| impl | device | issue | fp32 | fp16 | alu | smem | tensor | dram | bound (max) | max..sum | "
             "dominant |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        for dn, dv in DEV.items():
            cyc = dv["clk"] * dv["sms"]  # SM-cycles per second
            per = dict(issue=x["inst"] / 4, fp32=(x["fma"] - x["fp16"]) / (4 * dv["r32"]),
                       fp16=x["fp16"] / (4 * dv["r16"]), alu=x["alu"] / 2, smem=x["wf_ld"] + x["wf_st"],
                       tensor=x["hmma"] / (4 * dv["rT"]))
            us = {k: v * x["tok"] / cyc * 1e6 for k, v in per.items()}
            us["dram"] = 2 * x["kst"] * 16 * x["tok"] / dv["bw"] * 1e6
            mx = max(us.values())
            sm = max(us["dram"], us["fp32"] + us["fp16"] + us["alu"] + us["smem"] + us["tensor"])
            dom = max(us, key=us.get)
            L.append(f"| {x['name']} | {dn} | " + " | ".join(f"{us[k]:.0f}" for k in
                                                           ("issue", "fp32", "fp16", "alu", "smem", "tensor", "dram")) +
                     f" | {mx:.0f} | {mx:.0f}..{sm:.0f} | {dom} |")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
