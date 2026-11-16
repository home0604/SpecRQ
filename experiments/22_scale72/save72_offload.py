"""Collect the 72B synthetic offload numbers into results/E4/qwen72b_synth_offload.json:
CPU residual-plane sweep (bench_offload72.py --target cpu, 2 planes), PCIe split-stage round trips (72B stage sizes),
the real split-forward measurements (q72_synth.py --mode split) and the GPU-resident AR (q72_synth.py --mode gpu),
plus break-even tau = round_ms / T_AR (measured rounds only; no modeled speed-up)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from tools.results import save_result  # noqa: E402

D = "/data/lsh/temp/tmp/offload72"
R = os.path.join(HERE, "..", "..", "results", "E4")
cpu = json.load(open(f"{D}/cpu72_t16.json"))
pcie = json.load(open(f"{D}/pcie72.json"))
loads = {f: open(f"{D}/{f}").read() for f in sorted(os.listdir(D)) if f.startswith("load_")}
split = json.load(open(f"{R}/qwen72b_synth_split_c1.json"))["metrics"]["split"]
gpu = json.load(open(f"{R}/qwen72b_synth_gpu_c1.json"))["metrics"]["gpu"]

T_cpu = {r["M"]: {"wall_ms": 1e3 * r["wall_s"]["median"], "min_ms": 1e3 * r["wall_s"]["min"],
                  "max_ms": 1e3 * r["wall_s"]["max"], "GBps_codes": r["GBps_codes"]} for r in cpu["sweep"]["runs"]}
pc = pcie["pcie"]["stages"]
T_pcie = {M: sum(pc[f"{s}_M{M}"]["roundtrip_us"]["median"] for s in ("qkv", "o", "gateup", "down")) * 80 / 1e3
          for M in (1, 2, 3, 4, 5)}
T_ar_gpu_k2 = gpu["ar"][0]["step_us"]["median"] / 1e3
T_ar_off = split["ar_offloaded_k4"]["step_ms"]["median"]
be = []
for r in split["spec_round"]:
    g, Rm = r["gamma"], r["round_ms"]["median"]
    be.append({"gamma": g, "round_ms": Rm, "draft_ms": r["draft_ms"]["median"], "verify_ms": r["verify_ms"]["median"],
               "tau_breakeven_vs_offloaded_k4_AR": Rm / T_ar_off, "tau_breakeven_vs_gpu_k2_AR": Rm / T_ar_gpu_k2,
               "tau_max": g + 1})
for b in be:
    print(b)
print("T_cpu (2 planes, 80L x 7 lin, ms):", {M: round(v["wall_ms"], 1) for M, v in T_cpu.items()})
print("T_pcie (4 stage round trips x 80 layers, ms):", {M: round(v, 2) for M, v in T_pcie.items()})
print("T_AR gpu k2", T_ar_gpu_k2, "T_AR offloaded k4", T_ar_off)
print(save_result("E4", "qwen72b_synth_offload", config={
    "setting": "SYNTHETIC Qwen2.5-72B shapes, nested d8 K256: stages 1-2 GPU (C1 kernels, RTX 4090 GPU6, NUMA node1), "
               "stages 3-4 = 2 residual planes on CPU (node1, perm_i8_tile int8-LUT kernel, 16 OMP threads, cores 16-31)",
    "cpu_bench": "experiments/22_scale72/bench_offload72.py --target cpu --part stream sweep --kinds 3 --planes 2 "
                 "--Ms 1 2 3 4 5 8 --reps 5 (OMP_NUM_THREADS=16, OMP_PLACES=cores); the 32-thread run was aborted "
                 "(15 s per sweep under co-load)",
    "pcie_bench": "bench_offload72.py --target pcie --Ms 1..5 (bench_gpu.part_pcie with 72B stage sizes)",
    "split_forward": "q72_synth.py --mode split: REAL eager split forward; per linear (7 per layer, q/k/v separate) "
                     "synchronous D2H of the rotated input, CPU LUT kernels for planes 3 and 4, H2D add; GPU k2 part "
                     "runs async meanwhile. Not optimized (no q/k/v or gate/up sharing of the D2H, no pipelining).",
    "spec_round": "gamma k2 draft steps (one CUDA graph) + one eager k4 split verify of gamma+1 rows, forced "
                  "acceptance (round time only); break-even tau = round_ms / T_AR",
    "load_note": "node1 was heavily co-loaded during the CPU bench and the split run (load average 34-56; on node1: "
                 "train_draft_codebooks ~690% CPU, draft_target_eval ~460% CPU, xtx_dump ~100% CPU); CPU-side numbers "
                 "are pessimistic and noisy",
    "load_records": loads, "cpu_loadavg": [cpu.get("loadavg_start"), cpu.get("loadavg_end")],
    "split_loadavg": [split.get("loadavg_start"), split.get("loadavg_end")]},
    metrics={"T_cpu_2planes_ms": T_cpu, "cpu_stream": cpu["stream"], "T_pcie_4stage_x80_ms": T_pcie,
             "pcie_raw": pcie["pcie"], "split": split, "T_ar_gpu_k2_ms": T_ar_gpu_k2,
             "T_ar_offloaded_k4_ms": T_ar_off, "breakeven": be},
    model_dir="/data/models/qwen2.5_72b_instruct"))
