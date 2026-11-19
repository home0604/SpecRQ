"""Summarize Nsight Compute reports of ncu_point.py runs into a markdown table.
Usage: python ncu_summary.py <dir with *.ncu-rep> <out.md> [title]"""
import csv
import io
import subprocess
import sys
from pathlib import Path

NCU = "/usr/local/cuda-12.1/bin/ncu"
M = [("time us", "gpu__time_duration.sum", 1e-3), ("DRAM %peak", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 1),
     ("DRAM rd MB", "dram__bytes_read.sum", 1), ("DRAM wr MB", "dram__bytes_write.sum", 1),
     ("achieved occ %", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
     ("issue active %", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1),
     ("L1/smem thr %", "l1tex__throughput.avg.pct_of_peak_sustained_active", 1),
     ("FMA pipe %", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", 1),
     ("smem ld wavefronts M", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 1e-6),
     ("smem ld bank-conflict M", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", 1e-6),
     ("inst M", "smsp__inst_executed.sum", 1e-6)]
UNIT = {"usecond": 1e3, "msecond": 1e6, "nsecond": 1, "Mbyte": 1, "Gbyte": 1e3, "Kbyte": 1e-3, "byte": 1e-6}


def val(h, u, v, name, scale):
    i = h.index(name)
    x = float(v[i].replace(",", ""))
    if name == "gpu__time_duration.sum":
        return x * UNIT[u[i]] * scale
    if u[i] in UNIT and "bytes" in name:
        return x * UNIT[u[i]]
    return x * scale


rows = []
for f in sorted(Path(sys.argv[1]).glob("*.ncu-rep")):
    txt = subprocess.run([NCU, "-i", str(f), "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    r = list(csv.reader(io.StringIO(txt)))
    h, u, v = r[0], r[1], r[2]
    vals = [val(h, u, v, n, s) for _, n, s in M]
    st = sorted(((h[i][len("smsp__average_warps_issue_stalled_"):-len("_per_issue_active.ratio")],
                  float(v[i].replace(",", ""))) for i in range(len(h))
                 if h[i].startswith("smsp__average_warps_issue_stalled_") and h[i].endswith("_per_issue_active.ratio")
                 and v[i].replace(",", "").replace(".", "").isdigit()), key=lambda x: -x[1])
    st = [x for x in st if x[0] not in ("selected", "not_selected")][:4]
    rows.append((f.stem, vals, st))
TITLE = sys.argv[3] if len(sys.argv) > 3 else "E10 phase 0"
out = [f"# {TITLE}: Nsight Compute on representative points (RTX 4090, ncu 2023 / CUDA 12.1, --set full)\n",
       "Point name = k{KST}_n{tokens}_bh{B*Hkv}_c{chunk}; single launch (ncu serializes and flushes caches, so times "
       "differ from the graph-timed sweep).\n",
       "| point | " + " | ".join(m[0] for m in M) + " | top stalls (cycles per issued inst) |",
       "|---|" + "---|" * (len(M) + 1)]
for nm, vals, st in rows:
    out.append(f"| {nm} | " + " | ".join(f"{x:.1f}" for x in vals) + " | " +
               ", ".join(f"{a} {b:.2f}" for a, b in st) + " |")
Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print("\n".join(out))
