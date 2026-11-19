#!/usr/bin/env bash
# E10 phase 3: Nsight Compute instruction mix (per pipe) + smem wavefronts at the reference shape (B16 x Hkv2 = bh32,
# n 32768, chunk 8192 = splits 4) for original / p1 struct / p3 variants. Reports in /data/lsh/temp/e10_p3_ncu/.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-/data/lsh/temp/e10_p3_ncu}; mkdir -p "$OUT"
NCU=/usr/local/cuda-12.1/bin/ncu
MET=gpu__time_duration.sum,dram__bytes_read.sum,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_fma_type_fp16.sum,smsp__inst_executed_pipe_fmaheavy.sum,smsp__inst_executed_pipe_fmalite.sum,\
smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_xu.sum,\
smsp__inst_executed_pipe_tensor_op_hmma.sum,smsp__inst_executed_pipe_uniform.sum,smsp__inst_executed_pipe_cbu.sum,\
smsp__inst_executed_pipe_adu.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_inst_executed_op_shared_ld.sum,\
smsp__sass_inst_executed_op_shared_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
N=${N:-32768}; BH=${BH:-32}; CH=${CH:-8192}
for k in 1 2; do
  R=$([ $k = 1 ] && echo 8 || echo 2)
  for nm in orig_k$k p1:k${k}_w8_b1_r1_p1_l0_s2 k${k}_w8_v0_r1_h0_a0 k${k}_w8_v0_r1_h1_a0 k${k}_w16_v0_r1_h0_a0 \
            k${k}_w8_v1_r1_h0_a0 k${k}_w8_v1_r${R}_h0_a0 k${k}_w8_v1_r1_h1_a0 k${k}_w8_v1_r${R}_h1_a0 k${k}_w8_v2_r1_h0_a0 \
            k${k}_w8_v0_r1_h0_a1 k${k}_w8_v0_r1_h0_a2 k${k}_w8_v0_r1_h0_a3; do
    f="$OUT/${nm/:/_}_n${N}_bh${BH}_c${CH}"
    [ -e "$f.ncu-rep" ] && continue
    $NCU -k regex:'lut_' -s 2 -c 1 --metrics "$MET" -o "$f" python ncu_point_p3.py "$nm" $N $BH $CH > "$f.log" 2>&1 \
      || { echo "FAIL $nm"; tail -5 "$f.log"; }
    echo "done $nm"
  done
done
