#!/usr/bin/env bash
# E7 MagicDec repro: official MagicDec sweep, Llama-3.1-8B-Instruct, PG-19 (official loader), prefill 32672 (= 32k
# with the official (P-32) % 128 == 0 rule), max_len 32768, TP=4 on GPUs 4-7, --compile --benchmark.
# Run via: tools/run_tmux.sh magicdec_off_<x> 4,5,6,7 /data/lsh/envs/magicdec25 -- bash experiments/27_magicdec_repro/run_all.sh <set>
set -u
H=$(cd "$(dirname "$0")" && pwd)
M=/data/lsh/ckpts/magicdec/Llama-3.1-8B-Instruct/model.pth
MN=/data/models/llama3.1_8b_instruct
C="--model $M --model_name $MN --rank_group 0 1 2 3 --prefix_len 32672 --max_len 32768 --compile"
run() { local tag=$1; shift; [[ -e /data/lsh/logs/magicdec/$tag.co.json ]] && { echo "skip $tag (done)"; return; }
        bash "$H/run_official.sh" "$tag" "$@" && python "$H/parse_official.py" "$tag" --save | tail -3; }
case ${1:-main} in
  b8)
    run ar_b8      tests/baseline_benchmark.py $C --B 8
    run snap4097_g3_b8 tests/SnapKV/selfspec_benchmark.py $C --B 8 --gamma 3 --draft_budget 4097 --benchmark
    run snap4097_g4_b8 tests/SnapKV/selfspec_benchmark.py $C --B 8 --gamma 4 --draft_budget 4097 --benchmark
    run stream513_g3_b8 tests/StreamingLLM/selfspec_benchmark.py $C --B 8 --gamma 3 --draft_budget 513 --benchmark
    ;;
  b16)
    run ar_b16     tests/baseline_benchmark.py $C --B 16
    run snap4097_g3_b16 tests/SnapKV/selfspec_benchmark.py $C --B 16 --gamma 3 --draft_budget 4097 --benchmark
    run snap4097_g4_b16 tests/SnapKV/selfspec_benchmark.py $C --B 16 --gamma 4 --draft_budget 4097 --benchmark
    run stream513_g3_b16 tests/StreamingLLM/selfspec_benchmark.py $C --B 16 --gamma 3 --draft_budget 513 --benchmark
    run stream513_g4_b16 tests/StreamingLLM/selfspec_benchmark.py $C --B 16 --gamma 4 --draft_budget 513 --benchmark
    ;;
  b16b)  # SnapKV-4097 does not fit at B16 x 32k on 24 GB (OOM in prefill); smaller official budget
    run snap2049_g3_b16 tests/SnapKV/selfspec_benchmark.py $C --B 16 --gamma 3 --draft_budget 2049 --benchmark
    run snap2049_g4_b16 tests/SnapKV/selfspec_benchmark.py $C --B 16 --gamma 4 --draft_budget 2049 --benchmark
    ;;
  b16c)  # 2049 also OOMs (batch 3 / batch 2, SnapKV scoring softmax in prefill); budget 513 + expandable segments
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run snap513_g3_b16 tests/SnapKV/selfspec_benchmark.py $C --B 16 --gamma 3 --draft_budget 513 --benchmark
    ;;
  *) echo "unknown set $1"; exit 2;;
esac
