#!/usr/bin/env bash
# nsys per-op profile of the synthetic C1 decode step for the given model tags (72b, 32b, 8b) on the visible GPU.
set -eu
cd /hai/home/lsh/SpecRQ
O=/data/lsh/temp/prof72; mkdir -p $O
declare -A MP=([72b]=/data/models/qwen2.5_72b_instruct [32b]=/data/models/qwen2.5_32b_instruct [8b]=/data/models/llama3.1_8b_instruct)
declare -A RN=([72b]=qwen72b_synth_prof_c1 [32b]=q32b_synth_prof_c1 [8b]=llama8b_synth_prof_c1)
for t in "$@"; do
  echo "[prof] $t $(date -Is) co-tenants: $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr '\n' ';')"
  /usr/local/cuda/bin/nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop \
    -t cuda,nvtx -f true -o $O/$t python experiments/22_scale72/prof_synth.py --model ${MP[$t]} --meta $O/${t}_meta.json
  /usr/local/cuda/bin/nsys export --type sqlite -f true -o $O/$t.sqlite $O/$t.nsys-rep > /dev/null
  python experiments/22_scale72/prof_analyze.py --sqlite $O/$t.sqlite --meta $O/${t}_meta.json --name $t --run ${RN[$t]}
done
