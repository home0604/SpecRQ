cd /hai/home/lsh/SpecRQ
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
T="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/rqd_train.py --R 128 --level 2 --steps 1000 --init /data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt --layer_list 1"
CUDA_VISIBLE_DEVICES=6 $T --tag t12w_v2l1w3 --nest 3 > /data/lsh/logs/v9_l1fix_w3.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 $T --tag t12w_v2l1w10 --nest 10 > /data/lsh/logs/v9_l1fix_w10.log 2>&1 &
wait; grep -h heldout_after /data/lsh/logs/v9_l1fix_*.log
