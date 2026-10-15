#!/usr/bin/env bash
# DeepMind us_trees benchmark across all coordinate encoders.
# This benchmark has per-point timestamps; temporal models may use them directly.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail


RESULTS=/home/libe2152/projects/torchgeo-bench/results # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH

MODELS=(sincos satclip geoclip mind mind-small sinr climplicit gtloc "t_satclip/t_satclip_doy_1M")

for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset dm-us_trees \
        --split both \
        --resume \
        --output "$RESULTS/coordbench_dm_us_trees.csv" \
        "$@"
done