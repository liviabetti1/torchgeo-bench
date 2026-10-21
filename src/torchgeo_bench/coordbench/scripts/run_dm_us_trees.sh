#!/usr/bin/env bash
# DeepMind us_trees benchmark across all coordinate encoders.
# This benchmark has per-point timestamps; temporal models may use them directly.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail


RESULTS=/home/libe2152/projects/torchgeo-bench/results/coordbench/dm_us_trees # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(sincos satclip geoclip mind sinr climplicit gtloc "t_satclip/t_satclip_doy_1M")

out="$RESULTS/dm-us_trees.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset dm-us_trees \
        --split both \
        --resume \
        --output "$out" \
        "$@"
    rm -f "${out}.lock"
done