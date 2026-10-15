#!/usr/bin/env bash
# USA county-level daily electric load: spatial aggregation of sampled points per county.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail


RESULTS=/home/libe2152/projects/torchgeo-bench/results/coordbench/electrical_load # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
    climplicit
    gtloc
    "t_satclip/t_satclip_doy_1M"
)

out="$RESULTS/usa_electric_load_daily.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset usa_electric_load_daily \
        --split both \
        --from-polygon \
        --spatial-aggregation-methods statistical median_and_iqr \
        --resume \
        --output "$out" \
        "$@"
    rm -f "${out}.lock"
done
