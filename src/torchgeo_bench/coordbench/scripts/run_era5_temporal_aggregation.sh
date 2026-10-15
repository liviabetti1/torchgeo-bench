#!/usr/bin/env bash
# ERA5 ECMWF 2017 daily: temporal embedding aggregation at 4 window sizes.
# Each model encodes a sliding window of timestamps per location and mean-pools.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=/home/libe2152/projects/torchgeo-bench/results # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH

MODELS=(
    climplicit
    gtloc
    "t_satclip/t_satclip_doy_1M"
)

for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset era5_ecmwf \
        --split both \
        --temporal-discretization-methods 1_week 2_weeks 4_weeks 13_weeks \
        --resume \
        --output "$RESULTS/coordbench_era5_temporal.csv" \
        "$@"
done

