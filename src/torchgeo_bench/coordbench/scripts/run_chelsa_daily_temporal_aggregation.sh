#!/usr/bin/env bash
# CHELSA 2017 daily climate: temporal embedding aggregation at 4 window sizes.
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
        --dataset chelsa_daily \
        --split both \
        --temporal-discretization-methods 1_week 2_weeks 4_weeks 13_weeks \
        --resume \
        --output "$RESULTS/coordbench_chelsa_daily_temporal.csv" \
        "$@"
done
