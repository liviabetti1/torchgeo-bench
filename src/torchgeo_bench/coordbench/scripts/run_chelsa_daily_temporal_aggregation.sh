#!/usr/bin/env bash
# CHELSA 2017 daily climate: temporal embedding aggregation at 4 window sizes.
# Runs with and without Antarctica to compare impact on global climate models.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=/home/libe2152/projects/torchgeo-bench/results/coordbench/chelsa_daily_temporal_aggregation # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
    climplicit
    #gtloc
    "t_satclip/t_satclip_doy_1M"
)

out="$RESULTS/chelsa_daily.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset chelsa_daily \
        --split all \
        --temporal-discretization-methods 1_week 2_weeks 4_weeks 13_weeks \
        --resume \
        --output "$out" \
        "$@"

    torchgeo-bench coord \
        --model "$model" \
        --dataset chelsa_daily \
        --split all \
        --temporal-discretization-methods 1_week 2_weeks 4_weeks 13_weeks \
        --exclude-antarctica \
        --resume \
        --output "$out" \
        "$@"

    rm -f "${out}.lock"
done
