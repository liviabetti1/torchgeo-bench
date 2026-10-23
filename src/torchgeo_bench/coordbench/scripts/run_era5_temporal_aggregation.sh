#!/usr/bin/env bash
# ERA5 ECMWF 2017 daily: temporal embedding aggregation at 4 window sizes.
# Each model encodes a sliding window of timestamps per location and mean-pools.
# Runs with and without Antarctica to compare impact on global climate models.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=$RESULTS_HOME/era5_temporal_aggregation # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
    climplicit
    gtloc
    "t_satclip/t_satclip_doy_1M"
)

out="$RESULTS/era5_ecmwf.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset era5_ecmwf \
        --split all \
        --temporal-discretization-methods weekly biweekly 4_weeks 13_weeks \
        --resume \
        --output "$out" \
        "$@"

    torchgeo-bench coord \
        --model "$model" \
        --dataset era5_ecmwf \
        --split all \
        --temporal-discretization-methods weekly biweekly 4_weeks 13_weeks \
        --exclude-antarctica \
        --resume \
        --output "$out" \
        "$@"

    rm -f "${out}.lock"
done
