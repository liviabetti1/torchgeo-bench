#!/usr/bin/env bash
# CHELSA monthly climate 2011-2018: temporal aggregation at monthly/seasonal/annual scales.
# Runs with and without Antarctica to compare impact on global climate models.
# Requires access to the Temporal-Coordbench HuggingFace repo.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=/home/libe2152/projects/torchgeo-bench/results/coordbench/chelsa_monthly_temporal_aggregation # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
    climplicit
    #gtloc
    "t_satclip/t_satclip_doy_1M"
)

out="$RESULTS/chelsa_monthly.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset chelsa_monthly \
        --split all \
        --temporal-discretization-methods seasonally yearly \
        --resume \
        --output "$out" \
        "$@"

    torchgeo-bench coord \
        --model "$model" \
        --dataset chelsa_monthly \
        --split all \
        --temporal-discretization-methods seasonally yearly \
        --exclude-antarctica \
        --resume \
        --output "$out" \
        "$@"

    rm -f "${out}.lock"
done
