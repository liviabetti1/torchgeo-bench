#!/usr/bin/env bash
# CHELSA monthly climate 2011-2018: temporal aggregation at monthly/seasonal/annual scales.
# Requires access to the Temporal-Coordbench HuggingFace repo.
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
        --dataset chelsa_monthly \
        --split both \
        --temporal-discretization-methods monthly seasonally yearly \
        --resume \
        --output "$RESULTS/coordbench_chelsa_monthly_temporal.csv" \
        "$@"
done
