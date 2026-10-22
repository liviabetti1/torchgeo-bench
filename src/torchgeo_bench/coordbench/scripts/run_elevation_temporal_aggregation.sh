#!/usr/bin/env bash
# SatCLIP elevation: static encoders as a baseline, then temporal models with
# synthetic seasonal timestamps (year 2019, weekly resolution).
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=/home/libe2152/projects/torchgeo-bench/results/coordbench/elevation_temporal_aggregation # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

STATIC_MODELS=(sincos satclip geoclip mind sinr)
TEMPORAL_MODELS=(climplicit "t_satclip/t_satclip_doy_1M" gtloc)

out="$RESULTS/satclip-elevation.csv"
for model in "${STATIC_MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset satclip-elevation \
        --split all \
        --resume \
        --output "$out" \
        "$@"
    rm -f "${out}.lock"
done

for model in "${TEMPORAL_MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset satclip-elevation \
        --split all \
        --temporal-sampling-year 2019 \
        --temporal-sampling-methods year spring summer fall winter \
        --temporal-discretization-methods weekly \
        --skip-no-timestamps \
        --resume \
        --output "$out" \
        "$@"
    rm -f "${out}.lock"
done
