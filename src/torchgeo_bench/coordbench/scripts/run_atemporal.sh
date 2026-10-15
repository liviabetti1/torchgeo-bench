#!/usr/bin/env bash
# Standard CoordBench: 60 static benchmarks across all coordinate encoders.
# Pass --device cuda:0 (or cpu) as an extra arg when calling directly.
set -euo pipefail

RESULTS=/home/libe2152/projects/torchgeo-bench/results/coordbench/atemporal # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
    sincos
    satclip
    geoclip
    mind
    sinr
    climplicit
    gtloc
    "t_satclip/t_satclip_doy_1M"
)

DATASETS=(
    pdfm
    air_temp
    california_housing
    satclip
    sustainbench
    better_together
    cdc_places
    usavars
    country
    ecoregions
    worldclim
    soilgrids
    deepmind
)

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        out="$RESULTS/${dataset}.csv"
        torchgeo-bench coord \
            --model "$model" \
            --dataset "$dataset" \
            --split both \
            --resume \
            --output "$out" \
            "$@"
        rm -f "${out}.lock"
    done
done
