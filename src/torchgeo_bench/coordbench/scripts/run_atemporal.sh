#!/usr/bin/env bash
# Standard CoordBench: 60 static benchmarks across all coordinate encoders.
# Pass --device cuda:0 (or cpu) as an extra arg when calling directly.
set -euo pipefail

RESULTS=/home/libe2152/projects/torchgeo-bench/results # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH

MODELS=(
    sincos
    satclip
    geoclip
    mind
    mind-small
    sinr
    climplicit
    gtloc
    "t_satclip/t_satclip_doy_1M"
)

for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset pdfm air_temp california_housing satclip sustainbench better_together \
                  cdc_places usavars country ecoregions worldclim soilgrids deepmind \
        --split both \
        --resume \
        --output "$RESULTS/coordbench_atemporal.csv" \
        "$@"
done
