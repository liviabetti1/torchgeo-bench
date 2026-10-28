#!/usr/bin/env bash
# Standard CoordBench: 60 static benchmarks across all coordinate encoders. Spatiotemporal encoders
# are also run with timestamps sampled over 2017 (daily/weekly/monthly) into temporal_aggregation/.
# Pass --device cuda:0 (or cpu) as an extra arg when calling directly.
set -euo pipefail

RESULTS=results/coordbench/atemporal
mkdir -p "$RESULTS/temporal_aggregation"

SPATIAL_MODELS=(
    sincos
    satclip
    geoclip
    mind
    sinr
    tte
    taxabind
    sled
)

SPATIOTEMPORAL_MODELS=(
    climplicit
    gtloc
    t_satclip/t_satclip_doy_1M
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
)

for model in "${SPATIAL_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        out="$RESULTS/${dataset}.csv"
        torchgeo-bench coord \
            --model "$model" \
            --dataset "$dataset" \
            --split all \
            --resume \
            --output "$out" \
            "$@"
        rm -f "${out}.lock"
    done
done

for model in "${SPATIOTEMPORAL_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        out="$RESULTS/temporal_aggregation/${dataset}.csv"
        torchgeo-bench coord \
            --model "$model" \
            --dataset "$dataset" \
            --split all \
            --temporal-sampling-methods year \
            --temporal-sampling-year 2017 \
            --temporal-discretization-methods daily weekly monthly \
            --resume \
            --output "$out" \
            "$@"
        rm -f "${out}.lock"
    done
done
