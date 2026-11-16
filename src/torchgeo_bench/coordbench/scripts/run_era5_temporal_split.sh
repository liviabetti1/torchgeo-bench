#!/usr/bin/env bash
# ERA5 ECMWF 2017 daily: temporal-block cross-validation over embeddings aggregated at 4 window sizes.
# --temporal-discretization-methods sets the aggregation window; --time-units (unit,size pairs)
# set the width of the time blocks that are held out together in each fold.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=$RESULTS_HOME/era5_temporal_split # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
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
        --nested-splits temporal_checkerboard,random \
        --folds 5 \
        --time-units week,1 month,1 quarter,1 year,1 \
        --temporal-discretization-methods weekly biweekly 4_weeks 13_weeks \
        --resume \
        --output "$out" \
        "$@"

    rm -f "${out}.lock"
done
