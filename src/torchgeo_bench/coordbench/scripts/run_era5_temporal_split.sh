#!/usr/bin/env bash
# ERA5 ECMWF 2017 daily: temporal-block cross-validation over embeddings aggregated at 4 window sizes.
# --temporal-discretization-methods sets the aggregation window; --temporal-grid-sizes sets the
# width of the time blocks that are held out together in each fold.
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
        --split temporal \
        --temporal-grid-sizes weekly monthly seasonally yearly \
        --temporal-discretization-methods weekly biweekly 4_weeks 13_weeks \
        --exclude-antarctica \
        --resume \
        --output "$out" \
        "$@"

    rm -f "${out}.lock"
done
