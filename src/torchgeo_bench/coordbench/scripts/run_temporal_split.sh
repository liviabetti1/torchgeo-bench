#!/usr/bin/env bash
# Temporal-block cross-validation: every benchmark that carries real per-point
# timestamps, evaluated under temporal block CV with a sweep of block widths.
# Pass --device cuda:0 (or cpu) as an extra arg when calling directly.
set -euo pipefail

RESULTS=$RESULTS_HOME/temporal_split # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
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

# Timestamped benchmarks; the electric-load ones are sampled per county polygon
# and need spatial aggregation before evaluation, like run_electrical_load.sh.
DATASETS=(
    era5_ecmwf
    chelsa_daily
    chelsa_monthly
    usa_electric_load_daily
    usa_electric_load_weekly
    usa_census
    global_population
    dm-us_trees
)

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        out="$RESULTS/${dataset}.csv"
        extra_args=()
        case "$dataset" in
            usa_electric_load_daily | usa_electric_load_weekly)
                extra_args=(--from-polygon --spatial-aggregation-methods statistical median_and_iqr)
                ;;
        esac
        torchgeo-bench coord \
            --model "$model" \
            --dataset "$dataset" \
            --nested-splits temporal_checkerboard,random \
            --folds 5 \
            --test-size 1 \
            --time-units week,1 month,1 quarter,1 year,1 \
            "${extra_args[@]}" \
            --resume \
            --output "$out" \
            "$@"
        rm -f "${out}.lock"
    done
done
