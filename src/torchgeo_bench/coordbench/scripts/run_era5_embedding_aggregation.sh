#!/usr/bin/env bash
# ERA5 across temporal discretizations (weekly/biweekly/monthly/seasonally), one row per model.
set -euo pipefail

RESULTS_DIR=/u/lbetti/projects/torchgeo-bench/results/

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset era5_ecmwf --split both \
    --temporal-discretization-methods 1_week 2_weeks 4_weeks 13_weeks \
    --output "$RESULTS_DIR/coordbench_era5_embedding_aggregation.csv" \
    --device cuda:0 "$@"
done

