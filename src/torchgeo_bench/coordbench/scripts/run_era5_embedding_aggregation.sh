#!/usr/bin/env bash
# ERA5 across temporal discretizations (weekly/biweekly/monthly/seasonally), one row per model.
set -euo pipefail

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset era5_ecmwf --split both \
    --temporal-discretization-methods weekly biweekly monthly seasonally \
    --output "$RESULTS_DIR/coordbench_era5_embedding_aggregation.csv" \
    --device cuda:0 "$@"
done

