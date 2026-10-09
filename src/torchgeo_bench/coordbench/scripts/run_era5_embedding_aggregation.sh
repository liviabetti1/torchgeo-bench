#!/usr/bin/env bash
# ERA5 across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset era5_ecmwf --split both \
    --temporal-discretization-methods 1_week 2_week 4_week 13_week annual concat_four_seasons \
    --output "$RESULTS_DIR/coordbench_era5_embedding_aggregation.csv" "$@"
done
