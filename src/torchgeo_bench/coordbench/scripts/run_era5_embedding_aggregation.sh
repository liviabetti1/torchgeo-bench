#!/usr/bin/env bash
# ERA5 across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail
for model in climplicit t_satclip gtloc; do
  torchgeo-bench coord --model "$model" --dataset era5_ecmwf --split both \
    --temporal-aggregation-methods 1_week 2_week 4_week 13_week 52_week \
    --output results/coordbench_era5_spatiotemporal_encoders_aggregated_embeddings.csv "$@"
done
