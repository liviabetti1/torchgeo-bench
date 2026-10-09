#!/usr/bin/env bash
# CHELSA across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset chelsa --split both \
    --temporal-aggregation-methods weekly biweekly monthly seasonally \
    --output "$RESULTS_DIR/coordbench_chelsa_embedding_aggregation.csv" \
    --device cuda:0 "$@"
done
