#!/usr/bin/env bash
# CHELSA across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in sincos mind mind-small geoclip satclip sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --device cuda:0 --model "$model" --dataset chelsa --split both \
    --temporal-aggregation-methods 1_week 2_week 4_week 13_week 52_week \
    --no-aggregate-embeddings \
    --output "$RESULTS_DIR/coordbench_chelsa_timestamp_aggregation.csv" "$@"
done
