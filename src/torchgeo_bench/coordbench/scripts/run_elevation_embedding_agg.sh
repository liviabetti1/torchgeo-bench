#!/usr/bin/env bash
# Elevation (no native timestamp) across embedding-aggregation temporal methods
set -euo pipefail

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in sincos mind geoclip satclip sinr; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --temporally-aggregate-embeddings --no-skip-no-timestamps --output "$RESULTS_DIR/coordbench_satclip_elevation.csv" "$@"
done

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --temporal-aggregation-methods annual summer default_date_winter default_date_summer concat_four_seasons \
    --temporally-aggregate-embeddings --skip-no-timestamps --output "$RESULTS_DIR/coordbench_satclip_elevation.csv" "$@"
done
