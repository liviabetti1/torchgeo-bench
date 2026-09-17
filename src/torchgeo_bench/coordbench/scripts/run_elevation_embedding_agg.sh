#!/usr/bin/env bash
# Elevation (no native timestamp) across embedding-aggregation temporal methods
set -euo pipefail
out=results/coordbench_elevation_embedding_agg.csv

for model in sincos mind mind-small geoclip satclip sinr; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --no-aggregate-embeddings --no-skip-no-timestamps --output "$out" "$@"
done

for model in climplicit t_satclip gtloc; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --temporal-aggregation-methods annual summer default_date_winter default_date_summer concat_four_seasons \
    --aggregate-embeddings --no-skip-no-timestamps --output "$out" "$@"
done
