#!/usr/bin/env bash
# Elevation (no native timestamp) across embedding-aggregation temporal methods
# (annual, summer, default_date_winter, default_date_summer, concat_four_seasons).
set -euo pipefail
out=results/coordbench_elevation_embedding_agg.csv

for model in sincos mind mind-small geoclip satclip sinr; do
  torchgeo-bench run mode=coord model="$model" coord.names=satclip-elevation coord.split=both \
    coord.aggregate_embeddings=false coord.skip_no_timestamp=false coord.output="$out" "$@"
done

for model in climplicit t_satclip gtloc; do
  torchgeo-bench run mode=coord model="$model" coord.names=satclip-elevation coord.split=both \
    coord.temporal_aggregation_methods=[annual,summer,default_date_winter,default_date_summer,concat_four_seasons] \
    coord.aggregate_embeddings=true coord.skip_no_timestamp=false coord.output="$out" "$@"
done
