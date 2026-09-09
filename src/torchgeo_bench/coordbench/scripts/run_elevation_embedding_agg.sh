#!/usr/bin/env bash
# Elevation (no native timestamp) across embedding-aggregation settings: yearly/summer x freq.
set -euo pipefail
out=results/coordbench_elevation_embedding_agg.csv
for model in sincos geoclip satclip sinr; do
  torchgeo-bench run mode=coord model="$model" coord.names=satclip-elevation coord.output="$out" "$@" coord.skip_no_timestamp=false coord.split=both
done

for model in mind mind-small climplicit gtloc t_satclip; do
  for month in 1 8; do
    torchgeo-bench run mode=coord model="$model" coord.names=satclip-elevation \
      +model.default_month="$month" coord.output="$out" "$@" coord.skip_no_timestamp=false coord.split=both
  done
done

for model in gtloc t_satclip; do
  for freq in MS; do
    for summer in true false; do
      torchgeo-bench run mode=coord model="$model" coord.names=satclip-elevation \
        coord.aggregate_embeddings=true coord.aggregate_embeddings_freq="$freq" \
        coord.aggregate_embeddings_summer="$summer" coord.output="$out" "$@" \
        coord.skip_no_timestamp=false coord.split=both
    done
  done
done
