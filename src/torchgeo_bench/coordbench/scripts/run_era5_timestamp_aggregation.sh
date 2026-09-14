#!/usr/bin/env bash
# ERA5 across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
#for model in sincos mind mind-small geoclip satclip sinr climplicit gtloc t_satclip; do
for model in geoclip satclip sinr; do
  torchgeo-bench run device=cuda:0 mode=coord model="$model" coord.names=era5_ecmwf coord.split=both \
    coord.temporal_aggregation_methods=[1_week,2_week,4_week,13_week,52_week] \
    coord.aggregate_embeddings=false \
    coord.output=results/coordbench_era5_spatiotemporal_encoders.csv "$@"
done
