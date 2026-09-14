#!/usr/bin/env bash
# ERA5 across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail
#for model in climplicit gtloc t_satclip; do
for model in t_satclip gtloc; do
  torchgeo-bench run mode=coord model="$model" coord.names=era5_ecmwf coord.split=both \
    coord.temporal_aggregation_methods=[1_week,2_week,4_week,13_week,52_week] \
    coord.output=results/coordbench_era5_spatiotemporal_encoders_aggregated_embeddings.csv "$@"
done
