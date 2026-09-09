#!/usr/bin/env bash
# ERA5 across temporal aggregations (1/2/4/13-week), one row per model.
set -euo pipefail
for model in sincos mind mind-small geoclip satclip sinr climplicit gtloc t_satclip; do
  torchgeo-bench run mode=coord model="$model" coord.names=era5_ecmwf coord.split=both \
    coord.output=results/coordbench_era5.csv "$@"
done
