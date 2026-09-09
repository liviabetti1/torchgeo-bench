#!/usr/bin/env bash
# DeepMind us_trees benchmark, one row per model.
set -euo pipefail
for model in sincos mind mind-small geoclip satclip sinr climplicit gtloc t_satclip; do
  torchgeo-bench run mode=coord model="$model" coord.names=dm-us_trees coord.split=both \
    coord.output=results/coordbench_dm_us_trees.csv "$@"
done
