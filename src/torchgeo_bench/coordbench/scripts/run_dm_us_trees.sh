#!/usr/bin/env bash
# DeepMind us_trees benchmark, one row per model.

RESULTS_DIR=/projects/bgtj/t_satclip/results

set -euo pipefail

for model in gtloc t_satclip/t_satclip_doy_1M; do #sincos mind geoclip satclip sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord \
    --model "$model" \
    --dataset dm-us_trees \
    --split both \
    --output "$RESULTS_DIR/coordbench_dm_us_trees.csv" \
    "$@"
done