#!/usr/bin/env bash
# DeepMind us_trees benchmark, one row per model.
set -euo pipefail
for model in sincos mind mind-small geoclip satclip sinr climplicit gtloc t_satclip; do
  torchgeo-bench coord --model "$model" --dataset dm-us_trees --split both \
    --output results/coordbench_dm_us_trees.csv "$@"
done
