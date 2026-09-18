#!/usr/bin/env bash
# DeepMind us_trees benchmark, one row per model.
set -euo pipefail
for model in sincos; do
  torchgeo-bench coord --model "$model" --dataset usa_electric_usage --split both \
    --output results/coordbench_dm_us_trees.csv "$@"
done
