#!/usr/bin/env bash
# DeepMind US trees (per-point timestamps): native timestamps only. Pass --device cuda:0 (or cpu) as an extra arg.
set -euo pipefail
mkdir -p results/coordbench/temporal
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset dm-us_trees --split all --temporal-grid-sizes weekly days_30 13_weeks days_365 --resume --output results/coordbench/temporal/dm-us_trees.csv "$@"
done
