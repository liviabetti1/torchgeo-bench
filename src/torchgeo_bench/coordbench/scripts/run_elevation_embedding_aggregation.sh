#!/usr/bin/env bash
# Elevation (no native timestamp) with and without temporal discretization
set -euo pipefail

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in sincos mind geoclip satclip sinr; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --output "$RESULTS_DIR/coordbench_satclip_elevation.csv" "$@"
done

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --temporal-sampling-year 2019 --temporal-sampling-methods year --temporal-discretization-methods weekly \
    --skip-no-timestamps --output "$RESULTS_DIR/coordbench_satclip_elevation.csv" "$@"
done
