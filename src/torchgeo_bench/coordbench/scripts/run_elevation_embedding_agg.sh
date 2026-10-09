#!/usr/bin/env bash
# Elevation (no native timestamp) with and without temporal discretization
set -euo pipefail

TODO

RESULTS_DIR=/projects/bgtj/t_satclip/results

for model in sincos mind geoclip satclip sinr; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --no-skip-no-timestamps --output "$RESULTS_DIR/coordbench_satclip_elevation.csv" "$@"
done

for model in climplicit t_satclip/t_satclip_doy_1M gtloc; do
  torchgeo-bench coord --model "$model" --dataset satclip-elevation --split both \
    --temporal-discretization-methods yearly \
    --skip-no-timestamps --output "$RESULTS_DIR/coordbench_satclip_elevation.csv" "$@"
done
