#!/usr/bin/env bash
# ERA5 (daily, 2017): native timestamps only. Pass --device cuda:N --knn-device cuda:N as extra args.
set -euo pipefail
mkdir -p results/coordbench/temporal

MODELS=(
  "t_satclip/t_satclip_toy_y_300k_satcliploss"
  "t_satclip/t_satclip_toy_y_300k_softloss"
  "t_satclip/t_satclip_toy_y_100k"
  "t_satclip/t_satclip_toy_y_150k"
  "t_satclip/t_satclip_toy_y_200k"
  "t_satclip/t_satclip_toy_y_250k"
  "t_satclip/t_satclip_toy_y_300k"
  "t_satclip/t_satclip_y_300k"
  "t_satclip/t_satclip_toy_300k"
  climplicit
  gtloc
)

# Linear for every model first, then IDW.
for method in linear idw; do
  for model in "${MODELS[@]}"; do
    torchgeo-bench coord --model "$model" \
      --dataset era5_ecmwf \
      --methods "$method" \
      --nested-splits random,random temporal_checkerboard,random \
      --time-units month,1 \
      --resume \
      --output results/coordbench/temporal/era5_ecmwf.csv "$@"
  done
done
