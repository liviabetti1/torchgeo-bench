#!/usr/bin/env bash
# Weekly USA electrical load, 2016-2023. Pass --device cuda:N --knn-device cuda:N as extra args.
set -euo pipefail
source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME/year_patch # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

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
      --dataset usa_electric_load_weekly \
      --methods "$method" \
      --nested-splits random,random temporal_checkerboard,random \
      --time-units year,1 season,1 \
      --resume \
      --output "$RESULTS/usa_electrical_load_weekly.csv" "$@"
  done
done
