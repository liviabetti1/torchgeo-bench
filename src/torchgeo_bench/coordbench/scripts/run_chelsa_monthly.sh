#!/usr/bin/env bash
# CHELSA monthly. Pass --device cuda:N --knn-device cuda:N as extra args.
set -euo pipefail

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

out="$RESULTS/chelsa_monthly.csv"
SPLITS=(--nested-splits random,random temporal_checkerboard,random --time-units year,1 season,1)

for model in "${MODELS[@]}"; do
  torchgeo-bench coord --model "$model" --dataset chelsa_monthly --methods linear "${SPLITS[@]}" --resume --output "$out" "$@"
  rm -f "${out}.lock"
done

# IDW reads only coordinates, so it runs once without a model.
torchgeo-bench coord --dataset chelsa_monthly --methods idw "${SPLITS[@]}" --resume --output "$out" "$@"
rm -f "${out}.lock"
