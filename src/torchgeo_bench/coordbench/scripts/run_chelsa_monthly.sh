#!/usr/bin/env bash
# CHELSA monthly. Runs on CUDA; pass --device cuda:N --knn-device cuda:N (or cpu) to override.
set -euo pipefail

RESULTS=$RESULTS_HOME # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
  "climplicit"
  "t_satclip/t_satclip_toy_y_300k_satcliploss"
  "t_satclip/t_satclip_toy_y_300k_softloss"
  "t_satclip/t_satclip_toy_y_100k"
  "t_satclip/t_satclip_toy_y_150k"
  "t_satclip/t_satclip_toy_y_200k"
  "t_satclip/t_satclip_toy_y_250k"
  "t_satclip/t_satclip_toy_y_300k"
  "t_satclip/t_satclip_y_300k"
  "t_satclip/t_satclip_toy_300k"
  gtloc
)

out="$RESULTS/chelsa_monthly.csv"
SPLITS=(--nested-splits random,random temporal_checkerboard,random --time-units year,1 season,1)

# IDW reads only coordinates, so it runs once without a model.
torchgeo-bench coord --dataset chelsa_monthly --methods idw "${SPLITS[@]}" --resume --output "$out" --device cuda --knn-device cuda "$@"

for model in "${MODELS[@]}"; do
  torchgeo-bench coord --model "$model" --dataset chelsa_monthly --methods linear "${SPLITS[@]}" --resume --output "$out" --device cuda --knn-device cuda "$@"
  rm -f "${out}.lock"
done
