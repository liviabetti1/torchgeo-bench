#!/usr/bin/env bash
# Weekly USA electrical load, 2016-2023. Pass --device cuda:N --knn-device cuda:N as extra args.
set -euo pipefail
source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME/year_patch # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
  climplicit
  gtloc
  "t_satclip/t_satclip_doy_1M"
#  "t_satclip/t_satclip_landsat_toy_300k"
#  "t_satclip/t_satclip_landsat_toy_y_300k"
#  "t_satclip/t_satclip_landsat_y_300k"
  "t_satclip/t_satclip_toy_200k"
  "t_satclip/t_satclip_toy_300k"
  "t_satclip/t_satclip_toy_y_100k"
  "t_satclip/t_satclip_toy_y_150k"
  "t_satclip/t_satclip_toy_y_200k"
  "t_satclip/t_satclip_toy_y_200k_satcliploss"
  "t_satclip/t_satclip_toy_y_200k_softloss"
  "t_satclip/t_satclip_toy_y_250k"
#  "t_satclip/t_satclip_toy_y_250k_softloss" DOES NOT EXIST IN OUR CHECKPOINTS RIGHT NOW!
  "t_satclip/t_satclip_toy_y_300k"
  "t_satclip/t_satclip_toy_y_300k_satcliploss"
  "t_satclip/t_satclip_toy_y_300k_softloss"
  "t_satclip/t_satclip_y_200k"
  "t_satclip/t_satclip_y_300k"
)

SPLITS=(--nested-splits random,random temporal_checkerboard,random --time-units year,1 season,1)
out="$RESULTS/usa_electrical_load_weekly.csv"

for model in "${MODELS[@]}"; do
  torchgeo-bench coord --model "$model" --dataset usa_electric_load_weekly --methods linear "${SPLITS[@]}" --resume --output "$out" "$@"
done

# IDW reads only coordinates, so it runs once without a model.
torchgeo-bench coord --dataset usa_electric_load_weekly --methods idw "${SPLITS[@]}" --resume --output "$out" "$@"
