#!/usr/bin/env bash
# CHELSA monthly: each month embedded at weekly and daily timestamps, then pooled. Pass --device cuda:0 (or cpu) as an extra arg.
set -euo pipefail

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
#  "t_satclip/t_satclip_toy_y_250k_softloss"
  "t_satclip/t_satclip_toy_y_300k"
  "t_satclip/t_satclip_toy_y_300k_satcliploss"
  "t_satclip/t_satclip_toy_y_300k_softloss"
  "t_satclip/t_satclip_y_200k"
  "t_satclip/t_satclip_y_300k"
)

out="$RESULTS/chelsa_monthly.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset chelsa_monthly \
        --split all \
        --exclude-antarctica \
        --output "$out" \
        --temporal-checkerboard \
        "$@"

    rm -f "${out}.lock"
done
