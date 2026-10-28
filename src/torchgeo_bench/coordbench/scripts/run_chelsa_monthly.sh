#!/usr/bin/env bash
# CHELSA monthly climate 2011-2018: raw monthly embeddings, no temporal aggregation.
# Runs with and without Antarctica to compare impact on global climate models.
# Requires access to the Temporal-Coordbench HuggingFace repo.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=$RESULTS_HOME/factor_in_year # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
  climplicit
  gtloc
  "t_satclip/t_satclip_doy_1M"
  "t_satclip/t_satclip_landsat_toy_300k"
  "t_satclip/t_satclip_landsat_toy_y_300k"
  "t_satclip/t_satclip_landsat_y_300k"
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
        --temporal-grid-sizes monthly seasonally yearly 2_years 3_years 4_years \
        "$@"

    rm -f "${out}.lock"
done
