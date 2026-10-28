# Running two tasks:
# - 8 daily electrical load datasets, one per year from 2016-2023.
# - 1 weekly electrical load dataset that spans all of 2016-2023
#set -euo pipefail
source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME/testing # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
#  climplicit
#  gtloc
#  "t_satclip/t_satclip_doy_1M"
#  "t_satclip/t_satclip_landsat_toy_300k"
#  "t_satclip/t_satclip_landsat_toy_y_300k"
#  "t_satclip/t_satclip_landsat_y_300k"
#  "t_satclip/t_satclip_toy_200k"
#  "t_satclip/t_satclip_toy_300k"
#  "t_satclip/t_satclip_toy_y_100k"
#  "t_satclip/t_satclip_toy_y_150k"
#  "t_satclip/t_satclip_toy_y_200k"
#  "t_satclip/t_satclip_toy_y_200k_satcliploss"
#  "t_satclip/t_satclip_toy_y_200k_softloss"
#  "t_satclip/t_satclip_toy_y_250k"
#  "t_satclip/t_satclip_toy_y_250k_softloss"
  "t_satclip/t_satclip_toy_y_300k"
#  "t_satclip/t_satclip_toy_y_300k_satcliploss"
#  "t_satclip/t_satclip_toy_y_300k_softloss"
#  "t_satclip/t_satclip_y_200k"
#  "t_satclip/t_satclip_y_300k"
)

out="$RESULTS/usa_electrical_load_weekly.csv"
for model in "${MODELS[@]}"; do
  torchgeo-bench coord --model "$model" \
  --dataset usa_electric_load_weekly \
  --temporal-discretization-methods two_days \
  --subsample \
  --temporal-checkerboard \
  --output "$out" "$@"
done
