# Running USA census data
# yearly info from 2010-2025 at the county level
#set -euo pipefail

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
#  "t_satclip/t_satclip_toy_y_250k_softloss"
  "t_satclip/t_satclip_toy_y_300k"
  "t_satclip/t_satclip_toy_y_300k_satcliploss"
  "t_satclip/t_satclip_toy_y_300k_softloss"
  "t_satclip/t_satclip_y_200k"
  "t_satclip/t_satclip_y_300k"
)

# defaults to centroid; can do --from-polygon flag and --spatial-aggregation-methods to do spatial sampling
out="$RESULTS/usa_census.csv"
for model in "${MODELS[@]}"; do
  torchgeo-bench coord --model "$model" \
  --dataset usa_census \
  --nested-splits temporal_checkerboard,random \
  --folds 2 \
  --test-size 1 \
  --time-units year,1 year,2 year,4 \
  --temporal-discretization-methods monthly \
  --output "$out" "$@"
done
