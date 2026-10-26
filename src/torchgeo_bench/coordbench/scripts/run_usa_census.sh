# Running USA census data
# yearly info from 2010-2025 at the county level
#set -euo pipefail

source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

# defaults to centroid; can do --from-polygon flag and --spatial-aggregation-methods to do spatial sampling
out="$RESULTS/usa_census.csv"
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" \
  --dataset usa_census \
  --split all \
  --temporal-discretization-methods monthly \
  --temporal-grid-sizes weekly monthly seasonally yearly \
  --output "$out" "$@"
done
