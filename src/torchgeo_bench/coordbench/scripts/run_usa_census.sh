# Running USA census data
# yearly info from 2010-2025 at the county level
set -euo pipefail

source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME/temporal_split # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

out="$RESULTS/usa_census.csv"
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" --dataset usa_census --split both \
    --from-polygon --spatial-aggregation-methods statistical median_and_iqr mean \
    --output "$out" "$@"
done
