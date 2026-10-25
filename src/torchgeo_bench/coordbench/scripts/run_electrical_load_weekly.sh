# Running two tasks:
# - 8 daily electrical load datasets, one per year from 2016-2023.
# - 1 weekly electrical load dataset that spans all of 2016-2023
set -euo pipefail
source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME/dm_us_trees # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

out="$RESULTS/usa_electrical_load_weekly.csv"
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" \
  --dataset usa_electric_load_weekly \
  --temporal-discretization-methods two_days \
  --split all \
  --temporal-grid-sizes weekly monthly seasonally yearly \
  --output "$out" "$@"
done
