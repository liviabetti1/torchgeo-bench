# Running two tasks:
# - 8 daily electrical load datasets, one per year from 2016-2023.
# - 1 weekly electrical load dataset that spans all of 2016-2023
#set -euo pipefail
source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

out="$RESULTS/usa_electrical_load_daily.csv"
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" \
  --dataset usa_electric_load_daily \
  --nested-splits random,random spatial,random \
  --output "$out" "$@"
done
# Temporal block CV: every 5th block is held out, for each block width.
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" \
  --dataset usa_electric_load_daily \
  --nested-splits temporal_checkerboard,random \
  --folds 5 \
  --test-size 1 \
  --time-units week,1 month,1 quarter,1 \
  --output "$out" "$@"
done
