# Running two tasks:
# - 8 daily electrical load datasets, one per year from 2016-2023.
# - 1 weekly electrical load dataset that spans all of 2016-2023
set -euo pipefail
for model in climplicit; do
  torchgeo-bench coord --model "$model" --dataset usa_electric_load_daily usa_electric_load_weekly --split both \
    --from-polygon --spatial-aggregation-methods statistical median_and_iqr mean \
    --output results/coordbench_usa_electrical_load.csv "$@"
done
