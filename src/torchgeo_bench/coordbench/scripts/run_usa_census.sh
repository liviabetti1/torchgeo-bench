# Running USA census data
# yearly info from 2010-2025 at the county level
set -euo pipefail
for model in climplicit; do
  torchgeo-bench coord --model "$model" --dataset usa_census --split both \
    --from-polygon --spatial-aggregation-methods statistical median_and_iqr mean \
    --output results/coordbench_usa_electrical_load.csv "$@"
done
