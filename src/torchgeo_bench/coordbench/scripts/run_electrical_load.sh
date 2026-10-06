# ERA5 across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail
for model in climplicit; do
  torchgeo-bench coord --model "$model" --dataset usa_electric_usage --split both \
    --from-polygon --spatial-aggregation-methods statistical median_and_iqr \
    --output results/coordbench_usa_electrical_usage.csv "$@"
done
