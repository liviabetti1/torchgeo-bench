# ERA5 across temporal aggregations (1/2/4/13/52-week), one row per model.
set -euo pipefail
for model in t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" --dataset usa_electrical_usage --split both \
    --from_polygon True --spatial_aggregation_methods statistical median_and_iqr \
    --output results/coordbench_usa_electrical_usage.csv "$@"
done
