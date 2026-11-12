#!/usr/bin/env bash
# USA electric load (daily, per county polygon): native timestamps only. Pass --device cuda:0 (or cpu) as an extra arg.
set -euo pipefail
mkdir -p results/coordbench/temporal
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset usa_electric_load_daily --nested-splits random,random spatial,random --from-polygon --spatial-aggregation-methods statistical median_and_iqr --resume --output results/coordbench/temporal/usa_electric_load_daily.csv "$@"
done
# Temporal block CV: every 5th block is held out, for each block width.
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset usa_electric_load_daily --nested-splits temporal_checkerboard,random --folds 5 --test-size 1 --time-units week,1 quarter,1 day,30 day,365 --from-polygon --spatial-aggregation-methods statistical median_and_iqr --resume --output results/coordbench/temporal/usa_electric_load_daily.csv "$@"
done
