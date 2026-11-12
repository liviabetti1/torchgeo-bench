#!/usr/bin/env bash
# USA electric load (weekly, per county polygon): each week embedded at daily timestamps, then pooled. Pass --device cuda:0 (or cpu) as an extra arg.
set -euo pipefail
mkdir -p results/coordbench/temporal
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset usa_electric_load_weekly --nested-splits random,random spatial,random --temporal-discretization-methods daily --from-polygon --spatial-aggregation-methods statistical median_and_iqr --resume --output results/coordbench/temporal/usa_electric_load_weekly.csv "$@"
done
# Temporal block CV: every 5th block is held out, for each block width.
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset usa_electric_load_weekly --nested-splits temporal_checkerboard,random --folds 5 --test-size 1 --time-units week,1 quarter,1 day,30 day,365 --temporal-discretization-methods daily --from-polygon --spatial-aggregation-methods statistical median_and_iqr --resume --output results/coordbench/temporal/usa_electric_load_weekly.csv "$@"
done
