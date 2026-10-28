#!/usr/bin/env bash
# CHELSA daily (2017): native timestamps only. Pass --device cuda:0 (or cpu) as an extra arg.
set -euo pipefail
mkdir -p results/coordbench/temporal
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset chelsa_daily --split all --temporal-grid-sizes weekly days_30 13_weeks --resume --output results/coordbench/temporal/chelsa_daily.csv "$@"
done
