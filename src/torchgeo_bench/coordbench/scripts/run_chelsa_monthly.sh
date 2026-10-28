#!/usr/bin/env bash
# CHELSA monthly: each month embedded at weekly and daily timestamps, then pooled. Pass --device cuda:0 (or cpu) as an extra arg.
set -euo pipefail
mkdir -p results/coordbench/temporal
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset chelsa_monthly --split all --temporal-grid-sizes days_30 13_weeks days_365 --temporal-discretization-methods weekly daily --resume --output results/coordbench/temporal/chelsa_monthly.csv "$@"
done
