#!/usr/bin/env bash
# VIIRS night lights (annual): each year embedded at monthly, weekly and daily timestamps, then pooled. Pass --device cuda:0 (or cpu) as an extra arg.
# maybe add weekly or daily aggregation?
set -euo pipefail
mkdir -p results/coordbench/temporal
for m in sincos satclip geoclip mind sinr climplicit gtloc t_satclip/t_satclip_doy_1M; do
    torchgeo-bench coord --model "$m" --dataset viirs_night_lights --split all --temporal-grid-sizes days_365 --temporal-discretization-methods monthly --resume --output results/coordbench/temporal/viirs_night_lights.csv "$@"
done
