# Running a VIIRS nightlights task from 2015-2021
set -euo pipefail

source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME/temporal_split # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

out="$RESULTS/usa_census.csv"
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" --dataset viirs_night_lights --split both \
    --device cuda:0 --output "$out" "$@"
done
