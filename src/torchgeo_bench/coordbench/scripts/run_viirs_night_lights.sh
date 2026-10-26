# Running a VIIRS nightlights task from 2015-2021
#set -euo pipefail

source $TORCHGEO_ENV/bin/activate torchgeo-bench

RESULTS=$RESULTS_HOME # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

out="$RESULTS/viirs_night_lights.csv"
for model in climplicit gtloc t_satclip/t_satclip_doy_1M; do
  torchgeo-bench coord --model "$model" \
  --dataset viirs_night_lights \
  --split all \
  --exclude-antarctica \
  --temporal-discretization-methods monthly \
  --temporal-grid-sizes weekly monthly seasonally yearly \
  --output "$out" "$@"
done