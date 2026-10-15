# Running a VIIRS nightlights task from 2015-2021
set -euo pipefail
for model in climplicit; do
  torchgeo-bench coord --model "$model" --dataset viirs_night_lights --split both \
    --device cuda:0 --output results/viirs_night_lights.csv "$@"
done
