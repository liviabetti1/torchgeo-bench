#!/usr/bin/env bash
# TWSA monthly  2002 04 to 2026 08 (with gaps): raw monthly embeddings, no temporal aggregation.
# Runs with and without Antarctica.
# Requires access to the Temporal-Coordbench HuggingFace repo.
# Pass --device cuda:0 as an extra arg when calling directly.
set -euo pipefail

RESULTS=$RESULTS_HOME/year_patch # MAKE SURE TO EDIT THIS TO YOUR LOCAL RESULTS PATH
mkdir -p "$RESULTS"

MODELS=(
  climplicit
)

out="$RESULTS/twsa_monthly.csv"
for model in "${MODELS[@]}"; do
    torchgeo-bench coord \
        --model "$model" \
        --dataset twsa \
        --split random \
        --exclude-antarctica \
        --output "$out" \
        "$@"

    rm -f "${out}.lock"
done
