#!/usr/bin/env bash
# Submit the temporal-split runs: per dataset, IDW (array task 0) first, then the linear probe for every model.
# Usage: ./submit_temporal_splits.sh [--dry-run] [dataset ...]   (default: every dataset in _temporal_splits.sh)
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
source _temporal_splits.sh
mkdir -p slurm_logs

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

submit() {
    if $dry_run; then
        echo "sbatch $*" >&2
        echo "<jobid>"
    else
        sbatch --parsable "$@"
    fi
}

selected=("$@")
if ((${#selected[@]} == 0)); then
    selected=("${DATASETS[@]}")
fi

for dataset in "${selected[@]}"; do
    time_units_for "$dataset" >/dev/null
    idw_job=$(submit --array=0 "$dataset.job")
    linear_job=$(submit --array=1-${#MODELS[@]} --dependency=afterok:"$idw_job" "$dataset.job")
    echo "$dataset: idw=$idw_job linear=$linear_job"
done
