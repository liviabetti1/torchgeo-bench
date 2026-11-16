# Shared settings for the temporal-split jobs, sourced by <dataset>.job, the test job, and the submit script.
# Array task 0 runs IDW on coordinates alone; task i > 0 runs the linear probe for MODELS[i-1].
# submit_temporal_splits.sh submits task 0 first and makes the linear tasks wait for it.

set -euo pipefail

DATASETS=(
    era5_ecmwf
    chelsa_daily
    chelsa_monthly
    usa_electric_load_daily
    usa_electric_load_weekly
    global_population
    viirs_night_lights
)

MODELS=(
    climplicit
    gtloc
    "t_satclip/t_satclip_doy_1M"
    "t_satclip/t_satclip_toy_200k"
    "t_satclip/t_satclip_toy_300k"
    "t_satclip/t_satclip_toy_y_100k"
    "t_satclip/t_satclip_toy_y_150k"
    "t_satclip/t_satclip_toy_y_200k"
    "t_satclip/t_satclip_toy_y_200k_satcliploss"
    "t_satclip/t_satclip_toy_y_200k_softloss"
    "t_satclip/t_satclip_toy_y_250k"
    "t_satclip/t_satclip_toy_y_300k"
    "t_satclip/t_satclip_toy_y_300k_satcliploss"
    "t_satclip/t_satclip_toy_y_300k_softloss"
    "t_satclip/t_satclip_y_200k"
    "t_satclip/t_satclip_y_300k"
)

FOLDS=5

# Time units that are coarser than the data's resolution and finer than its span. temporal_checkerboard deals
# consecutive units to folds round-robin, so fewer units than FOLDS (year,2 over 7-8 years) means fewer folds.
time_units_for() {
    case "$1" in
        era5_ecmwf | chelsa_daily | usa_electric_load_daily) echo "week,1 month,1 season,1" ;;  # daily, one year
        chelsa_monthly) echo "month,1 season,1 year,1 year,2" ;;  # monthly, 2011-2018
        usa_electric_load_weekly) echo "week,1 month,1 season,1 year,1 year,2" ;;  # weekly, 2016-2023
        global_population | viirs_night_lights) echo "year,1 year,2" ;;  # yearly, 2015-2025 / 2015-2021
        *) echo "No time units defined for dataset '$1'" >&2; return 1 ;;
    esac
}

setup_env() {
    if [[ ! -f "${SLURM_SUBMIT_DIR:-.}/_temporal_splits.sh" ]]; then
        echo "Submit from coordbench/jobs/ (or use submit_temporal_splits.sh)" >&2
        return 1
    fi
    export OMP_NUM_THREADS=1
    REPO=${TORCHGEO_HOME:-$(cd "$SLURM_SUBMIT_DIR/../../../.." && pwd)}
    export CKPT_HOME=${CKPT_HOME:-/projects/bgtj/t_satclip/ckpts}
    RESULTS_ROOT=${RESULTS_HOME:-/projects/bgtj/t_satclip/results}
    # shellcheck disable=SC1091
    source "${TORCHGEO_ENV:-$REPO/.venv}/bin/activate"
}

# run_coord DATASET MODEL OUT_DIR [extra coord flags...]; an empty MODEL runs IDW.
run_coord() {
    local dataset=$1 model=$2 out_dir=$3
    shift 3
    local method_args name
    if [[ -z "$model" ]]; then
        method_args=(--methods idw)
        name=idw
    else
        method_args=(--model "$model" --methods linear)
        name=${model##*/}
    fi
    # shellcheck disable=SC2207
    local units=($(time_units_for "$dataset"))
    mkdir -p "$out_dir/$dataset"
    echo "dataset=$dataset model=${model:-idw} time_units=${units[*]}"
    torchgeo-bench coord \
        --dataset "$dataset" \
        "${method_args[@]}" \
        --nested-splits random,random temporal_checkerboard,random \
        --folds "$FOLDS" \
        --time-units "${units[@]}" \
        --resume \
        --output "$out_dir/$dataset/$name.csv" \
        --device cuda:0 \
        --knn-device cuda:0 \
        "$@"
}

# run_array_task DATASET [extra coord flags...]: dispatch on SLURM_ARRAY_TASK_ID.
run_array_task() {
    local dataset=$1
    shift
    setup_env
    local task=${SLURM_ARRAY_TASK_ID:-0} model=""
    if ((task > 0)); then
        model=${MODELS[task - 1]}
    fi
    run_coord "$dataset" "$model" "$RESULTS_ROOT/temporal_splits" "$@"
}
