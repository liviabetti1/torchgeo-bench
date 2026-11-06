#!/bin/bash
#SBATCH --mem=24g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpu_a100      # <- or cpu_amd, gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bgtj-tgirails
#SBATCH --time=1:00:00      # hh:mm:ss for the job
##SBATCH --mail-user=leca5365@colorado.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --job-name=tsatclip-all-exp
#SBATCH --array=0-119 #<==== CHANGE THIS TO THE APPROPRIATE RANGE
#SBATCH --export=ALL

set -x
export OMP_NUM_THREADS=1

# TODO: make sure huggingface is logged in somehow?
OUTPUT_EXPERIMENT_NAME="all"

# Setup the environment
cd $TORCHGEO_HOME/src/torchgeo_bench/coordbench/scripts/
source $TORCHGEO_ENV/bin/activate torchgeo-bench

# Ensure file paths are correct
export TORCHGEO_HOME=/u/leca5365/Documents/torchgeo-bench-temporal
#export TORCHGEO_ENV=/u/kplane19/torchgeo-bench/.venv
export SATCLIP_HOME=/u/leca5365/Documents/satclip
export RESULTS_HOME=/projects/bgtj/t_satclip/results
export CKPT_HOME=/projects/bgtj/t_satclip/ckpts

DATASETS=(
    "ghcn_daily"
    "chelsa_daily"
    "chelsa_monthly"
    "era5_ecmwf"
    "usa_census"
    "viirs_night_lights"
    "usa_electric_load_daily"
    "usa_electric_load_weekly"
    "global_population"
    "dm-us_trees"
)

MODELS=(
  climplicit
  gtloc
  "t_satclip/t_satclip_doy_1M"
  "t_satclip/t_satclip_landsat_toy_300k"
  "t_satclip/t_satclip_landsat_toy_y_300k"
  "t_satclip/t_satclip_landsat_y_300k"
  "t_satclip/t_satclip_landsat_toy_y_500k"
  #"t_satclip/t_satclip_toy_100k"
  #"t_satclip/t_satclip_toy_200k"
  "t_satclip/t_satclip_toy_300k"
  "t_satclip/t_satclip_toy_y_100k"
  "t_satclip/t_satclip_toy_y_200k"
  # "t_satclip/t_satclip_toy_y_200k_satcliploss"
  # "t_satclip/t_satclip_toy_y_200k_softloss"
  # "t_satclip/t_satclip_toy_y_250k_softloss"
  "t_satclip/t_satclip_toy_y_300k"
  # "t_satclip/t_satclip_toy_y_300k_satcliploss"
  # "t_satclip/t_satclip_toy_y_300k_softloss"
  #"t_satclip/t_satclip_y_100k"
  #"t_satclip/t_satclip_y_200k"
  "t_satclip/t_satclip_y_300k"
)

N_DATASETS=${#DATASETS[@]}
N_MODELS=${#MODELS[@]}

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / N_MODELS ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % N_MODELS ))

DATASET=${DATASETS[$DATASET_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "Starting: Dataset $DATASET | Model $MODEL | Task $SLURM_ARRAY_TASK_ID"

torchgeo-bench coord --model "$MODEL" \
	       --device cuda:0 \
	       --dataset "$DATASET" \
	       --temporal-checkerboard \
	       --exclude-antarctica \
	       --subsample \
	       --output /projects/bgtj/t_satclip/results/${OUTPUT_EXPERIMENT_NAME}.taskid_${SLURM_ARRAY_TASK_ID}.csv "$@"

echo "Finished: Dataset $DATASET | Model $MODEL | Task $SLURM_ARRAY_TASK_ID"
