#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpu      # <- or cpu_amd, gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bgtj-tgirails
#SBATCH --time=168:00:00      # hh:mm:ss for the job
##SBATCH --mail-user=leca5365@colorado.edu
##SBATCH --mail-type="BEGIN,END" See sbatch or srun man pages for more email options
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --job-name=viirs-runs
#SBATCH --array=0-11 #<==== CHANGE THIS TO THE APPROPRIATE RANGE
#SBATCH --export=ALL

set -x
export OMP_NUM_THREADS=1

cd $TORCHGEO_HOME/src/torchgeo_bench/coordbench/scripts/
source $TORCHGEO_ENV/bin/activate torchgeo-bench

# OUTPUT_EXPERIMENT_NAME="time_representation"
# MODELS=("t_satclip/t_satclip_y_200k"
#         "t_satclip/t_satclip_toy_200k"
#         "t_satclip/t_satclip_toy_y_200k"
#         "climplicit"
#         "gtloc")

# OUTPUT_EXPERIMENT_NAME="data_amount"
# MODELS=("t_satclip/t_satclip_toy_y_100k"
#         "t_satclip/t_satclip_toy_y_150k"
#         "t_satclip/t_satclip_toy_y_200k"
#         "t_satclip/t_satclip_toy_y_250k"
#         "t_satclip/t_satclip_toy_y_300k"
#         "climplicit"
#         "gtloc")

# OUTPUT_EXPERIMENT_NAME="loss_type"
# MODELS=("t_satclip/t_satclip_toy_y_200k_satcliploss"
#         "t_satclip/t_satclip_toy_y_200k_softloss"
#         "climplicit"
#         "gtloc")

OUTPUT_EXPERIMENT_NAME="all"
MODELS=("t_satclip/t_satclip_toy_y_300k_satcliploss"
        "t_satclip/t_satclip_toy_y_300k_softloss"
        "t_satclip/t_satclip_toy_y_100k"
        "t_satclip/t_satclip_toy_y_150k"
        "t_satclip/t_satclip_toy_y_200k"
        "t_satclip/t_satclip_toy_y_250k"
        "t_satclip/t_satclip_toy_y_300k"
        "t_satclip/t_satclip_y_300k"
        "t_satclip/t_satclip_toy_300k"
        "t_satclip/t_satclip_toy_y_300k"
        "climplicit"
        "gtloc")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL"

torchgeo-bench coord --model "$MODEL" --dataset viirs_night_lights --split both --device cuda:0 \
--temporal-discretization-methods 1_week 2_weeks 4_weeks 13_weeks yearly seasonally  \
--output /projects/bgtj/t_satclip/results/coordbench_chelsa_daily_agg_embeddings_${OUTPUT_EXPERIMENT_NAME}.taskid_${SLURM_ARRAY_TASK_ID}.csv "$@"
