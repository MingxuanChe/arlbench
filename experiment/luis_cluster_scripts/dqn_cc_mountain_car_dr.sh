#!/bin/bash

#SBATCH --array=0-7
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH -J arlb_dqn_mc_dr_part
#SBATCH -t 0-6:00:00
#SBATCH --mail-type all
#SBATCH --mail-user m.che@ai.uni-hannover.de
#SBATCH -p ai,tnt
#SBATCH --output experiment/luis_cluster_scripts/log/dqn_mountain_car_dr_partitioned_%A_%a.out
#SBATCH --error experiment/luis_cluster_scripts/log/dqn_mountain_car_dr_partitioned_%A_%a.err

# Partitioned Sobol Sequence: DQN on MountainCar with Domain Randomization
# Job array: 8 partitions (0-7) running in parallel
# Seeds: 42-91 (50 seeds)
# Total: 512 trials per seed (64 trials per partition)

echo "================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Partition: $((SLURM_ARRAY_TASK_ID + 1))/8"
echo "================================================="

# Configuration
N_PARTITIONS=8
TOTAL_TRIALS=512
TRIALS_PER_PARTITION=$((TOTAL_TRIALS / N_PARTITIONS))
CONFIG="part_sobol_ppo_cartpole_dr"
ALGO="dqn"
SEARCH_SPACE="dqn_cc"
ENV="cc_mountain_car_dr"
SOBOL_SEED=42
PARTITION_ID=$SLURM_ARRAY_TASK_ID

echo "Configuration:"
echo "  Total Trials: $TOTAL_TRIALS"
echo "  Trials per Partition: $TRIALS_PER_PARTITION"
echo "  N_Partitions: $N_PARTITIONS"
echo "  Partition ID: $PARTITION_ID"
echo "  Sobol Seed: $SOBOL_SEED"
echo ""

cd $BIGWORK/Repo/arlbench

# Build seed list (50 seeds: 42-91)
SEED_LIST=()
for SEED in {42..91}
do
  SEED_LIST+=($SEED)
done
echo "Using seeds array: ${SEED_LIST[@]}"
SEED_STRING="[$(IFS=,; echo "${SEED_LIST[*]}")]"
echo "Using seeds string for Hydra: ${SEED_STRING}"
echo ""

# Load conda and activate environment
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate arlbench

# Run experiment with partitioned Sobol
echo "Running partition $PARTITION_ID with $TRIALS_PER_PARTITION trials..."
python run_arlbench.py --config-name=$CONFIG -m \
    hydra.sweeper.n_trials=$TRIALS_PER_PARTITION \
    autorl.seed=$SEED_STRING \
    algorithm=$ALGO \
    search_space=$SEARCH_SPACE \
    environment=$ENV \
    hydra.sweeper.sweeper_kwargs.optimizer_kwargs.seed=$SOBOL_SEED \
    hydra.sweeper.sweeper_kwargs.optimizer_kwargs.n_sub=$N_PARTITIONS \
    hydra.sweeper.sweeper_kwargs.optimizer_kwargs.id_sub=$PARTITION_ID \
    hydra.sweep.dir="multirun/dqn_cc_mountain_car_dr/\${autorl.seed}/partition_${PARTITION_ID}_of_${N_PARTITIONS}"

echo "================================================="
echo "Job finished at: $(date)"
echo "================================================="

# ============================================================================
# MERGE INSTRUCTIONS (run after all 8 array jobs complete):
# ============================================================================
# Wait for all array jobs to finish, then merge partitions for each seed:
#
# for seed in {42..91}; do
#     pixi run python merge_sobol_partitions.py \
#         --base-dir multirun \
#         --algorithm dqn \
#         --environment cc_mountain_car_dr \
#         --seed $seed
# done
#
# Merged results will be in: multirun/dqn_cc_mountain_car_dr/{seed}/merged/
