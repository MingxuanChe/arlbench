#!/bin/bash

#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -J arlb_dqn_cartpole_dr
#SBATCH -t 0-12:00:00
#SBATCH --mail-type fail,end
#SBATCH --mail-user m.che@ai.uni-hannover.de
#SBATCH -p ai
#SBATCH --output experiment/cluster_scripts/log/dqn_cartpole_dr_%A.out
#SBATCH --error experiment/cluster_scripts/log/dqn_cartpole_dr_%A.err

# DQN on CartPole environment with Domain Randomization
# Seeds: 42-71 (30 seeds)
# 512 trials per seed

echo "================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "================================================="

NUM_TRIALS=512
CONFIG="rs_dqn_cartpole_dr"
ALGO="dqn"
SEARCH_SPACE="dqn_cc"
ENV="cc_cartpole_dr"

cd $BIGWORK/Repo/arlbench

# Build seed list
SEED_LIST=()
for SEED in {42..71}
do
  SEED_LIST+=($SEED)
done
echo "Using seeds array: ${SEED_LIST[@]}"
# Convert SEED_LIST to Hydra list format [42,43,44,...]
SEED_STRING="[$(IFS=,; echo "${SEED_LIST[*]}")]"
echo "Using seeds string for Hydra: ${SEED_STRING}"

# Load conda and activate environment
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate arlbench

# Run experiment
python run_arlbench.py --config-name=$CONFIG -m \
    hydra.sweeper.n_trials=$NUM_TRIALS \
    autorl.seed=$SEED_STRING \
    algorithm=$ALGO \
    search_space=$SEARCH_SPACE \
    environment=$ENV

echo "================================================="
echo "Job finished at: $(date)"
echo "================================================="
