#!/bin/bash

# Master script to submit all experiments to the Luis cluster
# Usage: bash submit_all.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================"
echo "Submitting all ARLBench experiments"
echo "========================================"
echo ""

# Submit DQN experiments
echo "--- Submitting DQN experiments ---"
bash "${SCRIPT_DIR}/submit_all_dqn.sh"
echo ""

# Submit PPO experiments
echo "--- Submitting PPO experiments ---"
bash "${SCRIPT_DIR}/submit_all_ppo.sh"
echo ""

# Submit SAC experiments
echo "--- Submitting SAC experiments ---"
bash "${SCRIPT_DIR}/submit_all_sac.sh"
echo ""

echo "========================================"
echo "All experiments submitted!"
echo "========================================"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all jobs with: scancel -u \$USER"
