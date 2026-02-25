#!/bin/bash

# Submit ALL Domain Randomization experiments to the Luis cluster
# Partitioned Sobol Sequence: 8 partitions (array jobs 0-7), 50 seeds (42-91), 512 trials total per seed
# Usage: bash submit_all_dr.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================="
echo "Submitting ALL Domain Randomization experiments"
echo "  Using partitioned Sobol sequence with 8 partitions"
echo "  Seeds: 42-91 (50 seeds)"
echo "  Trials: 512 per seed (64 per partition)"
echo "================================================="

# Submit DQN DR jobs
echo ""
echo "Submitting DQN with Domain Randomization..."
bash "${SCRIPT_DIR}/submit_all_dqn_dr.sh"

# Submit PPO DR jobs
echo ""
echo "Submitting PPO with Domain Randomization..."
bash "${SCRIPT_DIR}/submit_all_ppo_dr.sh"

# Submit SAC DR jobs
echo ""
echo "Submitting SAC with Domain Randomization..."
bash "${SCRIPT_DIR}/submit_all_sac_dr.sh"

echo ""
echo "================================================="
echo "All Domain Randomization jobs submitted!"
echo "Total: 10 experiments × 8 array tasks = 80 jobs"
echo "After completion, merge partitions with merge_sobol_partitions.py"
echo "================================================="
