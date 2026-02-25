#!/bin/bash

# Submit all DQN experiments with Domain Randomization to the Luis cluster
# Partitioned Sobol Sequence: 8 partitions (array jobs 0-7), 50 seeds (42-91), 512 trials total per seed
# Usage: bash submit_all_dqn_dr.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting all DQN experiments with Domain Randomization to the Luis cluster..."
echo "  Using partitioned Sobol sequence with 8 partitions"
echo "  Seeds: 42-91 (50 seeds)"
echo "  Trials: 512 per seed (64 per partition)"
echo ""

# List of DQN CC environments with domain randomization
envs=(
    "cc_acrobot_dr"
    "cc_cartpole_dr"
    "cc_mountain_car_dr"
)

# Submit each environment job
for env in "${envs[@]}"; do
    script_file="${SCRIPT_DIR}/dqn_${env}.sh"
    if [ -f "$script_file" ]; then
        echo "Submitting DQN ${env} (8 array jobs)..."
        sbatch "$script_file"
    else
        echo "Warning: Script not found: $script_file"
    fi
done

echo ""
echo "All DQN DR jobs submitted!"
echo "Note: Each job spawns 8 array tasks. After completion, merge with merge_sobol_partitions.py"
