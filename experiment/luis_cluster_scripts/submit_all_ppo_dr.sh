#!/bin/bash

# Submit all PPO experiments with Domain Randomization to the Luis cluster
# Usage: bash submit_all_ppo_dr.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting all PPO experiments with Domain Randomization to the Luis cluster..."

# List of PPO CC environments with domain randomization
envs=(
    "cc_acrobot_dr"
    "cc_cartpole_dr"
    "cc_continuous_mountain_car_dr"
    "cc_mountain_car_dr"
    "cc_pendulum_dr"
)

# Submit each environment job
for env in "${envs[@]}"; do
    script_file="${SCRIPT_DIR}/ppo_${env}.sh"
    if [ -f "$script_file" ]; then
        echo "Submitting PPO ${env}..."
        sbatch "$script_file"
    else
        echo "Warning: Script not found: $script_file"
    fi
done

echo "All PPO DR jobs submitted!"
