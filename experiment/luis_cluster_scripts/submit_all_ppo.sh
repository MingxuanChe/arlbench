#!/bin/bash

# Submit all PPO experiments to the Luis cluster
# Usage: bash submit_all_ppo.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting all PPO experiments to the Luis cluster..."

# List of PPO environments
envs=(
    "cc_acrobot"
    "cc_cartpole"
    "cc_continuous_mountain_car"
    "cc_mountain_car"
    "cc_pendulum"
    "brax_fast"
    "brax_halfcheetah"
    "minigrid_door_key"
    "xland_door_key"
    "xland_empty_random"
    "xland_four_rooms"
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

echo "All PPO jobs submitted!"
