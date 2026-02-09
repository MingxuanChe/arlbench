#!/bin/bash

# Submit all DQN experiments to the Luis cluster
# Usage: bash submit_all_dqn.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting all DQN experiments to the Luis cluster..."

# List of DQN environments
envs=(
    "cc_acrobot"
    "cc_cartpole"
    "cc_mountain_car"
    "minigrid_door_key"
    "xland_door_key"
    "xland_empty_random"
    "xland_four_rooms"
)

# Submit each environment job
for env in "${envs[@]}"; do
    script_file="${SCRIPT_DIR}/dqn_${env}.sh"
    if [ -f "$script_file" ]; then
        echo "Submitting DQN ${env}..."
        sbatch "$script_file"
    else
        echo "Warning: Script not found: $script_file"
    fi
done

echo "All DQN jobs submitted!"
