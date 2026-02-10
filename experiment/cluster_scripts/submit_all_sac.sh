#!/bin/bash

# Submit all SAC experiments to the Luis cluster
# Usage: bash submit_all_sac.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting all SAC experiments to the Luis cluster..."

# List of SAC environments (continuous action space only)
envs=(
    "cc_continuous_mountain_car"
    "cc_pendulum"
    "brax_fast"
    "brax_halfcheetah"
)

# Submit each environment job
for env in "${envs[@]}"; do
    script_file="${SCRIPT_DIR}/sac_${env}.sh"
    if [ -f "$script_file" ]; then
        echo "Submitting SAC ${env}..."
        sbatch "$script_file"
    else
        echo "Warning: Script not found: $script_file"
    fi
done

echo "All SAC jobs submitted!"
