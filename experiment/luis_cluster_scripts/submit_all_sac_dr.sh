#!/bin/bash

# Submit all SAC experiments with Domain Randomization to the Luis cluster
# Usage: bash submit_all_sac_dr.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Submitting all SAC experiments with Domain Randomization to the Luis cluster..."

# List of SAC CC environments with domain randomization (continuous action space only)
envs=(
    "cc_continuous_mountain_car_dr"
    "cc_pendulum_dr"
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

echo "All SAC DR jobs submitted!"
