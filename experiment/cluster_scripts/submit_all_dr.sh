#!/bin/bash

# Submit ALL Domain Randomization experiments to the Luis cluster
# Usage: bash submit_all_dr.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================="
echo "Submitting ALL Domain Randomization experiments"
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
echo "================================================="
