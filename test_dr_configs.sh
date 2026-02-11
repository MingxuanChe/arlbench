#!/bin/bash
# Test script for domain randomization configs with HyperSweeper

echo "Testing domain randomization configurations with HyperSweeper..."

# seed 42 to 71
SEED_LIST=()
for SEED in {42..71}
# for SEED in {42..42}
do
  SEED_LIST+=($SEED)
done
echo "Using seeds array: ${SEED_LIST[@]}"
# convert SEED_LIST to Hydra list format [42,43,44,...]
SEED_STRING="[$(IFS=,; echo "${SEED_LIST[*]}")]"
echo "Using seeds string for Hydra: ${SEED_STRING}"

# Test rs_dqn_cartpole_dr with minimal trials
echo "1. Testing rs_dqn_cartpole_dr..."
pixi run python run_arlbench.py --config-name=rs_dqn_cartpole_dr -m \
  hydra.sweeper.n_trials=1 autorl.seed=$SEED_STRING algorithm=dqn \
  search_space=dqn_cc environment=cc_cartpole_dr

if [ $? -eq 0 ]; then
    echo "✓ rs_dqn_cartpole_dr test passed"
else
    echo "✗ rs_dqn_cartpole_dr test failed"
    exit 1
fi

# Test rs_ppo_cartpole_dr with minimal trials
echo ""
echo "2. Testing rs_ppo_cartpole_dr..."
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m \
  hydra.sweeper.n_trials=1 autorl.seed=$SEED_STRING algorithm=ppo \
  search_space=ppo_cc environment=cc_cartpole_dr

if [ $? -eq 0 ]; then
    echo "✓ rs_ppo_cartpole_dr test passed"
else
    echo "✗ rs_ppo_cartpole_dr test failed"
    exit 1
fi

echo ""
echo "All domain randomization config tests passed successfully!"
