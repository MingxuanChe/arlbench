#!/bin/bash

RUNTIME_LIST=()
echo "Starting the ARLBench SAC dataset generation..."
# NUM_TRAILS=1
# NUM_TRAILS=16
NUM_TRAILS=512
CONFIG="rs_ppo_cartpole"
ALGO="sac"

cd $HOME/Repos/arlbench-5090

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

# SAC ONLY WORKS WITH CONTINUOUS ACTION SPACES

# rs_sac_mountaincar_continuous
EXP_LIST+=("rs_sac_mountaincar_continuous")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=sac_cc environment=cc_continuous_mountain_car
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_sac_pendulum
EXP_LIST+=("rs_sac_pendulum")START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=sac_cc environment=cc_pendulum
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_sac_brax_fast
EXP_LIST+=("rs_sac_brax_fast")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=sac_mujoco environment=brax_fast
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_sac_brax_halfcheetah
EXP_LIST+=("rs_sac_brax_halfcheetah")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=sac_mujoco environment=brax_halfcheetah
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))


# loop to print each experiment runtime
for i in "${!EXP_LIST[@]}"; do
    EXP_NAME=${EXP_LIST[$i]}
    RUNTIME=${RUNTIME_LIST[$i]}
    # convert to hours, minutes, seconds
    HOURS=$((RUNTIME / 3600))
    MINUTES=$(( (RUNTIME % 3600) / 60 ))
    SECONDS=$((RUNTIME % 60))
    echo "Experiment: $EXP_NAME - Runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
done


# SAC COMPATIBLE ENVIRONMENTS (CONTINUOUS ACTIONS)
# cc_continuous_mountain_car
# cc_pendulum
# brax_fast

# SAC SEARCH SPACES
# sac - generic
# sac_cc - classic control
# sac_mujoco - mujoco/brax environments
