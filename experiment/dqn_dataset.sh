#!/bin/bash

RUNTIME_LIST=()
echo "Starting the ARLBench DQN dataset generation..."
# NUM_TRAILS=1
# NUM_TRAILS=16
NUM_TRAILS=512
CONFIG="rs_dqn_cartpole"
ALGO="dqn"

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

# DQN ONLY WORKS WITH DISCRETE ACTION SPACES

# rs_dqn_acrobot
EXP_LIST+=("rs_dqn_acrobot")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=cc_acrobot
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_dqn_cartpole
EXP_LIST+=("rs_dqn_cartpole")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=cc_cartpole
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_dqn_mountaincar
EXP_LIST+=("rs_dqn_mountaincar")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=cc_mountain_car
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_dqn_minigrid_door_key
EXP_LIST+=("rs_dqn_minigrid_door_key")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=minigrid_door_key
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_dqn_xland_door_key
EXP_LIST+=("rs_dqn_xland_door_key")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=xland_door_key
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_dqn_xland_empty_random
EXP_LIST+=("rs_dqn_xland_empty_random")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=xland_empty_random
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_dqn_xland_four_rooms
EXP_LIST+=("rs_dqn_xland_four_rooms")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=dqn_cc environment=xland_four_rooms
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


# DQN COMPATIBLE ENVIRONMENTS (DISCRETE ACTIONS)
# cc_acrobot
# cc_cartpole
# cc_mountain_car
# minigrid_door_key
# xland_door_key
# xland_empty_random
# xland_four_rooms

# DQN SEARCH SPACES
# dqn - generic
# dqn_cc - classic control
# dqn_atari - atari (blocked by envpool)
# dqn_procgen - procgen (blocked by envpool)
