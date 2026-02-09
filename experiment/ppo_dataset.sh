#!/bin/bash

RUNTIME_LIST=()
echo "Starting the ARLBench test run..."
# NUM_TRAILS=1
# NUM_TRAILS=4
NUM_TRAILS=512
TRANSFER_SEEDS=30
CONFIG="rs_ppo_cartpole"
ALGO="ppo"

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

# rs_ppo_acrobot
EXP_LIST+=("rs_ppo_acrobot")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_cc environment=cc_acrobot
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_cartpole
EXP_LIST+=("rs_ppo_cartpole")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_cc environment=cc_cartpole
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_mountaincar_continuous
EXP_LIST+=("rs_ppo_mountaincar_continuous")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_cc environment=cc_continuous_mountain_car
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_mountaincar
EXP_LIST+=("rs_ppo_mountaincar")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_cc environment=cc_mountain_car
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_pendulum
EXP_LIST+=("rs_ppo_pendulum")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_cc environment=cc_pendulum
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_ppo_brax_halfcheetah
# EXP_LIST+=("rs_ppo_brax_halfcheetah")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=$CONFIG -m \
# hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
# search_space=ppo_mujoco environment=brax_halfcheetah
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_brax_fast
EXP_LIST+=("rs_ppo_brax_fast")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_mujoco environment=brax_fast
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_minigrid_door_key
EXP_LIST+=("rs_ppo_minigrid_door_key")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_minigrid environment=minigrid_door_key
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_xland_door_key
EXP_LIST+=("rs_ppo_xland_door_key")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_minigrid environment=xland_door_key
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_xland_empty_random
EXP_LIST+=("rs_ppo_xland_empty_random")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_minigrid environment=xland_empty_random
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_xland_four_rooms
EXP_LIST+=("rs_ppo_xland_four_rooms")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
hydra.sweeper.n_trials=$NUM_TRAILS autorl.seed=$SEED_STRING algorithm=$ALGO \
search_space=ppo_minigrid environment=xland_four_rooms
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


# WORKING SET
# brax_halfcheetah
# brax_fast
# cc_acrobot
# cc_cartpole
# cc_continuous_mountain_car
# cc_mountain_car
# cc_pendulum
# minigrid_door_key
# xland_door_key
# xland_empty_random
# xland_four_rooms

# SEARCH SPACES
# dqn
# dqn_atari
# dqn_cc
# dqn_procgen
# ppo_atari
# ppo_cc
# ppo_minigrid
# ppo_mujoco
# sac
# sac_cc
# sac_mujoco

# ALL ENVS
# atari_battle_zone
# atari_pong
# box2d_bipedal_walker
# box2d_lunar_lander
# brax_fast
# brax_halfcheetah # mujoco
# cc_acrobot
# cc_cartpole
# cc_continuous_mountain_car
# cc_mountain_car
# cc_pendulum
# minigrid_door_key
# mujoco_hopper
# procgen_bigfish_easy
# xland_door_key
# xland_empty_random
# xland_four_rooms

# WORKING SET
# brax_halfcheetah # mujoco
# brax_fast
# cc_acrobot
# cc_cartpole
# cc_continuous_mountain_car
# cc_mountain_car
# cc_pendulum
# minigrid_door_key
# xland_door_key
# xland_empty_random
# xland_four_rooms