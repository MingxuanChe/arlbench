

#!/bin/bash
RUNTIME_LIST=()
# EXP_LIST=("rs_dqn_acrobot" "rs_sac_pendulum" "rs_ppo_mountaincar_continuous" "rs_dqn_cartpole" "rs_ppo_cartpole" "rs_sac_mountaincar_continuous") 
# EXP_LIST=("rs_dqn_acrobot" "rs_sac_pendulum" "rs_ppo_mountaincar_continuous") 
echo "Starting the ARLBench test run..."
# NUM_TRAILS=2
# NUM_TRAILS=16
NUM_TRAILS=512
TRANSFER_SEEDS=30

# # rs_dqn_cartpole
# EXP_LIST+=("rs_dqn_cartpole")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=dqn search_space=dqn_cc environment=cc_cartpole
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_dqn_cartpole_transfer
# EXP_LIST+=("rs_dqn_cartpole_transfer")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole_transfer -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=dqn search_space=dqn_cc environment=cc_cartpole
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # transfer experiments
# EXP_LIST+=("transfer_dqn_cartpole")
# START_TIME=$(date +%s)
# cd examples && ./run_transfer_pipeline.sh \
#   --runhistory "../results/sobol/dqn_CartPole-v1/[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]/runhistory.csv" \
#   --algorithm dqn \
#   --env-name CartPole-v1 \
#   --n-seeds ${TRANSFER_SEEDS}
# cd ..
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# rs_ppo_cartpole
EXP_LIST+=("rs_ppo_cartpole")
START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
hydra.sweeper.n_trials=$NUM_TRAILS algorithm=ppo search_space=ppo_cc environment=cc_cartpole
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_ppo_cartpole_transfer
# EXP_LIST+=("rs_ppo_cartpole_transfer")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole_transfer -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=ppo search_space=ppo_cc environment=cc_cartpole
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# transfer experiments
EXP_LIST+=("transfer_ppo_cartpole")
START_TIME=$(date +%s)
cd examples && ./run_transfer_pipeline.sh \
  --runhistory "../results/sobol/ppo_CartPole-v1/[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]/runhistory.csv" \
  --algorithm ppo \
  --env-name CartPole-v1 \
  --n-seeds ${TRANSFER_SEEDS}
cd ..
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_sac_ cartpole
# EXP_LIST+=("rs_sac_cartpole")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=sac search_space=sac_cc environment=cc_cartpole
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))  
# # rs_sac_cartpole_transfer
# EXP_LIST+=("rs_sac_cartpole_transfer")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=sac search_space=sac_cc environment=cc_cartpole
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))  

# # transfer experiments
# EXP_LIST+=("transfer_sac_cartpole")
# START_TIME=$(date +%s)
# cd examples && ./run_transfer_pipeline.sh \
#   --runhistory "../results/sobol/sac_CartPole-v1/[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]/runhistory.csv" \
#   --algorithm sac \
#   --env-name CartPole-v1 \
#   --n-seeds ${TRANSFER_SEEDS}
# cd ..
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_sac_mountaincar_continuous
# EXP_LIST+=("rs_sac_mountaincar_continuous")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=sac search_space=sac_cc environment=cc_continuous_mountain_car
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))


# # rs_ppo_mountaincar_continuous
# EXP_LIST+=("rs_ppo_mountaincar_continuous")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole  -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=ppo search_space=ppo_cc environment=cc_continuous_mountain_car
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME))) 


# # rs_dqn_acrobot
# EXP_LIST+=("rs_dqn_acrobot")
# START_TIME=$(date +%s)
# # pixi run python run_arlbench.py --config-name=random_search_erahbo -m
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=dqn search_space=dqn_cc environment=cc_acrobot
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_sac_pendulum
# EXP_LIST+=("rs_sac_pendulum")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=sac search_space=sac_cc environment=cc_pendulum
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))

# # rs_ppo_mountaincar_continuous
# EXP_LIST+=("rs_ppo_mountaincar_continuous")
# START_TIME=$(date +%s)
# pixi run python run_arlbench.py --config-name=rs_dqn_cartpole  -m \
# hydra.sweeper.n_trials=$NUM_TRAILS algorithm=ppo search_space=ppo_cc environment=cc_continuous_mountain_car
# END_TIME=$(date +%s)
# RUNTIME_LIST+=($((END_TIME - START_TIME)))  

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



# dqn
# dqn_atari
# dqn_cc
# dqn_cc_pbt
# dqn_procgen
# ppo_atari
# ppo_cc
# ppo_minigrid
# ppo_mujoco
# sac
# sac_cc
# sac_cc_pbt
# sac_mujoco


# atari_battle_zone
# atari_pong
# box2d_bipedal_walker
# box2d_lunar_lander
# brax_fast
# brax_halfcheetah
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