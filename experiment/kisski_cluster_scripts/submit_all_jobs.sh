#!/bin/bash

# Master submission script for all ARLBench partitioned Sobol experiments
# Submits all 22 algorithm-environment combinations
# Each job runs 8 partitions in parallel with 50 seeds (42-91)

echo "========================================================================"
echo "ARLBench Dataset Generation - Master Submission Script"
echo "========================================================================"
echo "Starting submission at: $(date)"
echo ""
echo "Total Scripts: 22 (PPO: 11, SAC: 4, DQN: 7)"
echo "Seeds per experiment: 50 (42-91)"
echo "Partitions per experiment: 8"
echo "Total trials per seed: 512"
echo ""
echo "========================================================================"

# Navigate to the script directory
SCRIPT_DIR="experiment/kisski_cluster_scripts"
cd $HOME/Repos/arlbench || exit 1

# Array to store job IDs
declare -a JOB_IDS
declare -a JOB_NAMES

echo ""
echo "========== SUBMITTING PPO JOBS (11 environments) =========="
echo ""

# PPO Classic Control (5 environments)
echo "--- PPO Classic Control ---"
sbatch ${SCRIPT_DIR}/ppo_cc_acrobot_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_cc_acrobot")
echo ""

sbatch ${SCRIPT_DIR}/ppo_cc_cartpole_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_cc_cartpole")
echo ""

sbatch ${SCRIPT_DIR}/ppo_cc_continuous_mountain_car_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_cc_continuous_mountain_car")
echo ""

sbatch ${SCRIPT_DIR}/ppo_cc_mountain_car_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_cc_mountain_car")
echo ""

sbatch ${SCRIPT_DIR}/ppo_cc_pendulum_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_cc_pendulum")
echo ""

# PPO Brax (2 environments)
echo "--- PPO Brax/MuJoCo ---"
sbatch ${SCRIPT_DIR}/ppo_brax_fast_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_brax_fast")
echo ""

sbatch ${SCRIPT_DIR}/ppo_brax_halfcheetah_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_brax_halfcheetah")
echo ""

# PPO MiniGrid/XLand (4 environments)
echo "--- PPO MiniGrid/XLand ---"
sbatch ${SCRIPT_DIR}/ppo_minigrid_door_key_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_minigrid_door_key")
echo ""

sbatch ${SCRIPT_DIR}/ppo_xland_door_key_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_xland_door_key")
echo ""

sbatch ${SCRIPT_DIR}/ppo_xland_empty_random_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_xland_empty_random")
echo ""

sbatch ${SCRIPT_DIR}/ppo_xland_four_rooms_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("ppo_xland_four_rooms")
echo ""

echo ""
echo "========== SUBMITTING SAC JOBS (4 environments) =========="
echo ""

# SAC Classic Control (2 environments)
echo "--- SAC Classic Control ---"
sbatch ${SCRIPT_DIR}/sac_cc_continuous_mountain_car_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("sac_cc_continuous_mountain_car")
echo ""

sbatch ${SCRIPT_DIR}/sac_cc_pendulum_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("sac_cc_pendulum")
echo ""

# SAC Brax (2 environments)
echo "--- SAC Brax/MuJoCo ---"
sbatch ${SCRIPT_DIR}/sac_brax_fast_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("sac_brax_fast")
echo ""

sbatch ${SCRIPT_DIR}/sac_brax_halfcheetah_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("sac_brax_halfcheetah")
echo ""

echo ""
echo "========== SUBMITTING DQN JOBS (7 environments) =========="
echo ""

# DQN Classic Control (3 environments)
echo "--- DQN Classic Control ---"
sbatch ${SCRIPT_DIR}/dqn_cc_acrobot_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_cc_acrobot")
echo ""

sbatch ${SCRIPT_DIR}/dqn_cc_cartpole_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_cc_cartpole")
echo ""

sbatch ${SCRIPT_DIR}/dqn_cc_mountain_car_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_cc_mountain_car")
echo ""

# DQN MiniGrid/XLand (4 environments)
echo "--- DQN MiniGrid/XLand ---"
sbatch ${SCRIPT_DIR}/dqn_minigrid_door_key_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_minigrid_door_key")
echo ""

sbatch ${SCRIPT_DIR}/dqn_xland_door_key_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_xland_door_key")
echo ""

sbatch ${SCRIPT_DIR}/dqn_xland_empty_random_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_xland_empty_random")
echo ""

sbatch ${SCRIPT_DIR}/dqn_xland_four_rooms_partitioned.sh
JOB_IDS+=($?)
JOB_NAMES+=("dqn_xland_four_rooms")
echo ""

echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED!"
echo "========================================================================"
echo "Submission completed at: $(date)"
echo ""
echo "Total jobs submitted: ${#JOB_NAMES[@]}"
echo ""
echo "========================================================================"
echo "WHAT'S NEXT?"
echo "========================================================================"
echo ""
echo "1. Monitor job status:"
echo "   squeue -u \$USER"
echo ""
echo "2. Check specific job array status:"
echo "   squeue -u \$USER | grep arlb_"
echo ""
echo "3. View running/pending jobs by algorithm:"
echo "   squeue -u \$USER | grep ppo"
echo "   squeue -u \$USER | grep sac"
echo "   squeue -u \$USER | grep dqn"
echo ""
echo "4. After all array jobs complete, merge partitions for each experiment:"
echo "   cd experiment/kisski_cluster_scripts"
echo "   bash merge_all_experiments.sh"
echo ""
echo "5. Check logs for any errors:"
echo "   ls -lh ${SCRIPT_DIR}/log/*.err"
echo "   tail ${SCRIPT_DIR}/log/*.err"
echo ""
echo "========================================================================"
echo "EXPECTED COMPUTE RESOURCES"
echo "========================================================================"
echo "- Total array jobs: 22 experiments × 8 partitions = 176 parallel jobs"
echo "- Seeds per partition: 50 seeds"
echo "- Trials per partition: 64 trials"
echo "- Total trials across all jobs: 22 × 50 × 512 = 563,200 trials"
echo ""
echo "========================================================================"
