#!/bin/bash

# Merge script for all partitioned Sobol experiments
# Run this after all 8 partition jobs complete for all experiments

echo "========================================================================"
echo "ARLBench Dataset - Merge All Partitioned Results"
echo "========================================================================"
echo "Starting merge at: $(date)"
echo ""

cd $HOME/Repos/arlbench || exit 1

# Seed range
SEED_LIST=()
for SEED in {42..91}
do
  SEED_LIST+=($SEED)
done

echo "Merging 50 seeds: ${SEED_LIST[@]}"
echo ""
echo "========================================================================"

# Counter for progress
TOTAL_EXPERIMENTS=22
CURRENT=0

echo ""
echo "========== MERGING PPO EXPERIMENTS (11 environments) =========="
echo ""

# PPO Classic Control
echo "--- PPO Classic Control (5 environments) ---"
for env in "cc_acrobot" "cc_cartpole" "cc_continuous_mountain_car" "cc_mountain_car" "cc_pendulum"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging ppo_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm ppo \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed ppo_${env}"
    echo ""
done

# PPO Brax
echo "--- PPO Brax/MuJoCo (2 environments) ---"
for env in "brax_fast" "brax_halfcheetah"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging ppo_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm ppo \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed ppo_${env}"
    echo ""
done

# PPO MiniGrid/XLand
echo "--- PPO MiniGrid/XLand (4 environments) ---"
for env in "minigrid_door_key" "xland_door_key" "xland_empty_random" "xland_four_rooms"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging ppo_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm ppo \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed ppo_${env}"
    echo ""
done

echo ""
echo "========== MERGING SAC EXPERIMENTS (4 environments) =========="
echo ""

# SAC Classic Control
echo "--- SAC Classic Control (2 environments) ---"
for env in "cc_continuous_mountain_car" "cc_pendulum"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging sac_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm sac \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed sac_${env}"
    echo ""
done

# SAC Brax
echo "--- SAC Brax/MuJoCo (2 environments) ---"
for env in "brax_fast" "brax_halfcheetah"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging sac_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm sac \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed sac_${env}"
    echo ""
done

echo ""
echo "========== MERGING DQN EXPERIMENTS (7 environments) =========="
echo ""

# DQN Classic Control
echo "--- DQN Classic Control (3 environments) ---"
for env in "cc_acrobot" "cc_cartpole" "cc_mountain_car"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging dqn_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm dqn \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed dqn_${env}"
    echo ""
done

# DQN MiniGrid/XLand
echo "--- DQN MiniGrid/XLand (4 environments) ---"
for env in "minigrid_door_key" "xland_door_key" "xland_empty_random" "xland_four_rooms"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Merging dqn_${env}..."
    for seed in "${SEED_LIST[@]}"; do
        pixi run python merge_sobol_partitions.py \
            --base-dir multirun \
            --algorithm dqn \
            --environment $env \
            --seed $seed
    done
    echo "  ✓ Completed dqn_${env}"
    echo ""
done

echo ""
echo "========================================================================"
echo "ALL MERGES COMPLETED!"
echo "========================================================================"
echo "Merge completed at: $(date)"
echo ""
echo "Merged results are in:"
echo "  multirun/{algorithm}_{environment}/{seed}/merged/"
echo ""
echo "Verify merged datasets with:"
echo "  bash verify_merged_datasets.sh"
echo ""
echo "========================================================================"
