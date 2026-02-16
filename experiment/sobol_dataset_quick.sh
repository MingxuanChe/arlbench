#!/bin/bash
# Quick Test Version of Partitioned Sobol Dataset Script
# Uses minimal parameters for fast testing (< 10 minutes)

RUNTIME_LIST=()
EXP_LIST=()

echo "========================================================================"
echo "Quick Test: Partitioned Sobol Dataset Generation"
echo "========================================================================"
echo ""
echo "NOTE: This is a QUICK TEST version with minimal parameters"
echo "      For production use, see: experiment/sobol_dataset.sh"
echo ""

cd $HOME/Repos/arlbench-5090

# ============================================================================
# Configuration (Minimal for quick testing)
# ============================================================================
NUM_TRIALS=8   # Reduced from 16
N_PARTITIONS=2 # Reduced from 4
CONFIG="sobol_ppo_cartpole"
ALGO="ppo"
SEARCH_SPACE="ppo_cc"
SOBOL_SEED=42

# Only 3 seeds for quick test
SEED_LIST=(42 43 44)

echo "Quick Test Configuration:"
echo "  NUM_TRIALS: $NUM_TRIALS"
echo "  N_PARTITIONS: $N_PARTITIONS"
echo "  SEEDS: ${SEED_LIST[@]}"
echo "  SOBOL_SEED: $SOBOL_SEED"
echo ""

# Convert SEED_LIST to Hydra list format
SEED_STRING="[$(IFS=,; echo "${SEED_LIST[*]}")]"
echo "Hydra format: ${SEED_STRING}"
echo ""

# ============================================================================
# Experiment 1: Complete Sobol Sequence (Baseline)
# ============================================================================
echo "========================================================================"
echo "Step 1: Complete Sobol Sequence (n_sub=1) - BASELINE"
echo "========================================================================"
echo ""

EXP_NAME="complete_sobol"
EXP_LIST+=("$EXP_NAME")

START_TIME=$(date +%s)
pixi run python run_arlbench.py --config-name=$CONFIG -m \
  hydra.sweeper.n_trials=$NUM_TRIALS \
  autorl.seed=$SEED_STRING \
  algorithm=$ALGO \
  search_space=$SEARCH_SPACE \
  environment=cc_acrobot \
  hydra.sweeper.sweeper_kwargs.optimizer_kwargs.seed=$SOBOL_SEED \
  hydra.sweeper.sweeper_kwargs.optimizer_kwargs.n_sub=1 \
  hydra.sweeper.sweeper_kwargs.optimizer_kwargs.id_sub=0 \
  autorl.n_total_timesteps=10000 \
  autorl.n_eval_episodes=10 \
  hydra.sweep.dir="results/sobol_quick_test/complete/${ALGO}_cc_acrobot/\${autorl.seed}/complete"
END_TIME=$(date +%s)
RUNTIME_LIST+=($((END_TIME - START_TIME)))

echo ""
echo "✓ Complete sequence finished"
echo ""

# ============================================================================
# Experiment 2: Partitioned Sobol Sequences
# ============================================================================
echo "========================================================================"
echo "Step 2: Partitioned Sobol Sequences (n_sub=$N_PARTITIONS)"
echo "========================================================================"
echo ""

# Calculate trials per partition
TRIALS_PER_PARTITION=$((NUM_TRIALS / N_PARTITIONS))
echo "Trials per partition: $TRIALS_PER_PARTITION (total: $NUM_TRIALS / $N_PARTITIONS partitions)"
echo ""

for PARTITION_ID in $(seq 0 $((N_PARTITIONS - 1))); do
    echo "------------------------------------------------------------------------"
    echo "Partition $((PARTITION_ID + 1))/$N_PARTITIONS (id_sub=$PARTITION_ID)"
    echo "------------------------------------------------------------------------"
    
    EXP_NAME="partition_${PARTITION_ID}"
    EXP_LIST+=("$EXP_NAME")
    
    START_TIME=$(date +%s)
    pixi run python run_arlbench.py --config-name=$CONFIG -m \
      hydra.sweeper.n_trials=$TRIALS_PER_PARTITION \
      autorl.seed=$SEED_STRING \
      algorithm=$ALGO \
      search_space=$SEARCH_SPACE \
      environment=cc_acrobot \
      hydra.sweeper.sweeper_kwargs.optimizer_kwargs.seed=$SOBOL_SEED \
      hydra.sweeper.sweeper_kwargs.optimizer_kwargs.n_sub=$N_PARTITIONS \
      hydra.sweeper.sweeper_kwargs.optimizer_kwargs.id_sub=$PARTITION_ID \
      autorl.n_total_timesteps=10000 \
      autorl.n_eval_episodes=10 \
      hydra.sweep.dir="results/sobol_quick_test/partitioned/${ALGO}_cc_acrobot/\${autorl.seed}/partition_${PARTITION_ID}_of_${N_PARTITIONS}"
    END_TIME=$(date +%s)
    RUNTIME_LIST+=($((END_TIME - START_TIME)))
    
    echo ""
    echo "✓ Partition $((PARTITION_ID + 1))/$N_PARTITIONS completed"
    echo ""
done

# ============================================================================
# Step 3: Merge Partitioned Results
# ============================================================================
echo "========================================================================"
echo "Step 3: Merging Partitioned Results"
echo "========================================================================"
echo ""

# Hydra creates a single directory with all seeds: [42, 43, 44]  
printf -v SEED_STR '%s, ' "${SEED_LIST[@]}"
SEED_DIR_NAME="[${SEED_STR%, }]"  # Remove trailing ", "
echo "Merging results from directory: $SEED_DIR_NAME..."

pixi run python << EOF
import pandas as pd
from pathlib import Path
import json

n_partitions = ${N_PARTITIONS}
seed_dir_name = "${SEED_DIR_NAME}"
base_dir = Path(f"results/sobol_quick_test/partitioned/ppo_cc_acrobot/{seed_dir_name}")
output_dir = Path(f"results/sobol_quick_test/partitioned/ppo_cc_acrobot/{seed_dir_name}/merged")

partition_dirs = []
for i in range(n_partitions):
    partition_dir = base_dir / f"partition_{i}_of_{n_partitions}"
    if partition_dir.exists():
        partition_dirs.append(partition_dir)
        print(f"Found partition {i}")

if len(partition_dirs) != n_partitions:
    print(f"Warning: Expected {n_partitions} partitions, found {len(partition_dirs)}")

if not partition_dirs:
    print(f"No partitions found in {base_dir}")
    exit(1)

output_dir.mkdir(parents=True, exist_ok=True)

all_csv_files = {}
for partition_dir in partition_dirs:
    for csv_file in partition_dir.glob("**/*.csv"):
        rel_path = csv_file.relative_to(partition_dir)
        if rel_path not in all_csv_files:
            all_csv_files[rel_path] = []
        all_csv_files[rel_path].append(csv_file)

for rel_path, csv_files in all_csv_files.items():
    dfs = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"Merged {len(dfs)} files for {rel_path}")

merge_metadata = {
    "algorithm": "${ALGO}",
    "environment": "cc_acrobot",
    "seeds": [${SEED_LIST[*]}],
    "n_partitions": n_partitions,
    "sobol_seed": ${SOBOL_SEED},
}

with open(output_dir / "merge_metadata.json", "w") as f:
    json.dump(merge_metadata, f, indent=2)

print(f"✓ Merged results saved to {output_dir}")
EOF

echo ""
echo "✓ All results merged"
echo ""

# ============================================================================
# Step 4: Verification
# ============================================================================
echo "========================================================================"
echo "Step 4: Verification - Comparing Complete vs Partitioned Results"
echo "========================================================================"
echo ""

echo "Step 4.1: Checking result counts..."
echo "Step 4.1: Checking result counts..."
echo ""

printf -v SEED_STR '%s, ' "${SEED_LIST[@]}"
SEED_DIR_NAME="[${SEED_STR%, }]"

pixi run python << EOF
import pandas as pd
from pathlib import Path
import sys

n_trials = ${NUM_TRIALS}
n_partitions = ${N_PARTITIONS}
expected_per_partition = n_trials // n_partitions
seed_dir_name = "${SEED_DIR_NAME}"

print(f"Expected configs per partition: {expected_per_partition}")
print(f"Expected total when merged: {n_trials}")
print()

count_issues = False

# Check complete run
complete_csv = Path(f"results/sobol_quick_test/complete/ppo_cc_acrobot/{seed_dir_name}/complete/runhistory.csv")
if complete_csv.exists():
    df = pd.read_csv(complete_csv)
    if len(df) != n_trials:
        print(f"Complete: ✗ Expected {n_trials} configs, got {len(df)}")
        count_issues = True
    else:
        print(f"Complete: ✓ {len(df)} configs")
else:
    print(f"Complete: ✗ File not found")
    count_issues = True

# Check each partition
total_partition_configs = 0
for partition_id in range(n_partitions):
    partition_csv = Path(f"results/sobol_quick_test/partitioned/ppo_cc_acrobot/{seed_dir_name}/partition_{partition_id}_of_{n_partitions}/runhistory.csv")
    if partition_csv.exists():
        df = pd.read_csv(partition_csv)
        total_partition_configs += len(df)
        if len(df) != expected_per_partition:
            print(f"Partition {partition_id}: ✗ Expected {expected_per_partition} configs, got {len(df)}")
            count_issues = True
        else:
            print(f"Partition {partition_id}: ✓ {len(df)} configs")
    else:
        print(f"Partition {partition_id}: ✗ File not found")
        count_issues = True

if total_partition_configs > 0:
    if total_partition_configs != n_trials:
        print(f"Total partitioned: ✗ Expected {n_trials} configs, got {total_partition_configs}")
        count_issues = True
    else:
        print(f"Partitions: ✓ {n_partitions} × {expected_per_partition} = {total_partition_configs} configs")

print()
if count_issues:
    print("✗ Config count verification FAILED")
    sys.exit(1)
else:
    print("✓ Config count verification PASSED")
print()
EOF

COUNT_EXIT=$?
if [ $COUNT_EXIT -ne 0 ]; then
    echo "Verification failed at count check. Stopping."
    exit 1
fi

echo "Step 4.2: Comparing result data..."
echo ""

pixi run python << EOF
import pandas as pd
from pathlib import Path
import sys

seed_dir_name = "${SEED_DIR_NAME}"
complete_dir = Path(f"results/sobol_quick_test/complete/ppo_cc_acrobot/{seed_dir_name}/complete")
merged_dir = Path(f"results/sobol_quick_test/partitioned/ppo_cc_acrobot/{seed_dir_name}/merged")

if not complete_dir.exists() or not merged_dir.exists():
    print(f"✗ Missing directories")
    print(f"  Complete dir exists: {complete_dir.exists()}")
    print(f"  Merged dir exists: {merged_dir.exists()}")
    sys.exit(1)

complete_csvs = sorted(complete_dir.glob("**/*.csv"))
merged_csvs = sorted(merged_dir.glob("**/*.csv"))

if len(complete_csvs) == 0 or len(merged_csvs) == 0:
    print(f"✗ No CSV files found")
    print(f"  Complete CSVs: {len(complete_csvs)}")
    print(f"  Merged CSVs: {len(merged_csvs)}")
    sys.exit(1)

all_match = True
for complete_csv in complete_csvs:
    rel_path = complete_csv.relative_to(complete_dir)
    merged_csv = merged_dir / rel_path
    
    if not merged_csv.exists():
        print(f"✗ Missing file {rel_path}")
        all_match = False
        continue
    
    df_complete = pd.read_csv(complete_csv)
    df_merged = pd.read_csv(merged_csv)
    
    df_complete = df_complete.sort_values(by=list(df_complete.columns)).reset_index(drop=True)
    df_merged = df_merged.sort_values(by=list(df_merged.columns)).reset_index(drop=True)
    
    if df_complete.shape != df_merged.shape:
        print(f"✗ Shape mismatch for {rel_path}")
        print(f"  Complete: {df_complete.shape}, Merged: {df_merged.shape}")
        all_match = False
        continue
    
    try:
        pd.testing.assert_frame_equal(df_complete, df_merged, rtol=1e-10, atol=1e-10)
        print(f"✓ {rel_path}: MATCH")
    except AssertionError as e:
        print(f"✗ Data mismatch for {rel_path}")
        print(f"  {e}")
        all_match = False
        continue

print()
if all_match:
    print("="*72)
    print("✓ SUCCESS: All files match!")
    print("="*72)
else:
    print("="*72)
    print("✗ FAILURE: Some files do not match")
    print("="*72)
    sys.exit(1)
EOF

VERIFICATION_EXIT=$?
VERIFICATION_EXIT=$?

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo ""

for i in "${!EXP_LIST[@]}"; do
    EXP_NAME=${EXP_LIST[$i]}
    RUNTIME=${RUNTIME_LIST[$i]}
    MINUTES=$(( RUNTIME / 60 ))
    SECONDS=$((RUNTIME % 60))
    echo "Experiment: $EXP_NAME - Runtime: ${MINUTES}m ${SECONDS}s"
done

echo ""
echo "Results Location: results/sobol_quick_test/"
echo ""

if [ $VERIFICATION_EXIT -eq 0 ]; then
    echo "✓ QUICK TEST PASSED"
    echo ""
    echo "The partitioned Sobol implementation is working correctly!"
    echo "For production use with all 30 seeds, run: ./experiment/sobol_dataset.sh"
else
    echo "✗ QUICK TEST FAILED"
fi

echo ""
echo "To clean up test results:"
echo "  rm -rf results/sobol_quick_test/"
echo ""
