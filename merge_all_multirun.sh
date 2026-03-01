#!/bin/bash
# Merge all partitioned datasets in multirun/
# Handles multi-seed directory format: [42, 43, ..., 71]

echo "========================================================================"
echo "Merging All Partitioned Datasets in multirun/"
echo "========================================================================"
echo ""

BASE_DIR="multirun"
N_PARTITIONS=8  # All datasets use 8 partitions

# Build seed list and directory name
SEED_LIST=()
for SEED in {42..91}
do
  SEED_LIST+=($SEED)
done

# Format seed directory name as Hydra creates it: [42, 43, 44, ...]
printf -v SEED_STR '%s, ' "${SEED_LIST[@]}"
SEED_DIR_NAME="[${SEED_STR%, }]"  # Remove trailing ", "

echo "Configuration:"
echo "  Base dir:     $BASE_DIR"
echo "  Partitions:   $N_PARTITIONS"
echo "  Seeds:        $SEED_DIR_NAME"
echo ""

# Find all experiment directories
EXPERIMENTS=()
for exp_dir in "$BASE_DIR"/*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        EXPERIMENTS+=("$exp_name")
    fi
done

echo "Found ${#EXPERIMENTS[@]} experiment(s):"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo ""

# Parse algorithm and environment from directory name
merge_experiment() {
    local exp_name=$1
    
    # Extract algorithm and environment
    # Format: algorithm_environment (e.g., ppo_brax_fast)
    local algo=$(echo "$exp_name" | cut -d'_' -f1)
    local env=$(echo "$exp_name" | cut -d'_' -f2-)
    
    echo "========================================================================"
    echo "Merging: $exp_name"
    echo "  Algorithm:    $algo"
    echo "  Environment:  $env"
    echo "========================================================================"
    echo ""
    
    # Run Python merge script (same as aggregate_sobol_partition.sh)
    pixi run python << EOF
import pandas as pd
import shutil
from pathlib import Path
import json
import sys

n_partitions = ${N_PARTITIONS}
seed_dir_name = "${SEED_DIR_NAME}"
algo = "${algo}"
env = "${env}"
base_dir = Path("${BASE_DIR}")

# Find the experiment directory
exp_dir = base_dir / f"{algo}_{env}" / seed_dir_name
if not exp_dir.exists():
    print(f"✗ Experiment directory not found: {exp_dir}")
    sys.exit(1)

print(f"Found experiment directory: {exp_dir}")
print()

# Find partition directories
partition_dirs = []
for i in range(n_partitions):
    partition_dir = exp_dir / f"partition_{i}_of_{n_partitions}"
    if partition_dir.exists():
        partition_dirs.append(partition_dir)
        print(f"✓ Found partition {i}")
    else:
        print(f"✗ Missing partition {i}: {partition_dir}")

if len(partition_dirs) != n_partitions:
    print(f"Warning: Expected {n_partitions} partitions, found {len(partition_dirs)}")

if not partition_dirs:
    print(f"✗ No partitions found in {exp_dir}")
    sys.exit(1)

print()

# Create output directory
output_dir = exp_dir / "merged"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")
print()

# 1. Merge runhistory.csv with sequential config_id
print("Step 1: Merging runhistory.csv with sequential config_id...")
runhistory_dfs = []
for partition_dir in sorted(partition_dirs):
    runhistory_file = partition_dir / "runhistory.csv"
    if runhistory_file.exists():
        df = pd.read_csv(runhistory_file)
        runhistory_dfs.append(df)
        print(f"  ✓ Loaded partition: {partition_dir.name} ({len(df)} configs)")

if runhistory_dfs:
    merged_runhistory = pd.concat(runhistory_dfs, ignore_index=True)
    # Renumber config_id sequentially (0, 1, 2, ..., N-1)
    merged_runhistory['config_id'] = range(len(merged_runhistory))
    runhistory_output = output_dir / "runhistory.csv"
    merged_runhistory.to_csv(runhistory_output, index=False)
    print(f"✓ Merged runhistory: {len(merged_runhistory)} configs with sequential IDs (0-{len(merged_runhistory)-1})")
    print()
    
    # 2. Build incumbent history incrementally (tracks improvements)
    print("Step 2: Building incremental incumbent history...")
    incumbent_rows = []
    best_performance = float('-inf')
    for idx, row in merged_runhistory.iterrows():
        current_performance = row['mean_performance']
        if current_performance > best_performance:
            best_performance = current_performance
            incumbent_rows.append(row.to_dict())
    
    if incumbent_rows:
        incumbent_df = pd.DataFrame(incumbent_rows)
        incumbent_output = output_dir / "incumbent.csv"
        incumbent_df.to_csv(incumbent_output, index=False)
        print(f"✓ Incumbent history: {len(incumbent_rows)} improvements tracked")
        print()
else:
    print("✗ No runhistory.csv files found")
    sys.exit(1)

# 3. Copy job directories (0/, 1/, 2/, etc.)
print("Step 3: Copying job directories...")
job_count = 0
for partition_dir in sorted(partition_dirs):
    for job_dir in sorted(partition_dir.iterdir()):
        if job_dir.is_dir() and job_dir.name.isdigit():
            # Copy to merged with sequential numbering
            dest_dir = output_dir / str(job_count)
            if not dest_dir.exists():
                shutil.copytree(job_dir, dest_dir)
            job_count += 1
if job_count > 0:
    print(f"✓ Copied {job_count} job directories")
    print()

# 4. Merge other CSV files (evaluation CSVs, etc.)
print("Step 4: Merging other CSV files...")
all_csv_files = {}
for partition_dir in partition_dirs:
    for csv_file in partition_dir.glob("**/*.csv"):
        rel_path = csv_file.relative_to(partition_dir)
        # Skip files already handled
        if rel_path.name in ['runhistory.csv', 'incumbent.csv']:
            continue
        # Skip job directories (already copied)
        if rel_path.parts[0].isdigit():
            continue
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
            print(f"  ✗ Warning: Could not read {csv_file}: {e}")
    
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        output_file = output_dir / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)
        print(f"  ✓ Merged {rel_path} ({len(dfs)} files, {len(merged_df)} rows)")

if all_csv_files:
    print(f"✓ Merged {len(all_csv_files)} other CSV file(s)")
else:
    print("  No other CSV files to merge")
print()

# Save merge metadata
merge_metadata = {
    "algorithm": algo,
    "environment": env,
    "seeds": list(range(42, 72)),  # Seeds 42-71
    "n_partitions": n_partitions,
    "merged_configs": len(merged_runhistory) if 'merged_runhistory' in locals() else 0,
}

with open(output_dir / "merge_metadata.json", "w") as f:
    json.dump(merge_metadata, f, indent=2)

print("="*72)
print("MERGE COMPLETE")
print("="*72)
print(f"Output directory: {output_dir}")
print(f"  - runhistory.csv: Sequential config_id (0-{len(merged_runhistory)-1})")
print(f"  - incumbent.csv: Incremental improvement history")
print(f"  - Job directories: {job_count} directories copied")
print("="*72)
EOF
    
    local EXIT_CODE=$?
    echo ""
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $exp_name merged successfully"
        # Copy merged runhistory.csv to multirun/ root, named after the experiment
        local merged_rh="${BASE_DIR}/${exp_name}/${SEED_DIR_NAME}/merged/runhistory.csv"
        local dest_rh="${BASE_DIR}/${exp_name}_raw.csv"
        if [ -f "$merged_rh" ]; then
            cp "$merged_rh" "$dest_rh"
            echo "✓ Copied runhistory to $dest_rh"
        else
            echo "✗ Warning: merged runhistory not found at $merged_rh"
        fi
        return 0
    else
        echo "✗ $exp_name merge failed"
        return 1
    fi
}

# Merge all experiments
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_EXPS=()

for exp in "${EXPERIMENTS[@]}"; do
    if merge_experiment "$exp"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
        FAILED_EXPS+=("$exp")
    fi
    echo ""
done

# Summary
echo "========================================================================"
echo "FINAL SUMMARY"
echo "========================================================================"
echo "Total Experiments:  ${#EXPERIMENTS[@]}"
echo "Successfully Merged: $SUCCESS_COUNT"
echo "Failed:             $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failed Experiments:"
    for exp in "${FAILED_EXPS[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

echo "All merged results are in: $BASE_DIR/<experiment>/[seeds]/merged/"
echo "========================================================================"

if [ $FAIL_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi
