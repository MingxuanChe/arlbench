#!/bin/bash
# Automated transfer learning experiment pipeline
# 
# This script:
# 1. Finds incumbent configurations from HPO runhistory
# 2. Runs training with these configs on standard and transfer environments
# 3. Collects and analyzes results

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ALGORITHM="dqn"
ENV_NAME="CartPole-v1"
ALPHA=1.0
N_SEEDS=30
RUNHISTORY=""
OUTPUT_DIR="transfer_experiments"
SKIP_TRAINING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --runhistory)
            RUNHISTORY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --help)
            echo "Usage: $0 --runhistory <path> [options]"
            echo ""
            echo "Options:"
            echo "  --algorithm <name>      Algorithm (dqn, ppo, sac) [default: dqn]"
            echo "  --env-name <name>       Environment name [default: CartPole-v1]"
            echo "  --alpha <value>         Variance penalty [default: 1.0]"
            echo "  --n-seeds <num>         Number of seeds [default: 30]"
            echo "  --runhistory <path>     Path to runhistory.csv (required)"
            echo "  --output-dir <path>     Output directory [default: transfer_experiments]"
            echo "  --skip-training         Only find incumbents, skip training"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$RUNHISTORY" ]; then
    echo -e "${RED}Error: --runhistory is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$RUNHISTORY" ]; then
    echo -e "${RED}Error: Runhistory file not found: $RUNHISTORY${NC}"
    exit 1
fi

# Setup
ENV_SHORT=$(echo "$ENV_NAME" | sed 's/-v[0-9]$//' | tr '[:upper:]' '[:lower:]')
WORK_DIR="$OUTPUT_DIR/${ALGORITHM}_${ENV_SHORT}"
mkdir -p "$WORK_DIR"

echo "======================================================================"
echo "  TRANSFER LEARNING EXPERIMENT PIPELINE"
echo "======================================================================"
echo "Algorithm:      $ALGORITHM"
echo "Environment:    $ENV_NAME"
echo "Alpha:          $ALPHA"
echo "Seeds:          $N_SEEDS"
echo "Runhistory:     $RUNHISTORY"
echo "Output dir:     $WORK_DIR"
echo "======================================================================"
echo ""

# Step 1: Find incumbent configurations using Python script
echo -e "${GREEN}Step 1: Finding incumbent configurations...${NC}"
pixi run python run_transfer_experiment.py \
    --runhistory "$RUNHISTORY" \
    --algorithm "$ALGORITHM" \
    --env_name "$ENV_NAME" \
    --alpha "$ALPHA" \
    --n_seeds "$N_SEEDS" \
    --output_dir "$WORK_DIR" \
    --skip_training

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to find incumbent configurations${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Incumbent configurations saved to: $WORK_DIR/configs/${NC}"
echo ""

# Check if we should skip training
if [ "$SKIP_TRAINING" = true ]; then
    echo -e "${YELLOW}Skipping training (--skip-training flag set)${NC}"
    exit 0
fi

# Step 2: Copy incumbent configs to main configs directory
echo -e "${GREEN}Step 2: Preparing experiment configurations...${NC}"

CONFIG_DIR="$WORK_DIR/configs"
MAIN_CONFIG_DIR="configs"

# Find the two incumbent config files
MEAN_CONFIG="$CONFIG_DIR/${ALGORITHM}_${ENV_SHORT}_incumbent_mean.yaml"
MEANVAR_CONFIG="$CONFIG_DIR/${ALGORITHM}_${ENV_SHORT}_incumbent_mean_variance.yaml"

if [ ! -f "$MEAN_CONFIG" ]; then
    echo -e "${RED}Error: Mean incumbent config not found: $MEAN_CONFIG${NC}"
    exit 1
fi

if [ ! -f "$MEANVAR_CONFIG" ]; then
    echo -e "${RED}Error: Mean-variance incumbent config not found: $MEANVAR_CONFIG${NC}"
    exit 1
fi

# Function to create experiment config with multiple seeds
create_exp_config() {
    local base_config=$1
    local output_name=$2
    local n_seeds=$3
    
    # Generate seed list
    local seeds="["
    for i in $(seq 0 $((n_seeds - 1))); do
        if [ $i -gt 0 ]; then
            seeds="${seeds}, "
        fi
        seeds="${seeds}$((42 + i))"
    done
    seeds="${seeds}]"
    
    # Create config by adding autorl section
    cat "$base_config" > "$MAIN_CONFIG_DIR/$output_name.yaml"
    cat >> "$MAIN_CONFIG_DIR/$output_name.yaml" <<EOF

autorl:
  seed: $seeds
  env_framework: \${environment.framework}
  env_name: \${environment.name}
  env_kwargs: \${environment.kwargs}
  env_params: \${environment.env_params}
  eval_env_kwargs: \${environment.eval_kwargs}
  n_envs: \${environment.n_envs}
  algorithm: \${algorithm}
  cnn_policy: \${environment.cnn_policy}
  deterministic_eval: \${environment.deterministic_eval}
  nas_config: \${nas_config}
  n_total_timesteps: \${environment.n_total_timesteps}
  checkpoint: []
  checkpoint_name: default_checkpoint
  checkpoint_dir: /tmp
  state_features: []
  objectives:
    - reward_mean
  optimize_objectives: upper
  n_eval_steps: 10
  n_steps: 10
  n_eval_episodes: 128
EOF
    
    echo -e "  Created: $MAIN_CONFIG_DIR/$output_name.yaml"
}

# Create experiment configs
create_exp_config "$MEAN_CONFIG" "exp_transfer_mean" "$N_SEEDS"
create_exp_config "$MEANVAR_CONFIG" "exp_transfer_meanvar" "$N_SEEDS"

echo ""

# Step 3: Run experiments
echo -e "${GREEN}Step 3: Running training experiments...${NC}"
echo "This will run 4 experiments (2 incumbents × 2 environments)"
echo "Each with $N_SEEDS seeds - this may take a while!"
echo ""

# Create results directory
RESULTS_DIR="$WORK_DIR/results"
mkdir -p "$RESULTS_DIR"

# Function to run experiment
run_experiment() {
    local config_name=$1
    local env_override=$2
    local result_name=$3
    
    echo "======================================================================"
    echo "Running: $result_name"
    echo "======================================================================"
    
    local cmd="pixi run python run_arlbench.py --config-name=$config_name"
    if [ -n "$env_override" ]; then
        cmd="$cmd $env_override"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    # Run and save output
    if eval "$cmd" > "$RESULTS_DIR/${result_name}.log" 2>&1; then
        echo -e "${GREEN}✓ Completed: $result_name${NC}"
        
        # Copy performance file if it exists
        if [ -f "performance.csv" ]; then
            cp performance.csv "$RESULTS_DIR/${result_name}_performance.csv"
            echo "  Saved performance to: $RESULTS_DIR/${result_name}_performance.csv"
        fi
    else
        echo -e "${RED}✗ Failed: $result_name${NC}"
        echo "  See log: $RESULTS_DIR/${result_name}.log"
        return 1
    fi
    
    echo ""
}

# Run all 4 experiments
echo -e "${YELLOW}[1/4] Mean incumbent - Standard environment${NC}"
run_experiment "exp_transfer_mean" "" "mean_standard"

echo -e "${YELLOW}[2/4] Mean incumbent - Transfer environment${NC}"
run_experiment "exp_transfer_mean" "environment=cc_${ENV_SHORT}_transfer" "mean_transfer"

echo -e "${YELLOW}[3/4] Mean-Variance incumbent - Standard environment${NC}"
run_experiment "exp_transfer_meanvar" "" "mean_variance_standard"

echo -e "${YELLOW}[4/4] Mean-Variance incumbent - Transfer environment${NC}"
run_experiment "exp_transfer_meanvar" "environment=cc_${ENV_SHORT}_transfer" "mean_variance_transfer"

# Step 4: Collect and analyze results
echo -e "${GREEN}Step 4: Collecting and analyzing results...${NC}"

# Create Python script to collect results
cat > "$WORK_DIR/collect_results.py" <<'PYEOF'
import json
import sys
from pathlib import Path
import re

results_dir = Path(sys.argv[1])
output_file = results_dir.parent / "transfer_results.json"

results = {}
for perf_file in results_dir.glob("*_performance.csv"):
    name = perf_file.stem.replace("_performance", "")
    
    with open(perf_file, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            # NumPy array format: [1.2 3.4] or [500. 500.] without commas
            # Convert to proper list by adding commas between numbers
            # Replace all whitespace between numbers with comma+space
            content = re.sub(r'(\d\.?\d*)\s+(\d)', r'\1, \2', content)
            # Handle case where there might still be spaces after commas
            content = re.sub(r',\s*,', r',', content)
            perf_list = eval(content)
        else:
            perf_list = [float(content)]
    
    results[name] = perf_list
    print(f"Loaded {len(perf_list)} values for {name}")

# Save combined results
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved combined results to: {output_file}")
PYEOF

pixi run python "$WORK_DIR/collect_results.py" "$RESULTS_DIR"

# Generate plots
echo ""
echo -e "${GREEN}Step 5: Generating visualization...${NC}"

cat > "$WORK_DIR/generate_plot.py" <<'PLOTEOF'
import json
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import plotting function
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from run_transfer_experiment import plot_incumbent_comparison

# Load results
results_file = Path(sys.argv[1])
with open(results_file, 'r') as f:
    data = json.load(f)

# Convert to numpy arrays
results_dict = {k: np.array(v) for k, v in data.items()}

# Generate plot
save_dir = results_file.parent
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
plot_incumbent_comparison(results_dict, alpha=alpha, save_dir=save_dir)
print(f"Generated comparison plot: {save_dir}/incumbent_transfer_comparison.png")
PLOTEOF

pixi run python "$WORK_DIR/generate_plot.py" "$WORK_DIR/transfer_results.json" "$ALPHA"

# Cleanup temporary configs
echo ""
echo -e "${GREEN}Cleaning up temporary configs...${NC}"
rm -f "$MAIN_CONFIG_DIR/exp_transfer_mean.yaml"
rm -f "$MAIN_CONFIG_DIR/exp_transfer_meanvar.yaml"

echo ""
echo "======================================================================"
echo -e "${GREEN}  PIPELINE COMPLETE!${NC}"
echo "======================================================================"
echo "Results saved to: $WORK_DIR"
echo "  - Incumbent configs: $CONFIG_DIR"
echo "  - Experiment logs:   $RESULTS_DIR"
echo "  - Performance data:  $WORK_DIR/transfer_results.json"
echo "  - Plots:             $WORK_DIR/incumbent_transfer_comparison.png"
echo "======================================================================"
echo ""
echo "To visualize results, run the analysis script on the collected data."
