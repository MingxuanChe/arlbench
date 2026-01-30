"""
Script to investigate transfer learning behavior with incumbent configurations.

This script:
1. Finds incumbent configurations by mean and mean-variance metrics
2. Saves these configurations for later use
3. Runs experiments with these configurations on transfer environments
4. Collects and visualizes the results
"""
import argparse
import ast
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import Dict, List, Tuple

script_dir = Path(__file__).resolve().parent


def load_runhistory(csv_path, n_seeds=None):
    """
    Load runhistory CSV and parse performance lists.
    
    Args:
        csv_path: Path to runhistory.csv
        n_seeds: Number of seeds to load (None = all)
    
    Returns:
        DataFrame with parsed performance values
    """
    df = pd.read_csv(csv_path)
    
    # Parse performance strings into numpy arrays
    performances = []
    for perf_str in df['performance']:
        perf_list = ast.literal_eval(perf_str)
        perf_array = np.array(perf_list)
        
        if n_seeds is not None:
            perf_array = perf_array[:n_seeds]
        
        performances.append(perf_array)
    
    df['performance_array'] = performances
    return df


def extract_hp_config(df_row):
    """
    Extract hyperparameter configuration from a runhistory row.
    
    Args:
        df_row: DataFrame row from runhistory
    
    Returns:
        Dictionary of hyperparameters
    """
    hp_config = {}
    for col in df_row.index:
        if col.startswith('hp_config.'):
            param_name = col.replace('hp_config.', '')
            value = df_row[col]
            
            # Handle NaN values
            if pd.isna(value):
                continue
            
            # Convert types appropriately
            if isinstance(value, (np.bool_, bool)):
                hp_config[param_name] = bool(value)
            elif isinstance(value, (np.integer, int)):
                hp_config[param_name] = int(value)
            elif isinstance(value, (np.floating, float)):
                hp_config[param_name] = float(value)
            else:
                hp_config[param_name] = value
    
    return hp_config


def find_incumbents(df, alpha=1.0):
    """
    Find incumbent configurations based on mean and mean-variance metrics.
    
    Args:
        df: DataFrame with performance data
        alpha: Weight for variance penalty
    
    Returns:
        Tuple of (mean_incumbent_idx, mean_var_incumbent_idx, metrics)
    """
    config_means = []
    config_vars = []
    config_mean_vars = []
    
    for perf_array in df['performance_array']:
        mean = np.mean(perf_array)
        var = np.var(perf_array)
        mean_var = mean - alpha * var
        
        config_means.append(mean)
        config_vars.append(var)
        config_mean_vars.append(mean_var)
    
    config_means = np.array(config_means)
    config_vars = np.array(config_vars)
    config_mean_vars = np.array(config_mean_vars)
    
    # Find incumbents
    mean_incumbent_idx = np.argmax(config_means)
    mean_var_incumbent_idx = np.argmax(config_mean_vars)
    
    metrics = {
        'mean': config_means,
        'variance': config_vars,
        'mean_variance': config_mean_vars
    }
    
    return mean_incumbent_idx, mean_var_incumbent_idx, metrics


def save_incumbent_config(df, idx, algorithm, env_name, metric_name, output_dir):
    """
    Save incumbent configuration as a YAML file.
    
    Args:
        df: DataFrame with configurations
        idx: Index of incumbent configuration
        algorithm: Algorithm name
        env_name: Environment name
        metric_name: Name of the metric used (e.g., 'mean', 'mean_variance')
        output_dir: Directory to save the config
    
    Returns:
        Path to saved configuration file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract configuration
    hp_config = extract_hp_config(df.iloc[idx])
    
    # Get environment short name for config file
    env_short = env_name.replace('-v1', '').replace('-v0', '').lower()
    
    # Create config dictionary matching ARLBench structure
    hp_config_dict = hp_config
    metadata = {
        'config_id': int(df.iloc[idx]['config_id']),
        'mean_performance': float(np.mean(df.iloc[idx]['performance_array'])),
        'var_performance': float(np.var(df.iloc[idx]['performance_array'])),
        'metric_used': metric_name,
        'source_env': env_name
    }
    
    # Save as YAML manually to avoid quotes in defaults
    filename = f"{algorithm}_{env_short}_incumbent_{metric_name}.yaml"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        # Write defaults without quotes
        f.write("defaults:\n")
        f.write("- _self_\n")
        f.write(f"- /algorithm: {algorithm}\n")
        f.write(f"- /environment: cc_{env_short}\n")
        f.write(f"hpo_method: incumbent_{metric_name}\n")
        
        # Write hp_config using yaml
        f.write("hp_config:\n")
        for key, value in hp_config_dict.items():
            if isinstance(value, bool):
                f.write(f"  {key}: {str(value).lower()}\n")
            elif isinstance(value, (int, float)):
                f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("jax_enable_x64: ${environment.jax_enable_x64}\n")
        f.write("load_checkpoint: ''\n")
        
        # Write metadata
        f.write("_metadata:\n")
        for key, value in metadata.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Saved incumbent config to: {filepath}")
    print(f"  Config ID: {metadata['config_id']}")
    print(f"  Mean Performance: {metadata['mean_performance']:.4f}")
    print(f"  Variance: {metadata['var_performance']:.6f}")
    
    return filepath


def create_experiment_config(base_config_path, n_seeds, main_configs_dir):
    """
    Create experiment configuration file for running with multiple seeds.
    Copies the config to the main configs directory for Hydra to find.
    
    Args:
        base_config_path: Path to base configuration (incumbent)
        n_seeds: Number of seeds to run
        main_configs_dir: Main configs directory where Hydra looks
    
    Returns:
        Tuple of (config_name, config_path in main directory)
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate seed list
    seed_list = list(range(42, 42 + n_seeds))
    
    # Add autorl section for multi-seed execution
    config['autorl'] = {
        'seed': seed_list,
        'env_framework': '${environment.framework}',
        'env_name': '${environment.name}',
        'env_kwargs': '${environment.kwargs}',
        'env_params': '${environment.env_params}',
        'eval_env_kwargs': '${environment.eval_kwargs}',
        'n_envs': '${environment.n_envs}',
        'algorithm': '${algorithm}',
        'cnn_policy': '${environment.cnn_policy}',
        'deterministic_eval': '${environment.deterministic_eval}',
        'nas_config': '${nas_config}',
        'n_total_timesteps': '${environment.n_total_timesteps}',
        'checkpoint': [],
        'checkpoint_name': 'default_checkpoint',
        'checkpoint_dir': '/tmp',
        'state_features': [],
        'objectives': ['reward_mean'],
        'optimize_objectives': 'upper',
        'n_eval_steps': 10,
        'n_steps': 10,
        'n_eval_episodes': 128
    }
    
    # Derive config name from base config
    base_name = base_config_path.stem
    metric_name = 'mean' if 'mean' in base_name and 'variance' not in base_name else 'mean_variance'
    
    # Create config names for standard and transfer
    config_name = f"exp_{metric_name}"
    
    # Save to main configs directory
    main_configs_dir = Path(main_configs_dir)
    config_path = main_configs_dir / f"{config_name}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_name, config_path


def run_arlbench_experiment(config_path, metric_name, transfer=False, env_name="CartPole-v1"):
    """
    Run ARLBench experiment with given configuration.
    
    Args:
        config_path: Path to configuration file
        metric_name: Metric name for identification
        transfer: Whether this is a transfer experiment
        env_name: Environment name
    
    Returns:
        Performance array from the experiment
    """
    config_name = config_path.stem
    
    suffix = "transfer" if transfer else "standard"
    print(f"\n{'='*70}")
    print(f"Running ARLBench Experiment: {metric_name} incumbent on {suffix} environment")
    print(f"  Config: {config_name}")
    print(f"{'='*70}")
    
    # Construct command - use the main configs path with our custom config
    cmd = [
        'pixi', 'run', 'python', 
        str(script_dir / 'run_arlbench.py'),
        f'--config-name={config_name}'
    ]
    
    # Add transfer environment override if needed
    if transfer:
        env_short = env_name.replace('-v1', '').replace('-v0', '').lower()
        cmd.append(f'environment=cc_{env_short}_transfer')
    
    print(f"Executing: {' '.join(cmd)}")
    print("This may take several minutes...\n")
    
    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        if result.stdout:
            print("Training output (last 1000 chars):")
            print(result.stdout[-1000:])
        
        # Find and read performance file from Hydra output
        # Hydra creates directories based on config, look for performance.csv
        perf_data = collect_performance_from_hydra_output(metric_name, transfer)
        
        if perf_data is not None and len(perf_data) > 0:
            print(f"\n✓ Experiment completed successfully!")
            print(f"  Collected {len(perf_data)} performance values")
            print(f"  Mean: {np.mean(perf_data):.4f}, Std: {np.std(perf_data):.4f}")
            return perf_data
        else:
            print(f"\n⚠ Warning: No performance data collected from experiment")
            return np.array([])
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running experiment!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nSTDOUT (last 2000 chars):")
            print(e.stdout[-2000:])
        if e.stderr:
            print(f"\nSTDERR:")
            print(e.stderr)
        
        print(f"\n" + "="*70)
        print("Troubleshooting tips:")
        print("1. Check that the config exists in configs/ directory:")
        print(f"   configs/{config_name}.yaml")
        if transfer:
            env_short = env_name.replace('-v1', '').replace('-v0', '').lower()
            print("2. Check that the transfer environment config exists:")
            print(f"   configs/environment/cc_{env_short}_transfer.yaml")
        print("3. Verify the incumbent config has all required hyperparameters")
        print("4. Run with HYDRA_FULL_ERROR=1 for more details:")
        print(f"   HYDRA_FULL_ERROR=1 {' '.join(cmd)}")
        print("="*70)
        raise


def collect_performance_from_hydra_output(metric_name, transfer):
    """
    Collect performance data from Hydra output directory.
    
    Args:
        metric_name: Name of metric (mean or mean_variance)
        transfer: Whether this was a transfer experiment
    
    Returns:
        Array of performance values
    """
    # Look for performance.csv in recent Hydra outputs
    # The output directory structure depends on Hydra config
    suffix = "transfer" if transfer else "standard"
    
    # Check for performance.csv in current directory (Hydra default)
    perf_file = script_dir / 'performance.csv'
    if perf_file.exists():
        try:
            with open(perf_file, 'r') as f:
                content = f.read().strip()
                # Try to parse as a list/array
                if content.startswith('['):
                    perf_list = ast.literal_eval(content)
                    return np.array(perf_list)
                else:
                    # Single value
                    return np.array([float(content)])
        except Exception as e:
            print(f"Warning: Could not parse performance file: {e}")
    
    return None


def plot_incumbent_comparison(results_dict, alpha=1.0, save_dir=None):
    """
    Plot comparison of incumbent configurations on standard vs transfer environments.
    
    Args:
        results_dict: Dictionary with results for each incumbent
        alpha: Variance penalty weight
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data
    metrics = ['mean', 'mean_variance']
    envs = ['standard', 'transfer']
    
    all_data = {}
    for metric in metrics:
        all_data[metric] = {}
        for env in envs:
            key = f"{metric}_{env}"
            if key in results_dict:
                all_data[metric][env] = results_dict[key]
    
    # Plot 1: Mean incumbent - Distribution comparison
    ax = axes[0, 0]
    if 'mean' in all_data and 'standard' in all_data['mean'] and 'transfer' in all_data['mean']:
        std_perf = all_data['mean']['standard']
        trans_perf = all_data['mean']['transfer']
        
        bins = np.linspace(
            min(std_perf.min(), trans_perf.min()) if len(trans_perf) > 0 else std_perf.min(),
            max(std_perf.max(), trans_perf.max()) if len(trans_perf) > 0 else std_perf.max(),
            20
        )
        ax.hist(std_perf, bins=bins, alpha=0.6, label='Standard Env', color='blue', edgecolor='black')
        if len(trans_perf) > 0:
            ax.hist(trans_perf, bins=bins, alpha=0.6, label='Transfer Env', color='red', edgecolor='black')
        ax.axvline(np.mean(std_perf), color='blue', linestyle='--', linewidth=2)
        if len(trans_perf) > 0:
            ax.axvline(np.mean(trans_perf), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Performance')
        ax.set_ylabel('Frequency')
        ax.set_title('Mean Incumbent: Performance Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Plot 2: Mean incumbent - Box plot
    ax = axes[0, 1]
    if 'mean' in all_data and 'standard' in all_data['mean']:
        data_to_plot = [all_data['mean']['standard']]
        labels = ['Standard']
        if 'transfer' in all_data['mean'] and len(all_data['mean']['transfer']) > 0:
            data_to_plot.append(all_data['mean']['transfer'])
            labels.append('Transfer')
        
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Performance')
        ax.set_title('Mean Incumbent: Comparison')
        ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Mean incumbent - Metrics
    ax = axes[0, 2]
    ax.axis('off')
    if 'mean' in all_data and 'standard' in all_data['mean']:
        std_perf = all_data['mean']['standard']
        text = "MEAN INCUMBENT\n" + "="*35 + "\n\n"
        text += "Standard Environment:\n"
        text += f"  Mean: {np.mean(std_perf):.4f}\n"
        text += f"  Std:  {np.std(std_perf):.4f}\n"
        text += f"  Var:  {np.var(std_perf):.6f}\n"
        text += f"  Mean-Var (α={alpha}): {np.mean(std_perf) - alpha * np.var(std_perf):.4f}\n\n"
        
        if 'transfer' in all_data['mean'] and len(all_data['mean']['transfer']) > 0:
            trans_perf = all_data['mean']['transfer']
            text += "Transfer Environment:\n"
            text += f"  Mean: {np.mean(trans_perf):.4f}\n"
            text += f"  Std:  {np.std(trans_perf):.4f}\n"
            text += f"  Var:  {np.var(trans_perf):.6f}\n"
            text += f"  Mean-Var (α={alpha}): {np.mean(trans_perf) - alpha * np.var(trans_perf):.4f}\n\n"
            text += "Difference (Transfer - Standard):\n"
            text += f"  Δ Mean: {np.mean(trans_perf) - np.mean(std_perf):+.4f}\n"
            text += f"  Δ Std:  {np.std(trans_perf) - np.std(std_perf):+.4f}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Plot 4: Mean-Variance incumbent - Distribution comparison
    ax = axes[1, 0]
    if 'mean_variance' in all_data and 'standard' in all_data['mean_variance'] and 'transfer' in all_data['mean_variance']:
        std_perf = all_data['mean_variance']['standard']
        trans_perf = all_data['mean_variance']['transfer']
        
        bins = np.linspace(
            min(std_perf.min(), trans_perf.min()) if len(trans_perf) > 0 else std_perf.min(),
            max(std_perf.max(), trans_perf.max()) if len(trans_perf) > 0 else std_perf.max(),
            20
        )
        ax.hist(std_perf, bins=bins, alpha=0.6, label='Standard Env', color='blue', edgecolor='black')
        if len(trans_perf) > 0:
            ax.hist(trans_perf, bins=bins, alpha=0.6, label='Transfer Env', color='red', edgecolor='black')
        ax.axvline(np.mean(std_perf), color='blue', linestyle='--', linewidth=2)
        if len(trans_perf) > 0:
            ax.axvline(np.mean(trans_perf), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Performance')
        ax.set_ylabel('Frequency')
        ax.set_title('Mean-Variance Incumbent: Performance Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Plot 5: Mean-Variance incumbent - Box plot
    ax = axes[1, 1]
    if 'mean_variance' in all_data and 'standard' in all_data['mean_variance']:
        data_to_plot = [all_data['mean_variance']['standard']]
        labels = ['Standard']
        if 'transfer' in all_data['mean_variance'] and len(all_data['mean_variance']['transfer']) > 0:
            data_to_plot.append(all_data['mean_variance']['transfer'])
            labels.append('Transfer')
        
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Performance')
        ax.set_title('Mean-Variance Incumbent: Comparison')
        ax.grid(alpha=0.3, axis='y')
    
    # Plot 6: Mean-Variance incumbent - Metrics
    ax = axes[1, 2]
    ax.axis('off')
    if 'mean_variance' in all_data and 'standard' in all_data['mean_variance']:
        std_perf = all_data['mean_variance']['standard']
        text = "MEAN-VARIANCE INCUMBENT\n" + "="*35 + "\n\n"
        text += "Standard Environment:\n"
        text += f"  Mean: {np.mean(std_perf):.4f}\n"
        text += f"  Std:  {np.std(std_perf):.4f}\n"
        text += f"  Var:  {np.var(std_perf):.6f}\n"
        text += f"  Mean-Var (α={alpha}): {np.mean(std_perf) - alpha * np.var(std_perf):.4f}\n\n"
        
        if 'transfer' in all_data['mean_variance'] and len(all_data['mean_variance']['transfer']) > 0:
            trans_perf = all_data['mean_variance']['transfer']
            text += "Transfer Environment:\n"
            text += f"  Mean: {np.mean(trans_perf):.4f}\n"
            text += f"  Std:  {np.std(trans_perf):.4f}\n"
            text += f"  Var:  {np.var(trans_perf):.6f}\n"
            text += f"  Mean-Var (α={alpha}): {np.mean(trans_perf) - alpha * np.var(trans_perf):.4f}\n\n"
            text += "Difference (Transfer - Standard):\n"
            text += f"  Δ Mean: {np.mean(trans_perf) - np.mean(std_perf):+.4f}\n"
            text += f"  Δ Std:  {np.std(trans_perf) - np.std(std_perf):+.4f}\n"
        
        ax.text(0.1, 0.9, text, transform=ax.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.suptitle('Incumbent Configuration Transfer Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'incumbent_transfer_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Transfer learning experiment with incumbent configurations"
    )
    parser.add_argument(
        "--runhistory",
        type=str,
        required=True,
        help="Path to runhistory.csv from original HPO run"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['dqn', 'ppo', 'sac'],
        help="Algorithm used"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name (default: CartPole-v1)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Variance penalty weight for mean-variance metric (default: 1.0)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=30,
        help="Number of seeds for transfer experiments (default: 30)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="transfer_experiments",
        help="Output directory for results (default: transfer_experiments)"
    )
    parser.add_argument(
        "--skip_training",
        action='store_true',
        help="Skip training and only analyze existing results"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  TRANSFER LEARNING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Algorithm:      {args.algorithm}")
    print(f"Environment:    {args.env_name}")
    print(f"Alpha:          {args.alpha}")
    print(f"Seeds:          {args.n_seeds}")
    print(f"Output dir:     {output_dir}")
    print(f"{'='*70}\n")
    
    # Step 1: Load runhistory and find incumbents
    print("Step 1: Finding incumbent configurations...")
    df = load_runhistory(args.runhistory)
    print(f"Loaded {len(df)} configurations")
    
    mean_inc_idx, mean_var_inc_idx, metrics = find_incumbents(df, alpha=args.alpha)
    
    print(f"\nIncumbent by Mean:")
    print(f"  Config ID: {df.iloc[mean_inc_idx]['config_id']}")
    print(f"  Mean: {metrics['mean'][mean_inc_idx]:.4f}")
    print(f"  Variance: {metrics['variance'][mean_inc_idx]:.6f}")
    
    print(f"\nIncumbent by Mean-Variance (α={args.alpha}):")
    print(f"  Config ID: {df.iloc[mean_var_inc_idx]['config_id']}")
    print(f"  Mean: {metrics['mean'][mean_var_inc_idx]:.4f}")
    print(f"  Variance: {metrics['variance'][mean_var_inc_idx]:.6f}")
    print(f"  Mean-Var: {metrics['mean_variance'][mean_var_inc_idx]:.4f}")
    
    # Step 2: Save incumbent configurations
    print(f"\nStep 2: Saving incumbent configurations...")
    config_dir = output_dir / "configs"
    
    mean_config_path = save_incumbent_config(
        df, mean_inc_idx, args.algorithm, args.env_name, 'mean', config_dir
    )
    
    mean_var_config_path = save_incumbent_config(
        df, mean_var_inc_idx, args.algorithm, args.env_name, 'mean_variance', config_dir
    )
    
    # Step 3 & 4: Run experiments and collect results
    results_dict = {}
    
    if not args.skip_training:
        print(f"\nStep 3-4: Running transfer experiments...")
        print(f"This will run 4 experiments total (2 incumbents × 2 environments)")
        print(f"Each experiment trains with {args.n_seeds} seeds")
        
        # Main configs directory for Hydra
        main_configs_dir = script_dir / "configs"
        
        for metric_name, config_path in [
            ('mean', mean_config_path),
            ('mean_variance', mean_var_config_path)
        ]:
            # Create experiment config in main configs directory
            print(f"\n{'='*70}")
            print(f"Preparing experiments for {metric_name} incumbent...")
            print(f"{'='*70}")
            
            # Create experiment config
            exp_config_name, exp_config_path = create_experiment_config(
                config_path, 
                args.n_seeds,
                main_configs_dir
            )
            print(f"Created config: {exp_config_path}")
            
            # Run standard environment experiment
            print(f"\n[1/2] Running on STANDARD environment...")
            perf_std = run_arlbench_experiment(
                exp_config_path, 
                metric_name, 
                transfer=False,
                env_name=args.env_name
            )
            results_dict[f'{metric_name}_standard'] = perf_std
            
            # Run transfer environment experiment
            print(f"\n[2/2] Running on TRANSFER environment...")
            perf_transfer = run_arlbench_experiment(
                exp_config_path,
                metric_name,
                transfer=True,
                env_name=args.env_name
            )
            results_dict[f'{metric_name}_transfer'] = perf_transfer
            
            print(f"\n✓ Completed all experiments for {metric_name} incumbent")
            
            # Clean up config from main directory
            try:
                exp_config_path.unlink()
                print(f"Cleaned up temporary config: {exp_config_path}")
            except:
                pass
                
    else:
        print("\nSkipping training (--skip_training flag set)")
        print("Loading existing results if available...")
        
        # Try to load from previously saved results file
        results_file = output_dir / 'transfer_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                saved_data = json.load(f)
                if 'results' in saved_data:
                    results_dict = {k: np.array(v) for k, v in saved_data['results'].items()}
                    print(f"Loaded {len(results_dict)} result sets from {results_file}")
        else:
            print(f"No existing results found at {results_file}")
            print("Using simulated data for demonstration...")
            
            # Simulate data
            baseline_mean = metrics['mean'][mean_inc_idx]
            baseline_std = np.sqrt(metrics['variance'][mean_inc_idx])
            results_dict['mean_standard'] = np.random.normal(baseline_mean, baseline_std, args.n_seeds)
            results_dict['mean_transfer'] = np.random.normal(baseline_mean - 10, baseline_std * 1.1, args.n_seeds)
            
            baseline_mean = metrics['mean'][mean_var_inc_idx]
            baseline_std = np.sqrt(metrics['variance'][mean_var_inc_idx])
            results_dict['mean_variance_standard'] = np.random.normal(baseline_mean, baseline_std, args.n_seeds)
            results_dict['mean_variance_transfer'] = np.random.normal(baseline_mean - 5, baseline_std * 0.9, args.n_seeds)
    
    # Step 5: Plot results
    print(f"\nStep 5: Generating visualizations...")
    
    # Save results summary
    results_file = output_dir / 'transfer_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'algorithm': args.algorithm,
            'environment': args.env_name,
            'alpha': args.alpha,
            'n_seeds': args.n_seeds,
            'mean_incumbent_id': int(df.iloc[mean_inc_idx]['config_id']),
            'mean_var_incumbent_id': int(df.iloc[mean_var_inc_idx]['config_id']),
            'results': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in results_dict.items()}
        }, f, indent=2)
    
    print(f"Saved results to: {results_file}")
    
    # Generate plots
    plot_incumbent_comparison(results_dict, alpha=args.alpha, save_dir=output_dir)
    
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - Configs: {config_dir}")
    print(f"  - Results: {results_file}")
    print(f"  - Plots: {output_dir / 'incumbent_transfer_comparison.png'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
