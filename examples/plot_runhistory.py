"""
Script to plot the performance distribution from ARLBench runhistory.csv files.
"""
import argparse
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

script_dir = Path(__file__).resolve().parent

def load_runhistory(csv_path, n_seeds=None):
    """
    Load runhistory CSV and parse performance lists.
    
    Args:
        csv_path: Path to runhistory.csv
        n_seeds: Number of seeds to load from each performance list (None = all)
    
    Returns:
        DataFrame with parsed performance values
    """
    df = pd.read_csv(csv_path)
    
    # Parse performance strings into numpy arrays
    performances = []
    for perf_str in df['performance']:
        # Parse the string representation of the list
        perf_list = ast.literal_eval(perf_str)
        perf_array = np.array(perf_list)
        
        # Limit to n_seeds if specified
        if n_seeds is not None:
            perf_array = perf_array[:n_seeds]
        
        performances.append(perf_array)
    
    df['performance_array'] = performances
    return df


def compute_distribution_stats(performances):
    """
    Compute comprehensive statistics for a performance distribution.
    
    Args:
        performances: Array of performance values
    
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(performances),
        'median': np.median(performances),
        'std': np.std(performances),
        'min': np.min(performances),
        'max': np.max(performances),
        'q25': np.percentile(performances, 25),
        'q75': np.percentile(performances, 75),
        'iqr': np.percentile(performances, 75) - np.percentile(performances, 25),
        'count': len(performances)
    }


def compute_config_metrics(df, alpha=1.0):
    """
    Compute configuration-level metrics.
    
    Args:
        df: DataFrame with 'performance_array' column
        alpha: Weight for variance penalty in mean-variance metric
    
    Returns:
        Dictionary with metric arrays
    """
    config_means = []
    config_vars = []
    config_stds = []
    config_mean_var = []
    
    for perf_array in df['performance_array']:
        mean = np.mean(perf_array)
        var = np.var(perf_array)
        std = np.std(perf_array)
        mean_var = mean - alpha * var
        
        config_means.append(mean)
        config_vars.append(var)
        config_stds.append(std)
        config_mean_var.append(mean_var)
    
    return {
        'mean': np.array(config_means),
        'variance': np.array(config_vars),
        'std': np.array(config_stds),
        'mean_variance': np.array(config_mean_var)
    }


def plot_config_metric_distributions(df, alpha=1.0, save_path=None, name=""):
    """
    Plot distribution analysis for configuration-level metrics.
    
    Args:
        df: DataFrame with 'performance_array' column
        alpha: Weight for variance penalty
        save_path: Path to save the figure
        name: Name identifier for the plot
    """
    # Compute metrics
    metrics = compute_config_metrics(df, alpha=alpha)
    
    # Find incumbent (best config)
    incumbent_idx = np.argmax(metrics['mean_variance'])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Mean performance distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(metrics['mean'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(metrics['mean'][incumbent_idx], color='red', linestyle='--', linewidth=2, label='Incumbent')
    ax1.set_xlabel('Mean Performance')
    ax1.set_ylabel('# Configurations')
    ax1.set_title('Mean Performance per Config')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(metrics['mean'], ax=ax2, fill=True, alpha=0.5, color='steelblue')
    ax2.axvline(metrics['mean'][incumbent_idx], color='red', linestyle='--', linewidth=2, label='Incumbent')
    ax2.set_xlabel('Mean Performance')
    ax2.set_ylabel('Density')
    ax2.set_title('Mean Performance KDE')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([metrics['mean']], labels=['Configs'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightsteelblue')
    ax3.set_ylabel('Mean Performance')
    ax3.set_title('Mean Performance Box Plot')
    ax3.grid(alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(metrics['mean'], metrics['std'], alpha=0.5, s=30, c='steelblue')
    ax4.scatter(metrics['mean'][incumbent_idx], metrics['std'][incumbent_idx], 
                s=200, c='red', marker='*', edgecolors='black', linewidths=2, label='Incumbent')
    ax4.set_xlabel('Mean Performance')
    ax4.set_ylabel('Std Dev')
    ax4.set_title('Mean vs Std Scatter')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Row 2: Mean-Variance metric distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(metrics['mean_variance'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axvline(metrics['mean_variance'][incumbent_idx], color='red', linestyle='--', linewidth=2, label='Incumbent')
    ax5.set_xlabel(f'Mean - {alpha}×Variance')
    ax5.set_ylabel('# Configurations')
    ax5.set_title(f'Mean-Variance Metric (α={alpha})')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(metrics['mean_variance'], ax=ax6, fill=True, alpha=0.5, color='steelblue')
    ax6.axvline(metrics['mean_variance'][incumbent_idx], color='red', linestyle='--', linewidth=2, label='Incumbent')
    ax6.set_xlabel(f'Mean - {alpha}×Variance')
    ax6.set_ylabel('Density')
    ax6.set_title('Mean-Variance KDE')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    ax7 = fig.add_subplot(gs[1, 2])
    bp = ax7.boxplot([metrics['mean_variance']], labels=['Configs'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightsteelblue')
    ax7.set_ylabel(f'Mean - {alpha}×Variance')
    ax7.set_title('Mean-Variance Box Plot')
    ax7.grid(alpha=0.3, axis='y')
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(metrics['mean'], metrics['variance'], alpha=0.5, s=30, c='steelblue')
    ax8.scatter(metrics['mean'][incumbent_idx], metrics['variance'][incumbent_idx], 
                s=200, c='red', marker='*', edgecolors='black', linewidths=2, label='Incumbent')
    ax8.set_xlabel('Mean Performance')
    ax8.set_ylabel('Variance')
    ax8.set_title('Mean vs Variance Scatter')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    # Row 3: Top configurations and summary
    ax9 = fig.add_subplot(gs[2, :2])
    # Top 20 configs by mean-variance
    top_n = min(20, len(metrics['mean_variance']))
    top_indices = np.argsort(metrics['mean_variance'])[-top_n:][::-1]
    
    ax9.barh(range(top_n), metrics['mean_variance'][top_indices], color='steelblue', alpha=0.7)
    ax9.set_yticks(range(top_n))
    ax9.set_yticklabels([f'Config {i}' for i in top_indices])
    ax9.set_xlabel(f'Mean - {alpha}×Variance')
    ax9.set_title(f'Top {top_n} Configurations by Mean-Variance')
    ax9.grid(alpha=0.3, axis='x')
    ax9.invert_yaxis()
    
    # Highlight incumbent
    incumbent_rank = np.where(top_indices == incumbent_idx)[0]
    if len(incumbent_rank) > 0:
        ax9.get_children()[incumbent_rank[0]].set_color('red')
    
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    # Summary statistics
    summary = "CONFIGURATION METRIC SUMMARY\n" + "="*50 + "\n\n"
    summary += f"Total Configurations: {len(df)}\n"
    summary += f"Seeds per Config: {len(df['performance_array'].iloc[0])}\n\n"
    
    summary += "Mean Performance:\n"
    summary += f"  μ = {np.mean(metrics['mean']):.3f}\n"
    summary += f"  σ = {np.std(metrics['mean']):.3f}\n"
    summary += f"  Min = {np.min(metrics['mean']):.3f}\n"
    summary += f"  Max = {np.max(metrics['mean']):.3f}\n\n"
    
    summary += f"Mean-Variance (α={alpha}):\n"
    summary += f"  μ = {np.mean(metrics['mean_variance']):.3f}\n"
    summary += f"  σ = {np.std(metrics['mean_variance']):.3f}\n"
    summary += f"  Min = {np.min(metrics['mean_variance']):.3f}\n"
    summary += f"  Max = {np.max(metrics['mean_variance']):.3f}\n\n"
    
    summary += "Incumbent (by Mean-Variance):\n"
    summary += f"  Config ID: {incumbent_idx}\n"
    summary += f"  Mean: {metrics['mean'][incumbent_idx]:.3f}\n"
    summary += f"  Variance: {metrics['variance'][incumbent_idx]:.4f}\n"
    summary += f"  Std Dev: {metrics['std'][incumbent_idx]:.3f}\n"
    summary += f"  Mean-Var: {metrics['mean_variance'][incumbent_idx]:.3f}\n"
    
    ax10.text(0.05, 0.95, summary, transform=ax10.transAxes, 
              verticalalignment='top', fontfamily='monospace', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    title = f'Configuration Metric Distribution Analysis'
    if name:
        title = f'{name}: {title}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Config metrics figure saved to {save_path}")
    
    plt.close()
    
    return {
        'metrics': metrics,
        'incumbent_idx': incumbent_idx
    }


def plot_performance_distribution(df, save_path=None, n_seeds=None, name="", alpha=1.0):
    """
    Create visualization of performance distribution across configurations and seeds.
    
    Args:
        df: DataFrame with 'performance_array' column
        save_path: Optional path to save the figure
        n_seeds: Number of seeds used
        name: Name identifier for the plot
        alpha: Weight for variance penalty (for title info)
    """
    # Extract all individual seed performances
    all_performances = np.concatenate(df['performance_array'].values)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall distribution histogram
    ax = axes[0, 0]
    ax.hist(all_performances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Performance')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Overall Performance Distribution\n(n={len(all_performances)} evaluations)')
    ax.axvline(np.mean(all_performances), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_performances):.2f}')
    ax.axvline(np.median(all_performances), color='green', linestyle='--',
               label=f'Median: {np.median(all_performances):.2f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Violin plot per configuration
    ax = axes[0, 1]
    # Prepare data for violin plot (limit to first 30 configs for readability)
    n_configs_to_show = min(30, len(df))
    violin_data = []
    positions = []
    for idx in range(n_configs_to_show):
        violin_data.append(df['performance_array'].iloc[idx])
        positions.append(idx)
    
    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    ax.set_xlabel('Configuration ID')
    ax.set_ylabel('Performance')
    ax.set_title(f'Performance Distribution per Configuration\n(showing first {n_configs_to_show} configs)')
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Box plot with mean performance
    ax = axes[1, 0]
    mean_perfs = df['performance_array'].apply(np.mean)
    std_perfs = df['performance_array'].apply(np.std)
    
    ax.errorbar(range(len(mean_perfs)), mean_perfs, yerr=std_perfs, 
                fmt='o', alpha=0.6, capsize=3, markersize=4, color='steelblue')
    ax.set_xlabel('Configuration ID')
    ax.set_ylabel('Mean Performance ± Std')
    ax.set_title('Mean Performance with Std Error per Configuration')
    ax.grid(alpha=0.3)
    
    # Highlight best configuration
    best_idx = mean_perfs.idxmax()
    ax.scatter([best_idx], [mean_perfs[best_idx]], color='red', s=100, 
               marker='*', label=f'Best: Config {best_idx}', zorder=5)
    ax.legend()
    
    # 4. KDE plot of performance distribution
    ax = axes[1, 1]
    sns.kdeplot(all_performances, ax=ax, fill=True, alpha=0.5, color='steelblue')
    ax.set_xlabel('Performance')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimate of Performance')
    ax.axvline(np.mean(all_performances), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_performances):.2f}')
    ax.legend()
    ax.grid(alpha=0.3)

    # add number of seeds info
    if n_seeds is None:
        n_seeds = len(df['performance_array'].iloc[0])
    
    title = f'Performance Distribution across Configurations and Seeds (n_seeds={n_seeds})'
    if name:
        title = f'{name}: {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold')      
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance distribution figure saved to {save_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("PERFORMANCE STATISTICS")
    print("="*70)
    print(f"Number of configurations: {len(df)}")
    print(f"Seeds per configuration: {len(df['performance_array'].iloc[0])}")
    print(f"Total evaluations: {len(all_performances)}")
    print(f"\nOverall Performance:")
    print(f"  Mean: {np.mean(all_performances):.4f}")
    print(f"  Median: {np.median(all_performances):.4f}")
    print(f"  Std: {np.std(all_performances):.4f}")
    print(f"  Min: {np.min(all_performances):.4f}")
    print(f"  Max: {np.max(all_performances):.4f}")
    print(f"  25th percentile: {np.percentile(all_performances, 25):.4f}")
    print(f"  75th percentile: {np.percentile(all_performances, 75):.4f}")
    
    # Best configuration by mean
    mean_perfs = df['performance_array'].apply(np.mean)
    best_idx = mean_perfs.idxmax()
    print(f"\nBest Configuration (by mean):")
    print(f"  Config ID: {best_idx}")
    print(f"  Mean Performance: {mean_perfs[best_idx]:.4f}")
    print(f"  Performance across seeds: {df['performance_array'].iloc[best_idx]}")
    
    # Best configuration by mean-variance
    metrics = compute_config_metrics(df, alpha=alpha)
    incumbent_idx = np.argmax(metrics['mean_variance'])
    print(f"\nBest Configuration (by mean-{alpha}×variance):")
    print(f"  Config ID: {incumbent_idx}")
    print(f"  Mean: {metrics['mean'][incumbent_idx]:.4f}")
    print(f"  Variance: {metrics['variance'][incumbent_idx]:.4f}")
    print(f"  Std: {metrics['std'][incumbent_idx]:.4f}")
    print(f"  Mean-Variance: {metrics['mean_variance'][incumbent_idx]:.4f}")
    print(f"  Performance across seeds: {df['performance_array'].iloc[incumbent_idx]}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance distribution from ARLBench runhistory.csv"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to runhistory.csv file"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Number of seeds to load from each performance list (default: all)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for variance penalty in mean-variance metric (default: 1.0)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name identifier for the experiment (e.g., 'PPO_CartPole-v1')"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save figures (default: same directory as csv_path)"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    
    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = csv_path.parent / "runhistory_analysis"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base name from experiment name or csv location
    if args.name:
        base_name = args.name
    else:
        # Try to extract algorithm and env from path
        # e.g., results/sobol/ppo_CartPole-v1/[seeds]/runhistory.csv -> ppo_CartPole-v1
        parts = csv_path.parts
        for part in reversed(parts):
            if '_' in part and not part.startswith('['):
                base_name = part
                break
        else:
            base_name = "runhistory"
    
    print(f"\n{'='*70}")
    print(f"  RUNHISTORY ANALYSIS")
    print(f"{'='*70}")
    print(f"CSV Path:     {csv_path}")
    print(f"Save Dir:     {save_dir}")
    print(f"Name:         {base_name}")
    print(f"Alpha:        {args.alpha}")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"Loading runhistory from: {csv_path}")
    df = load_runhistory(csv_path, n_seeds=args.n_seeds)
    
    # Create performance distribution plot
    perf_dist_path = save_dir / f"{base_name}_performance_distribution.png"
    plot_performance_distribution(df, save_path=perf_dist_path, 
                                   n_seeds=args.n_seeds, name=base_name, alpha=args.alpha)
    
    # Create config metrics plot
    config_metrics_path = save_dir / f"{base_name}_config_metrics.png"
    plot_config_metric_distributions(df, alpha=args.alpha, 
                                      save_path=config_metrics_path, name=base_name)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Figures saved to: {save_dir}")
    print(f"  - {perf_dist_path.name}")
    print(f"  - {config_metrics_path.name}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
