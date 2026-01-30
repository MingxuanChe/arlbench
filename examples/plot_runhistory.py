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


def plot_performance_distribution(df, save_path=None, n_seeds=None):
    """
    Create visualization of performance distribution across configurations and seeds.
    
    Args:
        df: DataFrame with 'performance_array' column
        save_path: Optional path to save the figure
    """
    # Extract all individual seed performances
    all_performances = np.concatenate(df['performance_array'].values)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall distribution histogram
    ax = axes[0, 0]
    ax.hist(all_performances, bins=50, alpha=0.7, edgecolor='black')
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
                fmt='o', alpha=0.6, capsize=3, markersize=4)
    ax.set_xlabel('Configuration ID')
    ax.set_ylabel('Mean Performance ± Std')
    ax.set_title('Mean Performance with Std Error per Configuration')
    ax.grid(alpha=0.3)
    
    # Highlight best configuration
    best_idx = mean_perfs.idxmax()
    ax.scatter([best_idx], [mean_perfs[best_idx]], color='red', s=100, 
               marker='*', label=f'Best: Config {best_idx}')
    ax.legend()
    
    # 4. KDE plot of performance distribution
    ax = axes[1, 1]
    sns.kdeplot(all_performances, ax=ax, fill=True, alpha=0.5)
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
        fig.suptitle(f'Performance Distribution across Configurations and Seeds (n_seeds={n_seeds})', fontsize=16)      
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = script_dir / "performance_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    
    # plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
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
    
    # Best configuration
    mean_perfs = df['performance_array'].apply(np.mean)
    best_idx = mean_perfs.idxmax()
    print(f"\nBest Configuration (by mean):")
    print(f"  Config ID: {best_idx}")
    print(f"  Mean Performance: {mean_perfs[best_idx]:.4f}")
    print(f"  Performance across seeds: {df['performance_array'].iloc[best_idx]}")
    print("="*60)


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
        "--save",
        type=str,
        default=None,
        help="Path to save the figure (default: display only)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading runhistory from: {args.csv_path}")
    df = load_runhistory(args.csv_path, n_seeds=args.n_seeds)
    
    # Create plots
    plot_performance_distribution(df, save_path=args.save, n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
