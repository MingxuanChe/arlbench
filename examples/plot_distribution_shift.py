"""
Script to compare performance distribution shifts between standard and transfer learning.
Focuses on CartPole-v1 environment with multiple algorithms (PPO, SAC, DQN).
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


def compare_distributions(perf1, perf2, name1="Standard", name2="Transfer"):
    """
    Compare two performance distributions statistically.
    
    Args:
        perf1: First performance array
        perf2: Second performance array
        name1: Name of first distribution
        name2: Name of second distribution
    
    Returns:
        Dictionary of comparison statistics
    """
    # Statistical tests
    t_stat, t_pval = stats.ttest_ind(perf1, perf2)
    ks_stat, ks_pval = stats.ks_2samp(perf1, perf2)
    mw_stat, mw_pval = stats.mannwhitneyu(perf1, perf2, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(perf1)**2 + np.std(perf2)**2) / 2)
    cohens_d = (np.mean(perf2) - np.mean(perf1)) / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        't_pvalue': t_pval,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'mannwhitney_statistic': mw_stat,
        'mannwhitney_pvalue': mw_pval,
        'cohens_d': cohens_d,
        'mean_diff': np.mean(perf2) - np.mean(perf1),
        'median_diff': np.median(perf2) - np.median(perf1),
        'std_ratio': np.std(perf2) / np.std(perf1) if np.std(perf1) > 0 else np.inf
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


def plot_config_metric_distributions(algorithm, df_std, df_transfer, alpha=1.0, save_dir=None):
    """
    Plot distribution shifts for configuration-level metrics.
    
    Args:
        algorithm: Name of the algorithm
        df_std: DataFrame for standard training
        df_transfer: DataFrame for transfer learning
        alpha: Weight for variance penalty
        save_dir: Directory to save the figure
    """
    # Compute metrics for both datasets
    metrics_std = compute_config_metrics(df_std, alpha=alpha)
    metrics_transfer = compute_config_metrics(df_transfer, alpha=alpha)
    
    # Find incumbents (best configs)
    incumbent_std_idx = np.argmax(metrics_std['mean_variance'])
    incumbent_transfer_idx = np.argmax(metrics_transfer['mean_variance'])
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Mean performance distribution
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(
        min(metrics_std['mean'].min(), metrics_transfer['mean'].min()),
        max(metrics_std['mean'].max(), metrics_transfer['mean'].max()),
        30
    )
    ax1.hist(metrics_std['mean'], bins=bins, alpha=0.6, label='Standard', color='blue', edgecolor='black')
    ax1.hist(metrics_transfer['mean'], bins=bins, alpha=0.6, label='Transfer', color='red', edgecolor='black')
    ax1.axvline(metrics_std['mean'][incumbent_std_idx], color='blue', linestyle='--', linewidth=2, label='Std Incumbent')
    ax1.axvline(metrics_transfer['mean'][incumbent_transfer_idx], color='red', linestyle='--', linewidth=2, label='Trans Incumbent')
    ax1.set_xlabel('Mean Performance')
    ax1.set_ylabel('# Configurations')
    ax1.set_title('Mean Performance per Config')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.kdeplot(metrics_std['mean'], ax=ax2, fill=True, alpha=0.4, label='Standard', color='blue')
    sns.kdeplot(metrics_transfer['mean'], ax=ax2, fill=True, alpha=0.4, label='Transfer', color='red')
    ax2.set_xlabel('Mean Performance')
    ax2.set_ylabel('Density')
    ax2.set_title('Mean Performance KDE')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot([metrics_std['mean'], metrics_transfer['mean']], 
                      labels=['Standard', 'Transfer'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Mean Performance')
    ax3.set_title('Mean Performance Box Plot')
    ax3.grid(alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(metrics_std['mean'], metrics_std['std'], alpha=0.5, s=30, c='blue', label='Standard')
    ax4.scatter(metrics_transfer['mean'], metrics_transfer['std'], alpha=0.5, s=30, c='red', label='Transfer')
    ax4.scatter(metrics_std['mean'][incumbent_std_idx], metrics_std['std'][incumbent_std_idx], 
                s=200, c='blue', marker='*', edgecolors='black', linewidths=2, label='Std Inc.')
    ax4.scatter(metrics_transfer['mean'][incumbent_transfer_idx], metrics_transfer['std'][incumbent_transfer_idx], 
                s=200, c='red', marker='*', edgecolors='black', linewidths=2, label='Trans Inc.')
    ax4.set_xlabel('Mean Performance')
    ax4.set_ylabel('Std Dev')
    ax4.set_title('Mean vs Std Scatter')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    
    # Row 2: Mean-Variance metric distribution
    ax5 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(
        min(metrics_std['mean_variance'].min(), metrics_transfer['mean_variance'].min()),
        max(metrics_std['mean_variance'].max(), metrics_transfer['mean_variance'].max()),
        30
    )
    ax5.hist(metrics_std['mean_variance'], bins=bins, alpha=0.6, label='Standard', color='blue', edgecolor='black')
    ax5.hist(metrics_transfer['mean_variance'], bins=bins, alpha=0.6, label='Transfer', color='red', edgecolor='black')
    ax5.axvline(metrics_std['mean_variance'][incumbent_std_idx], color='blue', linestyle='--', linewidth=2, label='Std Incumbent')
    ax5.axvline(metrics_transfer['mean_variance'][incumbent_transfer_idx], color='red', linestyle='--', linewidth=2, label='Trans Incumbent')
    ax5.set_xlabel(f'Mean - {alpha}×Variance')
    ax5.set_ylabel('# Configurations')
    ax5.set_title(f'Mean-Variance Metric (α={alpha})')
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(metrics_std['mean_variance'], ax=ax6, fill=True, alpha=0.4, label='Standard', color='blue')
    sns.kdeplot(metrics_transfer['mean_variance'], ax=ax6, fill=True, alpha=0.4, label='Transfer', color='red')
    ax6.set_xlabel(f'Mean - {alpha}×Variance')
    ax6.set_ylabel('Density')
    ax6.set_title('Mean-Variance KDE')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    ax7 = fig.add_subplot(gs[1, 2])
    bp = ax7.boxplot([metrics_std['mean_variance'], metrics_transfer['mean_variance']], 
                      labels=['Standard', 'Transfer'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax7.set_ylabel(f'Mean - {alpha}×Variance')
    ax7.set_title('Mean-Variance Box Plot')
    ax7.grid(alpha=0.3, axis='y')
    
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(metrics_std['mean'], metrics_std['variance'], alpha=0.5, s=30, c='blue', label='Standard')
    ax8.scatter(metrics_transfer['mean'], metrics_transfer['variance'], alpha=0.5, s=30, c='red', label='Transfer')
    ax8.scatter(metrics_std['mean'][incumbent_std_idx], metrics_std['variance'][incumbent_std_idx], 
                s=200, c='blue', marker='*', edgecolors='black', linewidths=2, label='Std Inc.')
    ax8.scatter(metrics_transfer['mean'][incumbent_transfer_idx], metrics_transfer['variance'][incumbent_transfer_idx], 
                s=200, c='red', marker='*', edgecolors='black', linewidths=2, label='Trans Inc.')
    ax8.set_xlabel('Mean Performance')
    ax8.set_ylabel('Variance')
    ax8.set_title('Mean vs Variance Scatter')
    ax8.legend(fontsize=8)
    ax8.grid(alpha=0.3)
    
    # Row 3: Incumbent analysis and rankings
    ax9 = fig.add_subplot(gs[2, :2])
    # Rank correlation plot
    ranks_std = stats.rankdata(-metrics_std['mean_variance'])  # negative for descending
    ranks_transfer = stats.rankdata(-metrics_transfer['mean_variance'])
    
    ax9.scatter(ranks_std, ranks_transfer, alpha=0.3, s=20, c='gray')
    ax9.plot([1, max(len(ranks_std), len(ranks_transfer))], 
             [1, max(len(ranks_std), len(ranks_transfer))], 
             'k--', alpha=0.5, label='Perfect correlation')
    
    # Highlight incumbents
    ax9.scatter(ranks_std[incumbent_std_idx], ranks_transfer[incumbent_std_idx], 
                s=200, c='blue', marker='*', edgecolors='black', linewidths=2, 
                label=f'Std Incumbent (Std rank={int(ranks_std[incumbent_std_idx])})')
    ax9.scatter(ranks_std[incumbent_transfer_idx], ranks_transfer[incumbent_transfer_idx], 
                s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                label=f'Trans Incumbent (Trans rank={int(ranks_transfer[incumbent_transfer_idx])})')
    
    spearman_corr, spearman_p = stats.spearmanr(ranks_std, ranks_transfer)
    ax9.set_xlabel('Rank in Standard (1=best)')
    ax9.set_ylabel('Rank in Transfer (1=best)')
    ax9.set_title(f'Configuration Ranking Correlation (Spearman ρ={spearman_corr:.3f}, p={spearman_p:.4f})')
    ax9.legend(fontsize=8)
    ax9.grid(alpha=0.3)
    
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    # Summary statistics
    summary = "CONFIGURATION METRIC SUMMARY\n" + "="*50 + "\n\n"
    summary += "Mean Performance:\n"
    summary += f"  Standard:  μ={np.mean(metrics_std['mean']):.3f}, σ={np.std(metrics_std['mean']):.3f}\n"
    summary += f"  Transfer:  μ={np.mean(metrics_transfer['mean']):.3f}, σ={np.std(metrics_transfer['mean']):.3f}\n"
    summary += f"  Δ:         {np.mean(metrics_transfer['mean']) - np.mean(metrics_std['mean']):+.3f}\n\n"
    
    summary += f"Mean-Variance (α={alpha}):\n"
    summary += f"  Standard:  μ={np.mean(metrics_std['mean_variance']):.3f}, σ={np.std(metrics_std['mean_variance']):.3f}\n"
    summary += f"  Transfer:  μ={np.mean(metrics_transfer['mean_variance']):.3f}, σ={np.std(metrics_transfer['mean_variance']):.3f}\n"
    summary += f"  Δ:         {np.mean(metrics_transfer['mean_variance']) - np.mean(metrics_std['mean_variance']):+.3f}\n\n"
    
    summary += "Incumbents (by Mean-Variance):\n"
    summary += f"  Standard:  Config #{incumbent_std_idx}\n"
    summary += f"    Mean={metrics_std['mean'][incumbent_std_idx]:.3f}, Var={metrics_std['variance'][incumbent_std_idx]:.4f}\n"
    summary += f"    Mean-Var={metrics_std['mean_variance'][incumbent_std_idx]:.3f}\n"
    summary += f"  Transfer:  Config #{incumbent_transfer_idx}\n"
    summary += f"    Mean={metrics_transfer['mean'][incumbent_transfer_idx]:.3f}, Var={metrics_transfer['variance'][incumbent_transfer_idx]:.4f}\n"
    summary += f"    Mean-Var={metrics_transfer['mean_variance'][incumbent_transfer_idx]:.3f}\n\n"
    
    summary += "Ranking Correlation:\n"
    summary += f"  Spearman ρ:  {spearman_corr:.3f} (p={spearman_p:.4f})\n"
    summary += f"  Interpretation: {'Strong' if abs(spearman_corr) > 0.7 else 'Moderate' if abs(spearman_corr) > 0.4 else 'Weak'} correlation\n"
    
    ax10.text(0.05, 0.95, summary, transform=ax10.transAxes, 
              verticalalignment='top', fontfamily='monospace', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'{algorithm}: Configuration Metric Distribution Shift Analysis', 
                 fontsize=16, fontweight='bold')
    
    if save_dir:
        save_path = Path(save_dir) / f'{algorithm}_config_metrics_shift.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {algorithm} config metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'metrics_std': metrics_std,
        'metrics_transfer': metrics_transfer,
        'incumbent_std_idx': incumbent_std_idx,
        'incumbent_transfer_idx': incumbent_transfer_idx,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p
    }


def plot_algorithm_comparison(algorithm, df_std, df_transfer, save_dir=None):
    """
    Create detailed comparison plots for a single algorithm.
    
    Args:
        algorithm: Name of the algorithm
        df_std: DataFrame for standard training
        df_transfer: DataFrame for transfer learning
        save_dir: Directory to save the figure
    """
    # Extract all performances
    perf_std = np.concatenate(df_std['performance_array'].values)
    perf_transfer = np.concatenate(df_transfer['performance_array'].values)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overlapping histograms
    ax = axes[0, 0]
    bins = np.linspace(
        min(perf_std.min(), perf_transfer.min()),
        max(perf_std.max(), perf_transfer.max()),
        50
    )
    ax.hist(perf_std, bins=bins, alpha=0.6, label='Standard', color='blue', edgecolor='black')
    ax.hist(perf_transfer, bins=bins, alpha=0.6, label='Transfer', color='red', edgecolor='black')
    ax.set_xlabel('Performance')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{algorithm}: Performance Distribution Comparison')
    ax.axvline(np.mean(perf_std), color='blue', linestyle='--', linewidth=2, 
               label=f'Standard Mean: {np.mean(perf_std):.2f}')
    ax.axvline(np.mean(perf_transfer), color='red', linestyle='--', linewidth=2,
               label=f'Transfer Mean: {np.mean(perf_transfer):.2f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. KDE (Kernel Density Estimation) comparison
    ax = axes[0, 1]
    sns.kdeplot(perf_std, ax=ax, fill=True, alpha=0.4, label='Standard', color='blue')
    sns.kdeplot(perf_transfer, ax=ax, fill=True, alpha=0.4, label='Transfer', color='red')
    ax.set_xlabel('Performance')
    ax.set_ylabel('Density')
    ax.set_title(f'{algorithm}: Kernel Density Estimate')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Box plots comparison
    ax = axes[1, 0]
    data_to_plot = [perf_std, perf_transfer]
    bp = ax.boxplot(data_to_plot, labels=['Standard', 'Transfer'], 
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Performance')
    ax.set_title(f'{algorithm}: Box Plot Comparison')
    ax.grid(alpha=0.3, axis='y')
    
    # Add statistical annotation
    stats_comparison = compare_distributions(perf_std, perf_transfer)
    text_str = f"Mean Δ: {stats_comparison['mean_diff']:.3f}\n"
    text_str += f"Cohen's d: {stats_comparison['cohens_d']:.3f}\n"
    text_str += f"t-test p: {stats_comparison['t_pvalue']:.4f}"
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Violin plots comparison
    ax = axes[1, 1]
    positions = [1, 2]
    parts = ax.violinplot([perf_std, perf_transfer], positions=positions, 
                           showmeans=True, showmedians=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Standard', 'Transfer'])
    ax.set_ylabel('Performance')
    ax.set_title(f'{algorithm}: Violin Plot Comparison')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = Path(save_dir) / f'{algorithm}_distribution_shift.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {algorithm} comparison to {save_path}")
    # else:
    #     plt.show()
    
    plt.close()
    
    return stats_comparison


def plot_multi_algorithm_comparison(results_dict, save_dir=None):
    """
    Create a comprehensive comparison across all algorithms.
    
    Args:
        results_dict: Dictionary with algorithm names as keys and 
                     (df_std, df_transfer) tuples as values
        save_dir: Directory to save the figure
    """
    algorithms = list(results_dict.keys())
    n_algos = len(algorithms)
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Prepare data
    all_stats_std = []
    all_stats_transfer = []
    all_comparisons = []
    
    for algo in algorithms:
        df_std, df_transfer = results_dict[algo]
        perf_std = np.concatenate(df_std['performance_array'].values)
        perf_transfer = np.concatenate(df_transfer['performance_array'].values)
        
        all_stats_std.append(compute_distribution_stats(perf_std))
        all_stats_transfer.append(compute_distribution_stats(perf_transfer))
        all_comparisons.append(compare_distributions(perf_std, perf_transfer))
    
    # 1. Mean comparison
    ax = axes[0]
    means_std = [s['mean'] for s in all_stats_std]
    means_transfer = [s['mean'] for s in all_stats_transfer]
    x = np.arange(n_algos)
    width = 0.35
    ax.bar(x - width/2, means_std, width, label='Standard', alpha=0.8, color='blue')
    ax.bar(x + width/2, means_transfer, width, label='Transfer', alpha=0.8, color='red')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Performance')
    ax.set_title('Mean Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 2. Std comparison
    ax = axes[1]
    stds_std = [s['std'] for s in all_stats_std]
    stds_transfer = [s['std'] for s in all_stats_transfer]
    ax.bar(x - width/2, stds_std, width, label='Standard', alpha=0.8, color='blue')
    ax.bar(x + width/2, stds_transfer, width, label='Transfer', alpha=0.8, color='red')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variability Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Mean difference (Transfer - Standard)
    ax = axes[2]
    mean_diffs = [c['mean_diff'] for c in all_comparisons]
    colors = ['green' if d > 0 else 'orange' for d in mean_diffs]
    ax.bar(x, mean_diffs, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Mean Difference (Transfer - Standard)')
    ax.set_title('Transfer Learning Impact on Mean')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Cohen's d effect size
    ax = axes[3]
    cohens_d = [c['cohens_d'] for c in all_comparisons]
    colors = ['green' if d > 0 else 'orange' for d in cohens_d]
    ax.bar(x, cohens_d, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # small effect
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # medium effect
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # large effect
    ax.set_xlabel('Algorithm')
    ax.set_ylabel("Cohen's d")
    ax.set_title('Effect Size of Transfer Learning')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.grid(alpha=0.3, axis='y')
    
    # 5. Statistical significance (p-values)
    ax = axes[4]
    p_values = [c['t_pvalue'] for c in all_comparisons]
    colors = ['green' if p < 0.05 else 'orange' for p in p_values]
    ax.bar(x, [-np.log10(p) for p in p_values], color=colors, alpha=0.8)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', linewidth=1, label='p=0.01')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title('Statistical Significance (t-test)')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # 6. Distribution comparison summary
    ax = axes[5]
    ax.axis('off')
    summary_text = "Distribution Shift Summary\n" + "="*40 + "\n\n"
    for i, algo in enumerate(algorithms):
        summary_text += f"{algo}:\n"
        summary_text += f"  Δ Mean: {all_comparisons[i]['mean_diff']:+.3f}\n"
        summary_text += f"  Cohen's d: {all_comparisons[i]['cohens_d']:+.3f}\n"
        summary_text += f"  p-value: {all_comparisons[i]['t_pvalue']:.4f}\n"
        sig = "✓" if all_comparisons[i]['t_pvalue'] < 0.05 else "✗"
        summary_text += f"  Significant: {sig}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Multi-Algorithm Distribution Shift Analysis\nCartPole-v1: Standard vs Transfer Learning', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'multi_algorithm_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-algorithm comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_detailed_statistics(algorithm, df_std, df_transfer):
    """
    Print detailed statistical comparison for an algorithm.
    
    Args:
        algorithm: Name of the algorithm
        df_std: DataFrame for standard training
        df_transfer: DataFrame for transfer learning
    """
    perf_std = np.concatenate(df_std['performance_array'].values)
    perf_transfer = np.concatenate(df_transfer['performance_array'].values)
    
    stats_std = compute_distribution_stats(perf_std)
    stats_transfer = compute_distribution_stats(perf_transfer)
    comparison = compare_distributions(perf_std, perf_transfer)
    
    print(f"\n{'='*70}")
    print(f"  {algorithm.upper()} - DETAILED STATISTICS")
    print(f"{'='*70}")
    
    print(f"\nSTANDARD TRAINING:")
    print(f"  Mean:             {stats_std['mean']:.4f}")
    print(f"  Median:           {stats_std['median']:.4f}")
    print(f"  Std Dev:          {stats_std['std']:.4f}")
    print(f"  Min:              {stats_std['min']:.4f}")
    print(f"  Max:              {stats_std['max']:.4f}")
    print(f"  IQR:              {stats_std['iqr']:.4f}")
    print(f"  Total samples:    {stats_std['count']}")
    
    print(f"\nTRANSFER LEARNING:")
    print(f"  Mean:             {stats_transfer['mean']:.4f}")
    print(f"  Median:           {stats_transfer['median']:.4f}")
    print(f"  Std Dev:          {stats_transfer['std']:.4f}")
    print(f"  Min:              {stats_transfer['min']:.4f}")
    print(f"  Max:              {stats_transfer['max']:.4f}")
    print(f"  IQR:              {stats_transfer['iqr']:.4f}")
    print(f"  Total samples:    {stats_transfer['count']}")
    
    print(f"\nDISTRIBUTION SHIFT ANALYSIS:")
    print(f"  Mean difference:      {comparison['mean_diff']:+.4f}")
    print(f"  Median difference:    {comparison['median_diff']:+.4f}")
    print(f"  Std ratio (T/S):      {comparison['std_ratio']:.4f}")
    print(f"  Cohen's d:            {comparison['cohens_d']:+.4f}")
    
    # Interpret Cohen's d
    d_abs = abs(comparison['cohens_d'])
    if d_abs < 0.2:
        effect = "negligible"
    elif d_abs < 0.5:
        effect = "small"
    elif d_abs < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size:          {effect}")
    
    print(f"\nSTATISTICAL TESTS:")
    print(f"  t-test statistic:     {comparison['t_statistic']:.4f}")
    print(f"  t-test p-value:       {comparison['t_pvalue']:.4f}")
    sig = "YES" if comparison['t_pvalue'] < 0.05 else "NO"
    print(f"  Significant (α=0.05): {sig}")
    
    print(f"  KS statistic:         {comparison['ks_statistic']:.4f}")
    print(f"  KS p-value:           {comparison['ks_pvalue']:.4f}")
    
    print(f"  Mann-Whitney U:       {comparison['mannwhitney_statistic']:.4f}")
    print(f"  Mann-Whitney p-value: {comparison['mannwhitney_pvalue']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance distribution shifts for CartPole-v1 across algorithms"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing algorithm results (e.g., 'results/sobol')"
    )
    parser.add_argument(
        "--algorithms",
        nargs='+',
        default=['dqn', 'ppo', 'sac'],
        help="List of algorithms to compare (default: dqn ppo sac)"
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name (default: CartPole-v1)"
    )
    parser.add_argument(
        "--standard_suffix",
        type=str,
        default="",
        help="Suffix for standard training directories (default: empty)"
    )
    parser.add_argument(
        "--transfer_suffix",
        type=str,
        default="_transfer",
        help="Suffix for transfer learning directories (default: _transfer)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Number of seeds to load from each performance list (default: all)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save figures (default: base_dir/shift_analysis)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for variance penalty in mean-variance metric (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    # Setup save directory
    if args.save_dir is None:
        save_dir = base_dir / "shift_analysis"
    else:
        save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  DISTRIBUTION SHIFT ANALYSIS")
    print(f"{'='*70}")
    print(f"Environment:  {args.env_name}")
    print(f"Algorithms:   {', '.join(args.algorithms)}")
    print(f"Base dir:     {base_dir}")
    print(f"Save dir:     {save_dir}")
    print(f"Alpha (variance penalty): {args.alpha}")
    print(f"{'='*70}\n")
    
    # Load data for all algorithms
    results_dict = {}
    
    for algo in args.algorithms:
        # Construct paths
        std_dir = base_dir / f"{algo}_{args.env_name}{args.standard_suffix}"
        transfer_dir = base_dir / f"{algo}_{args.env_name}{args.transfer_suffix}"
        
        # Search for runhistory.csv in subdirectories (may be in seed-specific subdirs)
        def find_runhistory(directory):
            """Find runhistory.csv in directory or its subdirectories."""
            if not directory.exists():
                return None
            
            # Check direct path first
            direct_path = directory / "runhistory.csv"
            if direct_path.exists():
                return direct_path
            
            # Search in subdirectories (e.g., [42, 43, ...] or specific seed)
            for subdir in directory.iterdir():
                if subdir.is_dir():
                    csv_path = subdir / "runhistory.csv"
                    if csv_path.exists():
                        return csv_path
            
            return None
        
        std_csv = find_runhistory(std_dir)
        transfer_csv = find_runhistory(transfer_dir)
        
        # Check if files exist
        if std_csv is None:
            print(f"WARNING: Standard CSV not found for {algo} in directory: {std_dir}")
            print(f"         Searched in: {std_dir} and its subdirectories")
            continue
        if transfer_csv is None:
            print(f"WARNING: Transfer CSV not found for {algo} in directory: {transfer_dir}")
            print(f"         Searched in: {transfer_dir} and its subdirectories")
            continue
        
        print(f"Loading {algo.upper()}...")
        print(f"  Standard:  {std_csv}")
        print(f"  Transfer:  {transfer_csv}")
        
        # Load data
        df_std = load_runhistory(std_csv, n_seeds=args.n_seeds)
        df_transfer = load_runhistory(transfer_csv, n_seeds=args.n_seeds)
        
        results_dict[algo.upper()] = (df_std, df_transfer)
        
        # Print detailed statistics
        print_detailed_statistics(algo.upper(), df_std, df_transfer)
        
        # Create individual algorithm comparison plot
        plot_algorithm_comparison(algo.upper(), df_std, df_transfer, save_dir=save_dir)
        
        # Create configuration-level metric plots
        print(f"\nGenerating configuration metric plots for {algo.upper()}...")
        plot_config_metric_distributions(algo.upper(), df_std, df_transfer, 
                                         alpha=args.alpha, save_dir=save_dir)
    
    # Create multi-algorithm comparison if we have multiple algorithms
    if len(results_dict) > 1:
        print(f"\n{'='*70}")
        print("Creating multi-algorithm comparison plot...")
        plot_multi_algorithm_comparison(results_dict, save_dir=save_dir)
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {save_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
