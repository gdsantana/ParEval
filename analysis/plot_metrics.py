#!/usr/bin/env python3
"""Generate plots from metrics CSV file."""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV file with metrics")
    parser.add_argument("-o", "--output", type=str, default="metrics_plots.png", 
                        help="Output image file")
    return parser.parse_args()


def plot_metrics(df, output_file):
    """Create a comprehensive plot with multiple subplots for different metrics."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'Performance Metrics for {df["model"].iloc[0]}', 
                 fontsize=16, fontweight='bold')
    
    problem_types = df['problem type'].values
    x_pos = np.arange(len(problem_types))
    
    # Color palette
    colors = sns.color_palette("husl", len(problem_types))
    
    # 1. Build@k metrics
    ax = axes[0, 0]
    width = 0.2
    for i, k in enumerate([1, 5, 10, 20]):
        ax.bar(x_pos + i*width, df[f'build@{k}'], width, 
               label=f'build@{k}', alpha=0.8)
    ax.set_ylabel('Build Rate')
    ax.set_title('Build Success Rate @k')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Pass@k metrics
    ax = axes[0, 1]
    for i, k in enumerate([1, 5, 10, 20]):
        ax.bar(x_pos + i*width, df[f'pass@{k}'], width, 
               label=f'pass@{k}', alpha=0.8)
    ax.set_ylabel('Pass Rate')
    ax.set_title('Correctness Rate @k')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Speedup@1 vs Speedup_max@1
    ax = axes[0, 2]
    x = np.arange(len(problem_types))
    ax.bar(x - 0.2, df['speedup@1'], 0.4, label='speedup@1', alpha=0.8)
    ax.bar(x + 0.2, df['speedup_max@1'], 0.4, label='speedup_max@1', alpha=0.8)
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup @1 (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Speedup progression (k=1,5,10,20) for speedup@k
    ax = axes[1, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        ax.bar(x_pos + i*width, df[f'speedup@{k}'], width, 
               label=f'speedup@{k}', alpha=0.8)
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup @k (log scale)')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Efficiency@1 vs Efficiency_max@1
    ax = axes[1, 1]
    ax.bar(x - 0.2, df['efficiency@1'], 0.4, label='efficiency@1', alpha=0.8)
    ax.bar(x + 0.2, df['efficiency_max@1'], 0.4, label='efficiency_max@1', alpha=0.8)
    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency @1 (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Efficiency progression
    ax = axes[1, 2]
    for i, k in enumerate([1, 5, 10, 20]):
        ax.bar(x_pos + i*width, df[f'efficiency@{k}'], width, 
               label=f'efficiency@{k}', alpha=0.8)
    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency @k (log scale)')
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # 7. Build vs Pass @10
    ax = axes[2, 0]
    ax.scatter(df['build@10'], df['pass@10'], s=100, alpha=0.6, c=colors)
    for i, txt in enumerate(problem_types):
        ax.annotate(txt, (df['build@10'].iloc[i], df['pass@10'].iloc[i]), 
                   fontsize=8, ha='right')
    ax.set_xlabel('Build@10')
    ax.set_ylabel('Pass@10')
    ax.set_title('Build vs Pass Rate @10')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.grid(alpha=0.3)
    
    # 8. Pass@k comparison across k values
    ax = axes[2, 1]
    k_values = [1, 5, 10, 20]
    for i, ptype in enumerate(problem_types):
        pass_values = [df[f'pass@{k}'].iloc[i] for k in k_values]
        ax.plot(k_values, pass_values, marker='o', label=ptype, alpha=0.7)
    ax.set_xlabel('k')
    ax.set_ylabel('Pass@k')
    ax.set_title('Pass@k Progression')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(k_values)
    
    # 9. Summary heatmap
    ax = axes[2, 2]
    metrics_to_show = ['build@10', 'pass@10', 'speedup@10', 'efficiency@10']
    heatmap_data = df[metrics_to_show].T
    heatmap_data.columns = problem_types
    
    # Normalize each row to 0-1 for better visualization
    heatmap_normalized = heatmap_data.div(heatmap_data.max(axis=1), axis=0)
    
    im = ax.imshow(heatmap_normalized, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(problem_types)))
    ax.set_yticks(np.arange(len(metrics_to_show)))
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.set_yticklabels(metrics_to_show)
    ax.set_title('Normalized Metrics @10 Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(metrics_to_show)):
        for j in range(len(problem_types)):
            text = ax.text(j, i, f'{heatmap_normalized.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also create a summary statistics table
    print("\n=== Summary Statistics ===")
    print(f"\nModel: {df['model'].iloc[0]}")
    print(f"\nAverage metrics across all problem types:")
    summary_metrics = {
        'build@10': df['build@10'].mean(),
        'pass@10': df['pass@10'].mean(),
        'speedup@10': df['speedup@10'].mean(),
        'efficiency@10': df['efficiency@10'].mean(),
    }
    for metric, value in summary_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nBest performing problem types:")
    print(f"  Highest build@10: {df.loc[df['build@10'].idxmax(), 'problem type']} ({df['build@10'].max():.4f})")
    print(f"  Highest pass@10: {df.loc[df['pass@10'].idxmax(), 'problem type']} ({df['pass@10'].max():.4f})")
    print(f"  Highest speedup@10: {df.loc[df['speedup@10'].idxmax(), 'problem type']} ({df['speedup@10'].max():.4f})")


def main():
    args = get_args()
    
    # Read CSV
    df = pd.read_csv(args.input_csv)
    
    # Generate plots
    plot_metrics(df, args.output)


if __name__ == "__main__":
    main()
