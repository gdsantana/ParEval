#!/usr/bin/env python3
"""Generate plots from metrics CSV file."""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import re

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
    
    # Validate input
    if df.empty:
        print("Error: DataFrame is empty. Cannot generate plots.")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['model', 'problem type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    # Detect k values dynamically from column names
    k_values = []
    for col in df.columns:
        match = re.match(r'(build|pass|speedup|efficiency)@(\d+)', col)
        if match:
            k = int(match.group(2))
            if k not in k_values:
                k_values.append(k)
    k_values.sort()
    
    if not k_values:
        print("Error: No metrics with @k format found in CSV")
        sys.exit(1)
    
    # Validate that all required metric columns exist
    for k in k_values:
        for metric in ['build', 'pass', 'speedup', 'speedup_max', 'efficiency', 'efficiency_max']:
            col_name = f'{metric}@{k}'
            if col_name not in df.columns:
                print(f"Warning: Column {col_name} not found in CSV")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    model_name = df['model'].iloc[0] if 'model' in df.columns else 'Unknown Model'
    fig.suptitle(f'Performance Metrics for {model_name}', 
                 fontsize=16, fontweight='bold')
    
    problem_types = df['problem type'].values
    x_pos = np.arange(len(problem_types))
    
    # Color palette
    colors = sns.color_palette("husl", len(problem_types))
    
    # 1. Build@k metrics
    ax = axes[0, 0]
    width = 0.8 / len(k_values) if k_values else 0.2
    for i, k in enumerate(k_values):
        if f'build@{k}' in df.columns:
            ax.bar(x_pos + i*width, df[f'build@{k}'], width, 
                   label=f'build@{k}', alpha=0.8)
    ax.set_ylabel('Build Rate')
    ax.set_title('Build Success Rate @k')
    ax.set_xticks(x_pos + width * (len(k_values) - 1) / 2)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Pass@k metrics
    ax = axes[0, 1]
    for i, k in enumerate(k_values):
        if f'pass@{k}' in df.columns:
            ax.bar(x_pos + i*width, df[f'pass@{k}'], width, 
                   label=f'pass@{k}', alpha=0.8)
    ax.set_ylabel('Pass Rate')
    ax.set_title('Correctness Rate @k')
    ax.set_xticks(x_pos + width * (len(k_values) - 1) / 2)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Speedup@1 vs Speedup_max@1
    ax = axes[0, 2]
    x = np.arange(len(problem_types))
    k_first = k_values[0] if k_values else 1
    
    # Add small epsilon to avoid log(0)
    speedup_data = df[f'speedup@{k_first}'].replace(0, np.nan) if f'speedup@{k_first}' in df.columns else pd.Series([np.nan] * len(problem_types))
    speedup_max_data = df[f'speedup_max@{k_first}'].replace(0, np.nan) if f'speedup_max@{k_first}' in df.columns else pd.Series([np.nan] * len(problem_types))
    
    ax.bar(x - 0.2, speedup_data, 0.4, label=f'speedup@{k_first}', alpha=0.8)
    ax.bar(x + 0.2, speedup_max_data, 0.4, label=f'speedup_max@{k_first}', alpha=0.8)
    ax.set_ylabel('Speedup')
    ax.set_title(f'Speedup @{k_first} (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.001)  # Set minimum to avoid log scale issues
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Speedup progression (k=1,5,10,20) for speedup@k
    ax = axes[1, 0]
    for i, k in enumerate(k_values):
        if f'speedup@{k}' in df.columns:
            # Replace 0 with small value for log scale
            data = df[f'speedup@{k}'].replace(0, np.nan)
            ax.bar(x_pos + i*width, data, width, 
                   label=f'speedup@{k}', alpha=0.8)
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup @k (log scale)')
    ax.set_xticks(x_pos + width * (len(k_values) - 1) / 2)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.001)
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Efficiency@1 vs Efficiency_max@1
    ax = axes[1, 1]
    k_first = k_values[0] if k_values else 1
    
    # Replace 0 with NaN for log scale
    eff_data = df[f'efficiency@{k_first}'].replace(0, np.nan) if f'efficiency@{k_first}' in df.columns else pd.Series([np.nan] * len(problem_types))
    eff_max_data = df[f'efficiency_max@{k_first}'].replace(0, np.nan) if f'efficiency_max@{k_first}' in df.columns else pd.Series([np.nan] * len(problem_types))
    
    ax.bar(x - 0.2, eff_data, 0.4, label=f'efficiency@{k_first}', alpha=0.8)
    ax.bar(x + 0.2, eff_max_data, 0.4, label=f'efficiency_max@{k_first}', alpha=0.8)
    ax.set_ylabel('Efficiency')
    ax.set_title(f'Efficiency @{k_first} (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-10)
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Efficiency progression
    ax = axes[1, 2]
    for i, k in enumerate(k_values):
        if f'efficiency@{k}' in df.columns:
            # Replace 0 with NaN for log scale
            data = df[f'efficiency@{k}'].replace(0, np.nan)
            ax.bar(x_pos + i*width, data, width, 
                   label=f'efficiency@{k}', alpha=0.8)
    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency @k (log scale)')
    ax.set_xticks(x_pos + width * (len(k_values) - 1) / 2)
    ax.set_xticklabels(problem_types, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-10)
    ax.grid(axis='y', alpha=0.3)
    
    # 7. Build vs Pass @10 (or closest k value)
    ax = axes[2, 0]
    k_mid = k_values[len(k_values)//2] if k_values else 10
    
    if f'build@{k_mid}' in df.columns and f'pass@{k_mid}' in df.columns:
        ax.scatter(df[f'build@{k_mid}'], df[f'pass@{k_mid}'], s=100, alpha=0.6, c=colors)
        
        # Smart annotation positioning to avoid overlap
        for i, txt in enumerate(problem_types):
            x_val = df[f'build@{k_mid}'].iloc[i]
            y_val = df[f'pass@{k_mid}'].iloc[i]
            
            # Offset annotations slightly to reduce overlap using data coordinates
            offset_x = 0.02 if i % 2 == 0 else -0.02
            offset_y = 0.02 if i % 3 == 0 else -0.02
            
            ax.annotate(txt, (x_val, y_val), 
                       xytext=(x_val + offset_x, y_val + offset_y),
                       fontsize=7, ha='left' if i % 2 == 0 else 'right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.set_xlabel(f'Build@{k_mid}')
        ax.set_ylabel(f'Pass@{k_mid}')
        ax.set_title(f'Build vs Pass Rate @{k_mid}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
    ax.grid(alpha=0.3)
    
    # 8. Pass@k comparison across k values
    ax = axes[2, 1]
    for i, ptype in enumerate(problem_types):
        pass_values = []
        valid_k = []
        for k in k_values:
            if f'pass@{k}' in df.columns:
                pass_values.append(df[f'pass@{k}'].iloc[i])
                valid_k.append(k)
        if pass_values:
            ax.plot(valid_k, pass_values, marker='o', label=ptype, alpha=0.7)
    ax.set_xlabel('k')
    ax.set_ylabel('Pass@k')
    ax.set_title('Pass@k Progression')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(alpha=0.3)
    if k_values:
        ax.set_xticks(k_values)
    
    # 9. Summary heatmap
    ax = axes[2, 2]
    k_mid = k_values[len(k_values)//2] if k_values else 10
    
    metrics_to_show = [f'build@{k_mid}', f'pass@{k_mid}', f'speedup@{k_mid}', f'efficiency@{k_mid}']
    # Filter to only include metrics that exist
    metrics_to_show = [m for m in metrics_to_show if m in df.columns]
    
    if metrics_to_show:
        heatmap_data = df[metrics_to_show].T
        heatmap_data.columns = problem_types
        
        # Normalize each row to 0-1 for better visualization
        # Handle division by zero by replacing with 0
        max_values = heatmap_data.max(axis=1)
        max_values = max_values.replace(0, 1)  # Avoid division by zero
        heatmap_normalized = heatmap_data.div(max_values, axis=0)
        
        # Replace NaN with 0
        heatmap_normalized = heatmap_normalized.fillna(0)
        
        im = ax.imshow(heatmap_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(problem_types)))
        ax.set_yticks(np.arange(len(metrics_to_show)))
        ax.set_xticklabels(problem_types, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(metrics_to_show, fontsize=8)
        ax.set_title(f'Normalized Metrics @{k_mid} Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(metrics_to_show)):
            for j in range(len(problem_types)):
                value = heatmap_normalized.iloc[i, j]
                text_color = "white" if value > 0.5 else "black"
                text = ax.text(j, i, f'{value:.2f}',
                              ha="center", va="center", color=text_color, fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also create a summary statistics table
    print("\n=== Summary Statistics ===")
    print(f"\nModel: {model_name}")
    print(f"\nAverage metrics across all problem types:")
    
    k_mid = k_values[len(k_values)//2] if k_values else 10
    summary_metrics = {}
    
    for metric in [f'build@{k_mid}', f'pass@{k_mid}', f'speedup@{k_mid}', f'efficiency@{k_mid}']:
        if metric in df.columns:
            summary_metrics[metric] = df[metric].mean()
    
    for metric, value in summary_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nBest performing problem types:")
    if f'build@{k_mid}' in df.columns and not df[f'build@{k_mid}'].isna().all():
        best_build_idx = df[f'build@{k_mid}'].idxmax()
        print(f"  Highest build@{k_mid}: {df.loc[best_build_idx, 'problem type']} ({df[f'build@{k_mid}'].max():.4f})")
    
    if f'pass@{k_mid}' in df.columns and not df[f'pass@{k_mid}'].isna().all():
        best_pass_idx = df[f'pass@{k_mid}'].idxmax()
        print(f"  Highest pass@{k_mid}: {df.loc[best_pass_idx, 'problem type']} ({df[f'pass@{k_mid}'].max():.4f})")
    
    if f'speedup@{k_mid}' in df.columns and not df[f'speedup@{k_mid}'].isna().all():
        best_speedup_idx = df[f'speedup@{k_mid}'].idxmax()
        print(f"  Highest speedup@{k_mid}: {df.loc[best_speedup_idx, 'problem type']} ({df[f'speedup@{k_mid}'].max():.4f})")


def main():
    args = get_args()
    
    # Read CSV
    df = pd.read_csv(args.input_csv)
    
    # Generate plots
    plot_metrics(df, args.output)


if __name__ == "__main__":
    main()
