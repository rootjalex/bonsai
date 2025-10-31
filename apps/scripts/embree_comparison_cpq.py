import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List, Dict, Optional
import os


def compute_average_performance(df: pd.DataFrame, layout: str) -> pd.DataFrame:
    """
    Compute average CPQ time for each machine/scene combination for a given layout.
    Returns speedup compared to Embree (inverse of slowdown).
    """
    results = []

    # Get unique machine/scene combinations
    for (machine, scene), group_df in df.groupby(['machine', 'scene']):
        layout_df = group_df[group_df['layout'] == layout]

        if len(layout_df) == 0:
            continue

        # Average CPQ times
        avg_bonsai_cpq = layout_df['cpq-time-ms'].mean()
        avg_embree_cpq = layout_df['embree-cpq-time-ms'].mean()

        # Compute speedup (embree / bonsai) - inverse of slowdown
        if avg_bonsai_cpq > 0:
            speedup = avg_embree_cpq / avg_bonsai_cpq
        else:
            speedup = np.nan

        results.append({
            'machine': machine,
            'scene': scene,
            'label': f"{scene}\n{machine}",
            'avg_bonsai_cpq_ms': avg_bonsai_cpq,
            'avg_embree_cpq_ms': avg_embree_cpq,
            'speedup': speedup
        })

    return pd.DataFrame(results)


def create_bar_chart(df: pd.DataFrame, layout: str, machines: List[str],
                     scenes: List[str], output_file: str, show_title: bool = False):
    """
    Create a single bar chart showing speedup for selected machine/scene combinations.
    """

    # Filter data by machines and scenes
    filtered_df = df[
        (df['machine'].isin(machines)) &
        (df['scene'].isin(scenes))
    ]

    if len(filtered_df) == 0:
        print(
            f"No data found for layout '{layout}' with selected machines and scenes")
        return

    # Compute averages for this layout
    summary_df = compute_average_performance(filtered_df, layout)

    if len(summary_df) == 0:
        print(f"No data found for layout '{layout}'")
        return

    # Sort by speedup
    summary_df = summary_df.sort_values('speedup')

    # Create figure - much larger for 0.25x scaling
    fig, ax = plt.subplots(1, 1, figsize=(max(24, len(summary_df) * 2.4), 16))

    # Create bar positions
    x_pos = np.arange(len(summary_df))

    # Create bars with thicker edges
    bars = ax.bar(x_pos, summary_df['speedup'],
                  color='#0173B2', alpha=0.7, edgecolor='black', linewidth=4)

    # Add horizontal line at y=1 (Embree baseline) - much thicker
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=6,
               alpha=0.7, label='Embree Baseline')

    # Color bars: green if faster than Embree, red if slower
    for i, (bar, speedup) in enumerate(zip(bars, summary_df['speedup'])):
        if speedup > 1.0:
            bar.set_color('#2ECC71')  # Green for faster
        else:
            bar.set_color('#E74C3C')  # Red for slower

    # Add value labels on bars - much larger
    for i, (bar, speedup) in enumerate(zip(bars, summary_df['speedup'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom' if height >= 1.0 else 'top',
                fontsize=40, fontweight='bold')

    # Formatting - all fonts much larger
    if show_title:
        ax.set_title(f"Layout: {layout}",
                     fontsize=52, fontweight='bold', pad=40)
    ax.set_ylabel('Speedup vs Embree', fontsize=48, fontweight='bold')
    ax.set_xlabel('Scene / Machine', fontsize=48, fontweight='bold')

    # Set x-axis labels (scene + machine) - much larger
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df['label'], fontsize=40, fontweight='bold')

    ax.tick_params(axis='y', which='major', labelsize=40, width=3, length=10)
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=2)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # Set y-axis from 0.0 to 2.0
    ax.set_ylim(0.0, 2.0)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Saved: {output_file}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate bar chart comparing CPQ performance to Embree for a single layout')
    parser.add_argument(
        'input_csv', help='Input CSV file with CPQ benchmark data')
    parser.add_argument('--layout', required=True,
                        help='Layout to analyze')
    parser.add_argument('--machines', nargs='+', default=['x86', 'arm'],
                        help='Machines to include (default: x86 arm)')
    parser.add_argument('--scenes', nargs='+', required=True,
                        help='Scenes to include')
    parser.add_argument('--output', default='cpq_speedup',
                        help='Output file name (default: cpq_speedup)')
    parser.add_argument('--show-title', action='store_true',
                        help='Show title on the plot (default: False)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    print(f"Loaded {len(df)} rows")
    print(f"Analyzing layout: {args.layout}")
    print(f"Machines: {args.machines}")
    print(f"Scenes: {args.scenes}")

    # Generate chart
    create_bar_chart(df, args.layout, args.machines,
                     args.scenes, args.output, args.show_title)

    print(f"\nDone! Output saved to {args.output}")


if __name__ == "__main__":
    main()
