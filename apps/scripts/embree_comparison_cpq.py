import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List, Dict, Optional
import os


def compute_average_performance(df: pd.DataFrame, layout: str) -> pd.DataFrame:
    """
    Compute average CPQ time for each machine/scene combination for a given layout.
    Returns slowdown compared to Embree.
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

        # Compute slowdown (bonsai / embree)
        if avg_embree_cpq > 0:
            slowdown = avg_bonsai_cpq / avg_embree_cpq
        else:
            slowdown = np.nan

        results.append({
            'machine': machine,
            'scene': scene,
            'label': f"{scene}\n{machine.upper()}",
            'avg_bonsai_cpq_ms': avg_bonsai_cpq,
            'avg_embree_cpq_ms': avg_embree_cpq,
            'slowdown': slowdown
        })

    return pd.DataFrame(results)


def create_bar_chart(df: pd.DataFrame, layout: str, machines: List[str],
                     scenes: List[str], output_file: str):
    """
    Create a single bar chart showing slowdown for selected machine/scene combinations.
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

    # Sort by slowdown
    summary_df = summary_df.sort_values('slowdown')

    # Create figure - much larger for 0.25x scaling
    fig, ax = plt.subplots(1, 1, figsize=(max(24, len(summary_df) * 2.4), 16))

    # Create bar positions
    x_pos = np.arange(len(summary_df))

    # Create bars with thicker edges
    bars = ax.bar(x_pos, summary_df['slowdown'],
                  color='#0173B2', alpha=0.7, edgecolor='black', linewidth=4)

    # Add horizontal line at y=1 (Embree baseline) - much thicker
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=6,
               alpha=0.7, label='Embree Baseline')

    # Color bars: green if faster than Embree, red if slower
    for i, (bar, slowdown) in enumerate(zip(bars, summary_df['slowdown'])):
        if slowdown < 1.0:
            bar.set_color('#2ECC71')  # Green for faster
        else:
            bar.set_color('#E74C3C')  # Red for slower

    # Add value labels on bars - much larger
    for i, (bar, slowdown) in enumerate(zip(bars, summary_df['slowdown'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{slowdown:.2f}x',
                ha='center', va='bottom' if height >= 1.0 else 'top',
                fontsize=40, fontweight='bold')

    # Formatting - all fonts much larger
    ax.set_title(f"CPQ Performance vs Embree: {layout}",
                 fontsize=52, fontweight='bold', pad=40)
    ax.set_ylabel('Slowdown vs Embree', fontsize=48, fontweight='bold')
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

    # Set y-axis to start at 1.0
    y_max = max(summary_df['slowdown'].max() * 1.15, 1.2)
    ax.set_ylim(1.0, y_max)

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
    parser.add_argument('--output', default='cpq_slowdown',
                        help='Output file name (default: cpq_slowdown)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    print(f"Loaded {len(df)} rows")
    print(f"Analyzing layout: {args.layout}")
    print(f"Machines: {args.machines}")
    print(f"Scenes: {args.scenes}")

    # Generate chart
    create_bar_chart(df, args.layout, args.machines, args.scenes, args.output)

    print(f"\nDone! Output saved to {args.output}")


if __name__ == "__main__":
    main()
