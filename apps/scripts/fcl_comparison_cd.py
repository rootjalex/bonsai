import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import List, Dict, Optional
import os

from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    # "text.latex.preamble": r"\usepackage{amsmath}",  # Optional for math
})


def compute_average_performance(df: pd.DataFrame, layouts: List[str], machines: List[str]) -> pd.DataFrame:
    """
    Compute average collision detection time for each layout/machine combination.
    Returns speedup compared to FCL.
    """
    results = []

    for machine in machines:
        for layout in layouts:
            layout_df = df[(df['layout'] == layout) &
                           (df['machine'] == machine)]

            if len(layout_df) == 0:
                continue

            # Average collision times
            avg_bonsai_cd = layout_df['collision-time-ms'].mean()
            avg_fcl_cd = layout_df['fcl-collision-time-ms'].mean()

            # Compute speedup (fcl / bonsai) - higher is better
            if avg_bonsai_cd > 0:
                speedup = avg_fcl_cd / avg_bonsai_cd
            else:
                speedup = np.nan

            results.append({
                'layout': layout,
                'machine': machine,
                'label': f"{layout}\n{machine}",
                'avg_bonsai_cd_ms': avg_bonsai_cd,
                'avg_fcl_cd_ms': avg_fcl_cd,
                'speedup': speedup
            })

    return pd.DataFrame(results)


def create_bar_chart(df: pd.DataFrame, layouts: List[str], machines: List[str],
                     scene1: str, scene2: str, output_file: str, show_title: bool = False):
    """
    Create a single bar chart showing speedup over FCL for collision detection.
    Each bar represents a layout/machine combination.
    """

    # Filter data by scene combo
    filtered_df = df[
        (df['scene1'] == scene1) &
        (df['scene2'] == scene2)
    ]

    if len(filtered_df) == 0:
        print(f"No data found for scene pair '{scene1}' vs '{scene2}'")
        return

    # Compute averages
    summary_df = compute_average_performance(filtered_df, layouts, machines)

    if len(summary_df) == 0:
        print(f"No data found for specified layouts and machines")
        return

    # Sort by speedup
    summary_df = summary_df.sort_values('speedup', ascending=False)

    # Create figure - large for 0.25x scaling
    fig, ax = plt.subplots(1, 1, figsize=(min(24, len(summary_df) * 2.2), 16))

    # Create bar positions
    x_pos = np.arange(len(summary_df))

    # Create bars with thicker edges
    bars = ax.bar(x_pos, summary_df['speedup'],
                  color='#0173B2', alpha=0.7, edgecolor='black', linewidth=4)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=6,
               alpha=0.7, label='FCL Baseline')
    # Color bars: green if faster than FCL, red if slower
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
                ha='center', va='bottom',
                fontsize=32, fontweight='bold')

    # Formatting - all fonts much larger
    if show_title:
        ax.set_title(f"Scene(s): {scene1}, {scene2}",
                     fontsize=52, fontweight='bold', pad=40)
    ax.set_ylabel('Speedup vs FCL', fontsize=32, fontweight='bold')
    ax.set_xlabel('Layout / Machine', fontsize=32, fontweight='bold')

    # Set x-axis labels (layout + machine) - much larger
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df['label'], fontsize=24, fontweight='bold')

    ax.tick_params(axis='y', which='major', labelsize=24, width=3, length=10)
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', linewidth=2)

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # Set y-axis to start at 0 or slightly below minimum
    y_min = 0.0
    y_max = summary_df['speedup'].max() * 1.15
    ax.set_ylim(y_min, 1.9)

    # Adjust layout
    plt.tight_layout()

    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Saved: {output_file}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate bar chart comparing collision detection performance to FCL')
    parser.add_argument(
        'input_csv', help='Input CSV file with collision detection benchmark data')
    parser.add_argument('--layouts', nargs='+', required=True,
                        help='Layouts to compare')
    parser.add_argument('--machines', nargs='+', default=['x86', 'arm'],
                        help='Machines to include (default: x86 arm)')
    parser.add_argument('--scene1', required=True,
                        help='First scene in the collision pair')
    parser.add_argument('--scene2', required=True,
                        help='Second scene in the collision pair')
    parser.add_argument('--output_filename', default='cd_speedup',
                        help='Output file name (default: cd_speedup)')
    parser.add_argument('--show-title', action='store_true',
                        help='Show title on the plot (default: False)')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    print(f"Loaded {len(df)} rows")
    print(f"Analyzing scene pair: {args.scene1} vs {args.scene2}")
    print(f"Layouts: {args.layouts}")
    print(f"Machines: {args.machines}")

    # Generate chart
    create_bar_chart(df, args.layouts, args.machines,
                     args.scene1, args.scene2, args.output_filename, args.show_title)


if __name__ == "__main__":
    main()
