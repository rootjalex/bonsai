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
})


def format_scene_name(scene: str) -> str:
    """
    Format scene names for display.
    Converts 'san-miguel-x35-y22-z47' to 'san-miguel (35^{\circ}, 22^{\circ}, 47^{\circ})'
    """
    import re

    # Check if scene matches the pattern san-miguel-xN-yM-zP
    match = re.match(r'^(san-miguel)-x(\d+)-y(\d+)-z(\d+)$', scene)
    if match:
        base_name = match.group(1)
        x_val = match.group(2)
        y_val = match.group(3)
        z_val = match.group(4)
        return fr"{base_name} ({x_val}^{{\circ}}, {y_val}^{{\circ}}, {z_val}^{{\circ}})"

    return scene


def compute_average_performance(df: pd.DataFrame, layout: str) -> pd.DataFrame:
    """
    Compute average CPQ time for each machine/scene combination for a given layout.
    Returns speedup compared to FCPW (inverse of slowdown).
    """
    results = []

    # Get unique machine/scene combinations
    for (machine, scene), group_df in df.groupby(['machine', 'scene']):
        layout_df = group_df[group_df['layout'] == layout]

        if len(layout_df) == 0:
            continue

        # Average CPQ times
        avg_bonsai_cpq = layout_df['cpq-time-ms'].mean()
        avg_fcpw_cpq = layout_df['fcpw-cpq-time-ms'].mean()

        # Compute speedup (fcpw / bonsai) - inverse of slowdown
        if avg_bonsai_cpq > 0:
            speedup = avg_fcpw_cpq / avg_bonsai_cpq
        else:
            speedup = np.nan

        results.append({
            'machine': machine,
            'scene': scene,
            'avg_bonsai_cpq_ms': avg_bonsai_cpq,
            'avg_fcpw_cpq_ms': avg_fcpw_cpq,
            'speedup': speedup
        })

    return pd.DataFrame(results)


def create_grouped_bar_chart(df: pd.DataFrame, layout: str, scenes: List[str],
                             output_file: str, show_title: bool = False):
    """
    Create a grouped bar chart showing speedup for x86 and arm across all scenes.
    """

    # Filter data by scenes and layout
    filtered_df = df[df['scene'].isin(scenes)]

    if len(filtered_df) == 0:
        print(f"No data found for layout '{layout}' with selected scenes")
        return

    # Compute averages for this layout
    summary_df = compute_average_performance(filtered_df, layout)

    if len(summary_df) == 0:
        print(f"No data found for layout '{layout}'")
        return

    # Pivot to get x86 and arm data side by side
    pivot_df = summary_df.pivot(
        index='scene', columns='machine', values='speedup')

    # Reorder scenes to match input order
    pivot_df = pivot_df.reindex(scenes)

    # Remove any scenes with missing data
    pivot_df = pivot_df.dropna(how='all')

    if len(pivot_df) == 0:
        print(f"No valid data after pivoting")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(min(24, len(pivot_df) * 2), 8))

    # Create bar positions
    x_pos = np.arange(len(pivot_df))
    bar_width = 0.45

    # Create bars for x86 and arm
    x86_data = pivot_df['x86'].fillna(0)
    arm_data = pivot_df['arm'].fillna(0)

    bars_x86 = ax.bar(x_pos - bar_width/2, x86_data, bar_width,
                      label='x86', color='#0173B2', alpha=0.7,
                      edgecolor='black', linewidth=2)
    bars_arm = ax.bar(x_pos + bar_width/2, arm_data, bar_width,
                      label='arm', color='#DE8F05', alpha=0.7,
                      edgecolor='black', linewidth=2)

    # Add horizontal line at y=1 (FCPW baseline)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=3,
               alpha=0.7, label='FCPW Baseline')

    # Add value labels on bars
    for bars in [bars_x86, bars_arm]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                # Place label inside bar if below baseline (< 1.0), above if >= 1.0
                if height < 1.0:
                    # Inside the bar at the upper part
                    ax.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                            fr"$\mathbf{{{height:.2f}x}}$",
                            ha='center', va='top',
                            fontsize=16, color='black')
                else:
                    # Above the bar
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            fr"$\mathbf{{{height:.2f}x}}$",
                            ha='center', va='bottom',
                            fontsize=16)

    # Calculate y-axis limit based on tallest bar
    max_speedup = max(x86_data.max(), arm_data.max())
    y_limit = max_speedup * 1.15  # Add 15% padding above tallest bar

    # Formatting
    if show_title:
        ax.set_title(fr"$\mathbf{{Layout:\ {layout}}}$",
                     fontsize=52, pad=40)
    ax.set_ylabel(r"$\mathbf{Speedup\ vs\ FCPW}$", fontsize=32)
    ax.set_xlabel(r"$\mathbf{Scene}$", fontsize=32)

    # Set x-axis labels (scene names)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([fr"$\mathbf{{{format_scene_name(scene)}}}$" for scene in pivot_df.index],
                       fontsize=18, rotation=15, ha='right')

    ax.tick_params(axis='y', which='major', labelsize=20, width=3, length=10)
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Thicker spines
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # Set y-axis from 0.0 to calculated limit
    ax.set_ylim(0.0, y_limit)

    # Add legend
    ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.22, 1.0),
              frameon=True, edgecolor='black', framealpha=0.9)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Saved: {output_file}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate grouped bar chart comparing CPQ performance to FCPW for pbrt layout')
    parser.add_argument(
        'input_csv', help='Input CSV file with CPQ benchmark data')
    parser.add_argument('--scenes', nargs='+', required=True,
                        help='Scenes to include (in desired order)')
    parser.add_argument('--output', default='cpq_speedup_pbrt',
                        help='Output file name (default: cpq_speedup_pbrt)')
    parser.add_argument('--show-title', action='store_true',
                        help='Show title on the plot (default: False)')

    args = parser.parse_args()

    # Fixed layout
    layout = 'pbrt'

    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    print(f"Loaded {len(df)} rows")
    print(f"Analyzing layout: {layout}")
    print(f"Scenes: {args.scenes}")

    # Generate chart
    create_grouped_bar_chart(df, layout, args.scenes,
                             args.output, args.show_title)

    print(f"\nDone! Output saved to {args.output}")


if __name__ == "__main__":
    main()
