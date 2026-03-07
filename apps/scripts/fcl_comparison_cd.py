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
    Converts 'hairball_60_70_10' to 'hairball (60^{\circ}, 70^{\circ}, 10^{\circ})'
    """
    import re
    match = re.match(r'^(hairball)_(\d+)_(\d+)_(\d+)$', scene)
    if match:
        base_name = match.group(1)
        x_val = match.group(2)
        y_val = match.group(3)
        z_val = match.group(4)
        return fr"\texttt{{{base_name}}} ({x_val}^{{\circ}}, {y_val}^{{\circ}}, {z_val}^{{\circ}})"
    return fr"\texttt{{{scene}}}"


def compute_average_performance(df: pd.DataFrame, layouts: List[str], machines: List[str]) -> pd.DataFrame:
    """
    Compute average collision detection time for each layout/machine combination.
    Returns speedup compared to FCL.
    """
    results = []

    for layout in layouts:
        for machine in machines:
            layout_df = df[(df['layout'] == layout) &
                           (df['machine'] == machine)]

            if len(layout_df) == 0:
                continue

            avg_bonsai_cd = layout_df['collision-time-ms'].mean()
            avg_fcl_cd = layout_df['fcl-collision-time-ms'].mean()

            if avg_bonsai_cd > 0:
                speedup = avg_fcl_cd / avg_bonsai_cd
            else:
                speedup = np.nan

            results.append({
                'layout': layout,
                'machine': machine,
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

    filtered_df = df[
        (df['scene1'] == scene1) &
        (df['scene2'] == scene2)
    ]

    if len(filtered_df) == 0:
        print(f"No data found for scene pair '{scene1}' vs '{scene2}'")
        return

    summary_df = compute_average_performance(filtered_df, layouts, machines)

    if len(summary_df) == 0:
        print(f"No data found for specified layouts and machines")
        return

    pivot_df = summary_df.pivot(
        index='layout', columns='machine', values='speedup')

    pivot_df = pivot_df.reindex(layouts)
    pivot_df = pivot_df.dropna(how='all')

    if len(pivot_df) == 0:
        print(f"No valid data after pivoting")
        return

    fig, ax = plt.subplots(1, 1, figsize=(min(24, len(pivot_df) * 2), 4))

    x_pos = np.arange(len(pivot_df)) * 1.5

    available_machines = [m for m in pivot_df.columns if m in machines]
    machine_colors = {'x86': '#0173B2', 'arm': '#DE8F05', 'cuda': '#029E73'}
    n_machines = len(available_machines)
    bar_width = 0.70

    all_bars = []
    for i, machine in enumerate(available_machines):
        data = pivot_df[machine].fillna(0)
        offset = (i - (n_machines - 1) / 2) * bar_width
        color = machine_colors.get(machine, '#666666')
        bars = ax.bar(x_pos + offset, data, bar_width,
                      label=machine, color=color, alpha=0.7,
                      edgecolor='black', linewidth=2)
        all_bars.append(bars)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=3,
               alpha=0.7)

    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if height < 1.0:
                    ax.text(bar.get_x() + bar.get_width()/2., height * 0.95,
                            fr"$\mathbf{{{height:.2f}x}}$",
                            ha='center', va='top',
                            fontsize=18, color='black')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            fr"$\mathbf{{{height:.2f}x}}$",
                            ha='center', va='bottom',
                            fontsize=18)

    max_speedup = max(pivot_df[m].fillna(0).max() for m in available_machines)
    y_limit = max_speedup * 1.15

    if show_title:
        ax.set_title(f"${format_scene_name(scene1)}$, ${format_scene_name(scene2)}$",
                     fontsize=32, pad=40)
    ax.set_ylabel(r"$\mathbf{FCL \textsc{ } / \textsc{ Scion}}$", fontsize=25)
    ax.set_yticks([1])
    ax.set_xlabel(r"$\mathbf{Layout}$", fontsize=25)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([fr"$\mathbf{{\texttt{{{layout}}}}}$" for layout in pivot_df.index],
                       fontsize=24)

    ax.tick_params(axis='y', which='major', labelsize=26, width=3, length=10)
    ax.tick_params(axis='x', which='major', width=3, length=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.set_ylim(0.0, y_limit)

    ax.legend(fontsize=19, loc='lower left',
              frameon=True, edgecolor='black', framealpha=0.9)

    plt.tight_layout()

    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Saved: {output_file}.pdf")
    plt.close()


def print_fcl_comparison(csv_file, scene1, scene2, layout, machines=None):
    """Print summary statistics comparing collision-time-ms (Scion) vs fcl-collision-time-ms (FCL) for a given layout."""
    from scipy.stats import gmean

    df = pd.read_csv(csv_file)
    df = df[(df['layout'] == layout) &
            (df['scene1'] == scene1) &
            (df['scene2'] == scene2)]
    if machines is not None:
        df = df[df['machine'].isin(machines)]

    available_machines = sorted(df['machine'].unique())

    for machine in available_machines:
        mdf = df[df['machine'] == machine]

        scion_time = gmean(mdf['collision-time-ms'])
        fcl_time = gmean(mdf['fcl-collision-time-ms'])
        speedup = fcl_time / scion_time if scion_time > 0 else np.nan

        print("\n" + "="*80)
        print(f"FCL / Scion CD COMPARISON (layout: {layout}, machine: {machine})")
        print(f"Scenes: {scene1}, {scene2}")
        print("="*80)
        print(f"{'':>30} {'Scion (ms)':>12} {'FCL (ms)':>12} {'Speedup':>10}")
        print("-"*80)
        print(f"{'collision-time':>30} {scion_time:>12.1f} {fcl_time:>12.1f} {speedup:>9.2f}x")

        # Also show build times if available
        if 'build-time-ms' in mdf.columns and 'fcl-build-time-ms' in mdf.columns:
            scion_build = gmean(mdf['build-time-ms'])
            fcl_build = gmean(mdf['fcl-build-time-ms'])
            build_speedup = fcl_build / scion_build if scion_build > 0 else np.nan
            print(f"{'build-time':>30} {scion_build:>12.1f} {fcl_build:>12.1f} {build_speedup:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Generate bar chart comparing collision detection performance to FCL')
    parser.add_argument(
        'input_csv', help='Input CSV file with collision detection benchmark data')
    parser.add_argument('--layouts', nargs='+', default=None,
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
    parser.add_argument('--compare-fcl', action='store_true',
                        help='Print FCL vs Scion comparison summary statistics')
    parser.add_argument('--compare-fcl-layout', type=str, default='pbrt',
                        help='Layout to use for FCL comparison (default: pbrt)')

    args = parser.parse_args()

    if args.compare_fcl:
        print_fcl_comparison(args.input_csv, args.scene1, args.scene2,
                             args.compare_fcl_layout, machines=args.machines)
        return

    if not args.layouts:
        parser.error('--layouts is required unless using --compare-fcl')

    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    print(f"Loaded {len(df)} rows")
    print(f"Analyzing scene pair: {args.scene1} vs {args.scene2}")
    print(f"Layouts: {args.layouts}")
    print(f"Machines: {args.machines}")

    create_bar_chart(df, args.layouts, args.machines,
                     args.scene1, args.scene2, args.output_filename, args.show_title)


if __name__ == "__main__":
    main()
