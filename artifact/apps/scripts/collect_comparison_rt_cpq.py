#!/usr/bin/env python3
"""
Compare ray tracing and closest point query performance relative to fastest layout.
Creates bar plots showing speedup/slowdown for each layout.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict

from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    # "text.latex.preamble": r"\usepackage{amsmath}",  # Optional for math
})


def load_rt_data(csv_file, scenes=None, layouts=None, machines=None, ray_types=None, include_embree=False):
    """Load and process ray tracing data."""
    df = pd.read_csv(csv_file)

    # Exclude embree by default
    if not include_embree:
        df = df[~df['layout'].str.contains('embree', case=False, na=False)]

    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if layouts is not None:
        df = df[df['layout'].isin(layouts)]
    if machines is not None:
        df = df[df['machine'].isin(machines)]
    if ray_types is not None:
        df = df[df['ray_type'].isin(ray_types)]

    # Calculate average time per ray
    df['time_per_ray'] = df['trace_time-ms'] / df['ray_count']

    # Group by scene and layout, average across ray counts
    result = df.groupby(['scene', 'layout'])[
        'time_per_ray'].mean().reset_index()

    return result


def load_cpq_data(csv_file, scenes=None, layouts=None, machines=None, include_embree=False):
    """Load and process CPQ data."""
    df = pd.read_csv(csv_file)

    # Exclude embree by default
    if not include_embree:
        df = df[~df['layout'].str.contains('embree', case=False, na=False)]

    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if layouts is not None:
        df = df[df['layout'].isin(layouts)]
    if machines is not None:
        df = df[df['machine'].isin(machines)]

    # Calculate time per query
    df['time_per_query'] = df['cpq-time-ms'] / df['query-count']

    # Group by scene and layout
    result = df.groupby(['scene', 'layout'])[
        'time_per_query'].mean().reset_index()

    return result


def calculate_speedups(data, metric_col, reference_layout=None):
    """Calculate speedup relative to a reference layout (or fastest) for each scene."""
    speedups = {}

    for scene in data['scene'].unique():
        scene_data = data[data['scene'] == scene]

        if reference_layout is not None:
            ref_rows = scene_data[scene_data['layout'] == reference_layout]
            if ref_rows.empty:
                continue
            ref_time = ref_rows[metric_col].values[0]
        else:
            # Find the fastest time for THIS scene
            ref_time = scene_data[metric_col].min()

        # Calculate speedup relative to reference
        # Speedup > 1.0 means faster than reference
        # Speedup = 1.0 means same as reference
        # Speedup < 1.0 means slower than reference
        scene_speedups = {}
        for _, row in scene_data.iterrows():
            layout = row['layout']
            time = row[metric_col]
            speedup = ref_time / time
            scene_speedups[layout] = speedup

        speedups[scene] = scene_speedups

    return speedups


def plot_comparison(rt_speedups, cpq_speedups, scenes, layouts, output_file):
    """Create comparison bar plot."""
    n_scenes = len(scenes)
    n_layouts = len(layouts)

    # Create figure with appropriate size for scaling to 25%
    # Large fonts so they're readable when scaled
    fig, axes = plt.subplots(1, n_scenes, figsize=(8*n_scenes, 10))
    if n_scenes == 1:
        axes = [axes]

    # Color scheme
    rt_color = '#0173B2'
    cpq_color = '#DE8F05'

    # Bar width and positions
    bar_width = 0.35
    x = np.arange(n_layouts)

    for scene_idx, scene in enumerate(scenes):
        ax = axes[scene_idx]

        # Get speedups for this scene
        rt_values = [rt_speedups[scene].get(layout, 0) for layout in layouts]
        cpq_values = [cpq_speedups[scene].get(layout, 0) for layout in layouts]

        # Create bars
        bars1 = ax.bar(x - bar_width/2, rt_values, bar_width,
                       label='Ray Tracing', color=rt_color,
                       edgecolor='black', linewidth=2)
        bars2 = ax.bar(x + bar_width/2, cpq_values, bar_width,
                       label='CPQ', color=cpq_color,
                       edgecolor='black', linewidth=2)

        # Add horizontal line at y=1 (fastest)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=3, alpha=0.7)

        # Customize plot
        ax.set_xlabel('Layout', fontsize=32, fontweight='bold')
        ax.set_title(f"{scene.replace('_', ' ')}",
                     fontsize=28, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([l for l in layouts],
                           rotation=35, ha='right', fontsize=20)
        ax.tick_params(axis='y', labelsize=24)
        ax.legend(fontsize=24, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, axis='y', linestyle='--', linewidth=1, alpha=0.3)

        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=24, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}.pdf")
    plt.close()


def plot_comparison_grouped(rt_speedups, cpq_speedups, scenes, layouts, output_file, rt_data_all, cpq_data_all):
    """Create comparison bar plot with layouts grouped together."""
    n_scenes = len(scenes)
    n_layouts = len(layouts)

    # Find the dominant (fastest) layout for each scene and algorithm
    # If multiple layouts are within 1% of fastest, prefer one already in the list
    dominant_layouts = {}
    threshold = 1.01  # Within 1% is considered "really close"

    for scene in scenes:
        # Get layouts that exist in BOTH RT and CPQ datasets for this scene
        scene_rt_data = rt_data_all[rt_data_all['scene'] == scene]
        scene_cpq_data = cpq_data_all[cpq_data_all['scene'] == scene]

        rt_layouts = set(scene_rt_data['layout'].unique(
        )) if not scene_rt_data.empty else set()
        cpq_layouts = set(scene_cpq_data['layout'].unique(
        )) if not scene_cpq_data.empty else set()
        # Intersection - layouts in both datasets
        common_layouts = rt_layouts & cpq_layouts

        # Find fastest RT layout for this scene (only considering common layouts)
        if not scene_rt_data.empty and common_layouts:
            # Filter to only common layouts
            scene_rt_common = scene_rt_data[scene_rt_data['layout'].isin(
                common_layouts)]
            if not scene_rt_common.empty:
                min_time = scene_rt_common['time_per_ray'].min()
                # Get all layouts within threshold
                close_layouts = scene_rt_common[scene_rt_common['time_per_ray']
                                                <= min_time * threshold]
                # Prefer layouts already in the requested list
                preferred = None
                for _, row in close_layouts.iterrows():
                    if row['layout'] in layouts:
                        preferred = row['layout']
                        break
                # If no preferred found, use the fastest
                if preferred is None:
                    fastest_rt_idx = scene_rt_common['time_per_ray'].idxmin()
                    preferred = scene_rt_common.loc[fastest_rt_idx, 'layout']
                dominant_layouts[f'{scene}_rt'] = preferred

        # Find fastest CPQ layout for this scene (only considering common layouts)
        if not scene_cpq_data.empty and common_layouts:
            # Filter to only common layouts
            scene_cpq_common = scene_cpq_data[scene_cpq_data['layout'].isin(
                common_layouts)]
            if not scene_cpq_common.empty:
                min_time = scene_cpq_common['time_per_query'].min()
                # Get all layouts within threshold
                close_layouts = scene_cpq_common[scene_cpq_common['time_per_query']
                                                 <= min_time * threshold]
                # Prefer layouts already in the requested list
                preferred = None
                for _, row in close_layouts.iterrows():
                    if row['layout'] in layouts:
                        preferred = row['layout']
                        break
                # If no preferred found, use the fastest
                if preferred is None:
                    fastest_cpq_idx = scene_cpq_common['time_per_query'].idxmin(
                    )
                    preferred = scene_cpq_common.loc[fastest_cpq_idx, 'layout']
                dominant_layouts[f'{scene}_cpq'] = preferred

    # Combine requested layouts with dominant ones (avoiding duplicates)
    all_layouts_to_plot = []
    for scene in scenes:
        scene_layouts = []

        # Always add dominant RT layout first
        rt_dominant = dominant_layouts.get(f'{scene}_rt')
        if rt_dominant:
            scene_layouts.append(rt_dominant)

        # Always add dominant CPQ layout (if different from RT dominant)
        cpq_dominant = dominant_layouts.get(f'{scene}_cpq')
        if cpq_dominant and cpq_dominant != rt_dominant:
            scene_layouts.append(cpq_dominant)

        # Add user-requested layouts (avoiding duplicates)
        for layout in layouts:
            if layout not in scene_layouts:
                scene_layouts.append(layout)

        # Sort layouts alphabetically
        scene_layouts.sort()

        all_layouts_to_plot.append(scene_layouts)

    # Use the maximum number of layouts across all scenes for consistent sizing
    max_layouts = max(len(scene_layouts)
                      for scene_layouts in all_layouts_to_plot)

    # Special handling for 4 scenes - 2x2 grid
    if n_scenes == 4:
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.3)
        axes = [fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1])]
    else:
        # Side by side for other counts
        fig, axes = plt.subplots(1, n_scenes, figsize=(14*n_scenes, 10))
        if n_scenes == 1:
            axes = [axes]

    # Color scheme
    rt_color = '#E1BE6A'
    cpq_color = '#40B0A6'

    # Bar width and spacing
    bar_width = 0.35
    group_width = 1.0
    group_spacing = 0.3

    for scene_idx, scene in enumerate(scenes):
        ax = axes[scene_idx]

        # Use scene-specific layout list
        scene_layouts = all_layouts_to_plot[scene_idx]
        scene_n_layouts = len(scene_layouts)

        # Get speedups for this scene
        rt_values = [rt_speedups[scene].get(
            layout, 0) for layout in scene_layouts]
        cpq_values = [cpq_speedups[scene].get(
            layout, 0) for layout in scene_layouts]

        # Calculate x positions for grouped bars
        x_positions = []
        for i in range(scene_n_layouts):
            x_positions.append(i * (group_width + group_spacing))
        x = np.array(x_positions)

        # Create bars
        bars1 = ax.bar(x - bar_width/2, rt_values, bar_width,
                       label='RT', color=rt_color,
                       edgecolor='black', linewidth=2.5, alpha=0.9)
        bars2 = ax.bar(x + bar_width/2, cpq_values, bar_width,
                       label='CPQ', color=cpq_color,
                       edgecolor='black', linewidth=2.5, alpha=0.9)

        # Add horizontal line at y=1 (fastest)
        ax.axhline(y=1.0, color='red', linestyle='--',
                   linewidth=3.5, alpha=0.8)

        # Customize plot
        ax.set_title(scene.replace('_', ' '),
                     fontsize=28, fontweight='bold')
        ax.set_xticks(x)

        # Create layout labels without marking dominance
        layout_labels = []
        for layout in scene_layouts:
            label = layout.replace('_', '-')
            layout_labels.append(label)

        ax.set_xticklabels(layout_labels, rotation=45,
                           ha='right', fontsize=32, fontweight='bold')
        ax.tick_params(axis='y', labelsize=32)
        ax.tick_params(axis='both', width=2, length=8)

        ax.grid(True, axis='y', linestyle='--', linewidth=1.5, alpha=0.4)

        # Set y-axis to start from 0
        if rt_values and cpq_values:
            y_max = max(max(rt_values), max(cpq_values)) * 1.15
        else:
            y_max = 1.2
        ax.set_ylim(bottom=0, top=y_max)

        # Add value labels only on bars that are 1.0x (fastest)
        for bar in bars1:
            height = bar.get_height()
            if abs(height - 1.0) < 0.001:  # Check if it's 1.0 (fastest)
                ax.text(bar.get_x() + bar.get_width()/2., height + y_max*0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=28, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            if abs(height - 1.0) < 0.001:  # Check if it's 1.0 (fastest)
                ax.text(bar.get_x() + bar.get_width()/2., height + y_max*0.01,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=28, fontweight='bold')

        # Adjust spines
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    # Add legend in the center for 2x2 grid
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=rt_color, edgecolor='black',
              linewidth=2.5, label='RT'),
        Patch(facecolor=cpq_color, edgecolor='black',
              linewidth=2.5, label='CPQ')
    ]
    if n_scenes == 4:
        # Place legend in center of figure
        fig.legend(handles=legend_elements, loc='center', fontsize=28,
                   framealpha=0.95, edgecolor='black', fancybox=False, shadow=True,
                   bbox_to_anchor=(0.48, 0.470))
    elif n_scenes == 2:
        fig.legend(handles=legend_elements, loc='center', fontsize=28,
                   framealpha=0.95, edgecolor='black', fancybox=False, shadow=True,
                   bbox_to_anchor=(0.49, 0.150))
    else:
        raise NotImplementedError(f"n_scenes: {n_scenes}")

    plt.tight_layout()
    plt.savefig(output_file + ".pdf", dpi=1600, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    print(f"\nDominant layouts are selected from layouts that exist in BOTH RT and CPQ datasets.")
    print(f"Each subplot includes both the RT-dominant and CPQ-dominant layouts.")
    print(f"Layouts are displayed in alphabetical order.")
    print(f"Only bars with 1.0x performance (fastest) are labeled with values.")
    print(f"When layouts are within 1% performance, preference is given to already-requested layouts.")

    # Print dominant layouts for each scene
    print("\nDominant layouts per scene:")
    for scene in scenes:
        rt_dom = dominant_layouts.get(f'{scene}_rt', 'N/A')
        cpq_dom = dominant_layouts.get(f'{scene}_cpq', 'N/A')
        print(f"  {scene}: RT={rt_dom}, CPQ={cpq_dom}")

    plt.close()


def print_summary(rt_speedups, cpq_speedups, scenes, layouts):
    """Print summary statistics."""
    from scipy.stats import gmean

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)

    for scene in scenes:
        print(f"\nScene: {scene}")
        print("-"*80)
        print(f"{'Layout':<30} {'RT Speedup':>15} {'CPQ Speedup':>15}")
        print("-"*80)

        for layout in layouts:
            rt_speedup = rt_speedups[scene].get(layout, 0)
            cpq_speedup = cpq_speedups[scene].get(layout, 0)

            rt_str = f"{rt_speedup:.3f}x" if rt_speedup > 0 else "N/A"
            cpq_str = f"{cpq_speedup:.3f}x" if cpq_speedup > 0 else "N/A"

            print(f"{layout:<30} {rt_str:>15} {cpq_str:>15}")

    # Aggregate statistics across scenes
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS ACROSS SCENES")
    print("="*80)

    # Collect per-layout stats
    for label, speedups, metric in [("RT", rt_speedups, "RT"), ("CPQ", cpq_speedups, "CPQ")]:
        print(f"\n--- {label} ---")
        print(f"{'Layout':<30} {'GeoMean':>10} {'ArithMean':>10} {'Min':>10} {'Max':>10} {'Wins':>6}")
        print("-"*80)

        layout_geomeans = []
        for layout in layouts:
            vals = [speedups[s].get(layout, 0) for s in scenes]
            vals = [v for v in vals if v > 0]
            if not vals:
                print(f"{layout:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>6}")
                continue
            geo = gmean(vals)
            arith = np.mean(vals)
            mn = min(vals)
            mx = max(vals)
            wins = sum(1 for v in vals if abs(v - 1.0) < 0.001)
            layout_geomeans.append((layout, geo))
            print(f"{layout:<30} {geo:>9.3f}x {arith:>9.3f}x {mn:>9.3f}x {mx:>9.3f}x {wins:>6}")

        # Print ranking by geometric mean
        layout_geomeans.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{label} Ranking (by geometric mean speedup):")
        for rank, (layout, geo) in enumerate(layout_geomeans, 1):
            print(f"  {rank}. {layout:<28} {geo:.3f}x")


def print_fcpw_comparison(csv_file, scenes, layout, machines=None):
    """Print summary statistics comparing cpq-time-ms (Scion) vs fcpw-cpq-time-ms (FCPW) for a given layout."""
    from scipy.stats import gmean

    df = pd.read_csv(csv_file)
    df = df[df['layout'] == layout]
    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if machines is not None:
        df = df[df['machine'].isin(machines)]

    available_machines = sorted(df['machine'].unique())

    for machine in available_machines:
        mdf = df[df['machine'] == machine]

        # Group by scene, using geometric mean across runs
        grouped = mdf.groupby('scene').agg(
            scion_time=('cpq-time-ms', lambda x: gmean(x)),
            fcpw_time=('fcpw-cpq-time-ms', lambda x: gmean(x)),
            query_count=('query-count', 'first'),
        ).reset_index()

        grouped['speedup'] = grouped['fcpw_time'] / grouped['scion_time']

        print("\n" + "="*80)
        print(f"FCPW / Scion CPQ COMPARISON (layout: {layout}, machine: {machine})")
        print("="*80)
        print(f"{'Scene':<30} {'Scion (ms)':>12} {'FCPW (ms)':>12} {'Speedup':>10}")
        print("-"*80)

        speedups = []
        for _, row in grouped.iterrows():
            speedup = row['speedup']
            speedups.append(speedup)
            print(f"{row['scene']:<30} {row['scion_time']:>12.1f} {row['fcpw_time']:>12.1f} {speedup:>9.2f}x")

        print("-"*80)
        geo = gmean(speedups)
        arith = np.mean(speedups)
        print(f"{'Geometric Mean':<30} {'':>12} {'':>12} {geo:>9.2f}x")
        print(f"{'Arithmetic Mean':<30} {'':>12} {'':>12} {arith:>9.2f}x")
        print(f"{'Min':<30} {'':>12} {'':>12} {min(speedups):>9.2f}x")
        print(f"{'Max':<30} {'':>12} {'':>12} {max(speedups):>9.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Compare RT and CPQ performance across layouts and scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific scenes and layouts
  python3 compare_rt_cpq.py rt.csv cpq.csv --scenes lucy hairball --layouts pbrt pbrt-q16 bvh8
  
  # Filter by machine
  python3 compare_rt_cpq.py rt.csv cpq.csv --scenes lucy --layouts pbrt pbrt-q16 --machines x86
  
  # Specify output file
  python3 compare_rt_cpq.py rt.csv cpq.csv --scenes lucy --layouts pbrt pbrt-q16 --output comparison.pdf
        """
    )

    parser.add_argument('rt_csv', nargs='?', default=None,
                        help='Path to ray tracing CSV file')
    parser.add_argument('cpq_csv', nargs='?', default=None,
                        help='Path to CPQ CSV file')
    parser.add_argument('--scenes', nargs='+', required=True,
                        help='Scenes to compare (e.g., lucy hairball)')
    parser.add_argument('--layouts', nargs='+', default=None,
                        help='Layouts to compare (e.g., pbrt pbrt-q16 bvh8)')
    parser.add_argument('--machines', nargs='+', default=None,
                        help='Filter by machine (e.g., x86)')
    parser.add_argument('--ray-types', nargs='+', default=None,
                        help='Filter RT by ray types (e.g., primary secondary)')
    parser.add_argument('--include-embree', action='store_true',
                        help='Include Embree layouts in comparison (default: excluded)')
    parser.add_argument('--output_filename', type=str, default='rt_cpq_comparison',
                        help='Output file name (default: rt_cpq_comparison)')
    parser.add_argument('--reference-layout', type=str, default=None,
                        help='Reference layout for speedup calculation (default: fastest)')
    parser.add_argument('--compare-fcpw', type=str, default=None, metavar='CPQ_CSV',
                        help='Compare cpq-time-ms vs fcpw-cpq-time-ms for a layout (use with --compare-fcpw-layout)')
    parser.add_argument('--compare-fcpw-layout', type=str, default='pbrt',
                        help='Layout to use for FCPW comparison (default: pbrt)')

    args = parser.parse_args()

    if args.compare_fcpw:
        print_fcpw_comparison(args.compare_fcpw, args.scenes,
                              args.compare_fcpw_layout, machines=args.machines)
        return

    if not args.rt_csv or not args.cpq_csv:
        parser.error('rt_csv and cpq_csv are required unless using --compare-fcpw')
    if not args.layouts:
        parser.error('--layouts is required unless using --compare-fcpw')

    # Load data - first load ALL layouts to find global fastest
    print("Loading all ray tracing data for speedup calculation...")
    rt_data_all = load_rt_data(args.rt_csv,
                               scenes=args.scenes,
                               layouts=None,  # Load all layouts
                               machines=args.machines,
                               ray_types=args.ray_types,
                               include_embree=args.include_embree)

    print("Loading all CPQ data for speedup calculation...")
    cpq_data_all = load_cpq_data(args.cpq_csv,
                                 scenes=args.scenes,
                                 layouts=None,  # Load all layouts
                                 machines=args.machines,
                                 include_embree=args.include_embree)

    # Calculate speedups using all layouts
    ref = args.reference_layout
    if ref:
        print(f"Calculating speedups relative to '{ref}'...")
    else:
        print("Calculating speedups relative to fastest across all layouts...")
    rt_speedups = calculate_speedups(rt_data_all, 'time_per_ray', reference_layout=ref)
    cpq_speedups = calculate_speedups(cpq_data_all, 'time_per_query', reference_layout=ref)

    # Filter to only the requested layouts for display
    print(f"Filtering to requested layouts: {args.layouts}")

    # Print summary
    print_summary(rt_speedups, cpq_speedups, args.scenes, args.layouts)

    # Create plot
    print(f"\nGenerating comparison plot...")
    plot_comparison_grouped(rt_speedups, cpq_speedups, args.scenes,
                            args.layouts, args.output_filename, rt_data_all, cpq_data_all)


if __name__ == "__main__":
    main()
