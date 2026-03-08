#!/usr/bin/env python3
"""
Check if a given layout is Pareto-optimal within its group across all combinations.
"""

import pandas as pd
import argparse
from collections import defaultdict
import sys


def load_layout_groups():
    """Load layout groups from layout_grouping module or define inline."""
    try:
        import layout_grouping
        return layout_grouping.retrieve_layout_groups()
    except ImportError:
        # Define default groups if module not available
        return {
            'bvh2': ['pbrt', 'ptr', 'pbrt-align16', 'pbrt-soaos', 'pbrt-soaos-align16',
                     'pbrt-q16', 'pbrt-q16-soaos', 'sg-eq', 'sg-eq-align16'],
            'bvh8': ['bvh8', 'bvh8-align16', 'bvh8-q8', 'bvh8-q8-align16', 'bvh8-q8-ci',
                     'bvh8-q8-ci-align16', 'bvh8-q16', 'bvh8-q16-align16', 'bvh8-q16-ci',
                     'bvh8-q16-ci-align16']
        }


def find_group_for_layout(layout, layout_groups):
    """Find which group a layout belongs to."""
    for group_name, layouts in layout_groups.items():
        if layout in layouts:
            return group_name
    return None


def is_pareto_optimal(point, other_points):
    """
    Check if a point is Pareto-optimal.
    A point is Pareto-optimal if no other point dominates it.
    Point A dominates point B if A is better or equal in all objectives and strictly better in at least one.

    For minimization objectives (memory, time), lower is better.
    """
    memory, time = point

    for other_memory, other_time in other_points:
        # Check if this other point dominates our point
        if (other_memory <= memory and other_time <= time and
                (other_memory < memory or other_time < time)):
            return False

    return True


def check_pareto_optimal_collision(csv_file, layout, group_name, layout_groups):
    """Check Pareto optimality for collision detection data."""
    df = pd.read_csv(csv_file)

    # Get layouts in the group
    if group_name not in layout_groups:
        print(f"Error: Group '{group_name}' not found in layout groups")
        return

    group_layouts = layout_groups[group_name]

    if layout not in group_layouts:
        print(f"Error: Layout '{layout}' not in group '{group_name}'")
        return

    # Filter to only include layouts in the group
    df = df[df['layout'].isin(group_layouts)]

    # Get all unique combinations of (app, machine, scene1, scene2)
    combinations = df[['app', 'machine', 'scene1', 'scene2']].drop_duplicates()

    results = []

    for _, combo in combinations.iterrows():
        app = combo['app']
        machine = combo['machine']
        scene1 = combo['scene1']
        scene2 = combo['scene2']

        # Filter data for this combination
        combo_data = df[
            (df['app'] == app) &
            (df['machine'] == machine) &
            (df['scene1'] == scene1) &
            (df['scene2'] == scene2)
        ]

        if combo_data.empty:
            continue

        # Get the target layout's data
        target_data = combo_data[combo_data['layout'] == layout]

        if target_data.empty:
            results.append({
                'app': app,
                'machine': machine,
                'scene1': scene1,
                'scene2': scene2,
                'status': 'MISSING',
                'reason': f"No data for {layout}"
            })
            continue

        target_memory = target_data['triangle-memory-b'].values[0]
        target_time = target_data['collision-time-ms'].values[0]

        # Get all other points in the group
        other_points = []
        for _, row in combo_data.iterrows():
            if row['layout'] != layout:
                other_points.append(
                    (row['triangle-memory-b'], row['collision-time-ms']))

        # Check if Pareto-optimal
        is_optimal = is_pareto_optimal(
            (target_memory, target_time), other_points)

        if is_optimal:
            results.append({
                'app': app,
                'machine': machine,
                'scene1': scene1,
                'scene2': scene2,
                'status': 'PARETO-OPTIMAL',
                'memory': target_memory,
                'time': target_time
            })
        else:
            # Find which layout(s) dominate
            dominating = []
            dominating_metrics = []
            for _, row in combo_data.iterrows():
                if row['layout'] != layout:
                    other_mem = row['triangle-memory-b']
                    other_time = row['collision-time-ms']
                    if (other_mem <= target_memory and other_time <= target_time and
                            (other_mem < target_memory or other_time < target_time)):
                        dominating.append(row['layout'])
                        dominating_metrics.append((other_mem, other_time))

            # Use first dominator's metrics
            first_dominator_memory = dominating_metrics[0][0] if dominating_metrics else target_memory
            first_dominator_time = dominating_metrics[0][1] if dominating_metrics else target_time

            results.append({
                'app': app,
                'machine': machine,
                'scene1': scene1,
                'scene2': scene2,
                'status': 'DOMINATED',
                'memory': target_memory,
                'time': target_time,
                'dominated_by': dominating[0] if dominating else 'unknown',
                'dominator_memory': first_dominator_memory,
                'dominator_time': first_dominator_time
            })

    return results


def check_pareto_optimal_raytracing(csv_file, layout, group_name, layout_groups):
    """Check Pareto optimality for ray tracing data."""
    df = pd.read_csv(csv_file)

    # Get layouts in the group
    if group_name not in layout_groups:
        print(f"Error: Group '{group_name}' not found in layout groups")
        return

    group_layouts = layout_groups[group_name]

    if layout not in group_layouts:
        print(f"Error: Layout '{layout}' not in group '{group_name}'")
        return

    # Filter to only include layouts in the group
    df = df[df['layout'].isin(group_layouts)]

    # Compute average time per ray across ray counts
    aggregated = df.groupby(['machine', 'scene', 'ray_type', 'layout']).apply(
        lambda x: pd.Series({
            'avg_time_per_ray': (x['trace_time-ms'] / x['ray_count']).mean(),
            'memory': x['total-memory-b'].iloc[0] if 'total-memory-b' in x.columns else x['bvh-memory-b'].iloc[0]
        })
    ).reset_index()

    # Get all unique combinations of (machine, scene, ray_type)
    combinations = aggregated[['machine',
                               'scene', 'ray_type']].drop_duplicates()

    results = []

    for _, combo in combinations.iterrows():
        machine = combo['machine']
        scene = combo['scene']
        ray_type = combo['ray_type']

        # Filter data for this combination
        combo_data = aggregated[
            (aggregated['machine'] == machine) &
            (aggregated['scene'] == scene) &
            (aggregated['ray_type'] == ray_type)
        ]

        if combo_data.empty:
            continue

        # Get the target layout's data
        target_data = combo_data[combo_data['layout'] == layout]

        if target_data.empty:
            results.append({
                'machine': machine,
                'scene': scene,
                'ray_type': ray_type,
                'status': 'MISSING',
                'reason': f"No data for {layout}"
            })
            continue

        target_memory = target_data['memory'].values[0]
        target_time = target_data['avg_time_per_ray'].values[0]

        # Get all other points in the group
        other_points = []
        for _, row in combo_data.iterrows():
            if row['layout'] != layout:
                other_points.append((row['memory'], row['avg_time_per_ray']))

        # Check if Pareto-optimal
        is_optimal = is_pareto_optimal(
            (target_memory, target_time), other_points)

        if is_optimal:
            results.append({
                'machine': machine,
                'scene': scene,
                'ray_type': ray_type,
                'status': 'PARETO-OPTIMAL',
                'memory': target_memory,
                'time': target_time
            })
        else:
            # Find which layout(s) dominate
            dominating = []
            dominating_metrics = []
            for _, row in combo_data.iterrows():
                if row['layout'] != layout:
                    other_mem = row['memory']
                    other_time = row['avg_time_per_ray']
                    if (other_mem <= target_memory and other_time <= target_time and
                            (other_mem < target_memory or other_time < target_time)):
                        dominating.append(row['layout'])
                        dominating_metrics.append((other_mem, other_time))

            # Use first dominator's metrics
            first_dominator_memory = dominating_metrics[0][0] if dominating_metrics else target_memory
            first_dominator_time = dominating_metrics[0][1] if dominating_metrics else target_time

            results.append({
                'machine': machine,
                'scene': scene,
                'ray_type': ray_type,
                'status': 'DOMINATED',
                'memory': target_memory,
                'time': target_time,
                'dominated_by': dominating[0] if dominating else 'unknown',
                'dominator_memory': first_dominator_memory,
                'dominator_time': first_dominator_time
            })

    return results


def print_results(results, layout, group_name, data_type):
    """Print results in a readable format."""
    print(f"\n{'='*80}")
    print(f"Pareto Optimality Check for Layout: {layout}")
    print(f"Group: {group_name}")
    print(f"Data Type: {data_type}")
    print(f"{'='*80}\n")

    optimal_count = sum(1 for r in results if r['status'] == 'PARETO-OPTIMAL')
    dominated_count = sum(1 for r in results if r['status'] == 'DOMINATED')
    missing_count = sum(1 for r in results if r['status'] == 'MISSING')

    print(f"Summary:")
    print(f"  Pareto-Optimal: {optimal_count}/{len(results)}")
    print(f"  Dominated:      {dominated_count}/{len(results)}")
    print(f"  Missing Data:   {missing_count}/{len(results)}")
    print()

    # Print Pareto-optimal cases
    if optimal_count > 0:
        print(f"\nPARETO-OPTIMAL Cases ({optimal_count}):")
        print(f"{'-'*80}")
        for r in results:
            if r['status'] == 'PARETO-OPTIMAL':
                if data_type == 'collision':
                    print(
                        f"  {r['app']:15} | {r['machine']:8} | {r['scene1']:20} vs {r['scene2']:20}")
                else:
                    print(
                        f"  {r['machine']:8} | {r['scene']:30} | {r['ray_type']:10}")
                print(
                    f"    Memory: {r['memory']/1024/1024:8.2f} MB | Time: {r['time']:.6f} {'ms' if data_type == 'collision' else 'ms/ray'}")

    # Print dominated cases
    if dominated_count > 0:
        print(f"\nDOMINATED Cases ({dominated_count}):")
        print(f"{'-'*80}")
        for r in results:
            if r['status'] == 'DOMINATED':
                if data_type == 'collision':
                    print(
                        f"  {r['app']:15} | {r['machine']:8} | {r['scene1']:20} vs {r['scene2']:20}")
                else:
                    print(
                        f"  {r['machine']:8} | {r['scene']:30} | {r['ray_type']:10}")
                print(
                    f"    Memory: {r['memory']/1024/1024:8.2f} MB | Time: {r['time']:.6f} {'ms' if data_type == 'collision' else 'ms/ray'}")

                # Calculate ratios
                mem_ratio = r['memory'] / \
                    r['dominator_memory'] if 'dominator_memory' in r else 1.0
                time_ratio = r['time'] / \
                    r['dominator_time'] if 'dominator_time' in r else 1.0

                print(f"    Dominated by: {r['dominated_by']}")
                print(
                    f"    Ratios (target/dominator): Memory {mem_ratio:.2f}x, Time {time_ratio:.2f}x")

    # Print missing cases
    if missing_count > 0:
        print(f"\nMISSING Data ({missing_count}):")
        print(f"{'-'*80}")
        for r in results:
            if r['status'] == 'MISSING':
                if data_type == 'collision':
                    print(
                        f"  {r['app']:15} | {r['machine']:8} | {r['scene1']:20} vs {r['scene2']:20}")
                else:
                    print(
                        f"  {r['machine']:8} | {r['scene']:30} | {r['ray_type']:10}")
                print(f"    Reason: {r['reason']}")


def generate_latex_table(results, layout, group_name, data_type, output_file=None):
    """Generate LaTeX table for dominated cases."""
    dominated_results = [r for r in results if r['status'] == 'DOMINATED']

    if not dominated_results:
        print("\nNo dominated cases to generate LaTeX table.")
        return

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Dominated cases for \\texttt{" + layout.replace('_', '\\_') + "} in the \\texttt{" + group_name + "} group. "
                 "For each configuration, we show a single dominating layout and the relative performance difference "
                 "(ratio of \\texttt{" + layout.replace('_',
                                                        '\\_') + "} to dominating layout). "
                 "Bold indicates the dimension where the dominating layout is strictly better.}")
    latex.append("\\label{table:dominated-" + layout.replace('_', '-') + "}")

    if data_type == 'collision':
        latex.append("\\begin{tabular}{@{}lllll@{}}")
        latex.append("\\toprule")
        latex.append(
            "\\textbf{App} & \\textbf{Machine} & \\textbf{Scenes} & \\textbf{Dominated By} & \\textbf{Memory / Time} \\\\")
        latex.append("\\midrule")

        for r in dominated_results:
            app = r['app'].replace('_', '\\_')
            machine = r['machine'].replace('_', '\\_')
            scene1 = r['scene1'].replace('_', '\\_')
            scene2 = r['scene2'].replace('_', '\\_')
            scenes = f"{scene1} vs {scene2}"

            # Get first dominating layout
            first_dominator = r['dominated_by']
            first_dominator_escaped = first_dominator.replace('_', '\\_')

            # Calculate ratios (target / dominator)
            if 'dominator_memory' in r and 'dominator_time' in r:
                memory_ratio = r['memory'] / r['dominator_memory']
                time_ratio = r['time'] / r['dominator_time']

                # Determine which to bold (dominator is better when ratio > 1)
                if memory_ratio > 1.0 and time_ratio > 1.0:
                    # Both better, bold both
                    comparison = f"\\textbf{{{memory_ratio:.2f}$\\times$}} / \\textbf{{{time_ratio:.2f}$\\times$}}"
                elif memory_ratio > 1.0:
                    # Memory is better
                    comparison = f"\\textbf{{{memory_ratio:.2f}$\\times$}} / {time_ratio:.2f}$\\times$"
                elif time_ratio > 1.0:
                    # Time is better
                    comparison = f"{memory_ratio:.2f}$\\times$ / \\textbf{{{time_ratio:.2f}$\\times$}}"
                else:
                    # Neither better (shouldn't happen for dominated case)
                    comparison = f"{memory_ratio:.2f}$\\times$ / {time_ratio:.2f}$\\times$"
            else:
                comparison = "N/A"

            latex.append(
                f"{app} & {machine} & {scenes} & \\texttt{{{first_dominator_escaped}}} & {comparison} \\\\")

    else:  # raytracing
        latex.append("\\begin{tabular}{@{}lllll@{}}")
        latex.append("\\toprule")
        latex.append(
            "\\textbf{Machine} & \\textbf{Scene} & \\textbf{Ray Type} & \\textbf{Dominated By} & \\textbf{Memory / Time} \\\\")
        latex.append("\\midrule")

        for r in dominated_results:
            machine = r['machine'].replace('_', '\\_')
            scene = r['scene'].replace('_', '\\_')
            ray_type = r['ray_type'].replace('_', '\\_')

            # Get first dominating layout
            first_dominator = r['dominated_by']
            first_dominator_escaped = first_dominator.replace('_', '\\_')

            # Calculate ratios (target / dominator)
            if 'dominator_memory' in r and 'dominator_time' in r:
                memory_ratio = r['memory'] / r['dominator_memory']
                time_ratio = r['time'] / r['dominator_time']

                # Determine which to bold (dominator is better when ratio > 1)
                if memory_ratio > 1.0 and time_ratio > 1.0:
                    # Both better, bold both
                    comparison = f"\\textbf{{{memory_ratio:.2f}$\\times$}} / \\textbf{{{time_ratio:.2f}$\\times$}}"
                elif memory_ratio > 1.0:
                    # Memory is better
                    comparison = f"\\textbf{{{memory_ratio:.2f}$\\times$}} / {time_ratio:.2f}$\\times$"
                elif time_ratio > 1.0:
                    # Time is better
                    comparison = f"{memory_ratio:.2f}$\\times$ / \\textbf{{{time_ratio:.2f}$\\times$}}"
                else:
                    # Neither better (shouldn't happen for dominated case)
                    comparison = f"{memory_ratio:.2f}$\\times$ / {time_ratio:.2f}$\\times$"
            else:
                comparison = "N/A"

            latex.append(
                f"{machine} & {scene} & {ray_type} & \\texttt{{{first_dominator_escaped}}} & {comparison} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_output = '\n'.join(latex)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_output)
        print(f"\nLaTeX table written to: {output_file}")
    else:
        print("\n" + "="*80)
        print("LaTeX Table for Dominated Cases:")
        print("="*80)
        print(latex_output)

    return latex_output


def main():
    parser = argparse.ArgumentParser(
        description='Check if a layout is Pareto-optimal within its group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if pbrt-q16 is Pareto-optimal in bvh2 group for ray tracing
  python3 check_pareto_optimal.py raytracing.csv pbrt-q16 bvh2
  
  # Check if bvh8-q8 is Pareto-optimal in bvh8 group for collision detection
  python3 check_pareto_optimal.py collision.csv bvh8-q8 bvh8 --type collision
  
  # Auto-detect group for a layout
  python3 check_pareto_optimal.py raytracing.csv pbrt-q16
  
  # Generate LaTeX table for dominated cases
  python3 check_pareto_optimal.py raytracing.csv pbrt-q16 --latex-output dominated_table.tex
        """
    )

    parser.add_argument(
        'csv_file', help='Path to CSV file with benchmark data')
    parser.add_argument('layout', help='Layout name to check (e.g., pbrt-q16)')
    parser.add_argument('group', nargs='?', default=None,
                        help='Group name (e.g., bvh2, bvh8). If not provided, will auto-detect.')
    parser.add_argument('--type', choices=['raytracing', 'collision'], default='raytracing',
                        help='Type of data: raytracing or collision (default: raytracing)')
    parser.add_argument('--latex-output', type=str, default=None,
                        help='Output file for LaTeX table of dominated cases. If not provided, prints to stdout.')

    args = parser.parse_args()

    # Load layout groups
    layout_groups = load_layout_groups()

    # Auto-detect group if not provided
    if args.group is None:
        args.group = find_group_for_layout(args.layout, layout_groups)
        if args.group is None:
            print(f"Error: Could not find group for layout '{args.layout}'")
            print(f"Available groups: {list(layout_groups.keys())}")
            sys.exit(1)
        print(f"Auto-detected group: {args.group}")

    # Check Pareto optimality
    if args.type == 'collision':
        results = check_pareto_optimal_collision(
            args.csv_file, args.layout, args.group, layout_groups)
    else:
        results = check_pareto_optimal_raytracing(
            args.csv_file, args.layout, args.group, layout_groups)

    # Print results
    print_results(results, args.layout, args.group, args.type)

    # Generate LaTeX table if requested or if there are dominated cases
    dominated_count = sum(1 for r in results if r['status'] == 'DOMINATED')
    if dominated_count > 0:
        generate_latex_table(results, args.layout,
                             args.group, args.type, args.latex_output)


if __name__ == "__main__":
    main()
