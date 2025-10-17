import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from collections import defaultdict
import sys
import argparse
import layout_grouping


def load_csv_data(csv_file, machines=None, ray_types=None, scenes=None, layouts=None):
    df = pd.read_csv(csv_file)

    if machines is not None:
        df = df[df['machine'].isin(machines)]
    if ray_types is not None:
        df = df[df['ray_type'].isin(ray_types)]
    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if layouts is not None:
        df = df[df['layout'].isin(layouts)]

    raw_data = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))

    for _, row in df.iterrows():
        model = row['scene']
        machine = row['machine']
        ray_type = row['ray_type']
        layout = row['layout']
        ray_count = row['ray_count']
        trace_time = row['trace_time-ms']

        raw_data[model][machine][ray_type][layout][ray_count].append(
            trace_time)

    machine_type = df['machine'].iloc[0] if len(df) > 0 else None
    ray_type_val = df['ray_type'].iloc[0] if len(
        df) > 0 and 'ray_type' in df.columns else None

    if machines and len(machines) > 1:
        machine_type = ','.join(machines)
    if ray_types and len(ray_types) > 1:
        ray_type_val = ','.join(ray_types)

    return raw_data, machine_type, ray_type_val


def load_memory_data(csv_file, scenes=None, layouts=None, machines=None, ray_types=None, memory_column='bvh-memory-b'):
    df = pd.read_csv(csv_file)

    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if layouts is not None:
        df = df[df['layout'].isin(layouts)]
    if machines is not None:
        df = df[df['machine'].isin(machines)]
    if ray_types is not None:
        df = df[df['ray_type'].isin(ray_types)]

    memory_data = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))))

    for _, row in df.iterrows():
        model = row['scene']
        machine = row['machine']
        ray_type = row['ray_type']
        layout = row['layout']

        if pd.notna(row[memory_column]):
            memory_data[model][machine][ray_type][layout]['memory'] = row[memory_column]

    return memory_data


def calculate_average(values):
    if not values:
        return 0

    if len(values) <= 4:
        filtered_values = values
    else:
        sorted_values = sorted(values)
        filtered_values = sorted_values[2:-2]

    if not filtered_values:
        filtered_values = values

    return sum(filtered_values) / len(filtered_values)


def process_trace_data(raw_data, mean_strategy, ray_count_range=None):
    processed_data = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))))
    R = (0, sys.maxsize) if ray_count_range is None else ray_count_range

    for model in raw_data:
        for machine in raw_data[model]:
            for ray_type in raw_data[model][machine]:
                for layout in raw_data[model][machine][ray_type]:
                    time_per_ray_values = []
                    ray_counts = []
                    for ray_count_key in raw_data[model][machine][ray_type][layout]:
                        ray_count = int(ray_count_key) if isinstance(
                            ray_count_key, str) else ray_count_key
                        if not (ray_count >= R[0] and ray_count <= R[1]):
                            continue
                        values = raw_data[model][machine][ray_type][layout][ray_count_key]
                        avg_value = calculate_average(values)
                        if avg_value > 0:
                            time_per_ray = avg_value / ray_count
                            time_per_ray_values.append(time_per_ray)
                            ray_counts.append(ray_count)
                    if not time_per_ray_values:
                        continue
                    if mean_strategy == 'geo':
                        result = math.exp(
                            sum(math.log(t) for t in time_per_ray_values) / len(time_per_ray_values))
                    elif mean_strategy == 'wavg':
                        total_rays = sum(ray_counts)
                        result = sum(
                            t * r for t, r in zip(time_per_ray_values, ray_counts)) / total_rays
                    else:
                        assert mean_strategy == 'arithmetic', mean_strategy
                        result = sum(time_per_ray_values) / \
                            len(time_per_ray_values)
                    processed_data[model][machine][ray_type][layout] = result

    return processed_data


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, mean_strategy, ray_type, ray_count_range, label_dominated_points, memory_type, machines, ray_types):
    models = sorted(processed_data.keys())
    if len(models) == 0:
        return

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    group_colors = ['#0173B2',
                    '#DE8F05',
                    '#029E73',
                    '#CC78BC',
                    '#CA9161',
                    '#949494',
                    '#ECE133',
                    '#56B4E9',
                    '#D55E00',
                    '#F0E442']

    line_styles = [
        '-',
        '--',
        '-.',
        ':',
        (0, (5, 2, 1, 2)),
        (0, (3, 1, 1, 1)),
        (0, (5, 5)),
        (0, (1, 1)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1))
    ]
    marker_styles = ['o', 's', '^', 'D', 'v', '>', 'p', '*', 'h', 'X']

    multiple_machines = machines and len(machines) > 1
    multiple_ray_types = ray_types and len(ray_types) > 1

    for idx, model in enumerate(models):
        ax = axes[idx]

        if model not in memory_data:
            ax.text(0.5, 0.5, f'No data for {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        all_points = []
        all_labels = []
        all_colors = []
        all_groups = []

        for machine in memory_data[model]:
            for ray_type_key in memory_data[model][machine]:
                for group_idx, (group_name, layouts) in enumerate(layout_groups.items()):
                    group_color = group_colors[group_idx % len(group_colors)]

                    for layout in layouts:
                        if machine not in processed_data[model]:
                            continue
                        if ray_type_key not in processed_data[model][machine]:
                            continue
                        if layout not in processed_data[model][machine][ray_type_key]:
                            continue
                        if machine not in memory_data[model]:
                            continue
                        if ray_type_key not in memory_data[model][machine]:
                            continue
                        if layout not in memory_data[model][machine][ray_type_key]:
                            continue

                        time_per_ray = processed_data[model][machine][ray_type_key][layout]
                        memory = memory_data[model][machine][ray_type_key][layout]['memory']

                        if time_per_ray > 0 and memory > 0:
                            all_points.append((memory, time_per_ray))
                            all_labels.append(layout.upper())
                            all_colors.append(group_color)

                            group_key = group_name
                            if multiple_machines:
                                group_key = f"{machine}-{group_key}"
                            if multiple_ray_types:
                                group_key = f"{group_key}-{ray_type_key}"
                            all_groups.append(group_key)

        if not all_points:
            print(
                f"  Warning: No valid data points for model '{model}' - skipping plot")
            ax.text(0.5, 0.5, f'No valid data for {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        print(f"  Plotting {len(all_points)} data points for {model}")

        points = np.array(all_points)
        x = points[:, 0]
        y = points[:, 1]

        memory_values = x / 1024 / 1024
        memory_unit = 'MB'
        time_per_ray_values = y * 1e6
        time_unit = 'ns/ray'

        unique_groups = sorted(set(all_groups))
        for group_idx, group_key in enumerate(unique_groups):
            group_color = group_colors[group_idx % len(group_colors)]
            group_linestyle = line_styles[group_idx % len(line_styles)]
            group_marker = marker_styles[group_idx % len(marker_styles)]

            group_mask = np.array([g == group_key for g in all_groups])
            if not np.any(group_mask):
                continue

            group_points = points[group_mask]
            group_labels = [all_labels[i]
                            for i in range(len(all_labels)) if group_mask[i]]

            is_pareto = np.ones(len(group_points), dtype=bool)
            for i in range(len(group_points)):
                if not is_pareto[i]:
                    continue
                for j in range(len(group_points)):
                    if i == j:
                        continue
                    if (group_points[j, 0] <= group_points[i, 0] and
                        group_points[j, 1] <= group_points[i, 1] and
                            (group_points[j, 0] < group_points[i, 0] or group_points[j, 1] < group_points[i, 1])):
                        is_pareto[i] = False
                        break
            group_memory = memory_values[group_mask]
            group_time_per_ray = time_per_ray_values[group_mask]
            if label_dominated_points and np.any(~is_pareto):
                ax.scatter(group_memory[~is_pareto], group_time_per_ray[~is_pareto],
                           c='lightgray', s=80, alpha=0.5,
                           marker='o', edgecolors='gray', linewidth=1, zorder=1)

            ax.scatter(group_memory[is_pareto], group_time_per_ray[is_pareto],
                       c=group_color, s=180, alpha=0.9,
                       edgecolors='black', linewidth=2.5,
                       marker=group_marker, label=group_key, zorder=3)

            pareto_indices = np.where(is_pareto)[0]
            if len(pareto_indices) > 1:
                pareto_sorted = sorted(
                    pareto_indices, key=lambda i: group_memory[i])
                pareto_x = [group_memory[i] for i in pareto_sorted]
                pareto_y = [group_time_per_ray[i] for i in pareto_sorted]
                ax.plot(pareto_x, pareto_y, color=group_color,
                        linestyle=group_linestyle, alpha=0.8, linewidth=3, zorder=2)

            for i in pareto_indices:
                ax.annotate(group_labels[i],
                            xy=(group_memory[i], group_time_per_ray[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=8,
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='yellow',
                                      alpha=0.7,
                                      edgecolor='black',
                                      linewidth=1),
                            zorder=4)

            if label_dominated_points and np.any(~is_pareto):
                dominated_indices = np.where(~is_pareto)[0]
                for i in dominated_indices:
                    ax.annotate(group_labels[i],
                                xy=(group_memory[i], group_time_per_ray[i]),
                                textcoords="offset points",
                                xytext=(0, 6),
                                ha='center',
                                fontsize=7,
                                color='gray',
                                alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='white',
                                          alpha=0.5,
                                          edgecolor='gray',
                                          linewidth=0.5),
                                zorder=2)

        memory_label = 'Memory Utilization (excluding primitives)' if memory_type == 'bvh' else 'Memory Utilization'
        ax.set_xlabel(
            f'{memory_label} ({memory_unit})', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'Time per Ray ({time_unit})',
                      fontweight='bold', fontsize=11)
        ax.set_title(f'{model.title()}', fontweight='bold', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=9, loc='best', framealpha=0.9,
                  edgecolor='black', fancybox=False, shadow=True)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, p: f'{y:,.1f}'))

    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    title = 'Time per Ray vs Memory Utilization'
    if ray_count_range is not None:
        title += f'\n{mean_strategy} ({ray_count_range[0]:,} - {ray_count_range[1]:,})'
    else:
        title += f'\n{mean_strategy} (all)'
    if machine_type:
        title += f' - {machine_type}'
    if ray_type:
        title += f' - {ray_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if len(all_points) > 0:
        x_range = max(memory_values) - min(memory_values)
        padding = x_range * 0.02
        x_min = min(memory_values) - padding
        x_max = max(memory_values) + padding
        ax.set_xlim(x_min, x_max)
        ax.margins(x=0)

    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(results_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(output_path))[0]

    if ray_count_range is not None:
        def log2(x): return int(math.log2(x))
        output_file = os.path.join(
            results_dir, f'{name}-{mean_strategy}{log2(ray_count_range[0])}-{log2(ray_count_range[1])}.pdf')
    else:
        output_file = os.path.join(
            results_dir, f'{name}-{mean_strategy}.pdf')

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()


def calculate_speedups(processed_data, memory_data, layout_groups, filename, mean_strategy, machines, ray_types):
    speedups = {}

    for model in processed_data:
        speedups[model] = {}

        for machine in processed_data[model]:
            speedups[model][machine] = {}

            for ray_type in processed_data[model][machine]:
                layouts = processed_data[model][machine][ray_type]

                slowest_layout = max(layouts.items(), key=lambda x: x[1])
                slowest_time = slowest_layout[1]
                slowest_name = slowest_layout[0]

                model_speedups = {}
                model_memories = {}
                for layout, time_per_ray in layouts.items():
                    speedup = slowest_time / time_per_ray
                    model_speedups[layout] = speedup
                    if machine in memory_data[model] and ray_type in memory_data[model][machine] and layout in memory_data[model][machine][ray_type]:
                        model_memories[layout] = memory_data[model][machine][ray_type][layout]['memory']

                result = {
                    'overall': {
                        'speedups': model_speedups,
                        'memories': model_memories,
                        'slowest_layout': slowest_name,
                        'slowest_time': slowest_time
                    }
                }

                group_speedups = {}
                for group_name, group_layouts in layout_groups.items():
                    group_data = {layout: layouts[layout] for layout in group_layouts
                                  if layout in layouts}

                    if not group_data:
                        continue

                    group_slowest = max(group_data.items(), key=lambda x: x[1])
                    group_slowest_time = group_slowest[1]
                    group_slowest_name = group_slowest[0]

                    group_speedup_data = {}
                    group_memory_data = {}
                    for layout, time_per_ray in group_data.items():
                        speedup = group_slowest_time / time_per_ray
                        group_speedup_data[layout] = speedup
                        if machine in memory_data[model] and ray_type in memory_data[model][machine] and layout in memory_data[model][machine][ray_type]:
                            group_memory_data[layout] = memory_data[model][machine][ray_type][layout]['memory']

                    group_speedups[group_name] = {
                        'speedups': group_speedup_data,
                        'memories': group_memory_data,
                        'slowest_layout': group_slowest_name,
                        'slowest_time': group_slowest_time
                    }

                result['groups'] = group_speedups
                speedups[model][machine][ray_type] = result

    results_dir = os.path.dirname(
        filename) if os.path.dirname(filename) else '.'
    name = os.path.splitext(os.path.basename(filename))[0]
    speedup_file = os.path.join(
        results_dir, f'{name}-{mean_strategy}.speedups.txt')
    save_speedups_to_file(speedups, speedup_file, layout_groups)


def save_speedups_to_file(speedups, output_path, layout_groups):
    with open(output_path, 'w') as f:
        for model in sorted(speedups.keys()):
            f.write(f"\n{'='*90}\n")
            f.write(f"Model: {model.upper()}\n")
            f.write(f"{'='*90}\n")

            for machine in sorted(speedups[model].keys()):
                f.write(f"\nMachine: {machine.upper()}\n")
                f.write(f"{'-'*90}\n")

                for ray_type in sorted(speedups[model][machine].keys()):
                    ray_type_data = speedups[model][machine][ray_type]
                    overall = ray_type_data['overall']

                    f.write(f"\nRay Type: {ray_type.upper()}\n")
                    f.write(
                        f"Overall Baseline (slowest): {overall['slowest_layout']} @ {overall['slowest_time']*1e6:.3f} ns/ray\n")

                    if layout_groups and 'groups' in ray_type_data:
                        for group_name, _ in layout_groups.items():
                            if group_name not in ray_type_data['groups']:
                                continue

                            group_info = ray_type_data['groups'][group_name]
                            group_speedups = group_info['speedups']
                            group_memories = group_info['memories']
                            group_slowest = group_info['slowest_layout']
                            group_slowest_time = group_info['slowest_time']

                            f.write(
                                f"\n{group_name.upper()} Layouts (within-group speedups):\n")
                            f.write(
                                f"  Group Baseline: {group_slowest} @ {group_slowest_time*1e6:.3f} ns/ray\n")
                            f.write(
                                f"  {'Layout':<30} {'Speedup':>10} {'Time (ns)':>12} {'Memory (KB)':>12} {'vs Overall':>12}\n")
                            f.write(
                                f"  {'-'*30} {'-'*10} {'-'*12} {'-'*12} {'-'*12}\n")
                            sorted_items = sorted(
                                group_speedups.items(), key=lambda x: x[1], reverse=True)

                            for layout, speedup in sorted_items:
                                time_ns = (group_slowest_time / speedup) * 1e6
                                overall_speedup = overall['speedups'][layout]
                                memory_mb = group_memories.get(
                                    layout, 0) / 1024
                                marker = " *" if layout == group_slowest else ""
                                f.write(
                                    f"  {layout:<30} {speedup:>10.3f}x {time_ns:>11.2f} {memory_mb:>11.1f} {overall_speedup:>11.3f}x{marker}\n")
                    else:
                        f.write(
                            f"\n  {'Layout':<30} {'Speedup':>10} {'Time (ns)':>12} {'Memory (KB)':>12}\n")
                        f.write(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*12}\n")

                        sorted_items = sorted(
                            overall['speedups'].items(), key=lambda x: x[1], reverse=True)

                        for layout, speedup in sorted_items:
                            time_ns = (overall['slowest_time'] / speedup) * 1e6
                            memory_mb = overall['memories'].get(
                                layout, 0) / 1024
                            marker = " *" if layout == overall['slowest_layout'] else ""
                            f.write(
                                f"  {layout:<30} {speedup:>10.3f}x {time_ns:>11.2f} {memory_mb:>11.1f}{marker}\n")

        f.write(f"\n{'='*90}\n")
        f.write("* = baseline (slowest layout in group)\n")

    print(f"Speedups saved to: {output_path}")


def parse_layout_groups(group_str):
    if not group_str:
        return {}

    groups = {}
    for group_def in group_str.split(';'):
        group_def = group_def.strip()
        if ':' not in group_def:
            continue
        group_name, layouts_str = group_def.split(':', 1)
        layouts = [l.strip() for l in layouts_str.split(',')]
        groups[group_name.strip()] = layouts

    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Pareto frontier plots from benchmark CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python3 collect_trace.py results.csv
  
  # Specify mean strategy
  python3 collect_trace.py results.csv --mean wavg
  
  # Filter by machine and ray type
  python3 collect_trace.py results.csv --machines x86 cuda --ray-types primary
  
  # Filter by specific scenes
  python3 collect_trace.py results.csv --scenes lucy hairball
  
  # Filter by specific layouts
  python3 collect_trace.py results.csv --layouts bvh8 bvh8-q8 pbrt
  
  # Define custom layout groups
  python3 collect_trace.py results.csv --layout-groups "bvh8:bvh8,bvh8-q8;bvh2:pbrt,sg-eq"
  
  # Combine multiple filters
  python3 collect_trace.py results.csv --machines x86 --ray-types primary --mean geo --scenes lucy
        """
    )

    parser.add_argument(
        'csv_file', help='Path to CSV file with benchmark data')
    parser.add_argument('--mean', choices=['wavg', 'geo', 'arithmetic'], default='wavg',
                        help='Mean strategy to use (default: wavg)')
    parser.add_argument('--machines', nargs='+',
                        help='Filter by machine types (e.g., x86 cuda)')
    parser.add_argument('--ray-types', nargs='+',
                        help='Filter by ray types (e.g., primary secondary)')
    parser.add_argument('--scenes', nargs='+',
                        help='Filter by scene names (e.g., lucy hairball)')
    parser.add_argument('--layouts', nargs='+',
                        help='Filter by layout names (e.g., bvh8 pbrt)')
    parser.add_argument('--include-embree',
                        action='store_true',
                        help='Include Embree in the results')
    parser.add_argument('--layout-groups', type=str,
                        help='Define layout groups as "group1:layout1,layout2;group2:layout3,layout4"')
    parser.add_argument('--label-dominated', action='store_true',
                        help='Label dominated points on the plot')
    parser.add_argument('--memory-type', choices=['bvh', 'total'], default='total',
                        help='Memory type to use: bvh (excludes primitives) or total (includes primitives) (default: total)')

    args = parser.parse_args()

    if args.layout_groups:
        layout_groups = parse_layout_groups(args.layout_groups)
    else:
        layout_groups = layout_grouping.retrieve_layout_groups()
    if not args.include_embree:
        del layout_groups['embree']

    all_group_layouts = set()
    for layouts in layout_groups.values():
        all_group_layouts.update(layouts)

    layouts_filter = args.layouts if args.layouts else list(all_group_layouts)

    memory_column = 'bvh-memory-b' if args.memory_type == 'bvh' else 'total-memory-b'

    raw_data, machine_type, ray_type = load_csv_data(
        args.csv_file,
        machines=args.machines,
        ray_types=args.ray_types,
        scenes=args.scenes,
        layouts=layouts_filter
    )
    memory_utilization = load_memory_data(
        args.csv_file,
        scenes=args.scenes,
        layouts=layouts_filter,
        machines=args.machines,
        ray_types=args.ray_types,
        memory_column=memory_column
    )

    ray_count_range = None
    trace_data = process_trace_data(
        raw_data, mean_strategy=args.mean, ray_count_range=None)

    print(f"Models found: {list(trace_data.keys())}")
    for model in trace_data:
        print(f"\n{model}:")
        for machine in trace_data[model]:
            for ray_type_key in trace_data[model][machine]:
                layouts = list(trace_data[model][machine][ray_type_key].keys())
                print(f"  {machine}/{ray_type_key}: {layouts}")

    calculate_speedups(
        trace_data,
        memory_utilization,
        layout_groups,
        args.csv_file,
        args.mean,
        args.machines,
        args.ray_types
    )

    plot_pareto_frontiers(
        trace_data,
        memory_utilization,
        layout_groups,
        args.csv_file,
        machine_type,
        args.mean,
        ray_type,
        ray_count_range,
        args.label_dominated,
        args.memory_type,
        args.machines,
        args.ray_types
    )
