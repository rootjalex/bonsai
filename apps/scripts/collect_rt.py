import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
from collections import defaultdict
import sys
import argparse

from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    # "text.latex.preamble": r"\usepackage{amsmath}",  # optional, for Latex math.
})


def load_csv_data(csv_file, machines=None, ray_types=None, scenes=None, layouts=None, intersect=None):
    """
    # Columns:
    # - app:            application, e.g., 'rt'
    # - machine:        architecture, e.g., 'x86'
    # - scene:          model, e.g., 'lucy'
    # - layout:         physical specification, e.g., 'pbrt'
    # - intersect:      ray-triangle intersection method, e.g., 'pc' (Plücker coordinates)
    # - ray_count:      number of rays for this trial
    # - hits:           number of model hits recorded
    # - group:          one of {'bvh2' (system), 'bvh8' (system), 'embree'}
    # - ray_type:       ray distribution, e.g., 'primary'
    # - total_memory_b: total memory utilized
    # - bvh_memory_b:   total BVH memory utilized (minus 36B * # triangles)
    #                   (note: Embree may have negative numbers due to aggressive vertex compression)
    """
    df = pd.read_csv(csv_file)

    if machines is not None:
        df = df[df['machine'].isin(machines)]
    if ray_types is not None:
        df = df[df['ray_type'].isin(ray_types)]
    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if layouts is not None:
        df = df[df['layout'].isin(layouts)]
    if intersect is not None:
        df = df[df['intersect'] == intersect]

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


def load_memory_data(csv_file, scenes=None, layouts=None, machines=None, ray_types=None, memory_column='bvh-memory-b', intersect=None):
    df = pd.read_csv(csv_file)

    if scenes is not None:
        df = df[df['scene'].isin(scenes)]
    if layouts is not None:
        df = df[df['layout'].isin(layouts)]
    if machines is not None:
        df = df[df['machine'].isin(machines)]
    if ray_types is not None:
        df = df[df['ray_type'].isin(ray_types)]
    if intersect is not None:
        df = df[df['intersect'] == intersect]

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


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, mean_strategy, ray_type, ray_count_range, label_dominated_points, memory_type, machines, ray_types, output_filename, remove_legend, remove_title):
    models = sorted(processed_data.keys())
    if len(models) == 0:
        return

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    # Larger figure for bigger fonts
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16 * n_cols, 13 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    # Color palettes for different dimensions
    machine_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    ray_type_colors = ['#CA9161', '#56B4E9']
    layout_colors = ['#D55E00', '#F0E442', '#009E73', '#E69F00',
                     '#56B4E9', '#CC79A7', '#0173B2', '#DE8F05']

    # Marker styles for different dimensions
    machine_markers = ['o', 's', '^', 'D']
    ray_type_markers = ['v', 'p', '*', '>']
    layout_markers = ['h', 'X', '<', 'P', 'd', '8', 'H', 'o']

    # Line styles
    line_styles = ['-', '--', '-.', ':', (0, (5, 2, 1, 2)), (0, (3, 1, 1, 1))]

    # Gray for dominated points
    dominated_color = '#CCCCCC'

    multiple_machines = machines and len(machines) > 1
    multiple_ray_types = ray_types and len(ray_types) > 1
    multiple_layouts = layout_groups and len(layout_groups) > 1

    # Create consistent mappings based on alphabetical order of all possible values
    # This ensures consistency across different subsets of machines/ray_types.
    all_possible_machines = ['arm', 'cuda', 'x86']
    all_possible_ray_types = ['primary', 'secondary']

    machine_to_idx = {m: i for i, m in enumerate(all_possible_machines)}
    ray_type_to_idx = {rt: i for i, rt in enumerate(all_possible_ray_types)}

    # Create layout to index mapping
    layout_list = []
    for group_name, layouts in layout_groups.items():
        layout_list.extend(layouts)
    unique_layouts = sorted(set(layout_list))
    layout_to_idx = {l: i for i, l in enumerate(unique_layouts)}

    for idx, model in enumerate(models):
        ax = axes[idx]

        if model not in memory_data:
            ax.text(0.5, 0.5, f'No data for {model}',
                    ha='center', va='center', fontsize=32)
            if not remove_title:
                ax.set_title(f'{model}', fontsize=36, fontweight='bold')
            continue

        all_points = []
        all_labels = []
        all_machines = []
        all_ray_types_list = []
        all_layouts = []
        all_groups = []

        for machine in memory_data[model]:
            for ray_type_key in memory_data[model][machine]:
                for group_idx, (group_name, layouts) in enumerate(layout_groups.items()):
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
                            all_labels.append(layout)
                            all_machines.append(machine)
                            all_ray_types_list.append(ray_type_key)
                            all_layouts.append(layout)

                            group_key = group_name
                            if multiple_machines:
                                group_key = f"{machine}-{group_key}"
                            if multiple_ray_types:
                                group_key = f"{ray_type_key}"
                            all_groups.append(group_key)

        if not all_points:
            print(
                f"  Warning: No valid data points for model '{model}' - skipping plot")
            ax.text(0.5, 0.5, f'No valid data for {model}',
                    ha='center', va='center', fontsize=32)
            if not remove_title:
                ax.set_title(f'{model}', fontsize=36, fontweight='bold')
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
        labels_to_add = []
        for group_idx, group_key in enumerate(unique_groups):
            # Determine color based on dominant factor.
            if multiple_machines and multiple_ray_types and multiple_layouts:
                # Use machine color as primary, ray type for marker, layout for line style.
                machine_name = group_key.split('-')[0]
                color = machine_colors[machine_to_idx[machine_name] % len(
                    machine_colors)]
                ray_type_str = group_key.split('-')[-1]
                marker = ray_type_markers[ray_type_to_idx[ray_type_str] % len(
                    ray_type_markers)]
                linestyle = line_styles[group_idx % len(line_styles)]
            elif multiple_machines and multiple_ray_types:
                # Machine determines color, ray type determines marker.
                machine_name = group_key.split('-')[0]
                color = machine_colors[machine_to_idx[machine_name] % len(
                    machine_colors)]
                ray_type_str = group_key.split('-')[-1]
                marker = ray_type_markers[ray_type_to_idx[ray_type_str] % len(
                    ray_type_markers)]
                linestyle = line_styles[group_idx % len(line_styles)]
            elif multiple_machines:
                # Machine determines both color and marker.
                machine_name = group_key.split('-')[0]
                color = machine_colors[machine_to_idx[machine_name] % len(
                    machine_colors)]
                marker = machine_markers[machine_to_idx[machine_name] % len(
                    machine_markers)]
                linestyle = line_styles[group_idx % len(line_styles)]
            elif multiple_ray_types:
                # Ray type determines both color and marker.
                ray_type_str = group_key.split('-')[-1]
                color = ray_type_colors[ray_type_to_idx[ray_type_str] % len(
                    ray_type_colors)]
                marker = ray_type_markers[ray_type_to_idx[ray_type_str] % len(
                    ray_type_markers)]
                linestyle = line_styles[group_idx % len(line_styles)]
            else:
                # Layout determines color and marker.
                color = layout_colors[group_idx % len(layout_colors)]
                marker = layout_markers[group_idx % len(layout_markers)]
                linestyle = line_styles[group_idx % len(line_styles)]
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
            # Plot dominated points in gray only if requested.
            if label_dominated_points and np.any(~is_pareto):
                ax.scatter(group_memory[~is_pareto], group_time_per_ray[~is_pareto],
                           c=dominated_color, s=180, alpha=0.6,
                           marker='o', edgecolors='gray', linewidth=2.5, zorder=6)
            # Plot Pareto-optimal points with their assigned colors/markers.
            ax.scatter(group_memory[is_pareto], group_time_per_ray[is_pareto],
                       c=color, s=180, alpha=0.9,
                       edgecolors='black', linewidth=3.5,
                       marker=marker, label=group_key, zorder=3)

            pareto_indices = np.where(is_pareto)[0]
            if len(pareto_indices) > 1:
                pareto_sorted = sorted(
                    pareto_indices, key=lambda i: group_memory[i])
                pareto_x = [group_memory[i] for i in pareto_sorted]
                pareto_y = [group_time_per_ray[i] for i in pareto_sorted]
                ax.plot(pareto_x, pareto_y, color=color,
                        linestyle=linestyle, alpha=0.8, linewidth=5, zorder=2)

            for i in pareto_indices:
                labels_to_add.append({
                    'x': group_memory[i],
                    'y': group_time_per_ray[i],
                    'label': group_labels[i],
                    'is_pareto': True
                })

            # Only add dominated point labels if requested.
            if label_dominated_points and np.any(~is_pareto):
                dominated_indices = np.where(~is_pareto)[0]
                for i in dominated_indices:
                    labels_to_add.append({
                        'x': group_memory[i],
                        'y': group_time_per_ray[i],
                        'label': group_labels[i],
                        'is_pareto': False
                    })
        memory_label = 'Memory Utilization (excluding primitives)' if memory_type == 'bvh' else 'Memory Utilization'
        ax.set_xlabel(
            rf"\textbf{{{memory_label} ({memory_unit})}}", fontweight='bold', fontsize=32)
        ax.tick_params(axis='x', labelsize=28)
        ax.tick_params(axis='y', labelsize=28)
        ax.set_ylabel(rf"\textbf{{Latency ({time_unit})}}",
                      fontweight='bold', fontsize=32)
        if not remove_title:
            ax.set_title(rf"\textbf{{{model}}}",
                         fontweight='bold', fontsize=36)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        if not remove_legend:
            ax.legend(fontsize=26, loc='best', framealpha=0.9,
                      edgecolor='black', fancybox=False, shadow=True)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, p: f'{y:,.1f}'))

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        x_padding = x_range * 0.025
        y_padding = y_range * 0.025
        ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
        ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

        for label_info in labels_to_add:
            x_pos = label_info['x']
            y_pos = label_info['y']

            if label_info['is_pareto']:
                ax.annotate(
                    label_info['label'],
                    xy=(x_pos, y_pos),
                    textcoords="offset pixels",
                    xytext=(-20, 45),
                    ha='left',
                    va='center',
                    fontsize=32,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='#C8FACD',
                        edgecolor='black',
                        linewidth=2,
                        alpha=0.9
                    )
                )
            else:
                ax.annotate(
                    label_info['label'],
                    xy=(x_pos, y_pos),
                    textcoords="offset pixels",
                    xytext=(-20, -45),
                    ha='left',
                    va='center',
                    fontsize=38,
                    fontweight='normal',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='#EEEEEE',
                        edgecolor='gray',
                        linewidth=1.5,
                        alpha=0.8
                    )
                )

    plt.tight_layout()

    if output_filename:
        output_file = f'{output_filename}.pdf'
    else:
        base_csv = os.path.splitext(os.path.basename(output_path))[0]
        output_file = f'{base_csv}_{mean_strategy}_pareto.pdf'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()


def calculate_speedups(trace_data, memory_data, layout_groups, csv_file, mean_strategy, machines=None, ray_types=None, output_filename=None):
    """Additional script for optionally calculating speedups between layouts."""
    speedups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model in trace_data:
        for machine in trace_data[model]:
            for ray_type in trace_data[model][machine]:
                if machine not in memory_data[model]:
                    continue
                if ray_type not in memory_data[model][machine]:
                    continue

                layouts_data = trace_data[model][machine][ray_type]

                if not layouts_data:
                    continue

                slowest_layout = max(layouts_data, key=layouts_data.get)
                slowest_time = layouts_data[slowest_layout]

                overall_speedups = {}
                overall_memories = {}

                for layout, time_per_ray in layouts_data.items():
                    speedup = slowest_time / time_per_ray
                    overall_speedups[layout] = speedup

                    if layout in memory_data[model][machine][ray_type]:
                        memory = memory_data[model][machine][ray_type][layout]['memory']
                        overall_memories[layout] = memory

                speedups[model][machine][ray_type]['overall'] = {
                    'slowest_layout': slowest_layout,
                    'slowest_time': slowest_time,
                    'speedups': overall_speedups,
                    'memories': overall_memories
                }

                if layout_groups:
                    group_speedups_data = {}

                    for group_name, group_layouts in layout_groups.items():
                        group_times = {
                            layout: layouts_data[layout]
                            for layout in group_layouts if layout in layouts_data
                        }

                        if not group_times:
                            continue

                        group_slowest_layout = max(
                            group_times, key=group_times.get)
                        group_slowest_time = group_times[group_slowest_layout]

                        group_speedups = {}
                        group_memories = {}

                        for layout, time_per_ray in group_times.items():
                            speedup = group_slowest_time / time_per_ray
                            group_speedups[layout] = speedup

                            if layout in memory_data[model][machine][ray_type]:
                                memory = memory_data[model][machine][ray_type][layout]['memory']
                                group_memories[layout] = memory

                        group_speedups_data[group_name] = {
                            'slowest_layout': group_slowest_layout,
                            'slowest_time': group_slowest_time,
                            'speedups': group_speedups,
                            'memories': group_memories
                        }

                    speedups[model][machine][ray_type]['groups'] = group_speedups_data

    if output_filename:
        output_path = f'{output_filename}_speedups.txt'
    else:
        base_csv = os.path.splitext(os.path.basename(csv_file))[0]
        output_path = f'{base_csv}_{mean_strategy}_speedups.txt'

    with open(output_path, 'w') as f:
        f.write(f"SPEEDUP ANALYSIS\n")
        f.write(f"Mean Strategy: {mean_strategy}\n")
        f.write(f"{'='*90}\n\n")

        for model in sorted(speedups.keys()):
            f.write(f"Model: {model}\n")
            f.write(f"{'='*90}\n")

            for machine in sorted(speedups[model].keys()):
                f.write(f"\nMachine: {machine}\n")
                f.write(f"{'-'*90}\n")

                for ray_type in sorted(speedups[model][machine].keys()):
                    ray_type_data = speedups[model][machine][ray_type]
                    overall = ray_type_data['overall']

                    f.write(f"\nRay Type: {ray_type}\n")
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
                                f"\n{group_name} Layouts (within-group speedups):\n")
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


def parse_layout_groups(s: str):
    """Useful when trying to parse a subset of layouts."""
    if not s:
        return {}

    groups = {}
    for d in s.split(';'):
        d = d.strip()
        if ':' not in d:
            continue
        name, layouts_str = d.split(':', 1)
        layouts = [l.strip() for l in layouts_str.split(',')]
        groups[name.strip()] = layouts
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Pareto frontier plots from benchmark CSV data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python3 collect_rt.py results.csv
  
  # Specify mean strategy.
  python3 collect_rt.py results.csv --mean wavg
  
  # Filter by machine and ray type.
  python3 collect_rt.py results.csv --machines x86 cuda --ray-types primary
  
  # Filter by specific scenes.
  python3 collect_rt.py results.csv --scenes lucy hairball
  
  # Filter by specific layouts.
  python3 collect_rt.py results.csv --layouts bvh8 bvh8-q8 pbrt
  
  # Define custom layout groups.
  python3 collect_rt.py results.csv --layout-groups "bvh8:bvh8,bvh8-q8;bvh2:pbrt,sg-eq"
  
  # Combine multiple filters.
  python3 collect_rt.py results.csv --machines x86 --ray-types primary --mean geo --scenes lucy

  # (complex example) script for collecting data-dependent example in the Evaluation:
  python3 collect_rt.py rt-results.csv \
    --machines cuda --ray-types primary \
    --scenes lucy power-plant \
    --layout-groups "bvh2:pbrt-q16,pbrt-q16-soaos,pbrt,sg-eq" \
    --label-dominated \
    --output_filename data-dependent1 \
    --remove-legend
""")

    parser.add_argument(
        'csv_file', help='Path to CSV file with benchmark data')
    parser.add_argument('--mean', choices=['wavg', 'geo', 'arithmetic'], default='wavg',
                        help='Mean strategy to use (default: weighted average)')
    parser.add_argument('--machines', nargs='+',
                        help='Filter by machine types (e.g., x86 cuda arm)')
    parser.add_argument('--ray-types', nargs='+',
                        help='Filter by ray types (e.g., primary secondary)')
    parser.add_argument('--scenes', nargs='+',
                        help='Filter by scene names (e.g., lucy hairball)')
    parser.add_argument('--layouts', nargs='+',
                        help='Filter by layout names or group names (e.g., bvh8 pbrt bvh2)')
    parser.add_argument('--include-embree', action='store_true',
                        help='Include Embree in the results')
    parser.add_argument('--layout-groups', type=str,
                        help='Define layout groups as "group1:layout1,layout2;group2:layout3,layout4"')
    parser.add_argument('--label-dominated', action='store_true',
                        help='Label dominated points on the plot')
    parser.add_argument('--memory-type', choices=['bvh', 'total'], default='total',
                        help='Memory type to use: bvh (excludes primitives) or total (includes primitives) (default: total)')
    parser.add_argument(
        '--output_filename', help='name of file to be saved', default=None)
    parser.add_argument('--remove-legend',
                        action='store_true',
                        help='remove the legend from the plot')
    parser.add_argument('--remove-title',
                        action='store_true',
                        help='remove the title from the plot', default=False)
    parser.add_argument('--intersect', type=str, default='mt',
                        help='Intersect method to filter by (default: mt)')
    args = parser.parse_args()

    if args.layout_groups:
        layout_groups = parse_layout_groups(args.layout_groups)
    else:
        layout_groups = {
            'bvh8': ['bvh8', 'bvh8-align16', 'bvh8-q8',
                     'bvh8-q16', 'bvh8-q16-ci', 'bvh8-q8-align16', 'bvh8-q8-ci-align16',
                     'bvh8-q8-ci', 'bvh8-q16-align16', 'bvh8-q16-ci-align16'],
            'bvh2': ['sg-eq', 'pbrt', 'pbrt-align16', 'sg-eq-align16',
                     'ptr', 'pbrt-soaos', 'pbrt-soaos-align16',
                     'pbrt-q16-soaos', 'pbrt-q16'],
            'embree': ['embree-qbvh8i', 'embree-bvh8i'],
        }
    if 'embree' in layout_groups and not args.include_embree:
        del layout_groups['embree']

    all_group_layouts = set()
    for layouts in layout_groups.values():
        all_group_layouts.update(layouts)

    if args.layouts:
        layouts_filter = set()
        for item in args.layouts:
            if item in layout_groups:
                layouts_filter.update(layout_groups[item])
            else:
                layouts_filter.add(item)
        layouts_filter = list(layouts_filter)
    else:
        layouts_filter = list(all_group_layouts)

    memory_column = 'bvh-memory-b' if args.memory_type == 'bvh' else 'total-memory-b'
    raw_data, machine_type, ray_type = load_csv_data(
        args.csv_file,
        machines=args.machines,
        ray_types=args.ray_types,
        scenes=args.scenes,
        layouts=layouts_filter,
        intersect=args.intersect
    )
    memory_utilization = load_memory_data(
        args.csv_file,
        scenes=args.scenes,
        layouts=layouts_filter,
        machines=args.machines,
        ray_types=args.ray_types,
        memory_column=memory_column,
        intersect=args.intersect
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
        args.ray_types,
        args.output_filename
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
        args.ray_types,
        args.output_filename,
        args.remove_legend,
        args.remove_title
    )
