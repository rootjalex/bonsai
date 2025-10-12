import matplotlib.pyplot as plt
import numpy as np
import re
import math
import os
from collections import defaultdict
import sys


def parse_trace_scaling_data(data_text):
    """Parse trace time scaling data with multiple runs per configuration."""
    lines = data_text.strip().split('\n')
    parsed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    current_model = None
    current_layout = None
    current_ray_count = None
    machine_type = None
    ray_count_sequence = []
    current_run_index = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Model name
        if ',' not in line and ':' not in line and not line.isdigit() and line != '---':
            current_model = line
            current_layout = None
            ray_count_sequence = []
            current_run_index = 0
            i += 1
            continue

        # Configuration line (rt, cpu, layout)
        if ',' in line:
            config_parts = [part.strip() for part in line.split(',')]
            if len(config_parts) >= 3:
                if machine_type is None and len(config_parts) >= 2:
                    machine_type = config_parts[1]

                new_layout = config_parts[2]
                if new_layout != current_layout:
                    current_layout = new_layout
                    ray_count_sequence = []
                    current_run_index = 0
                else:
                    current_run_index = 0
            i += 1
            continue

        # Ray count
        if line.isdigit():
            current_ray_count = int(line)
            if current_run_index < len(ray_count_sequence):
                if ray_count_sequence[current_run_index] != current_ray_count:
                    print(f"Warning: Unexpected ray count {current_ray_count}")
            else:
                ray_count_sequence.append(current_ray_count)
            current_run_index += 1
            i += 1
            continue

        # Trace time measurement
        if ':' in line and 'trace time' in line.lower():
            time_match = re.search(r'(\d+)\s*ms', line)
            if time_match and current_model and current_layout and current_ray_count:
                time_value = int(time_match.group(1))
                parsed_data[current_model][current_layout][current_ray_count].append(
                    time_value)
            i += 1
            continue

        if line == '---':
            i += 1
            continue

        i += 1

    return parsed_data, machine_type


def calculate_average(values):
    """Calculate average with outlier removal."""
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


def process_trace_data(raw_data, ray_count_range=None):
    """Process raw data to compute averages."""
    processed_data = defaultdict(lambda: defaultdict(dict))
    # geometric mean over all rays, otherwise
    R = (0, sys.maxsize) if ray_count_range is None else ray_count_range

    for model in raw_data:
        for layout in raw_data[model]:
            times = []
            for ray_count in raw_data[model][layout]:
                if not (ray_count >= R[0] and ray_count <= R[1]):
                    continue
                values = raw_data[model][layout][ray_count]
                avg_value = calculate_average(values)
                if avg_value > 0:
                    times.append(avg_value)
            assert times
            geomean = math.exp(sum(math.log(t)
                                   for t in times) / len(times))
            processed_data[model][layout] = geomean

    return processed_data


def process_trace_data_geomean(raw_data):
    return process_trace_data(raw_data, ray_count_range=None)


def format_unit(values, threshold, small_unit, large_unit):
    """Format values with appropriate unit based on threshold."""
    # Check if any value exceeds threshold to determine unit
    if np.any(values >= threshold):
        return values / threshold, large_unit
    return values, small_unit


def plot_pareto_frontiers_geomean(processed_data, memory_data, layout_groups, output_path, machine_type):
    return plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count_range=None)


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count_range=None, label_dominated_points=True):
    """
    Plot Pareto frontiers for multiple layout groups on the same graph for each model.

    Args:
        processed_data: Dict[model][layout] = trace_time
        memory_data: Dict[model][layout] = {'memory': bytes, 'nodes': {...}}
        layout_groups: Dict[group_name] = [list of layouts]
        output_path: Path for output file
        machine_type: Machine type for title
        label_dominated_points: Whether to label dominated points
    """
    models = sorted(processed_data.keys())
    if len(models) == 0:
        return

    # Create figure with one subplot per model
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    # Colorblind-friendly color scheme (based on Wong 2011 palette + extensions)
    # Extended to ensure 10+ distinct colors
    group_colors = ['#0173B2',  # Blue
                    '#DE8F05',  # Orange
                    '#029E73',  # Green
                    '#CC78BC',  # Purple
                    '#CA9161',  # Brown
                    '#949494',  # Gray
                    '#ECE133',  # Yellow
                    '#56B4E9',  # Sky blue
                    '#D55E00',  # Vermillion
                    '#F0E442']  # Light yellow

    # Distinct line styles - each clearly different
    # Using varied dash patterns to ensure distinguishability
    line_styles = [
        '-',                    # Solid
        '--',                   # Dashed
        '-.',                   # Dash-dot
        ':',                    # Dotted
        (0, (5, 2, 1, 2)),     # Dash-dot-dot
        (0, (3, 1, 1, 1)),     # Dense dash-dot
        (0, (5, 5)),           # Long dash
        (0, (1, 1)),           # Very short dash
        (0, (3, 5, 1, 5)),     # Dash-dot with gaps
        (0, (3, 1, 1, 1, 1, 1))  # Multiple dots
    ]

    # Marker styles for additional distinction
    marker_styles = ['o', 's', '^', 'D', 'v', '>', 'p', '*', 'h', 'X']

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

        # Collect points from all groups
        for group_idx, (group_name, layouts) in enumerate(layout_groups.items()):
            group_color = group_colors[group_idx % len(group_colors)]

            for layout in layouts:
                if layout not in processed_data[model]:
                    continue
                if layout not in memory_data[model]:
                    continue

                trace_time = processed_data[model][layout]
                memory = memory_data[model][layout]['memory']

                if trace_time > 0 and memory > 0:
                    all_points.append((memory, trace_time))
                    all_labels.append(layout.upper())
                    all_colors.append(group_color)
                    all_groups.append(group_name)

        if not all_points:
            print(
                f"  Warning: No valid data points for model '{model}' - skipping plot")
            ax.text(0.5, 0.5, f'No valid data for {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        print(f"  Plotting {len(all_points)} data points for {model}")

        points = np.array(all_points)
        x = points[:, 0]  # memory
        y = points[:, 1]  # time

        # Format units: bytes -> MB
        memory_values = x / 1024 / 1024
        memory_unit = 'MB'
        time_values = y
        time_unit = 'ms'

        # Compute Pareto frontier for each group
        for group_idx, (group_name, layouts) in enumerate(layout_groups.items()):
            group_color = group_colors[group_idx % len(group_colors)]
            group_linestyle = line_styles[group_idx % len(line_styles)]
            group_marker = marker_styles[group_idx % len(marker_styles)]

            # Filter points for this group
            group_mask = np.array([g == group_name for g in all_groups])
            if not np.any(group_mask):
                continue

            group_points = points[group_mask]
            group_labels = [all_labels[i]
                            for i in range(len(all_labels)) if group_mask[i]]

            # Compute Pareto frontier
            is_pareto = np.ones(len(group_points), dtype=bool)
            for i in range(len(group_points)):
                if not is_pareto[i]:
                    continue
                for j in range(len(group_points)):
                    if i == j:
                        continue
                    # Point j dominates point i if lower/equal memory AND lower/equal time
                    if (group_points[j, 0] <= group_points[i, 0] and
                        group_points[j, 1] <= group_points[i, 1] and
                            (group_points[j, 0] < group_points[i, 0] or group_points[j, 1] < group_points[i, 1])):
                        is_pareto[i] = False
                        break

            # Get memory and time values for this group (with proper units)
            group_memory = memory_values[group_mask]
            group_time = time_values[group_mask]

            # Plot dominated points
            if label_dominated_points and np.any(~is_pareto):
                ax.scatter(group_memory[~is_pareto], group_time[~is_pareto],
                           c='lightgray', s=80, alpha=0.5,
                           marker='o', edgecolors='gray', linewidth=1, zorder=1)

            # Plot Pareto frontier points
            ax.scatter(group_memory[is_pareto], group_time[is_pareto],
                       c=group_color, s=180, alpha=0.9,
                       edgecolors='black', linewidth=2.5,
                       marker=group_marker, label=group_name, zorder=3)

            # Connect Pareto points
            pareto_indices = np.where(is_pareto)[0]
            if len(pareto_indices) > 1:
                pareto_sorted = sorted(
                    pareto_indices, key=lambda i: group_memory[i])
                pareto_x = [group_memory[i] for i in pareto_sorted]
                pareto_y = [group_time[i] for i in pareto_sorted]
                ax.plot(pareto_x, pareto_y, color=group_color,
                        linestyle=group_linestyle, alpha=0.8, linewidth=3, zorder=2)

            # Annotate Pareto frontier points
            for i in pareto_indices:
                ax.annotate(group_labels[i],
                            xy=(group_memory[i], group_time[i]),
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

            # Optionally annotate dominated (non-Pareto) points
            if label_dominated_points and np.any(~is_pareto):
                dominated_indices = np.where(~is_pareto)[0]
                for i in dominated_indices:
                    ax.annotate(group_labels[i],
                                xy=(group_memory[i], group_time[i]),
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
        # Formatting
        ax.set_xlabel(
            f'Memory Utilization ({memory_unit})', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'Trace Time ({time_unit})',
                      fontweight='bold', fontsize=11)
        ax.set_title(f'{model.title()}', fontweight='bold', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=9, loc='best', framealpha=0.9,
                  edgecolor='black', fancybox=False, shadow=True)

        # Format axis with commas for large numbers
        ax.ticklabel_format(style='plain')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, p: f'{y:,.1f}'))

    # Hide extra subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    # Overall title
    title = 'Pareto Frontiers: Trace Time vs Memory Uitlization'
    if ray_count_range is not None:
        title += f'\nGeometric Mean (Ray Counts: {ray_count_range[0]:,} - {ray_count_range[1]:,})'
    else:
        title += '\nGeometric Mean Across All Ray Counts'
    if machine_type:
        title += f' - {machine_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save figure
    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(results_dir, exist_ok=True)
    name = filename.split('/')[-1].split('.')[0]
    if ray_count_range is not None:
        def log2(x): return int(math.log2(x))
        output_file = os.path.join(
            results_dir, f'{name}-geomean{log2(ray_count_range[0])}-{log2(ray_count_range[1])}.pdf')
    else:
        output_file = os.path.join(
            results_dir, f'{name}-geomean.pdf')

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pareto_plotter.py <data_file> [ray_count]")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, 'r') as file:
            data_text = file.read()
        print(f"Successfully loaded data from: {filename}\n")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

    # Parse data
    raw_data, machine_type = parse_trace_scaling_data(data_text)
    memory_utilization = {'sheep': {'bvh8-align16': {'memory': 153480896, 'nodes': {'primitives': 2967664, 'interiors': 182207}}, 'bvh8': {'memory': 153480896, 'nodes': {'primitives': 2967664, 'interiors': 182207}}, 'cl-bvh8-align16': {'memory': 133073712, 'nodes': {'primitives': 2967664, 'interiors': 182207}}, 'cl-bvh8-idx-align16': {'memory': 127243088, 'nodes': {'primitives': 2967664, 'interiors': 182207}}, 'cl-bvh8-idx': {'memory': 125785432, 'nodes': {'primitives': 2967664, 'interiors': 182207}}, 'cl-bvh8': {'memory': 131616056, 'nodes': {'primitives': 2967664, 'interiors': 182207}}, 'eq-align16': {'memory': 116034896, 'nodes': {'primitives': 2967664, 'nodes': 574937}}, 'eq': {'memory': 113735148, 'nodes': {'primitives': 2967664, 'nodes': 574937}}, 'pbrt-align16': {'memory': 125233888, 'nodes': {'primitives': 2967664, 'nodes': 574937}}, 'pbrt': {'memory': 125233888, 'nodes': {'primitives': 2967664, 'nodes': 574937}}, 'ptr': {'memory': 132133132, 'nodes': {'primitives': 2967664, 'nodes': 25297228}}, 'soaos-align16': {'memory': 134432880, 'nodes': {'primitives': 2967664, 'aabbs': 574937, 'nodes': 574937}}, 'soaos': {'memory': 125233888, 'nodes': {'primitives': 2967664, 'aabbs': 574937, 'nodes': 574937}}}, 'lucy': {'bvh8-align16': {'memory': 1444637376, 'nodes': {'primitives': 28055728, 'interiors': 1697778}}, 'bvh8': {'memory': 1444637376, 'nodes': {'primitives': 28055728, 'interiors': 1697778}}, 'cl-bvh8-align16': {'memory': 1254486240, 'nodes': {'primitives': 28055728, 'interiors': 1697778}}, 'cl-bvh8': {'memory': 1240904016, 'nodes': {'primitives': 28055728, 'interiors': 1697778}}, 'eq-align16': {'memory': 1090001744, 'nodes': {'primitives': 28055728, 'nodes': 4999721}}, 'eq': {'memory': 1070002860, 'nodes': {'primitives': 28055728, 'nodes': 4999721}}, 'pbrt-align16': {'memory': 1169997280, 'nodes': {'primitives': 28055728, 'nodes': 4999721}}, 'pbrt': {'memory': 1169997280, 'nodes': {'primitives': 28055728, 'nodes': 4999721}}, 'ptr': {'memory': 1229993932, 'nodes': {'primitives': 28055728, 'nodes': 219987724}}, 'soaos-align16': {'memory': 1249992816, 'nodes': {'primitives': 28055728, 'aabbs': 4999721, 'nodes': 4999721}}, 'soaos': {'memory': 1169997280, 'nodes': {'primitives': 28055728, 'aabbs': 4999721, 'nodes': 4999721}}}, 'san-miguel-x35-y22-z47': {'bvh8-align16': {'memory': 507870560, 'nodes': {'primitives': 9832024, 'interiors': 601241}}, 'bvh8': {'memory': 507870560, 'nodes': {'primitives': 9832024, 'interiors': 601241}}, 'cl-bvh8-align16': {'memory': 440531568, 'nodes': {'primitives': 9832024, 'interiors': 601241}}, 'cl-bvh8-idx-align16': {'memory': 421291856, 'nodes': {'primitives': 9832024, 'interiors': 601241}}, 'cl-bvh8-idx': {'memory': 416481928, 'nodes': {'primitives': 9832024, 'interiors': 601241}}, 'cl-bvh8': {'memory': 435721640, 'nodes': {'primitives': 9832024, 'interiors': 601241}}, 'eq-align16': {'memory': 384025936, 'nodes': {'primitives': 9832024, 'nodes': 1879567}}, 'eq': {'memory': 376507668, 'nodes': {'primitives': 9832024, 'nodes': 1879567}}, 'pbrt-align16': {'memory': 414099008, 'nodes': {'primitives': 9832024, 'nodes': 1879567}}, 'pbrt': {'memory': 414099008, 'nodes': {'primitives': 9832024, 'nodes': 1879567}}, 'ptr': {'memory': 436653812, 'nodes': {'primitives': 9832024, 'nodes': 82700948}}, 'soaos-align16': {'memory': 444172080, 'nodes': {'primitives': 9832024, 'aabbs': 1879567, 'nodes': 1879567}}, 'soaos': {'memory': 414099008, 'nodes': {'primitives': 9832024, 'aabbs': 1879567, 'nodes': 1879567}}}, 'hairball': {'bvh8-align16': {'memory': 151729920, 'nodes': {'primitives': 2880000, 'interiors': 187695}}, 'bvh8': {'memory': 151729920, 'nodes': {'primitives': 2880000, 'interiors': 187695}}, 'cl-bvh8-align16': {'memory': 130708080, 'nodes': {'primitives': 2880000, 'interiors': 187695}}, 'cl-bvh8-idx-align16': {'memory': 124701840, 'nodes': {'primitives': 2880000, 'interiors': 187695}}, 'cl-bvh8-idx': {'memory': 123200280, 'nodes': {'primitives': 2880000, 'interiors': 187695}}, 'cl-bvh8': {'memory': 129206520, 'nodes': {'primitives': 2880000, 'interiors': 187695}}, 'eq-align16': {'memory': 112518672,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'nodes': {'primitives': 2880000, 'nodes': 552417}}, 'eq': {'memory': 110309004, 'nodes': {'primitives': 2880000, 'nodes': 552417}}, 'pbrt-align16': {'memory': 121357344, 'nodes': {'primitives': 2880000, 'nodes': 552417}}, 'pbrt': {'memory': 121357344, 'nodes': {'primitives': 2880000, 'nodes': 552417}}, 'ptr': {'memory': 127986348, 'nodes': {'primitives': 2880000, 'nodes': 24306348}}, 'soaos-align16': {'memory': 130196016, 'nodes': {'primitives': 2880000, 'aabbs': 552417, 'nodes': 552417}}, 'soaos': {'memory': 121357344, 'nodes': {'primitives': 2880000, 'aabbs': 552417, 'nodes': 552417}}}, 'white-oak': {'bvh8-align16': {'memory': 1904736, 'nodes': {'primitives': 36760, 'interiors': 2271}}, 'bvh8': {'memory': 1904736, 'nodes': {'primitives': 36760, 'interiors': 2271}}, 'cl-bvh8-align16': {'memory': 1650384, 'nodes': {'primitives': 36760, 'interiors': 2271}}, 'cl-bvh8-idx-align16': {'memory': 1577712, 'nodes': {'primitives': 36760, 'interiors': 2271}}, 'cl-bvh8-idx': {'memory': 1559544, 'nodes': {'primitives': 36760, 'interiors': 2271}}, 'cl-bvh8': {'memory': 1632216, 'nodes': {'primitives': 36760, 'interiors': 2271}}, 'eq-align16': {'memory': 1432592, 'nodes': {'primitives': 36760, 'nodes': 6827}}, 'eq': {'memory': 1405284, 'nodes': {'primitives': 36760, 'nodes': 6827}}, 'pbrt-align16': {'memory': 1541824, 'nodes': {'primitives': 36760, 'nodes': 6827}}, 'pbrt': {'memory': 1541824, 'nodes': {'primitives': 36760, 'nodes': 6827}}, 'ptr': {'memory': 1623748, 'nodes': {'primitives': 36760, 'nodes': 300388}}, 'soaos-align16': {'memory': 1651056, 'nodes': {'primitives': 36760, 'aabbs': 6827, 'nodes': 6827}}, 'soaos': {'memory': 1541824, 'nodes': {'primitives': 36760, 'aabbs': 6827, 'nodes': 6827}}}, 'sponza': {'bvh8-align16': {'memory': 13226316, 'nodes': {'primitives': 262267, 'interiors': 14784}}, 'bvh8': {'memory': 13226316, 'nodes': {'primitives': 262267, 'interiors': 14784}}, 'cl-bvh8-align16': {'memory': 11570508, 'nodes': {'primitives': 262267, 'interiors': 14784}}, 'cl-bvh8-idx-align16': {'memory': 11097420, 'nodes': {'primitives': 262267, 'interiors': 14784}}, 'cl-bvh8-idx': {'memory': 10979148, 'nodes': {'primitives': 262267, 'interiors': 14784}}, 'cl-bvh8': {'memory': 11452236, 'nodes': {'primitives': 262267, 'interiors': 14784}}, 'eq-align16': {'memory': 10185116, 'nodes': {'primitives': 262267, 'nodes': 46469}}, 'eq': {'memory': 9999240, 'nodes': {'primitives': 262267, 'nodes': 46469}}, 'pbrt-align16': {'memory': 10928620, 'nodes': {'primitives': 262267, 'nodes': 46469}}, 'pbrt': {'memory': 10928620, 'nodes': {'primitives': 262267, 'nodes': 46469}}, 'ptr': {'memory': 11486248, 'nodes': {'primitives': 262267, 'nodes': 2044636}}, 'soaos-align16': {'memory': 11672124, 'nodes': {'primitives': 262267, 'aabbs': 46469, 'nodes': 46469}}, 'soaos': {'memory': 10928620, 'nodes': {'primitives': 262267, 'aabbs': 46469, 'nodes': 46469}}}, 'power-plant': {'bvh8-align16': {'memory': 658321656, 'nodes': {'primitives': 12759246, 'interiors': 777300}}, 'bvh8': {'memory': 658321656, 'nodes': {'primitives': 12759246, 'interiors': 777300}}, 'cl-bvh8-align16': {'memory': 571264056, 'nodes': {'primitives': 12759246, 'interiors': 777300}}, 'cl-bvh8-idx-align16': {'memory': 546390456, 'nodes': {'primitives': 12759246, 'interiors': 777300}}, 'cl-bvh8-idx': {'memory': 540172056, 'nodes': {'primitives': 12759246, 'interiors': 777300}}, 'cl-bvh8': {'memory': 565045656, 'nodes': {'primitives': 12759246, 'interiors': 777300}}, 'eq-align16': {'memory': 494524296, 'nodes': {'primitives': 12759246, 'nodes': 2199465}}, 'eq': {'memory': 485726436, 'nodes': {'primitives': 12759246, 'nodes': 2199465}}, 'pbrt-align16': {'memory': 529715736, 'nodes': {'primitives': 12759246, 'nodes': 2199465}}, 'pbrt': {'memory': 529715736, 'nodes': {'primitives': 12759246, 'nodes': 2199465}}, 'ptr': {'memory': 556109316, 'nodes': {'primitives': 12759246, 'nodes': 96776460}}, 'soaos-align16': {'memory': 564907176, 'nodes': {'primitives': 12759246, 'aabbs': 2199465, 'nodes': 2199465}}, 'soaos': {'memory': 529715736, 'nodes': {'primitives': 12759246, 'aabbs': 2199465, 'nodes': 2199465}}}}
    embree_mu = {'lucy': {'embree-bvh8i': {'memory': 1184515489, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 2022579372, 'nodes': {'aabbs': 1}}}, 'sheep': {'embree-bvh8i': {'memory': 85013299, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 170263576, 'nodes': {'aabbs': 1}}}, 'san-miguel-x35-y22-z47': {'embree-bvh8i': {'memory': 374480044, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 675439181, 'nodes': {'aabbs': 1}}}, 'hairball': {'embree-bvh8i': {'memory': 99327410,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 191183716, 'nodes': {'aabbs': 1}}}, 'white-oak': {'embree-bvh8i': {'memory': 1367343, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 2521825, 'nodes': {'aabbs': 1}}}, 'sponza': {'embree-bvh8i': {'memory': 10135535, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 17718837, 'nodes': {'aabbs': 1}}}, 'power-plant': {'embree-bvh8i': {'memory': 458345152, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 833888452, 'nodes': {'aabbs': 1}}}}
    for model, layouts in embree_mu.items():
        assert model in memory_utilization
        # Object exists, add the new layouts
        memory_utilization[model].update(layouts)

    trace_data = process_trace_data_geomean(raw_data)
    # Print what was parsed
    print("\n=== Parsed Data Summary ===")
    print(f"Models found: {list(trace_data.keys())}")
    for model in trace_data:
        layouts = list(trace_data[model].keys())
        print(f"\n{model}:")
        print(f"  Layouts with trace data: {layouts}")
        if model.lower() in {k.lower() for k in memory_utilization.keys()}:
            mem_model = [k for k in memory_utilization.keys()
                         if k.lower() == model.lower()][0]
            print(
                f"  Layouts with memory data: {list(memory_utilization[mem_model].keys())}")
        else:
            print(f"  No memory data found")
    print("\n=========================\n")

    # Define layout groups
    layout_groups = {
        'bvh8': ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16',
                 'cl-bvh8-idx', 'cl-bvh8-idx-align16',
                 # 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16',
                 ],
        'bvh2': ['eq', 'pbrt', 'pbrt-align16', 'eq-align16',
                 'ptr', 'soaos-align16', 'soaos',
                 ],
        'embree': ['embree-bvh8i', 'embree-bvh8v'],
    }

    # Generate plots
    plot_pareto_frontiers_geomean(trace_data, memory_utilization, layout_groups,
                                  filename, machine_type)

    cpu_ray_count_ranges = [
        # (2**19, 2**22),
        (2**18, 2**22),
    ]
    gpu_ray_count_ranges = [
        # (2**20, 2**25),
        (2**22, 2**25),
    ]
    for R in cpu_ray_count_ranges:
        if machine_type != "cpu":
            continue
        trace_data = process_trace_data(raw_data, R)
        plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                              filename, machine_type, R)
    for R in gpu_ray_count_ranges:
        if machine_type != "cuda":
            continue
        trace_data = process_trace_data(raw_data, R)
        plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                              filename, machine_type, R)
