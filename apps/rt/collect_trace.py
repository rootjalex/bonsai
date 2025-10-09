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


def parse_layout_memory_and_nodes(data_text):
    """Parse layout data to extract memory utilization per model."""
    lines = data_text.strip().split('\n')
    data = defaultdict(lambda: defaultdict(lambda: {'memory': 0, 'nodes': {}}))

    current_model = None
    current_layout = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == '---':
            i += 1
            continue

        # Model name
        if (',' not in line and ':' not in line and not line.isdigit()
                and not line.startswith(';;') and not line.startswith('./')):
            current_model = line
            current_layout = None
            i += 1
            continue

        # Configuration line
        if line.startswith('rt, cpu,'):
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 3:
                current_layout = parts[2]
                if current_model and current_layout:
                    data[current_model][current_layout] = {
                        'memory': 0, 'nodes': {}}
            i += 1
            continue

        # Node definition line
        if line.startswith(';;') and current_model and current_layout:
            match = re.match(r';;\s*(\S+):\s*(\d+),(\d+)', line)
            if match:
                node_type = match.group(1)
                size = int(match.group(2))
                count = int(match.group(3))

                data[current_model][current_layout]['nodes'][node_type] = count

                # Add to memory utilization (exclude primitives)
                if node_type != 'primitives':
                    data[current_model][current_layout]['memory'] += size * count
            i += 1
            continue

        i += 1

    # Convert to regular dict
    result = {}
    for model in data:
        result[model] = {}
        for layout in data[model]:
            if data[model][layout]['memory'] > 0:
                result[model][layout] = {
                    'memory': data[model][layout]['memory'],
                    'nodes': dict(data[model][layout]['nodes'])
                }

    return result


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


def process_trace_data(raw_data, ray_count=None):
    """Process raw data to compute averages."""
    processed_data = defaultdict(lambda: defaultdict(dict))

    for model in raw_data:
        for layout in raw_data[model]:
            if ray_count is not None:
                if ray_count in raw_data[model][layout]:
                    values = raw_data[model][layout][ray_count]
                    avg_value = calculate_average(values)
                    processed_data[model][layout] = avg_value
            else:
                # Geometric mean across all ray counts
                times = []
                for rc in raw_data[model][layout]:
                    values = raw_data[model][layout][rc]
                    avg_value = calculate_average(values)
                    if avg_value > 0:
                        times.append(avg_value)

                if times:
                    geomean = math.exp(sum(math.log(t)
                                       for t in times) / len(times))
                    processed_data[model][layout] = geomean

    return processed_data


def format_unit(values, threshold, small_unit, large_unit):
    """Format values with appropriate unit based on threshold."""
    # Check if any value exceeds threshold to determine unit
    if np.any(values >= threshold):
        return values / threshold, large_unit
    return values, small_unit


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, ray_count=None, machine_type=None):
    """
    Plot Pareto frontiers for multiple layout groups on the same graph for each model.

    Args:
        processed_data: Dict[model][layout] = trace_time
        memory_data: Dict[model][layout] = {'memory': bytes, 'nodes': {...}}
        layout_groups: Dict[group_name] = [list of layouts]
        output_path: Path for output file
        ray_count: Specific ray count (None for geometric mean)
        machine_type: Machine type for title
    """
    models = sorted(processed_data.keys())

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
            ax.text(0.5, 0.5, f'No valid data for {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        points = np.array(all_points)
        x = points[:, 0]  # memory
        y = points[:, 1]  # time

        # Format units
        max_memory = np.max(x)
        max_time = np.max(y)

        memory_values, memory_unit = format_unit(x, 10000, 'Bytes', 'MB')
        time_values, time_unit = format_unit(y, 10000, 'ms', 's')

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
            if np.any(~is_pareto):
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

                # Label the Pareto frontier line at its midpoint
                mid_idx = len(pareto_sorted) // 2
                mid_x = pareto_x[mid_idx]
                mid_y = pareto_y[mid_idx]

                # Calculate angle for text rotation (approximate slope)
                if mid_idx > 0 and mid_idx < len(pareto_x) - 1:
                    dx = pareto_x[mid_idx + 1] - pareto_x[mid_idx - 1]
                    dy = pareto_y[mid_idx + 1] - pareto_y[mid_idx - 1]
                    angle = np.degrees(np.arctan2(dy, dx))
                else:
                    angle = 0

                # Add text label along the line
                ax.text(mid_x, mid_y, f'  {group_name}  ',
                        fontsize=9, fontweight='bold',
                        color=group_color,
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor='white',
                                  edgecolor=group_color,
                                  alpha=0.85,
                                  linewidth=2),
                        rotation=angle,
                        ha='center', va='center',
                        zorder=5)

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
    title = 'Pareto Frontiers: Trace Time vs Memory'
    if ray_count is not None:
        title += f'\nRay Count: {ray_count:,}'
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

    if ray_count is not None:
        output_file = os.path.join(
            results_dir, f'pareto_frontiers_rc{ray_count}.png')
    else:
        output_file = os.path.join(
            results_dir, f'pareto_frontiers_geomean.png')

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pareto_plotter.py <data_file> [ray_count]")
        sys.exit(1)

    filename = sys.argv[1]
    ray_count = int(sys.argv[2]) if len(sys.argv) > 2 else None

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
    memory_utilization = {'hairball': {'eb-align16': {'memory': 54336384, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'eb': {'memory': 54336384, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-align16': {'memory': 53450496, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl-align16': {'memory': 30496544, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl-idx-align16': {'memory': 23741408, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl-idx': {'memory': 22028016, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl': {'memory': 28783152, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'bvh8-align16': {'memory': 53653504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'bvh8': {'memory': 53653504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-align16': {'memory': 30180096, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-idx-align16': {'memory': 23473408, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-idx': {'memory': 21796736, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8': {'memory': 28503424, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-cw-bvh8-idx-align16': {'memory': 20120064, 'nodes': {'primitives': 2880000, 'aabbs': 209584}}, 'cl-cw-bvh8-idx': {'memory': 20120064, 'nodes': {'primitives': 2880000, 'aabbs': 209584}}, 'eq-align16': {'memory': 9836624, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'eq': {'memory': 7377468, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt-align16': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'soa-align16': {'memory': 29509872, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}, 'soa': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}}, 'white-oak': {'eb-align16': {'memory': 686896, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'eb': {'memory': 686896, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-align16': {'memory': 661408, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl-align16': {'memory': 384432, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl-idx-align16': {'memory': 299632, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl-idx': {'memory': 277724, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl': {'memory': 362524, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'bvh8-align16': {'memory': 679936, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'bvh8': {'memory': 679936, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-align16': {'memory': 382464, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-idx-align16': {'memory': 297472, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-idx': {'memory': 276224, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8': {'memory': 361216, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-cw-bvh8-idx-align16': {'memory': 254976, 'nodes': {'primitives': 36760, 'aabbs': 2656}}, 'cl-cw-bvh8-idx': {'memory': 254976, 'nodes': {'primitives': 36760, 'aabbs': 2656}}, 'eq-align16': {'memory': 127408, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'eq': {'memory': 95556, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt-align16': {'memory': 254816, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt': {'memory': 254816, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'soa-align16': {'memory': 382224, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}, 'soa': {'memory': 254816, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}}, 'power-plant': {'eb-align16': {'memory': 265928944, 'nodes': {
        'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'eb': {'memory': 265928944, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-align16': {'memory': 244427296, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl-align16': {'memory': 147942544, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl-idx-align16': {'memory': 115597328, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl-idx': {'memory': 106913756, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl': {'memory': 139258972, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'bvh8-align16': {'memory': 261846528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'bvh8': {'memory': 261846528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-align16': {'memory': 147288672, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-idx-align16': {'memory': 114557856, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-idx': {'memory': 106375152, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8': {'memory': 139105968, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-cw-bvh8-idx-align16': {'memory': 98192448, 'nodes': {'primitives': 12759246, 'aabbs': 1022838}}, 'cl-cw-bvh8-idx': {'memory': 98192448, 'nodes': {'primitives': 12759246, 'aabbs': 1022838}}, 'eq-align16': {'memory': 43702608, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'eq': {'memory': 32776956, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt-align16': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'soa-align16': {'memory': 131107824, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}, 'soa': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}}, 'sponza': {'eb-align16': {'memory': 5040464, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'eb': {'memory': 5040464, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-align16': {'memory': 4698464, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl-align16': {'memory': 2809136, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl-idx-align16': {'memory': 2193328, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl-idx': {'memory': 2029876, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl': {'memory': 2645684, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'bvh8-align16': {'memory': 4948224, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'bvh8': {'memory': 4948224, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-align16': {'memory': 2783376, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-idx-align16': {'memory': 2164848, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-idx': {'memory': 2010216, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8': {'memory': 2628744, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-cw-bvh8-idx-align16': {'memory': 1855584, 'nodes': {'primitives': 262267, 'aabbs': 19329}}, 'cl-cw-bvh8-idx': {'memory': 1855584, 'nodes': {'primitives': 262267, 'aabbs': 19329}}, 'eq-align16': {'memory': 897264, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'eq': {'memory': 672948, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt-align16': {'memory': 1794528, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt': {'memory': 1794528, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'soa-align16': {'memory': 2691792, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}, 'soa': {'memory': 1794528, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}}}

    processed_data = process_trace_data(raw_data, ray_count)

    # Define layout groups
    layout_groups = {
        'bvh8': ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16',
                 'cl-bvh8-idx', 'cl-bvh8-idx-align16',
                 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16'],
        'bvh2': ['eq', 'pbrt', 'pbrt-align16', 'eq-align16',
                 'ptr', 'soa-align16', 'soa'],
        # 'mixed-bvh8': ['ebq-align16', 'eb-align16', 'ebq-cl',
        #                 'ebq-cl-align16', 'ebq-cl-idx', 'ebq-cl-idx-align16']
    }

    # Generate plots
    plot_pareto_frontiers(processed_data, memory_utilization, layout_groups,
                          filename, ray_count, machine_type)

    # Also generate with geometric mean if specific ray count was provided
    if ray_count is not None:
        processed_data_geomean = process_trace_data(raw_data, None)
        plot_pareto_frontiers(processed_data_geomean, memory_utilization, layout_groups,
                              filename, None, machine_type)
