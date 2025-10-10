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


def process_trace_data_geomean(raw_data):
    return process_trace_data(raw_data, ray_count=None)


def format_unit(values, threshold, small_unit, large_unit):
    """Format values with appropriate unit based on threshold."""
    # Check if any value exceeds threshold to determine unit
    if np.any(values >= threshold):
        return values / threshold, large_unit
    return values, small_unit


def plot_pareto_frontiers_geomean(processed_data, memory_data, layout_groups, output_path, machine_type):
    return plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count=None)


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count=None):
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

        # Format units
        memory_values = x / 10**6
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
    memory_utilization =
    embree_mu = {'lucy': {'embree-bvh4': {'memory': 1890677948, 'nodes': {'aabbs': 1}}, 'embree-bvh8': {'memory': 2022644383, 'nodes': {'aabbs': 1}}}, 'sheep': {'embree-bvh4': {'memory': 165597413, 'nodes': {'aabbs': 1}}, 'embree-bvh8': {'memory': 170256236, 'nodes': {'aabbs': 1}}}, 'san-miguel-x35-y22-z47': {'embree-bvh4': {'memory': 642591490, 'nodes': {'aabbs': 1}}, 'embree-bvh8': {'memory': 675466444, 'nodes': {
        'aabbs': 1}}}, 'hairball': {'embree-bvh4': {'memory': 180888797, 'nodes': {'aabbs': 1}}, 'embree-bvh8': {'memory': 191193153, 'nodes': {'aabbs': 1}}}, 'white-oak': {'embree-bvh4': {'memory': 2398093, 'nodes': {'aabbs': 1}}, 'embree-bvh8': {'memory': 2517630, 'nodes': {'aabbs': 1}}}, 'sponza': {'embree-bvh4': {'memory': 16804478, 'nodes': {'aabbs': 1}}, 'embree-bvh8': {'memory': 17710448, 'nodes': {'aabbs': 1}}}}
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
                 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16',
                 ],
        'bvh2': ['eq', 'pbrt', 'pbrt-align16', 'eq-align16',
                 'ptr', 'soa-align16', 'soa',
                 ],
        'embree': ['embree-bvh4', 'embree-bvh8'],
    }

    # Generate plots
    plot_pareto_frontiers_geomean(trace_data, memory_utilization, layout_groups,
                                  filename, machine_type)

    for ray_count in [int(2**20), int(2**25)]:
        trace_data = process_trace_data(raw_data, ray_count)
        plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                              filename, machine_type, ray_count)
