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


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count=None, label_dominated_points=False):
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
    memory_utilization = {'lucy': {'eq-align16': {'memory': 1101193232, 'nodes': {'primitives': 28055728, 'nodes': 5699189}}, 'eq': {'memory': 1078396476, 'nodes': {'primitives': 28055728, 'nodes': 5699189}}, 'pbrt-align16': {'memory': 1192380256, 'nodes': {'primitives': 28055728, 'nodes': 5699189}}, 'pbrt': {'memory': 1192380256, 'nodes': {'primitives': 28055728, 'nodes': 5699189}}, 'ptr': {'memory': 1260770524, 'nodes': {'primitives': 28055728, 'nodes': 250764316}}, 'soa-align16': {'memory': 1283567280, 'nodes': {'primitives': 28055728, 'aabbs': 5699189, 'nodes': 5699189}}, 'soa': {'memory': 1192380256, 'nodes': {'primitives': 28055728, 'aabbs': 5699189, 'nodes': 5699189}}, 'bvh8-align16': {'memory': 1460297920, 'nodes': {'primitives': 28055728, 'interiors': 1758952}}, 'bvh8': {'memory': 1460297920, 'nodes': {'primitives': 28055728, 'interiors': 1758952}}, 'cl-bvh8-align16': {'memory': 1263295296, 'nodes': {'primitives': 28055728, 'interiors': 1758952}}, 'cl-bvh8-idx-align16': {'memory': 1207008832, 'nodes': {'primitives': 28055728, 'interiors': 1758952}}, 'cl-bvh8-idx': {'memory': 1192937216, 'nodes': {'primitives': 28055728, 'interiors': 1758952}}, 'cl-bvh8': {'memory': 1249223680, 'nodes': {'primitives': 28055728, 'interiors': 1758952}}}, 'sheep': {'eq-align16': {'memory': 117007536, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'eq': {'memory': 114464628, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'pbrt-align16': {'memory': 127179168, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'pbrt': {'memory': 127179168, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'ptr': {'memory': 134807892, 'nodes': {'primitives': 2967664, 'nodes': 27971988}}, 'soa-align16': {'memory': 137350800, 'nodes': {'primitives': 2967664, 'aabbs': 635727, 'nodes': 635727}}, 'soa': {'memory': 127179168, 'nodes': {'primitives': 2967664, 'aabbs': 635727, 'nodes': 635727}}, 'bvh8-align16': {'memory': 163034048, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'bvh8': {'memory': 163034048, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'cl-bvh8-align16': {'memory': 138447360, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'cl-bvh8-idx-align16': {'memory': 131422592, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'cl-bvh8-idx': {'memory': 129666400, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'cl-bvh8': {'memory': 136691168, 'nodes': {'primitives': 2967664, 'interiors': 219524}}}, 'san-miguel-x35-y22-z47': {'eq-align16': {'memory': 388913456, 'nodes': {'primitives': 9832536, 'nodes': 2183885}}, 'eq': {'memory': 380177916, 'nodes': {'primitives': 9832536, 'nodes': 2183885}}, 'pbrt-align16': {'memory': 423855616, 'nodes': {'primitives': 9832536, 'nodes': 2183885}}, 'pbrt': {'memory': 423855616, 'nodes': {'primitives': 9832536, 'nodes': 2183885}}, 'ptr': {'memory': 450062236, 'nodes': {'primitives': 9832536, 'nodes': 96090940}}, 'soa-align16': {'memory': 458797776, 'nodes': {'primitives': 9832536, 'aabbs': 2183885, 'nodes': 2183885}}, 'soa': {'memory': 423855616, 'nodes': {'primitives': 9832536, 'aabbs': 2183885, 'nodes': 2183885}}, 'bvh8-align16': {'memory': 541145952, 'nodes': {'primitives': 9832536, 'interiors': 731151}}, 'bvh8': {'memory': 541145952, 'nodes': {'primitives': 9832536, 'interiors': 731151}}, 'cl-bvh8-align16': {'memory': 459257040, 'nodes': {'primitives': 9832536, 'interiors': 731151}}, 'cl-bvh8-idx-align16': {'memory': 435860208, 'nodes': {'primitives': 9832536, 'interiors': 731151}}, 'cl-bvh8-idx': {'memory': 430011000, 'nodes': {'primitives': 9832536, 'interiors': 731151}}, 'cl-bvh8': {'memory': 453407832, 'nodes': {'primitives': 9832536, 'interiors': 731151}}}, 'hairball': {'eq-align16': {'memory': 113516624, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'eq': {'memory': 111057468, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt-align16': {'memory': 123353248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt': {'memory': 123353248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'ptr': {'memory': 130730716, 'nodes': {'primitives': 2880000, 'nodes': 27050716}}, 'soa-align16': {'memory': 133189872, 'nodes': {'primitives': 2880000, 'aabbs': 614789,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'nodes': 614789}}, 'soa': {'memory': 123353248, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}, 'bvh8-align16': {'memory': 157333504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'bvh8': {'memory': 157333504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-align16': {'memory': 133860096, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-idx-align16': {'memory': 127153408, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-idx': {'memory': 125476736, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8': {'memory': 132183424, 'nodes': {'primitives': 2880000, 'interiors': 209584}}}, 'white-oak': {'eq-align16': {'memory': 1450768, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'eq': {'memory': 1418916, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt-align16': {'memory': 1578176, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt': {'memory': 1578176, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'ptr': {'memory': 1673732, 'nodes': {'primitives': 36760, 'nodes': 350372}}, 'soa-align16': {'memory': 1705584, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}, 'soa': {'memory': 1578176, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}, 'bvh8-align16': {'memory': 2003296, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'bvh8': {'memory': 2003296, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-align16': {'memory': 1705824, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-idx-align16': {'memory': 1620832, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-idx': {'memory': 1599584, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8': {'memory': 1684576, 'nodes': {'primitives': 36760, 'interiors': 2656}}}, 'sponza': {'eq-align16': {'memory': 10338876, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'eq': {'memory': 10114560, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt-align16': {'memory': 11236140, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt': {'memory': 11236140, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'ptr': {'memory': 11909088, 'nodes': {'primitives': 262267, 'nodes': 2467476}}, 'soa-align16': {'memory': 12133404, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}, 'soa': {'memory': 11236140, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}, 'bvh8-align16': {'memory': 14389836, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'bvh8': {'memory': 14389836, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-align16': {'memory': 12224988, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-idx-align16': {'memory': 11606460, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-idx': {'memory': 11451828, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8': {'memory': 12070356, 'nodes': {'primitives': 262267, 'interiors': 19329}}}, 'power-plant': {'eq-align16': {'memory': 503035464, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'eq': {'memory': 492109812, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt-align16': {'memory': 546738072, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt': {'memory': 546738072, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'ptr': {'memory': 579515028, 'nodes': {'primitives': 12759246, 'nodes': 120182172}}, 'soa-align16': {'memory': 590440680, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}, 'soa': {'memory': 546738072, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}, 'bvh8-align16': {'memory': 721179384, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'bvh8': {'memory': 721179384, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-align16': {'memory': 606621528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-idx-align16': {'memory': 573890712, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-idx': {'memory': 565708008, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8': {'memory': 598438824, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}}}
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
                 # 'cl-bvh8-idx', 'cl-bvh8-idx-align16',
                 # 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16',
                 ],
        'bvh2': ['eq', 'pbrt', 'pbrt-align16', 'eq-align16',
                 'ptr', 'soa-align16', 'soa',
                 ],
        'embree': ['embree-bvh4', 'embree-bvh8'],
    }

    # Generate plots
    plot_pareto_frontiers_geomean(trace_data, memory_utilization, layout_groups,
                                  filename, machine_type)

    for ray_count in [int(2**22), int(2**25)]:
        trace_data = process_trace_data(raw_data, ray_count)
        plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                              filename, machine_type, ray_count)
