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

    # Track ray count sequence for each layout to handle multiple runs
    ray_count_sequence = []
    current_run_index = 0

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if it's a model name (no commas, no colons, not a number)
        if ',' not in line and ':' not in line and not line.isdigit() and line != '---':
            current_model = line
            current_layout = None  # Reset layout when new model starts
            ray_count_sequence = []
            current_run_index = 0
            i += 1
            continue

        # Check if it's a configuration line (rt, cpu, layout)
        if ',' in line:
            config_parts = [part.strip() for part in line.split(',')]
            if len(config_parts) >= 3:
                if machine_type is None and len(config_parts) >= 2:
                    machine_type = config_parts[1]

                # Check if we're starting a new layout or continuing runs of the same layout
                new_layout = config_parts[2]
                if new_layout != current_layout:
                    current_layout = new_layout
                    ray_count_sequence = []
                    current_run_index = 0
                else:
                    # Same layout, this is another run - reset to beginning of sequence
                    current_run_index = 0
            i += 1
            continue

        # Check if it's a ray count
        if line.isdigit():
            current_ray_count = int(line)

            # Track the sequence of ray counts for this layout
            if current_run_index < len(ray_count_sequence):
                # We're in a subsequent run, verify ray count matches expected sequence
                if ray_count_sequence[current_run_index] != current_ray_count:
                    print(
                        f"Warning: Unexpected ray count {current_ray_count} at position {current_run_index}")
            else:
                # First run for this layout, build the sequence
                ray_count_sequence.append(current_ray_count)

            current_run_index += 1
            i += 1
            continue

        # Check if it's a trace time measurement
        if ':' in line and 'trace time' in line.lower():
            # Extract the time value
            time_match = re.search(r'(\d+)\s*ms', line)
            if time_match and current_model and current_layout and current_ray_count:
                time_value = int(time_match.group(1))
                parsed_data[current_model][current_layout][current_ray_count].append(
                    time_value)
            i += 1
            continue

        # Handle separator lines
        if line == '---':
            i += 1
            continue

        i += 1

    return parsed_data, machine_type


def calculate_average(values, method='arithmetic'):
    """Calculate average with outlier removal."""
    if not values:
        return 0

    if len(values) <= 4:
        filtered_values = values
    else:
        sorted_values = sorted(values)
        # Drop lowest 2 and highest 2
        filtered_values = sorted_values[2:-2]

    if not filtered_values:
        filtered_values = values

    if method == 'geometric':
        # For geometric mean, skip zero values
        non_zero_values = [v for v in filtered_values if v > 0]
        if not non_zero_values:
            return 0
        return math.exp(sum(math.log(v) for v in non_zero_values) / len(non_zero_values))
    else:
        return sum(filtered_values) / len(filtered_values)


def process_trace_data(raw_data, blacklist, method):
    """Process raw data to compute averages."""
    processed_data = defaultdict(lambda: defaultdict(dict))

    for model in raw_data:
        for layout in raw_data[model]:
            if blacklist is not None and layout == blacklist:
                continue
            for ray_count in raw_data[model][layout]:
                values = raw_data[model][layout][ray_count]
                avg_value = calculate_average(values, method)
                processed_data[model][layout][ray_count] = avg_value

    return processed_data


def format_ray_count(ray_count):
    """Format ray count for display."""
    if (ray_count & (ray_count - 1)) == 0:
        exponent = int(math.log2(ray_count))
        return f"$2^{{{exponent}}}$"
    return f"{ray_count:,}"


def parse_layout_memory_and_nodes(data_text):
    """
    Parse layout data to extract memory utilization and node counts per model.

    Returns:
        dict: {model: {layout: {'memory': int, 'nodes': {node_type: count}}}}
    """
    lines = data_text.strip().split('\n')

    # Structure: model -> layout -> {'memory': int, 'nodes': {node_type: count}}
    data = defaultdict(lambda: defaultdict(lambda: {'memory': 0, 'nodes': {}}))

    current_model = None
    current_layout = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and separator
        if not line or line == '---':
            i += 1
            continue

        # Check if it's a model name (no commas, no colons, not a number, not starting with ;;)
        if (',' not in line and ':' not in line and not line.isdigit()
                and not line.startswith(';;') and not line.startswith('./')):
            current_model = line
            current_layout = None
            i += 1
            continue

        # Check if it's a configuration line (rt, cpu, layout)
        if line.startswith('rt, cpu,'):
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 3:
                current_layout = parts[2]
                # Reset for new layout
                if current_model and current_layout:
                    data[current_model][current_layout] = {
                        'memory': 0, 'nodes': {}}
            i += 1
            continue

        # Check if it's a node definition line
        if line.startswith(';;') and current_model and current_layout:
            # Parse node line: ";; node_type: size,count"
            match = re.match(r';;\s*(\S+):\s*(\d+),(\d+)', line)
            if match:
                node_type = match.group(1)
                size = int(match.group(2))
                count = int(match.group(3))

                # Store node count
                data[current_model][current_layout]['nodes'][node_type] = count

                # Add to memory utilization (exclude primitives)
                if node_type != 'primitives':
                    data[current_model][current_layout]['memory'] += size * count

            i += 1
            continue

        # Skip other lines (ray counts, hits, trace time)
        i += 1

    # Convert defaultdict to regular dict for cleaner output
    result = {}
    for model in data:
        result[model] = {}
        for layout in data[model]:
            if data[model][layout]['memory'] == 0:
                continue
            result[model][layout] = {
                'memory': data[model][layout]['memory'],
                'nodes': dict(data[model][layout]['nodes'])
            }

    return result


def create_scaling_plots(data, machine_type, output_path, baseline_layout, method='arithmetic'):
    """Create speedup plots comparing layouts to baseline."""
    models = sorted(data.keys())
    all_layouts = set()
    for model in data:
        all_layouts.update(data[model].keys())
    layouts = sorted(all_layouts)

    # Generate colors and styles dynamically
    color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#8B5A3C', '#4A90E2',
                     '#6B4C8A', '#E85D75', '#3AA655', '#F4B942', '#D64545']
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    marker_styles = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

    model_colors = {}
    for i, model in enumerate(models):
        model_colors[model] = color_palette[i % len(color_palette)]

    layout_styles = {}
    layout_markers = {}
    for i, layout in enumerate(layouts):
        layout_styles[layout] = line_styles[i % len(line_styles)]
        layout_markers[layout] = marker_styles[i % len(marker_styles)]

    # Create figure with one speedup plot per model
    n_models = len(models)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(12 * n_cols, 8 * n_rows))
    title = f'Layout Speedup vs {baseline_layout.upper()}'
    if machine_type:
        title += f' - {machine_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for idx, model in enumerate(models):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        assert baseline_layout in data[
            model], f"Baseline layout '{baseline_layout}' not found for model '{model}'"
        baseline_data = data[model][baseline_layout]

        for layout in layouts:
            if layout != baseline_layout and layout in data[model]:
                ray_counts = sorted(set(baseline_data.keys())
                                    & set(data[model][layout].keys()))
                speedups = []
                valid_ray_counts = []

                for rc in ray_counts:
                    if baseline_data[rc] > 0 and data[model][layout][rc] > 0:
                        speedup = baseline_data[rc] / data[model][layout][rc]
                        speedups.append(speedup)
                        valid_ray_counts.append(rc)
                    elif data[model][layout][rc] == 0 and baseline_data[rc] == 0:
                        speedups.append(1.0)
                        valid_ray_counts.append(rc)

                if speedups:
                    style = layout_styles[layout]
                    marker = layout_markers[layout]
                    ax.plot(valid_ray_counts, speedups,
                            marker=marker, markersize=6, linewidth=2,
                            linestyle=style, label=layout.upper(), alpha=0.8)

        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xscale('log', base=2)  # Set log scale with base 2
        ax.set_xlabel('Number of Rays')
        ax.set_ylabel(f'Speedup vs {baseline_layout.upper()}')
        ax.set_title(f'{model.title()}')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)

    plt.tight_layout()

    # Save figure
    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(results_dir, exist_ok=True)

    method_suffix = '_geomean' if method == 'geometric' else '_arithmetic'
    output_file = os.path.join(
        results_dir, f'speedup_vs_{baseline_layout}{method_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close()


def plot_pareto_raw(processed_data, memory_data, output_path='.', ray_count=None):
    """
    Plot Pareto frontier of trace time vs memory utilization for each model separately.
    """
    import os
    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for model in sorted(processed_data.keys()):
        print(f"\nProcessing Pareto frontier for model: {model}")

        points = []
        labels = []
        point_metadata = []

        # Collect data points for all layouts in this model
        for layout in processed_data[model]:
            # Get trace time
            if ray_count is not None:
                if ray_count not in processed_data[model][layout]:
                    continue
                avg_time = processed_data[model][layout][ray_count]
            else:
                times = [t for t in processed_data[model]
                         [layout].values() if t > 0]
                if not times:
                    continue
                avg_time = math.exp(sum(math.log(t)
                                    for t in times) / len(times))

            # Get memory utilization
            if model not in memory_data or layout not in memory_data[model]:
                print(
                    f"Warning: Memory data not found for model '{model}', layout '{layout}'")
                continue

            memory = memory_data[model][layout]['memory']

            if avg_time > 0 and memory > 0:
                points.append((memory, avg_time))
                labels.append(layout)
                point_metadata.append({
                    'model': model,
                    'layout': layout,
                    'memory': memory,
                    'time': avg_time
                })

        if len(points) == 0:
            print(f"Error: No valid data points to plot for model '{model}'.")
            continue

        points = np.array(points)
        x = points[:, 0]  # memory
        y = points[:, 1]  # time (ms)

        # Compute Pareto frontier for this model
        # Lower memory AND lower time is better
        # A point dominates another if it has (lower/equal memory AND lower/equal time)
        # with at least one strictly better
        is_pareto = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if not is_pareto[i]:
                continue
            for j in range(len(points)):
                if i == j:
                    continue
                # Point j dominates point i if it has lower/equal memory and lower/equal time
                if (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
                        (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                    is_pareto[i] = False
                    break

        # Create matplotlib figure for this model
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot dominated points
        if np.any(~is_pareto):
            ax.scatter(x[~is_pareto], y[~is_pareto],
                       c='lightgray', s=100, alpha=0.6,
                       label='Dominated Configurations',
                       zorder=1, marker='o')

        # Plot Pareto frontier points
        ax.scatter(x[is_pareto], y[is_pareto],
                   c='#A23B72', s=200,
                   edgecolors='black', linewidth=2.5,
                   label='Pareto Frontier',
                   zorder=3, marker='o')

        # Sort Pareto points by memory for connecting line
        pareto_indices = np.where(is_pareto)[0]
        dominated_indices = np.where(~is_pareto)[0]
        pareto_sorted = sorted(pareto_indices, key=lambda i: x[i])

        # Draw line connecting Pareto points
        if len(pareto_sorted) > 1:
            pareto_x = [x[i] for i in pareto_sorted]
            pareto_y = [y[i] for i in pareto_sorted]
            ax.plot(pareto_x, pareto_y,
                    'k--', alpha=0.4, linewidth=2,
                    zorder=2)

        # Annotate all Pareto frontier points with layout names
        for i in pareto_indices:
            layout_name = labels[i].upper()

            # Add text annotation with background box
            ax.annotate(layout_name,
                        xy=(x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 15),
                        ha='center',
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='yellow',
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1.5),
                        zorder=4)

        # Annotate all dominated points with layout names
        for i in dominated_indices:
            layout_name = labels[i].upper()

            # Add text annotation with lighter background box
            ax.annotate(layout_name,
                        xy=(x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 15),
                        ha='center',
                        fontsize=9,
                        fontweight='normal',
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor='lightgray',
                                  alpha=0.6,
                                  edgecolor='gray',
                                  linewidth=1),
                        zorder=2)

        # Set axis labels
        ax.set_xlabel("Memory Utilization (bytes, excluding primitives)",
                      fontsize=13, fontweight='bold')
        ax.set_ylabel("Trace Time (ms)",
                      fontsize=13, fontweight='bold')

        # Set title
        title = f"{model.title()} - Trace Time vs Memory (Pareto Frontier)"
        if ray_count is not None:
            title += f"\nRay Count: {ray_count}"
        else:
            title += "\nGeometric Mean Across All Ray Counts"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Add grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Add legend
        ax.legend(fontsize=12, loc='best', framealpha=0.9)

        # Format x-axis with commas for large numbers
        ax.ticklabel_format(style='plain', axis='x')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        output_path = os.path.dirname(
            output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_path, exist_ok=True)
        if ray_count is not None:
            output_path = os.path.join(
                output_path, f'pareto_time_{model}_rc{ray_count}.png')
        else:
            output_path = os.path.join(
                output_path, f'pareto_time_{model}_geomean.png')

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

        # Store results for this model
        pareto_points = []
        for i in pareto_indices:
            pareto_points.append(point_metadata[i])

        pareto_points.sort(key=lambda p: p['memory'])

        all_results[model] = {
            'pareto_points': pareto_points,
            'total_points': len(points),
            'pareto_count': len(pareto_indices)
        }

    return all_results


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(
            "Usage: python trace_scaling.py <data_file> <baseline-layout> <blacklist-layout>? ")
        sys.exit(1)

    filename = sys.argv[1]
    baseline_layout = sys.argv[2]
    blacklist = sys.argv[3] if len(sys.argv) > 3 else None
    method = 'arithmetic'

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

    # Parse and process data
    raw_data, machine_type = parse_trace_scaling_data(data_text)
    processed_data = process_trace_data(raw_data, blacklist, method)

    # Print summary with details about runs per configuration
    print(f"Found {len(processed_data)} models")
    for model in sorted(processed_data.keys()):
        layouts = list(processed_data[model].keys())
        ray_counts = set()
        for layout in layouts:
            ray_counts.update(processed_data[model][layout].keys())
        print(f"  {model}: {len(layouts)} layouts, {len(ray_counts)} ray counts")

        # Show number of runs for each layout
        for layout in sorted(layouts):
            first_ray_count = min(
                raw_data[model][layout].keys()) if raw_data[model][layout] else None
            if first_ray_count:
                n_runs = len(raw_data[model][layout][first_ray_count])
                print(f"    {layout}: {n_runs} runs per ray count")

    # Generate pareto
    # print(parse_layout_memory_and_nodes(data_text))
    memory_utilization = {'hairball': {'eb-align16': {'memory': 53676368, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'eb': {'memory': 53676368, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'ebq-align16': {'memory': 52921952, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'ebq-cl-align16': {'memory': 30135328, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'ebq-cl-idx-align16': {'memory': 23457216, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'ebq-cl-idx': {'memory': 21766732, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'ebq-cl': {'memory': 28444844, 'nodes': {'primitives': 2880000, 'aabbs': 203452, 'obbs': 5239}}, 'bvh8-align16': {'memory': 53653504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'bvh8': {'memory': 53653504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-align16': {'memory': 30180096, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8': {'memory': 28503424, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'eq': {'memory': 7377468, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt-align16': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'soa-align16': {'memory': 29509872, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}, 'soa-align32': {'memory': 39346496, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}, 'soa': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}}, 'white-oak': {'eb-align16': {'memory': 681312, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'eb': {'memory': 681312, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'ebq-align16': {'memory': 667200, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'ebq-cl-align16': {'memory': 382160, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'ebq-cl-idx-align16': {'memory': 297584, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'ebq-cl-idx': {'memory': 276048, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'ebq-cl': {'memory': 360624, 'nodes': {'primitives': 36760, 'aabbs': 2545, 'obbs': 98}}, 'bvh8-align16': {'memory': 679936, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'bvh8': {'memory': 679936, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-align16': {'memory': 382464, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8': {'memory': 361216, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'eq': {'memory': 95556, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt-align16': {'memory': 254816, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt': {'memory': 254816, 'nodes': {'primitives': 36760, 'nodes': 7963}},  'soa-align16': {'memory': 382224, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}, 'soa-align32': {'memory': 509632, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}, 'soa': {'memory': 254816, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}}, 'power-plant': {'eb-align16': {'memory': 263453920, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'eb': {'memory': 263453920, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'ebq-align16': {'memory': 254849344, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'ebq-cl-align16': {'memory': 147535536, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'ebq-cl-idx-align16': {'memory': 114962320, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'ebq-cl-idx': {'memory': 106580000, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'ebq-cl': {'memory': 139153216, 'nodes': {'primitives': 12759246, 'aabbs': 958159, 'obbs': 59754}}, 'bvh8-align16': {'memory': 261846528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'bvh8': {'memory': 261846528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}},
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'cl-bvh8-align16': {'memory': 147288672, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8': {'memory': 139105968, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'eq': {'memory': 32776956, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt-align16': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'soa-align16': {'memory': 131107824, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}, 'soa-align32': {'memory': 174810432, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}, 'soa': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}}, 'sheep': {'eb-align16': {'memory': 56229040, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'eb': {'memory': 56229040, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'ebq-align16': {'memory': 56066464, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'ebq-cl-align16': {'memory': 31616416, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'ebq-cl-idx-align16': {'memory': 24594560, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'ebq-cl-idx': {'memory': 22834580, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'ebq-cl': {'memory': 29856436, 'nodes': {'primitives': 2967664, 'aabbs': 218304, 'obbs': 1129}}, 'bvh8-align16': {'memory': 56198144, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'bvh8': {'memory': 56198144, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'cl-bvh8-align16': {'memory': 31611456, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'cl-bvh8': {'memory': 29855264, 'nodes': {'primitives': 2967664, 'interiors': 219524}}, 'eq': {'memory': 7628724, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'pbrt-align16': {'memory': 20343264, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'pbrt': {'memory': 20343264, 'nodes': {'primitives': 2967664, 'nodes': 635727}}, 'soa-align16': {'memory': 30514896, 'nodes': {'primitives': 2967664, 'aabbs': 635727, 'nodes': 635727}}, 'soa-align32': {'memory': 40686528, 'nodes': {'primitives': 2967664, 'aabbs': 635727, 'nodes': 635727}}, 'soa': {'memory': 20343264, 'nodes': {'primitives': 2967664, 'aabbs': 635727, 'nodes': 635727}}}, 'sponza': {'eb-align16': {'memory': 4969344, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'eb': {'memory': 4969344, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'ebq-align16': {'memory': 4806912, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'ebq-cl-align16': {'memory': 2782848, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'ebq-cl-idx-align16': {'memory': 2168448, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'ebq-cl-idx': {'memory': 2010336, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'ebq-cl': {'memory': 2624736, 'nodes': {'primitives': 262267, 'aabbs': 18072, 'obbs': 1128}}, 'bvh8-align16': {'memory': 4948224, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'bvh8': {'memory': 4948224, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-align16': {'memory': 2783376, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8': {'memory': 2628744, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'eq': {'memory': 672948, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt-align16': {'memory': 1794528, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt': {'memory': 1794528, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'soa-align16': {'memory': 2691792, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}, 'soa-align32': {'memory': 3589056, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}, 'soa': {'memory': 1794528, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}}}
    plot_pareto_raw(processed_data, memory_utilization,
                    filename, ray_count=2 ** 25)
    plot_pareto_raw(processed_data, memory_utilization,
                    filename, ray_count=None)
    # Generate outputs
    create_scaling_plots(processed_data, machine_type,
                         filename, baseline_layout, method)
