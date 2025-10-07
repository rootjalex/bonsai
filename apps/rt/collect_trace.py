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


def plot_normalized_performance(processed_data, layouts, baseline_layout, output_path, ray_count=None, machine_type=None, memory_data=None):
    """
    Create bar graphs showing performance of selected layouts normalized to baseline.
    Optionally includes a single memory comparison chart.

    Args:
        processed_data: Processed trace time data
        layouts: List of layout names to compare
        baseline_layout: Layout to normalize against
        output_path: Path for output file
        ray_count: Specific ray count to plot (None for geometric mean across all)
        machine_type: Machine type for title
        memory_data: Optional memory utilization data (from parse_layout_memory_and_nodes)
    """
    models = sorted(processed_data.keys())

    # Validate that baseline is in the layouts list
    if baseline_layout not in layouts:
        print(
            f"Warning: Baseline layout '{baseline_layout}' not in provided layouts list. Adding it.")
        layouts = [baseline_layout] + \
            [l for l in layouts if l != baseline_layout]

    # Determine if we're showing memory
    show_memory = memory_data is not None

    # Create figure with subplots
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    if show_memory:
        # Add one extra row for the single memory chart
        fig = plt.figure(figsize=(8 * n_cols, 6 * (n_rows + 1)))
        gs = fig.add_gridspec(n_rows + 1, n_cols, hspace=0.3, wspace=0.3)
        axes = None  # Not used in memory mode
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]
        gs = None  # Not used in non-memory mode

    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#8B5A3C', '#4A90E2',
              '#6B4C8A', '#E85D75', '#3AA655', '#F4B942', '#D64545']

    for idx, model in enumerate(models):
        if show_memory:
            # Create subplot for this model
            row = idx // n_cols
            col = idx % n_cols
            ax_perf = fig.add_subplot(gs[row, col])
        else:
            ax_perf = axes[idx]

        # Check if baseline exists for this model
        if baseline_layout not in processed_data[model]:
            ax_perf.text(0.5, 0.5, f'No baseline data\nfor {model}',
                         ha='center', va='center', fontsize=12)
            ax_perf.set_title(f'{model.title()}')
            continue

        # Get baseline performance
        if ray_count is not None:
            if ray_count not in processed_data[model][baseline_layout]:
                ax_perf.text(0.5, 0.5, f'Ray count {ray_count}\nnot found for {model}',
                             ha='center', va='center', fontsize=12)
                ax_perf.set_title(f'{model.title()}')
                continue
            baseline_time = processed_data[model][baseline_layout][ray_count]
        else:
            # Geometric mean across all ray counts
            times = [t for t in processed_data[model]
                     [baseline_layout].values() if t > 0]
            if not times:
                ax_perf.text(0.5, 0.5, f'No valid data\nfor {model}',
                             ha='center', va='center', fontsize=12)
                ax_perf.set_title(f'{model.title()}')
                continue
            baseline_time = math.exp(sum(math.log(t)
                                     for t in times) / len(times))

        # Collect normalized performance for each layout
        layout_names = []
        normalized_values = []
        bar_colors = []

        for i, layout in enumerate(layouts):
            if layout not in processed_data[model]:
                continue

            # Get performance for this layout
            if ray_count is not None:
                if ray_count not in processed_data[model][layout]:
                    continue
                layout_time = processed_data[model][layout][ray_count]
            else:
                times = [t for t in processed_data[model]
                         [layout].values() if t > 0]
                if not times:
                    continue
                layout_time = math.exp(sum(math.log(t)
                                       for t in times) / len(times))

            if layout_time > 0 and baseline_time > 0:
                # Normalized performance (baseline / layout, so >1 means faster than baseline)
                normalized = baseline_time / layout_time
                layout_names.append(layout.upper())
                normalized_values.append(normalized)

                # Color baseline differently
                if layout == baseline_layout:
                    bar_colors.append('#808080')  # Gray for baseline
                else:
                    bar_colors.append(colors[i % len(colors)])

        if not layout_names:
            ax_perf.text(0.5, 0.5, f'No valid data\nfor {model}',
                         ha='center', va='center', fontsize=12)
            ax_perf.set_title(f'{model.title()}')
            continue

        # Create performance bar chart
        x_pos = np.arange(len(layout_names))
        bars = ax_perf.bar(x_pos, normalized_values, color=bar_colors,
                           edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, normalized_values):
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                         f'{val:.2f}x',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add baseline reference line
        ax_perf.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                        alpha=0.7, label=f'{baseline_layout.upper()} (baseline)')

        # Formatting for performance chart
        ax_perf.set_xticks(x_pos)
        ax_perf.set_xticklabels(layout_names, rotation=45, ha='right')
        ax_perf.set_ylabel(
            f'Speedup vs {baseline_layout.upper()}', fontweight='bold')
        ax_perf.set_title(f'{model.title()}', fontweight='bold', fontsize=12)
        ax_perf.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax_perf.legend(loc='best', fontsize=9)
        ax_perf.set_ylim(bottom=0)

    # Create single memory bar chart if data available
    if show_memory:
        # Use any model that has memory data (they should all be the same)
        reference_model = None
        for model in models:
            if model in memory_data and baseline_layout in memory_data[model]:
                reference_model = model
                break

        if reference_model:
            # Create memory chart spanning the entire bottom row
            ax_mem = fig.add_subplot(gs[n_rows, :])

            baseline_memory = memory_data[reference_model][baseline_layout]['memory']

            # Collect memory data for layouts
            layout_names = []
            memory_values = []
            bar_colors = []

            for i, layout in enumerate(layouts):
                if layout not in memory_data[reference_model]:
                    continue

                layout_memory = memory_data[reference_model][layout]['memory']

                if baseline_memory > 0 and layout_memory > 0:
                    # Normalize memory (baseline / layout, so >1 means less memory than baseline)
                    memory_norm = baseline_memory / layout_memory
                    layout_names.append(layout.upper())
                    memory_values.append(memory_norm)

                    # Color baseline differently
                    if layout == baseline_layout:
                        bar_colors.append('#808080')
                    else:
                        bar_colors.append(colors[i % len(colors)])

            if layout_names:
                x_pos = np.arange(len(layout_names))
                bars_mem = ax_mem.bar(x_pos, memory_values, color=bar_colors,
                                      edgecolor='black', linewidth=1.5, alpha=0.8)

                # Add value labels on bars
                for bar, val in zip(bars_mem, memory_values):
                    height = bar.get_height()
                    ax_mem.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.2f}x',
                                ha='center', va='bottom', fontsize=9, fontweight='bold')

                # Add baseline reference line
                ax_mem.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                               alpha=0.7, label=f'{baseline_layout.upper()} (baseline)')

                # Formatting for memory chart
                ax_mem.set_xticks(x_pos)
                ax_mem.set_xticklabels(layout_names, rotation=45, ha='right')
                ax_mem.set_ylabel(
                    f'Memory Reduction vs {baseline_layout.upper()}', fontweight='bold', fontsize=12)
                ax_mem.set_title('Memory Utilization (All Models)',
                                 fontweight='bold', fontsize=14)
                ax_mem.grid(True, axis='y', alpha=0.3, linestyle='--')
                ax_mem.legend(loc='best', fontsize=10)
                ax_mem.set_ylim(bottom=0)

    # Hide extra subplots if any (only for non-memory case)
    if not show_memory and axes is not None:
        for idx in range(len(models), len(axes)):
            axes[idx].axis('off')

    # Overall title
    if show_memory:
        title = f'Performance and Memory Normalized to {baseline_layout.upper()}'
    else:
        title = f'Performance Normalized to {baseline_layout.upper()}'
    if ray_count is not None:
        title += f' (Ray Count: {ray_count:,})'
    else:
        title += ' (Geometric Mean)'
    if machine_type:
        title += f'\n{machine_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save figure
    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(results_dir, exist_ok=True)

    mem_suffix = '_with_memory' if show_memory else ''
    if ray_count is not None:
        output_file = os.path.join(
            results_dir, f'normalized_performance{mem_suffix}_rc{ray_count}_{baseline_layout}.png')
    else:
        output_file = os.path.join(
            results_dir, f'normalized_performance{mem_suffix}_geomean_{baseline_layout}.png')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Normalized performance plot saved to: {output_file}")
    plt.close()


def plot_pareto_normalized(processed_data, memory_data, layouts, baseline_layout, output_path, ray_count=None, machine_type=None):
    """
    Plot Pareto frontiers for selected layouts showing normalized performance vs memory.
    Each model gets its own subplot.

    Args:
        processed_data: Processed trace time data
        memory_data: Memory utilization data
        layouts: List of layout names to include
        baseline_layout: Layout to normalize against
        output_path: Path for output file
        ray_count: Specific ray count to plot (None for geometric mean across all)
        machine_type: Machine type for title
    """
    models = sorted(processed_data.keys())

    # Validate baseline
    if baseline_layout not in layouts:
        print(
            f"Warning: Baseline layout '{baseline_layout}' not in provided layouts list. Adding it.")
        layouts = [baseline_layout] + \
            [l for l in layouts if l != baseline_layout]

    # Create figure with one subplot per model
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#8B5A3C', '#4A90E2',
              '#6B4C8A', '#E85D75', '#3AA655', '#F4B942', '#D64545']

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Check if baseline exists
        if baseline_layout not in processed_data[model]:
            ax.text(0.5, 0.5, f'No baseline data\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        if model not in memory_data or baseline_layout not in memory_data[model]:
            ax.text(0.5, 0.5, f'No memory data\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        # Get baseline values
        if ray_count is not None:
            if ray_count not in processed_data[model][baseline_layout]:
                ax.text(0.5, 0.5, f'Ray count {ray_count}\nnot found for {model}',
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{model.title()}')
                continue
            baseline_time = processed_data[model][baseline_layout][ray_count]
        else:
            times = [t for t in processed_data[model]
                     [baseline_layout].values() if t > 0]
            if not times:
                ax.text(0.5, 0.5, f'No valid data\nfor {model}',
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{model.title()}')
                continue
            baseline_time = math.exp(sum(math.log(t)
                                     for t in times) / len(times))

        baseline_memory = memory_data[model][baseline_layout]['memory']

        if baseline_time <= 0 or baseline_memory <= 0:
            ax.text(0.5, 0.5, f'Invalid baseline\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        # Collect normalized data points for selected layouts
        points = []
        labels = []
        point_colors = []

        for i, layout in enumerate(layouts):
            if layout not in processed_data[model]:
                continue
            if model not in memory_data or layout not in memory_data[model]:
                continue

            # Get performance
            if ray_count is not None:
                if ray_count not in processed_data[model][layout]:
                    continue
                layout_time = processed_data[model][layout][ray_count]
            else:
                times = [t for t in processed_data[model]
                         [layout].values() if t > 0]
                if not times:
                    continue
                layout_time = math.exp(sum(math.log(t)
                                       for t in times) / len(times))

            # Get memory
            layout_memory = memory_data[model][layout]['memory']

            if layout_time > 0 and layout_memory > 0:
                # Normalize: higher is better (speedup and memory reduction)
                perf_norm = baseline_time / layout_time  # >1 means faster
                mem_norm = baseline_memory / layout_memory  # >1 means less memory

                points.append((mem_norm, perf_norm))
                labels.append(layout.upper())

                # Color baseline differently
                if layout == baseline_layout:
                    point_colors.append('#808080')
                else:
                    point_colors.append(colors[i % len(colors)])

        if not points:
            ax.text(0.5, 0.5, f'No valid data\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        points = np.array(points)
        x = points[:, 0]  # normalized memory (higher = less memory used)
        y = points[:, 1]  # normalized performance (higher = faster)

        # Compute Pareto frontier
        # A point is on the frontier if no other point dominates it
        # Point j dominates point i if it has >= memory reduction AND >= speedup
        # with at least one strictly better
        is_pareto = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if not is_pareto[i]:
                continue
            for j in range(len(points)):
                if i == j:
                    continue
                # Point j dominates point i if both dimensions are better or equal
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                        (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break

        # Plot dominated points
        if np.any(~is_pareto):
            for i in np.where(~is_pareto)[0]:
                ax.scatter(x[i], y[i], c='lightgray', s=100, alpha=0.6,
                           marker='o', edgecolors='gray', linewidth=1, zorder=1)

        # Plot Pareto frontier points
        for i in np.where(is_pareto)[0]:
            ax.scatter(x[i], y[i], c=point_colors[i], s=200,
                       edgecolors='black', linewidth=2.5,
                       marker='o', zorder=3)

        # Sort Pareto points and connect with line
        pareto_indices = np.where(is_pareto)[0]
        if len(pareto_indices) > 1:
            pareto_sorted = sorted(pareto_indices, key=lambda i: x[i])
            pareto_x = [x[i] for i in pareto_sorted]
            pareto_y = [y[i] for i in pareto_sorted]
            ax.plot(pareto_x, pareto_y, 'k--',
                    alpha=0.4, linewidth=2, zorder=2)

        # Annotate Pareto frontier points
        for i in pareto_indices:
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor='yellow',
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1.5),
                        zorder=4)

        # Annotate dominated points
        for i in np.where(~is_pareto)[0]:
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=8,
                        fontweight='normal',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightgray',
                                  alpha=0.6,
                                  edgecolor='gray',
                                  linewidth=1),
                        zorder=2)

        # Add reference lines at baseline (1.0, 1.0)
        ax.axhline(y=1.0, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.5)
        ax.axvline(x=1.0, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.5)

        # Formatting
        ax.set_xlabel(
            f'Memory Reduction vs {baseline_layout.upper()}', fontweight='bold', fontsize=11)
        ax.set_ylabel(
            f'Speedup vs {baseline_layout.upper()}', fontweight='bold', fontsize=11)
        ax.set_title(f'{model.title()}', fontweight='bold', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Set limits to show context around data
        x_margin = (x.max() - x.min()) * 0.1 if x.max() > x.min() else 0.1
        y_margin = (y.max() - y.min()) * 0.1 if y.max() > y.min() else 0.1
        ax.set_xlim(max(0, x.min() - x_margin), x.max() + x_margin)
        ax.set_ylim(max(0, y.min() - y_margin), y.max() + y_margin)

    # Hide extra subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    # Overall title
    title = f'Pareto Frontier: Performance vs Memory (Normalized to {baseline_layout.upper()})'
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
            results_dir, f'pareto_normalized_rc{ray_count}_{baseline_layout}.png')
    else:
        output_file = os.path.join(
            results_dir, f'pareto_normalized_geomean_{baseline_layout}.png')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Normalized Pareto frontier plot saved to: {output_file}")
    plt.close()


def plot_pareto_normalized(processed_data, memory_data, layouts, baseline_layout, output_path, ray_count=None, machine_type=None):
    """
    Plot Pareto frontiers for selected layouts showing normalized performance vs memory.
    Each model gets its own subplot.

    Args:
        processed_data: Processed trace time data
        memory_data: Memory utilization data
        layouts: List of layout names to include
        baseline_layout: Layout to normalize against
        output_path: Path for output file
        ray_count: Specific ray count to plot (None for geometric mean across all)
        machine_type: Machine type for title
    """
    models = sorted(processed_data.keys())

    # Validate baseline
    if baseline_layout not in layouts:
        print(
            f"Warning: Baseline layout '{baseline_layout}' not in provided layouts list. Adding it.")
        layouts = [baseline_layout] + \
            [l for l in layouts if l != baseline_layout]

    # Create figure with one subplot per model
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#8B5A3C', '#4A90E2',
              '#6B4C8A', '#E85D75', '#3AA655', '#F4B942', '#D64545']

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Check if baseline exists
        if baseline_layout not in processed_data[model]:
            ax.text(0.5, 0.5, f'No baseline data\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        if model not in memory_data or baseline_layout not in memory_data[model]:
            ax.text(0.5, 0.5, f'No memory data\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        # Get baseline values
        if ray_count is not None:
            if ray_count not in processed_data[model][baseline_layout]:
                ax.text(0.5, 0.5, f'Ray count {ray_count}\nnot found for {model}',
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{model.title()}')
                continue
            baseline_time = processed_data[model][baseline_layout][ray_count]
        else:
            times = [t for t in processed_data[model]
                     [baseline_layout].values() if t > 0]
            if not times:
                ax.text(0.5, 0.5, f'No valid data\nfor {model}',
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{model.title()}')
                continue
            baseline_time = math.exp(sum(math.log(t)
                                     for t in times) / len(times))

        baseline_memory = memory_data[model][baseline_layout]['memory']

        if baseline_time <= 0 or baseline_memory <= 0:
            ax.text(0.5, 0.5, f'Invalid baseline\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        # Collect normalized data points for selected layouts
        points = []
        labels = []
        point_colors = []

        for i, layout in enumerate(layouts):
            if layout not in processed_data[model]:
                continue
            if model not in memory_data or layout not in memory_data[model]:
                continue

            # Get performance
            if ray_count is not None:
                if ray_count not in processed_data[model][layout]:
                    continue
                layout_time = processed_data[model][layout][ray_count]
            else:
                times = [t for t in processed_data[model]
                         [layout].values() if t > 0]
                if not times:
                    continue
                layout_time = math.exp(sum(math.log(t)
                                       for t in times) / len(times))

            # Get memory
            layout_memory = memory_data[model][layout]['memory']

            if layout_time > 0 and layout_memory > 0:
                # Normalize: higher is better (speedup and memory reduction)
                perf_norm = baseline_time / layout_time  # >1 means faster
                mem_norm = baseline_memory / layout_memory  # >1 means less memory

                points.append((mem_norm, perf_norm))
                labels.append(layout.upper())

                # Color baseline differently
                if layout == baseline_layout:
                    point_colors.append('#808080')
                else:
                    point_colors.append(colors[i % len(colors)])

        if not points:
            ax.text(0.5, 0.5, f'No valid data\nfor {model}',
                    ha='center', va='center', fontsize=12)
            ax.set_title(f'{model.title()}')
            continue

        points = np.array(points)
        x = points[:, 0]  # normalized memory (higher = less memory used)
        y = points[:, 1]  # normalized performance (higher = faster)

        # Compute Pareto frontier
        # A point is on the frontier if no other point dominates it
        # Point j dominates point i if it has >= memory reduction AND >= speedup
        # with at least one strictly better
        is_pareto = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if not is_pareto[i]:
                continue
            for j in range(len(points)):
                if i == j:
                    continue
                # Point j dominates point i if both dimensions are better or equal
                if (points[j, 0] >= points[i, 0] and points[j, 1] >= points[i, 1] and
                        (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])):
                    is_pareto[i] = False
                    break

        # Plot dominated points
        if np.any(~is_pareto):
            for i in np.where(~is_pareto)[0]:
                ax.scatter(x[i], y[i], c='lightgray', s=100, alpha=0.6,
                           marker='o', edgecolors='gray', linewidth=1, zorder=1)

        # Plot Pareto frontier points
        for i in np.where(is_pareto)[0]:
            ax.scatter(x[i], y[i], c=point_colors[i], s=200,
                       edgecolors='black', linewidth=2.5,
                       marker='o', zorder=3)

        # Sort Pareto points and connect with line
        pareto_indices = np.where(is_pareto)[0]
        if len(pareto_indices) > 1:
            pareto_sorted = sorted(pareto_indices, key=lambda i: x[i])
            pareto_x = [x[i] for i in pareto_sorted]
            pareto_y = [y[i] for i in pareto_sorted]
            ax.plot(pareto_x, pareto_y, 'k--',
                    alpha=0.4, linewidth=2, zorder=2)

        # Annotate Pareto frontier points
        for i in pareto_indices:
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4',
                                  facecolor='yellow',
                                  alpha=0.8,
                                  edgecolor='black',
                                  linewidth=1.5),
                        zorder=4)

        # Annotate dominated points
        for i in np.where(~is_pareto)[0]:
            ax.annotate(labels[i],
                        xy=(x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=8,
                        fontweight='normal',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightgray',
                                  alpha=0.6,
                                  edgecolor='gray',
                                  linewidth=1),
                        zorder=2)

        # Add reference lines at baseline (1.0, 1.0)
        ax.axhline(y=1.0, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.5)
        ax.axvline(x=1.0, color='red', linestyle='--',
                   linewidth=1.5, alpha=0.5)

        # Formatting
        ax.set_xlabel(
            f'Memory Reduction vs {baseline_layout.upper()}', fontweight='bold', fontsize=11)
        ax.set_ylabel(
            f'Speedup vs {baseline_layout.upper()}', fontweight='bold', fontsize=11)
        ax.set_title(f'{model.title()}', fontweight='bold', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Set limits to show context around data
        x_margin = (x.max() - x.min()) * 0.1 if x.max() > x.min() else 0.1
        y_margin = (y.max() - y.min()) * 0.1 if y.max() > y.min() else 0.1
        ax.set_xlim(max(0, x.min() - x_margin), x.max() + x_margin)
        ax.set_ylim(max(0, y.min() - y_margin), y.max() + y_margin)

    # Hide extra subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    # Overall title
    title = f'Pareto Frontier: Performance vs Memory (Normalized to {baseline_layout.upper()})'
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
            results_dir, f'pareto_normalized_rc{ray_count}_{baseline_layout}.png')
    else:
        output_file = os.path.join(
            results_dir, f'pareto_normalized_geomean_{baseline_layout}.png')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Normalized Pareto frontier plot saved to: {output_file}")
    plt.close()


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
    memory_utilization = {'hairball': {'eb-align16': {'memory': 54336384, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'eb': {'memory': 54336384, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-align16': {'memory': 53450496, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl-align16': {'memory': 30496544, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl-idx-align16': {'memory': 23741408, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl-idx': {'memory': 22028016, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'ebq-cl': {'memory': 28783152, 'nodes': {'primitives': 2880000, 'aabbs': 204946, 'obbs': 6152}}, 'bvh8-align16': {'memory': 53653504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'bvh8': {'memory': 53653504, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-align16': {'memory': 30180096, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-idx-align16': {'memory': 23473408, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8-idx': {'memory': 21796736, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-bvh8': {'memory': 28503424, 'nodes': {'primitives': 2880000, 'interiors': 209584}}, 'cl-cw-bvh8-idx-align16': {'memory': 20120064, 'nodes': {'primitives': 2880000, 'aabbs': 209584}}, 'cl-cw-bvh8-idx': {'memory': 20120064, 'nodes': {'primitives': 2880000, 'aabbs': 209584}}, 'eq-align16': {'memory': 9836624, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'eq': {'memory': 7377468, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt-align16': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'pbrt': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'nodes': 614789}}, 'soa-align16': {'memory': 29509872, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}, 'soa': {'memory': 19673248, 'nodes': {'primitives': 2880000, 'aabbs': 614789, 'nodes': 614789}}}, 'white-oak': {'eb-align16': {'memory': 686896, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'eb': {'memory': 686896, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-align16': {'memory': 661408, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl-align16': {'memory': 384432, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl-idx-align16': {'memory': 299632, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl-idx': {'memory': 277724, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'ebq-cl': {'memory': 362524, 'nodes': {'primitives': 36760, 'aabbs': 2473, 'obbs': 177}}, 'bvh8-align16': {'memory': 679936, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'bvh8': {'memory': 679936, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-align16': {'memory': 382464, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-idx-align16': {'memory': 297472, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8-idx': {'memory': 276224, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-bvh8': {'memory': 361216, 'nodes': {'primitives': 36760, 'interiors': 2656}}, 'cl-cw-bvh8-idx-align16': {'memory': 254976, 'nodes': {'primitives': 36760, 'aabbs': 2656}}, 'cl-cw-bvh8-idx': {'memory': 254976, 'nodes': {'primitives': 36760, 'aabbs': 2656}}, 'eq-align16': {'memory': 127408, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'eq': {'memory': 95556, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt-align16': {'memory': 254816, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'pbrt': {'memory': 254816, 'nodes': {'primitives': 36760, 'nodes': 7963}}, 'soa-align16': {'memory': 382224, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}, 'soa': {'memory': 254816, 'nodes': {'primitives': 36760, 'aabbs': 7963, 'nodes': 7963}}}, 'power-plant': {'eb-align16': {'memory': 265928944, 'nodes': {
        'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'eb': {'memory': 265928944, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-align16': {'memory': 244427296, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl-align16': {'memory': 147942544, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl-idx-align16': {'memory': 115597328, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl-idx': {'memory': 106913756, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'ebq-cl': {'memory': 139258972, 'nodes': {'primitives': 12759246, 'aabbs': 861471, 'obbs': 149317}}, 'bvh8-align16': {'memory': 261846528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'bvh8': {'memory': 261846528, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-align16': {'memory': 147288672, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-idx-align16': {'memory': 114557856, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8-idx': {'memory': 106375152, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-bvh8': {'memory': 139105968, 'nodes': {'primitives': 12759246, 'interiors': 1022838}}, 'cl-cw-bvh8-idx-align16': {'memory': 98192448, 'nodes': {'primitives': 12759246, 'aabbs': 1022838}}, 'cl-cw-bvh8-idx': {'memory': 98192448, 'nodes': {'primitives': 12759246, 'aabbs': 1022838}}, 'eq-align16': {'memory': 43702608, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'eq': {'memory': 32776956, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt-align16': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'pbrt': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'nodes': 2731413}}, 'soa-align16': {'memory': 131107824, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}, 'soa': {'memory': 87405216, 'nodes': {'primitives': 12759246, 'aabbs': 2731413, 'nodes': 2731413}}}, 'sponza': {'eb-align16': {'memory': 5040464, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'eb': {'memory': 5040464, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-align16': {'memory': 4698464, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl-align16': {'memory': 2809136, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl-idx-align16': {'memory': 2193328, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl-idx': {'memory': 2029876, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'ebq-cl': {'memory': 2645684, 'nodes': {'primitives': 262267, 'aabbs': 16869, 'obbs': 2375}}, 'bvh8-align16': {'memory': 4948224, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'bvh8': {'memory': 4948224, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-align16': {'memory': 2783376, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-idx-align16': {'memory': 2164848, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8-idx': {'memory': 2010216, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-bvh8': {'memory': 2628744, 'nodes': {'primitives': 262267, 'interiors': 19329}}, 'cl-cw-bvh8-idx-align16': {'memory': 1855584, 'nodes': {'primitives': 262267, 'aabbs': 19329}}, 'cl-cw-bvh8-idx': {'memory': 1855584, 'nodes': {'primitives': 262267, 'aabbs': 19329}}, 'eq-align16': {'memory': 897264, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'eq': {'memory': 672948, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt-align16': {'memory': 1794528, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'pbrt': {'memory': 1794528, 'nodes': {'primitives': 262267, 'nodes': 56079}}, 'soa-align16': {'memory': 2691792, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}, 'soa': {'memory': 1794528, 'nodes': {'primitives': 262267, 'aabbs': 56079, 'nodes': 56079}}}}
    plot_pareto_raw(processed_data, memory_utilization,
                    filename, ray_count=2 ** 25)
    plot_pareto_raw(processed_data, memory_utilization,
                    filename, ray_count=None)
    # Generate outputs
    create_scaling_plots(processed_data, machine_type,
                         filename, baseline_layout, method)
    plot_normalized_performance(
        processed_data,
        ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16', 'cl-bvh8-idx',
            'cl-bvh8-idx-align16', 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16'],
        "bvh8",  # baseline layout
        filename,
        ray_count=2**20,
        machine_type=machine_type,
        memory_data=memory_utilization

    )

    plot_pareto_normalized(
        processed_data,
        memory_utilization,
        ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16', 'cl-bvh8-idx',
            'cl-bvh8-idx-align16', 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16'],
        "bvh8",
        filename,
        ray_count=2**20,
        machine_type=machine_type
    )
    plot_pareto_normalized(
        processed_data,
        memory_utilization,
        ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16', 'cl-bvh8-idx',
            'cl-bvh8-idx-align16', 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16'],
        "bvh8",
        filename,
        ray_count=None,
        machine_type=machine_type
    )

    plot_normalized_performance(
        processed_data,
        ['eq', 'pbrt', 'pbrt-align16', 'eq-align16', 'ptr', 'soa-align16', 'soa'],
        "eq",  # baseline layout
        filename,
        ray_count=2**20,
        machine_type=machine_type,
        memory_data=memory_utilization

    )

    plot_pareto_normalized(
        processed_data,
        memory_utilization,
        ['eq', 'pbrt', 'pbrt-align16', 'eq-align16', 'ptr', 'soa-align16', 'soa'],
        "pbrt",
        filename,
        ray_count=2**20,
        machine_type=machine_type
    )
    plot_pareto_normalized(
        processed_data,
        memory_utilization,
        ['eq', 'pbrt', 'pbrt-align16', 'eq-align16', 'ptr', 'soa-align16', 'soa'],
        "pbrt",
        filename,
        ray_count=None,
        machine_type=machine_type
    )

    plot_pareto_normalized(
        processed_data,
        memory_utilization,
        ['ebq-align16', 'eb-align16', 'eq', 'ebq-cl',
            'ebq-cl-align16', 'ebq-cl-idx', 'ebq-cl-idx-align16'],
        "eb",
        filename,
        ray_count=2**20,
        machine_type=machine_type
    )
    plot_pareto_normalized(
        processed_data,
        memory_utilization,
        ['ebq-align16', 'eb-align16', 'eq', 'ebq-cl',
            'ebq-cl-align16', 'ebq-cl-idx', 'ebq-cl-idx-align16'],
        "eb",
        filename,
        ray_count=None,
        machine_type=machine_type
    )
