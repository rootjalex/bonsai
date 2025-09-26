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
        if any(v <= 0 for v in filtered_values):
            return 0
        return math.exp(sum(math.log(v) for v in filtered_values) / len(filtered_values))
    else:
        return sum(filtered_values) / len(filtered_values)


def process_trace_data(raw_data, method='arithmetic'):
    """Process raw data to compute averages."""
    processed_data = defaultdict(lambda: defaultdict(dict))

    for model in raw_data:
        for layout in raw_data[model]:
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


def create_scaling_plots(data, machine_type, output_path, method='arithmetic'):
    """Create comprehensive scaling plots."""
    models = sorted(data.keys())
    all_layouts = set()
    for model in data:
        all_layouts.update(data[model].keys())
    layouts = sorted(all_layouts)

    # Generate colors and styles dynamically
    color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#8B5A3C', '#4A90E2',
                     '#6B4C8A', '#E85D75', '#3AA655', '#F4B942', '#D64545']
    line_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.']

    model_colors = {}
    for i, model in enumerate(models):
        model_colors[model] = color_palette[i % len(color_palette)]

    layout_styles = {}
    for i, layout in enumerate(layouts):
        layout_styles[layout] = line_styles[i % len(line_styles)]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    method_str = 'Geometric Mean' if method == 'geometric' else 'Arithmetic Mean'
    title = f'Ray Tracing Performance Scaling ({method_str})'
    if machine_type:
        title += f' - {machine_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Combined log-log plot (top left)
    ax1 = plt.subplot(2, 3, 1)
    for model in models:
        for layout in layouts:
            if layout in data[model]:
                ray_counts = sorted(data[model][layout].keys())
                trace_times = [data[model][layout][rc] for rc in ray_counts]

                # Filter out zero values for log scale
                valid_indices = [i for i, t in enumerate(trace_times) if t > 0]
                if valid_indices:
                    valid_ray_counts = [ray_counts[i] for i in valid_indices]
                    valid_trace_times = [trace_times[i] for i in valid_indices]

                    label = f'{model}-{layout}'
                    color = model_colors.get(model, '#666666')
                    style = layout_styles.get(layout, '-')
                    ax1.loglog(valid_ray_counts, valid_trace_times,
                               marker='o', markersize=4, linewidth=1.5,
                               linestyle=style, color=color, label=label, alpha=0.7)

    ax1.set_xlabel('Number of Rays')
    ax1.set_ylabel('Trace Time (ms)')
    ax1.set_title('All Configurations (Log-Log Scale)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=8, ncol=2, loc='upper left')

    # 2. Per-model plots (remaining subplots)
    n_models = len(models)
    # Limit to 4 models for layout
    for idx, model in enumerate(models[:min(4, n_models)]):
        ax = plt.subplot(2, 3, idx + 2)

        for layout in layouts:
            if layout in data[model]:
                ray_counts = sorted(data[model][layout].keys())
                trace_times = [data[model][layout][rc] for rc in ray_counts]

                # Plot with both linear and markers
                color = model_colors.get(model, '#666666')
                style = layout_styles.get(layout, '-')
                ax.plot(ray_counts, trace_times,
                        marker='o', markersize=6, linewidth=2,
                        linestyle=style, label=layout.upper(), alpha=0.8)

        ax.set_xlabel('Number of Rays')
        ax.set_ylabel('Trace Time (ms)')
        ax.set_title(f'{model.title()}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Use log scale for x-axis if range is large
        if ray_counts and max(ray_counts) / min(ray_counts) > 100:
            ax.set_xscale('log')
            ax.set_xticks(ray_counts)
            ax.set_xticklabels([format_ray_count(rc).replace('$', '').replace('{', '').replace('}', '')
                               for rc in ray_counts], rotation=45)

    # 3. Speedup analysis plot (bottom right)
    ax_speedup = plt.subplot(2, 3, 6)

    # Calculate speedup relative to ptr layout for each model
    for model in models:
        if 'ptr' in data[model]:
            ptr_data = data[model]['ptr']

            for layout in layouts:
                if layout != 'ptr' and layout in data[model]:
                    ray_counts = sorted(set(ptr_data.keys()) & set(
                        data[model][layout].keys()))
                    speedups = []

                    for rc in ray_counts:
                        if ptr_data[rc] > 0:
                            speedup = ptr_data[rc] / data[model][layout][rc]
                            speedups.append(speedup)
                        else:
                            speedups.append(1.0)

                    if speedups:
                        color = model_colors.get(model, '#666666')
                        style = layout_styles.get(layout, '-')
                        ax_speedup.semilogx(ray_counts, speedups,
                                            marker='s', markersize=4, linewidth=1.5,
                                            linestyle=style, color=color,
                                            label=f'{model}-{layout}', alpha=0.7)

    ax_speedup.axhline(y=1.0, color='black', linestyle='-',
                       linewidth=0.5, alpha=0.5)
    ax_speedup.set_xlabel('Number of Rays')
    ax_speedup.set_ylabel('Speedup vs PTR')
    ax_speedup.set_title('Layout Performance Relative to PTR')
    ax_speedup.grid(True, alpha=0.3, which='both')
    ax_speedup.legend(fontsize=8, ncol=2, loc='best')

    plt.tight_layout()

    # Save figure
    results_dir = os.path.dirname(output_path)
    os.makedirs(results_dir, exist_ok=True)

    method_suffix = '_geomean' if method == 'geometric' else '_arithmetic'
    output_file = os.path.join(
        results_dir, f'trace_scaling{method_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    plt.close()


def print_scaling_table(data, method='arithmetic'):
    """Print summary table of trace times."""
    models = sorted(data.keys())
    all_layouts = set()
    all_ray_counts = set()

    for model in data:
        all_layouts.update(data[model].keys())
        for layout in data[model]:
            all_ray_counts.update(data[model][layout].keys())

    layouts = sorted(all_layouts)
    ray_counts = sorted(all_ray_counts)

    method_str = 'Geometric Mean' if method == 'geometric' else 'Arithmetic Mean'
    print(f"\nTrace Time Scaling Summary ({method_str}):")
    print("=" * 120)

    for model in models:
        print(f"\n{model.upper()}")
        print("-" * 120)

        # Header
        header = f"{'Layout':<12}"
        for rc in ray_counts:
            header += f"{format_ray_count(rc).replace('$', '').replace('{', '').replace('}', ''):>10}"
        print(header)
        print("-" * 120)

        # Data rows
        for layout in layouts:
            if layout in data[model]:
                row = f"{layout.upper():<12}"
                for rc in ray_counts:
                    if rc in data[model][layout]:
                        value = data[model][layout][rc]
                        row += f"{value:>10.1f}"
                    else:
                        row += f"{'--':>10}"
                print(row)

    print("\n")


def analyze_scaling_behavior(data):
    """Analyze and report scaling behavior."""
    print("\nScaling Analysis:")
    print("=" * 80)

    for model in sorted(data.keys()):
        print(f"\n{model.upper()}:")

        for layout in sorted(data[model].keys()):
            ray_counts = sorted(data[model][layout].keys())
            trace_times = [data[model][layout][rc] for rc in ray_counts]

            # Filter valid data points
            valid_points = [(rc, tt) for rc, tt in zip(
                ray_counts, trace_times) if tt > 0]

            if len(valid_points) >= 2:
                # Calculate scaling exponent (log-log regression)
                log_rays = np.log([p[0] for p in valid_points])
                log_times = np.log([p[1] for p in valid_points])

                # Linear regression in log-log space
                coeffs = np.polyfit(log_rays, log_times, 1)
                scaling_exponent = coeffs[0]

                # Determine scaling behavior
                if scaling_exponent < 0.9:
                    behavior = "Sub-linear"
                elif scaling_exponent < 1.1:
                    behavior = "Linear"
                elif scaling_exponent < 1.5:
                    behavior = "Super-linear"
                else:
                    behavior = "Quadratic or worse"

                print(
                    f"  {layout.upper():<15} Exponent: {scaling_exponent:.3f} ({behavior})")

    print("\n")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: python trace_scaling.py <data_file> [arithmetic|geometric]")
        sys.exit(1)

    filename = sys.argv[1]
    method = 'arithmetic'
    if len(sys.argv) == 3:
        if sys.argv[2] in ['arithmetic', 'geometric']:
            method = sys.argv[2]
        else:
            print("Method must be 'arithmetic' or 'geometric'")
            sys.exit(1)

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
    processed_data = process_trace_data(raw_data, method)

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

    # Generate outputs
    print_scaling_table(processed_data, method)
    analyze_scaling_behavior(processed_data)
    create_scaling_plots(processed_data, machine_type, filename, method)
