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
            if layout in blacklist:
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

    method_str = 'Geometric Mean' if method == 'geometric' else 'Arithmetic Mean'
    title = f'Layout Speedup vs {baseline_layout.upper()} ({method_str})'
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
        ax.set_xscale('log')
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


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python trace_scaling.py <data_file> <baseline-layout> <blacklist-layout> [arithmetic|geometric]")
        sys.exit(1)

    filename = sys.argv[1]
    baseline_layout = sys.argv[2]
    blacklist = sys.argv[3]
    method = 'arithmetic'
    if len(sys.argv) == 5:
        if sys.argv[4] in ['arithmetic', 'geometric']:
            method = sys.argv[4]
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

    # Generate outputs
    create_scaling_plots(processed_data, machine_type,
                         filename, baseline_layout, method)
