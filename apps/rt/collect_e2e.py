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
    hit_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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

        # Check if it's a hits line
        if ':' in line and 'hits' in line.lower():
            # Extract the hit count value
            hit_match = re.search(r'hits\s*:\s*(\d+)', line, re.IGNORECASE)
            if hit_match and current_model and current_layout and current_ray_count:
                hit_value = int(hit_match.group(1))
                hit_data[current_model][current_layout][current_ray_count].append(
                    hit_value)
            i += 1
            continue

        # Handle separator lines
        if line == '---':
            i += 1
            continue

        i += 1

    return parsed_data, hit_data, machine_type


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


def process_hit_data(hit_data):
    """Process hit data to compute totals and statistics."""
    processed_hits = defaultdict(lambda: defaultdict(dict))

    for model in hit_data:
        for layout in hit_data[model]:
            for ray_count in hit_data[model][layout]:
                values = hit_data[model][layout][ray_count]
                # Sum hits across all runs for this configuration
                total_hits = sum(values)
                processed_hits[model][layout][ray_count] = total_hits

    return processed_hits


def analyze_hit_consistency(hit_data, expected_variance=5):
    """Analyze hit count consistency across layouts and report statistics.

    Args:
        hit_data: Dictionary of hit counts by model/layout/ray_count
        expected_variance: Threshold for reporting differences (default: 5)
    """
    print("\nHit Count Analysis:")
    print(
        f"(Reporting only differences > {expected_variance} hits - small differences are expected due to FP precision)")
    print("=" * 120)

    models = sorted(hit_data.keys())
    significant_diffs_found = False

    for model in models:
        model_has_issues = False
        model_output = []

        # Get all ray counts across all layouts for this model
        all_ray_counts = set()
        for layout in hit_data[model]:
            all_ray_counts.update(hit_data[model][layout].keys())

        ray_counts = sorted(all_ray_counts)

        # For each ray count, compare hits across layouts
        for rc in ray_counts:
            hits_by_layout = {}
            for layout in sorted(hit_data[model].keys()):
                if rc in hit_data[model][layout]:
                    hits_by_layout[layout] = hit_data[model][layout][rc]

            if not hits_by_layout:
                continue

            # Calculate statistics
            hit_values = list(hits_by_layout.values())
            min_hits = min(hit_values)
            max_hits = max(hit_values)
            mean_hits = sum(hit_values) / len(hit_values)
            diff = max_hits - min_hits

            # Only report if difference exceeds expected variance
            if diff > expected_variance:
                if not model_has_issues:
                    model_output.append(f"\n{model.upper()}:")
                    model_output.append("-" * 120)
                    model_has_issues = True

                pct_diff = (diff / mean_hits * 100) if mean_hits > 0 else 0

                model_output.append(f"  Ray Count {rc:,}:")
                model_output.append(
                    f"    Min: {min_hits:,} | Max: {max_hits:,} | Mean: {mean_hits:,.1f}")
                model_output.append(
                    f"    Difference: {diff:,} hits ({pct_diff:.3f}%) *** EXCEEDS THRESHOLD ***")
                model_output.append(f"    Per layout:")
                for layout in sorted(hits_by_layout.keys()):
                    hits = hits_by_layout[layout]
                    deviation = hits - mean_hits
                    pct_dev = (deviation / mean_hits *
                               100) if mean_hits > 0 else 0
                    model_output.append(
                        f"      {layout.upper():<15} {hits:,} ({pct_dev:+.3f}%, {deviation:+.1f} from mean)")

        if model_has_issues:
            significant_diffs_found = True
            for line in model_output:
                print(line)
            print()

    if not significant_diffs_found:
        print(
            f"\n  ✓ All hit count differences are within expected variance (≤{expected_variance} hits)")
        print("    This is normal due to floating-point precision in BVH traversal.\n")

    print()


def print_hit_table(hit_data):
    """Print summary table of hit counts."""
    models = sorted(hit_data.keys())
    all_layouts = set()
    all_ray_counts = set()

    for model in hit_data:
        all_layouts.update(hit_data[model].keys())
        for layout in hit_data[model]:
            all_ray_counts.update(hit_data[model][layout].keys())

    layouts = sorted(all_layouts)
    ray_counts = sorted(all_ray_counts)

    print(f"\nHit Count Summary (Total Hits Across All Runs):")
    print("=" * 120)

    for model in models:
        print(f"\n{model.upper()}")
        print("-" * 120)

        # Header
        header = f"{'Layout':<12}"
        for rc in ray_counts:
            header += f"{format_ray_count(rc).replace('$', '').replace('{', '').replace('}', ''):>15}"
        print(header)
        print("-" * 120)

        # Data rows
        for layout in layouts:
            if layout in hit_data[model]:
                row = f"{layout.upper():<12}"
                for rc in ray_counts:
                    if rc in hit_data[model][layout]:
                        value = hit_data[model][layout][rc]
                        row += f"{value:>15,}"
                    else:
                        row += f"{'--':>15}"
                print(row)

    print("\n")


def format_ray_count(ray_count):
    """Format ray count for display."""
    if (ray_count & (ray_count - 1)) == 0:
        exponent = int(math.log2(ray_count))
        return f"$2^{{{exponent}}}$"
    return f"{ray_count:,}"


def create_scaling_plots(data, machine_type, output_path, baseline_layout, method='arithmetic'):
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
    marker_styles = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']

    model_colors = {}
    for i, model in enumerate(models):
        model_colors[model] = color_palette[i % len(color_palette)]

    layout_styles = {}
    layout_markers = {}
    for i, layout in enumerate(layouts):
        layout_styles[layout] = line_styles[i % len(line_styles)]
        layout_markers[layout] = marker_styles[i % len(marker_styles)]

    # Determine subplot layout based on number of models
    n_models = len(models)
    fig = plt.figure(figsize=(45, 45))
    n_rows = 4
    n_cols = 4

    method_str = 'Geometric Mean' if method == 'geometric' else 'Arithmetic Mean'
    title = f'Ray Tracing Performance Scaling ({method_str})'
    if machine_type:
        title += f' - {machine_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 2. Per-model plots
    for idx, model in enumerate(models):
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        for layout in layouts:
            if layout in data[model]:
                ray_counts = sorted(data[model][layout].keys())
                trace_times = [data[model][layout][rc] for rc in ray_counts]

                # Plot with both linear and markers
                style = layout_styles[layout]
                marker = layout_markers[layout]
                ax.plot(ray_counts, trace_times,
                        marker=marker, markersize=6, linewidth=2,
                        linestyle=style, label=layout.upper(), alpha=0.8)

        ax.set_xlabel('Number of Rays')
        ax.set_ylabel('Trace Time (ms)')
        ax.set_title(f'{model.title()}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Use log scale for x-axis if range is large
        ax.set_xscale('log')
        ax.set_xticks(ray_counts)
        ax.set_xticklabels([format_ray_count(rc).replace('$', '').replace('{', '').replace('}', '')
                            for rc in ray_counts], rotation=45)

    # 3. Additional analysis plot (if space available)
    if n_models <= 4:
        idx = 4
        for model in models:
            # Speedup analysis plot for small number of models
            ax_speedup = plt.subplot(n_rows, n_cols, idx + 1)
            idx += 1

            # Calculate speedup relative to first layout (as baseline)
            assert baseline_layout in data[model], baseline_layout
            baseline_data = data[model][baseline_layout]
            for layout in layouts:
                if layout != baseline_layout and layout in data[model]:
                    ray_counts = sorted(set(baseline_data.keys()) & set(
                        data[model][layout].keys()))
                    speedups = []

                    for rc in ray_counts:
                        if baseline_data[rc] > 0 and data[model][layout][rc] > 0:
                            speedup = baseline_data[rc] / \
                                data[model][layout][rc]
                            speedups.append(speedup)
                        elif data[model][layout][rc] == 0 and baseline_data[rc] == 0:
                            # Both are 0, treat as equal
                            speedups.append(1.0)
                        elif data[model][layout][rc] == 0:
                            # Don't add infinite speedup, skip this point
                            pass
                        else:
                            # baseline is 0, can't compute speedup
                            speedups.append(1.0)

                    if speedups and len(speedups) == len(ray_counts):
                        # Only plot if we have speedup values for all ray counts
                        color = model_colors[model]
                        style = layout_styles[layout]
                        marker = layout_markers[layout]
                        ax_speedup.plot(ray_counts, speedups,
                                        marker=marker, markersize=4, linewidth=1.5,
                                        linestyle=style, color=color,
                                        label=f'{model}-{layout}', alpha=0.7)

                        ax_speedup.set_xscale(
                            'log', base=2)   # log2 scaling

        ax_speedup.axhline(y=1.0, color='black',
                           linestyle='-', linewidth=0.5, alpha=0.5)
        ax_speedup.set_xlabel('Number of Rays')
        ax_speedup.set_ylabel(f'Speedup vs {baseline_layout.upper()}')
        ax_speedup.set_title(
            f'Layout Performance Relative to {baseline_layout.upper()} ({model})')
        ax_speedup.grid(True, alpha=0.3, which='both')
        ax_speedup.legend(fontsize=8, ncol=2, loc='best')

    plt.tight_layout()

    # Save figure
    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
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
                        if value == 0:
                            row += f"{'<1':>10}"  # Show <1 instead of 0
                        else:
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

            # Filter valid data points (non-zero times)
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
            else:
                print(
                    f"  {layout.upper():<15} Insufficient non-zero data points for analysis")

    print("\n")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python trace_scaling.py <data_file> <baseline_layout> [arithmetic|geometric]")
        sys.exit(1)

    filename = sys.argv[1]
    baseline_layout = sys.argv[2]
    method = 'arithmetic'
    if len(sys.argv) == 4:
        if sys.argv[3] in ['arithmetic', 'geometric']:
            method = sys.argv[3]
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
    raw_data, raw_hit_data, machine_type = parse_trace_scaling_data(data_text)
    processed_data = process_trace_data(raw_data, method)
    processed_hits = process_hit_data(raw_hit_data)

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

    # Hit count analysis
    if processed_hits:
        print_hit_table(processed_hits)
        analyze_hit_consistency(processed_hits)

    create_scaling_plots(processed_data, machine_type,
                         filename, baseline_layout, method)
