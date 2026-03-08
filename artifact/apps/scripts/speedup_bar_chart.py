import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict

from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})


def load_csv_data(csv_file, machines=None, ray_types=None, scenes=None, layouts=None, intersect=None):
    """Load and filter CSV data."""
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

    return raw_data


def calculate_average(values):
    """Calculate average with outlier filtering."""
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


def process_trace_data(raw_data, mean_strategy='wavg'):
    """Process raw trace data to compute time per ray."""
    processed_data = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))))

    for model in raw_data:
        for machine in raw_data[model]:
            for ray_type in raw_data[model][machine]:
                for layout in raw_data[model][machine][ray_type]:
                    time_per_ray_values = []
                    ray_counts = []
                    for ray_count_key in raw_data[model][machine][ray_type][layout]:
                        ray_count = int(ray_count_key) if isinstance(
                            ray_count_key, str) else ray_count_key
                        values = raw_data[model][machine][ray_type][layout][ray_count_key]
                        avg_value = calculate_average(values)
                        if avg_value > 0:
                            time_per_ray = avg_value / ray_count
                            time_per_ray_values.append(time_per_ray)
                            ray_counts.append(ray_count)

                    if not time_per_ray_values:
                        continue

                    if mean_strategy == 'geo':
                        import math
                        result = math.exp(
                            sum(math.log(t) for t in time_per_ray_values) / len(time_per_ray_values))
                    elif mean_strategy == 'wavg':
                        total_rays = sum(ray_counts)
                        result = sum(
                            t * r for t, r in zip(time_per_ray_values, ray_counts)) / total_rays
                    else:
                        result = sum(time_per_ray_values) / \
                            len(time_per_ray_values)

                    processed_data[model][machine][ray_type][layout] = result

    return processed_data


def plot_relative_performance_bar_chart(processed_data, scene, machine, layouts, baseline_layout, ray_types, output_filename):
    """Create grouped bar chart showing performance relative to baseline layout across ray types."""

    if scene not in processed_data:
        print(f"Error: Scene '{scene}' not found in data")
        return
    if machine not in processed_data[scene]:
        print(f"Error: Machine '{machine}' not found for scene '{scene}'")
        return

    # Prepare data
    relative_perf = defaultdict(dict)

    for ray_type in ray_types:
        if ray_type not in processed_data[scene][machine]:
            print(
                f"Warning: Ray type '{ray_type}' not found for scene '{scene}'")
            continue

        scene_data = processed_data[scene][machine][ray_type]

        if baseline_layout not in scene_data:
            print(
                f"Warning: Baseline layout '{baseline_layout}' not found for {scene}/{ray_type}")
            continue

        baseline_time = scene_data[baseline_layout]

        for layout in layouts:
            if layout in scene_data:
                # Calculate relative performance (higher is better)
                # Positive = slower than baseline, negative = faster than baseline
                rel_perf = (
                    (scene_data[layout] - baseline_time) / baseline_time) * 100
                relative_perf[ray_type][layout] = rel_perf
            else:
                print(
                    f"Warning: Layout '{layout}' not found for {scene}/{ray_type}")

    if not relative_perf:
        print("Error: No valid data to plot")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(5.5, 4))

    x = np.arange(len(ray_types))
    width = 0.3  # Increased width to reduce white space

    # Single red color for all bars
    bar_color = '#A52A2A'

    # Plot bars for each layout
    for i, layout in enumerate(layouts):
        values = [relative_perf[rt].get(layout, 0) for rt in ray_types]
        offset = (i - len(layouts)/2 + 0.5) * width

        bars = ax.bar(x + offset, values, width,
                      color=bar_color,
                      edgecolor='black', linewidth=1.5,
                      label=layout)

        # Add value labels INSIDE bars
        for bar, val in zip(bars, values):
            if val != 0:
                height = bar.get_height()
                # Place text inside the bar
                y_pos = height / 2
                # Use minus sign for positive values (slowdown)
                label = f'${-abs(val):.1f}$\\%' if val > 0 else f'${abs(val):.1f}$\\%'
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                        label,
                        ha='center', va='center', fontsize=14,
                        fontweight='bold', color='white')

    # Add baseline reference line - dotted style
    ax.axhline(y=0, color='black', linestyle=':', linewidth=2.5,
               label=f'{baseline_layout} (baseline)', zorder=1)

    # Customize plot
    ax.set_ylabel(r'\textbf{Relative Performance}',
                  fontsize=20, fontweight='bold')
    ax.set_xlabel(r'\textbf{Ray Distribution}', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [r'\textbf{' + rt + '}' for rt in ray_types], fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='best', framealpha=0.9, edgecolor='black')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

    # Invert y-axis so positive (slower) goes down
    ax.invert_yaxis()

    # Add space above the baseline
    ylim = ax.get_ylim()
    y_range = ylim[0] - ylim[1]  # Remember y-axis is inverted
    ax.set_ylim(ylim[0], ylim[1] - y_range * 0.15)  # Add 15% space above

    plt.tight_layout()

    plt.tight_layout()

    # Save plot
    output_file = f'{output_filename}.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Bar chart saved to: {output_file}")

    # Print summary
    print(f"\nRelative Performance Summary (vs {baseline_layout}):")
    print("=" * 60)
    for ray_type in ray_types:
        print(f"\n{ray_type.capitalize()} rays:")
        for layout in layouts:
            val = relative_perf[ray_type].get(layout, 0)
            if val != 0:
                comparison = "slower" if val > 0 else "faster"
                print(f"  {layout:20s}: {abs(val):6.2f}% {comparison}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate bar chart showing relative performance across ray distributions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 relative_performance_bar.py results.csv \\
    --scene lucy \\
    --machine cuda \\
    --layouts pbrt-q16-soaos sg-eq \\
    --baseline pbrt-q16 \\
    --ray-types primary secondary \\
    --output lucy_relative_perf
""")

    parser.add_argument(
        'csv_file', help='Path to CSV file with benchmark data')
    parser.add_argument('--scene', type=str, required=True,
                        help='Scene name (e.g., lucy)')
    parser.add_argument('--machine', type=str, default='cuda',
                        help='Machine type (default: cuda)')
    parser.add_argument('--layouts', nargs='+', required=True,
                        help='Layout names to compare (e.g., pbrt-q16-soaos sg-eq)')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline layout for comparison (e.g., pbrt-q16)')
    parser.add_argument('--ray-types', nargs='+', default=['primary', 'secondary'],
                        help='Ray types to compare (default: primary secondary)')
    parser.add_argument('--mean', choices=['wavg', 'geo', 'arithmetic'], default='wavg',
                        help='Mean strategy to use (default: wavg)')
    parser.add_argument('--intersect', type=str, default='mt',
                        help='Intersect method to filter by (default: mt)')
    parser.add_argument('--output', type=str, default='relative_performance',
                        help='Output filename (without extension, default: relative_performance)')

    args = parser.parse_args()

    # Include baseline in layouts for data loading
    all_layouts = args.layouts + [args.baseline]

    # Load data
    raw_data = load_csv_data(
        args.csv_file,
        machines=[args.machine],
        ray_types=args.ray_types,
        scenes=[args.scene],
        layouts=all_layouts,
        intersect=args.intersect
    )

    # Process data
    trace_data = process_trace_data(raw_data, mean_strategy=args.mean)

    # Generate plot
    plot_relative_performance_bar_chart(
        trace_data,
        args.scene,
        args.machine,
        args.layouts,
        args.baseline,
        args.ray_types,
        args.output
    )
