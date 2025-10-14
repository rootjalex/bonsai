import matplotlib.pyplot as plt
import numpy as np
import re
import math
import os
from collections import defaultdict
import sys


def parse_trace_scaling_data(data_text):
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

        if ',' not in line and ':' not in line and not line.isdigit() and line != '---':
            current_model = line
            current_layout = None
            ray_count_sequence = []
            current_run_index = 0
            i += 1
            continue
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


def process_trace_data(raw_data, mean_strategy, ray_count_range=None):
    processed_data = defaultdict(lambda: defaultdict(dict))
    R = (0, sys.maxsize) if ray_count_range is None else ray_count_range

    for model in raw_data:
        for layout in raw_data[model]:
            time_per_ray_values = []
            ray_counts = []
            for ray_count in raw_data[model][layout]:
                if not (ray_count >= R[0] and ray_count <= R[1]):
                    continue
                values = raw_data[model][layout][ray_count]
                avg_value = calculate_average(values)
                if avg_value > 0:
                    time_per_ray = avg_value / ray_count
                    time_per_ray_values.append(time_per_ray)
                    ray_counts.append(ray_count)
            assert time_per_ray_values
            if mean_strategy == 'geo':
                result = math.exp(
                    sum(math.log(t) for t in time_per_ray_values) / len(time_per_ray_values))
            elif mean_strategy == 'wavg':
                total_rays = sum(ray_counts)
                result = sum(
                    t * r for t, r in zip(time_per_ray_values, ray_counts)) / total_rays
            else:
                assert mean_strategy == 'arithmetic', mean_strategy
                result = sum(time_per_ray_values) / len(time_per_ray_values)
            processed_data[model][layout] = result

    return processed_data


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, mean_strategy, ray_count_range=None, label_dominated_points=True):
    models = sorted(processed_data.keys())
    if len(models) == 0:
        return

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]

    group_colors = ['#0173B2',
                    '#DE8F05',
                    '#029E73',
                    '#CC78BC',
                    '#CA9161',
                    '#949494',
                    '#ECE133',
                    '#56B4E9',
                    '#D55E00',
                    '#F0E442']

    line_styles = [
        '-',
        '--',
        '-.',
        ':',
        (0, (5, 2, 1, 2)),
        (0, (3, 1, 1, 1)),
        (0, (5, 5)),
        (0, (1, 1)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1, 1, 1))
    ]
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

        for group_idx, (group_name, layouts) in enumerate(layout_groups.items()):
            group_color = group_colors[group_idx % len(group_colors)]

            for layout in layouts:
                if layout not in processed_data[model]:
                    continue
                if layout not in memory_data[model]:
                    continue

                time_per_ray = processed_data[model][layout]
                memory = memory_data[model][layout]['memory']

                if time_per_ray > 0 and memory > 0:
                    all_points.append((memory, time_per_ray))
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
        x = points[:, 0]
        y = points[:, 1]

        memory_values = x / 1024 / 1024
        memory_unit = 'MB'
        time_per_ray_values = y * 1e6
        time_unit = 'ns/ray'

        for group_idx, (group_name, layouts) in enumerate(layout_groups.items()):
            group_color = group_colors[group_idx % len(group_colors)]
            group_linestyle = line_styles[group_idx % len(line_styles)]
            group_marker = marker_styles[group_idx % len(marker_styles)]

            group_mask = np.array([g == group_name for g in all_groups])
            if not np.any(group_mask):
                continue

            group_points = points[group_mask]
            group_labels = [all_labels[i]
                            for i in range(len(all_labels)) if group_mask[i]]

            is_pareto = np.ones(len(group_points), dtype=bool)
            for i in range(len(group_points)):
                if not is_pareto[i]:
                    continue
                for j in range(len(group_points)):
                    if i == j:
                        continue
                    if (group_points[j, 0] <= group_points[i, 0] and
                        group_points[j, 1] <= group_points[i, 1] and
                            (group_points[j, 0] < group_points[i, 0] or group_points[j, 1] < group_points[i, 1])):
                        is_pareto[i] = False
                        break
            group_memory = memory_values[group_mask]
            group_time_per_ray = time_per_ray_values[group_mask]
            if label_dominated_points and np.any(~is_pareto):
                ax.scatter(group_memory[~is_pareto], group_time_per_ray[~is_pareto],
                           c='lightgray', s=80, alpha=0.5,
                           marker='o', edgecolors='gray', linewidth=1, zorder=1)

            ax.scatter(group_memory[is_pareto], group_time_per_ray[is_pareto],
                       c=group_color, s=180, alpha=0.9,
                       edgecolors='black', linewidth=2.5,
                       marker=group_marker, label=group_name, zorder=3)

            pareto_indices = np.where(is_pareto)[0]
            if len(pareto_indices) > 1:
                pareto_sorted = sorted(
                    pareto_indices, key=lambda i: group_memory[i])
                pareto_x = [group_memory[i] for i in pareto_sorted]
                pareto_y = [group_time_per_ray[i] for i in pareto_sorted]
                ax.plot(pareto_x, pareto_y, color=group_color,
                        linestyle=group_linestyle, alpha=0.8, linewidth=3, zorder=2)

            for i in pareto_indices:
                ax.annotate(group_labels[i],
                            xy=(group_memory[i], group_time_per_ray[i]),
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

            if label_dominated_points and np.any(~is_pareto):
                dominated_indices = np.where(~is_pareto)[0]
                for i in dominated_indices:
                    ax.annotate(group_labels[i],
                                xy=(group_memory[i], group_time_per_ray[i]),
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
        ax.set_xlabel(
            f'Memory Utilization ({memory_unit})', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'Time per Ray ({time_unit})',
                      fontweight='bold', fontsize=11)
        ax.set_title(f'{model.title()}', fontweight='bold', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.legend(fontsize=9, loc='best', framealpha=0.9,
                  edgecolor='black', fancybox=False, shadow=True)

        ax.ticklabel_format(style='plain')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x:,.1f}'))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, p: f'{y:,.1f}'))

    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')

    title = 'Time per Ray vs Memory Utilization'
    if ray_count_range is not None:
        title += f'\n{mean_strategy} ({ray_count_range[0]:,} - {ray_count_range[1]:,})'
    else:
        title += f'\n{mean_strategy} (all)'
    if machine_type:
        title += f' - {machine_type}'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    results_dir = os.path.dirname(
        output_path) if os.path.dirname(output_path) else '.'
    os.makedirs(results_dir, exist_ok=True)
    name = filename.split('/')[-1].split('.')[0]
    if ray_count_range is not None:
        def log2(x): return int(math.log2(x))
        output_file = os.path.join(
            results_dir, f'{name}-{mean_strategy}{log2(ray_count_range[0])}-{log2(ray_count_range[1])}.pdf')
    else:
        output_file = os.path.join(
            results_dir, f'{name}-{mean_strategy}.pdf')

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pareto_plotter.py <data_file> [mean_strategy]")
        print("  mean_strategy: 'wavg' (default), 'geo', or 'arithmetic'")
        sys.exit(1)

    filename = sys.argv[1]
    mean_strategy = sys.argv[2] if len(sys.argv) > 2 else 'wavg'

    if mean_strategy not in ['geo', 'wavg', 'arithmetic']:
        print(
            f"error: strategy must be 'geo', 'wavg', or 'arithmetic', got '{mean_strategy}'")
        sys.exit(1)

    try:
        with open(filename, 'r') as file:
            data_text = file.read()
    except FileNotFoundError:
        print(f"error: File '{filename}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"error reading file '{filename}': {e}")
        sys.exit(1)

    raw_data, machine_type = parse_trace_scaling_data(data_text)
    memory_utilization = {'lucy': {'eq-align16': {'memory': 1095462416, 'nodes': {'primitives': 28055728, 'nodes': 5341013}}, 'eq': {'memory': 1074098364, 'nodes': {'primitives': 28055728, 'nodes': 5341013}}, 'pbrt-align16': {'memory': 1180918624, 'nodes': {'primitives': 28055728, 'nodes': 5341013}}, 'pbrt-q16-soaos': {'memory': 1095462416, 'nodes': {'primitives': 28055728, 'q_aabbs': 5341013, 'nodes': 5341013}}, 'pbrt-q16': {'memory': 1095462416, 'nodes': {'primitives': 28055728, 'nodes': 5341013}}, 'pbrt-soaos-align16': {'memory': 1266374832, 'nodes': {'primitives': 28055728, 'aabbs': 5341013, 'nodes': 5341013}}, 'pbrt-soaos': {'memory': 1180918624, 'nodes': {'primitives': 28055728, 'aabbs': 5341013, 'nodes': 5341013}}, 'pbrt': {'memory': 1180918624, 'nodes': {'primitives': 28055728, 'nodes': 5341013}}, 'ptr': {'memory': 1245010780, 'nodes': {'primitives': 28055728, 'nodes': 235004572}}, 'bvh8-align16': {'memory': 1101218752, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'bvh8': {'memory': 1101218752, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-align16': {'memory': 1061313264, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-idx-align16': {'memory': 1049911696, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-idx': {'memory': 1047061304, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-u16-align16': {'memory': 1078415616, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-u16-idx-align16': {'memory': 1078415616, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-u16-idx': {'memory': 1075565224, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8-u16': {'memory': 1075565224, 'nodes': {'primitives': 28055728, 'interiors': 356299}}, 'cl-bvh8': {'memory': 1058462872, 'nodes': {'primitives': 28055728, 'interiors': 356299}}}, 'sheep': {'eq-align16': {'memory': 116611728, 'nodes': {'primitives': 2967664, 'nodes': 610989}}, 'eq': {'memory': 114167772, 'nodes': {'primitives': 2967664, 'nodes': 610989}}, 'pbrt-align16': {'memory': 126387552, 'nodes': {'primitives': 2967664, 'nodes': 610989}}, 'pbrt-q16-soaos': {'memory': 116611728, 'nodes': {'primitives': 2967664, 'q_aabbs': 610989, 'nodes': 610989}}, 'pbrt-q16': {'memory': 116611728, 'nodes': {'primitives': 2967664, 'nodes': 610989}}, 'pbrt-soaos-align16': {'memory': 136163376, 'nodes': {'primitives': 2967664, 'aabbs': 610989, 'nodes': 610989}}, 'pbrt-soaos': {'memory': 126387552, 'nodes': {'primitives': 2967664, 'aabbs': 610989, 'nodes': 610989}}, 'pbrt': {'memory': 126387552, 'nodes': {'primitives': 2967664, 'nodes': 610989}}, 'ptr': {'memory': 133719420, 'nodes': {'primitives': 2967664, 'nodes': 26883516}}, 'bvh8-align16': {'memory': 119498176, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'bvh8': {'memory': 119498176, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-align16': {'memory': 113958432, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-idx-align16': {'memory': 112375648, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-idx': {'memory': 111979952, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-u16-align16': {'memory': 116332608, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-u16-idx-align16': {'memory': 116332608, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-u16-idx': {'memory': 115936912, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8-u16': {'memory': 115936912, 'nodes': {'primitives': 2967664, 'interiors': 49462}}, 'cl-bvh8': {'memory': 113562736, 'nodes': {'primitives': 2967664, 'interiors': 49462}}}, 'san-miguel-x35-y22-z47': {'eq-align16': {'memory': 386278960, 'nodes': {'primitives': 9832536, 'nodes': 2019229}}, 'eq': {'memory': 378202044, 'nodes': {'primitives': 9832536, 'nodes': 2019229}}, 'pbrt-align16': {'memory': 418586624, 'nodes': {'primitives': 9832536, 'nodes': 2019229}}, 'pbrt-q16-soaos': {'memory': 386278960, 'nodes': {'primitives': 9832536, 'q_aabbs': 2019229, 'nodes': 2019229}}, 'pbrt-q16': {'memory': 386278960, 'nodes': {'primitives': 9832536, 'nodes': 2019229}}, 'pbrt-soaos-align16': {'memory': 450894288, 'nodes': {'primitives': 9832536, 'aabbs': 2019229, 'nodes': 2019229}}, 'pbrt-soaos': {'memory': 418586624, 'nodes': {'primitives': 9832536, 'aabbs': 2019229, 'nodes': 2019229}}, 'pbrt': {'memory': 418586624, 'nodes': {'primitives': 9832536, 'nodes': 2019229}}, 'ptr': {'memory': 442817372, 'nodes': {'primitives': 9832536, 'nodes': 88846076}}, 'bvh8-align16': {'memory': 394949216, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'bvh8': {'memory': 394949216, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-align16': {'memory': 377013312, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-idx-align16': {'memory': 371888768, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-idx': {'memory': 370607632, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-u16-align16': {'memory': 384700128, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-u16-idx-align16': {'memory': 384700128, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-u16-idx': {'memory': 383418992, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8-u16': {'memory': 383418992, 'nodes': {'primitives': 9832024, 'interiors': 160142}}, 'cl-bvh8': {'memory': 375732176, 'nodes': {'primitives': 9832024, 'interiors': 160142}}}, 'hairball': {'eq-align16': {'memory': 112958992, 'nodes': {'primitives': 2880000, 'nodes': 579937}}, 'eq': {'memory': 110639244, 'nodes': {'primitives': 2880000, 'nodes': 579937}}, 'pbrt-align16': {'memory': 122237984, 'nodes': {'primitives': 2880000, 'nodes': 579937}}, 'pbrt-q16-soaos': {'memory': 112958992, 'nodes': {'primitives': 2880000, 'q_aabbs': 579937, 'nodes': 579937}}, 'pbrt-q16': {'memory': 112958992, 'nodes': {'primitives': 2880000, 'nodes': 579937}}, 'pbrt-soaos-align16': {'memory': 131516976, 'nodes': {'primitives': 2880000, 'aabbs': 579937, 'nodes': 579937}}, 'pbrt-soaos': {'memory': 122237984, 'nodes': {'primitives': 2880000, 'aabbs': 579937, 'nodes': 579937}}, 'pbrt': {'memory': 122237984, 'nodes': {'primitives': 2880000, 'nodes': 579937}}, 'ptr': {
        'memory': 129197228, 'nodes': {'primitives': 2880000, 'nodes': 25517228}}, 'bvh8-align16': {'memory': 115344896, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'bvh8': {'memory': 115344896, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-align16': {'memory': 110241504, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-idx-align16': {'memory': 108783392, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-idx': {'memory': 108418864, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-u16-align16': {'memory': 112428672, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-u16-idx-align16': {'memory': 112428672, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-u16-idx': {'memory': 112064144, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8-u16': {'memory': 112064144, 'nodes': {'primitives': 2880000, 'interiors': 45566}}, 'cl-bvh8': {'memory': 109876976, 'nodes': {'primitives': 2880000, 'interiors': 45566}}}, 'white-oak': {'eq-align16': {'memory': 1440304, 'nodes': {'primitives': 36760, 'nodes': 7309}}, 'eq': {'memory': 1411068, 'nodes': {'primitives': 36760, 'nodes': 7309}}, 'pbrt-align16': {'memory': 1557248, 'nodes': {'primitives': 36760, 'nodes': 7309}}, 'pbrt-q16-soaos': {'memory': 1440304, 'nodes': {'primitives': 36760, 'q_aabbs': 7309, 'nodes': 7309}}, 'pbrt-q16': {'memory': 1440304, 'nodes': {'primitives': 36760, 'nodes': 7309}}, 'pbrt-soaos-align16': {'memory': 1674192, 'nodes': {'primitives': 36760, 'aabbs': 7309, 'nodes': 7309}}, 'pbrt-soaos': {'memory': 1557248, 'nodes': {'primitives': 36760, 'aabbs': 7309, 'nodes': 7309}}, 'pbrt': {'memory': 1557248, 'nodes': {'primitives': 36760, 'nodes': 7309}}, 'ptr': {'memory': 1644956, 'nodes': {'primitives': 36760, 'nodes': 321596}}, 'bvh8-align16': {'memory': 1468512, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'bvh8': {'memory': 1468512, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-align16': {'memory': 1405008, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-idx-align16': {'memory': 1386864, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-idx': {'memory': 1382328, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-u16-align16': {'memory': 1432224, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-u16-idx-align16': {'memory': 1432224, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-u16-idx': {'memory': 1427688, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8-u16': {'memory': 1427688, 'nodes': {'primitives': 36760, 'interiors': 567}}, 'cl-bvh8': {'memory': 1400472, 'nodes': {'primitives': 36760, 'interiors': 567}}}, 'sponza': {'eq-align16': {'memory': 10280380, 'nodes': {'primitives': 262267, 'nodes': 52423}}, 'eq': {'memory': 10070688, 'nodes': {'primitives': 262267, 'nodes': 52423}}, 'pbrt-align16': {'memory': 11119148, 'nodes': {'primitives': 262267, 'nodes': 52423}}, 'pbrt-q16-soaos': {'memory': 10280380, 'nodes': {'primitives': 262267, 'q_aabbs': 52423, 'nodes': 52423}}, 'pbrt-q16': {'memory': 10280380, 'nodes': {'primitives': 262267, 'nodes': 52423}}, 'pbrt-soaos-align16': {'memory': 11957916, 'nodes': {'primitives': 262267, 'aabbs': 52423, 'nodes': 52423}}, 'pbrt-soaos': {'memory': 11119148, 'nodes': {'primitives': 262267, 'aabbs': 52423, 'nodes': 52423}}, 'pbrt': {'memory': 11119148, 'nodes': {'primitives': 262267, 'nodes': 52423}}, 'ptr': {'memory': 11748224, 'nodes': {'primitives': 262267, 'nodes': 2306612}}, 'bvh8-align16': {'memory': 10460492, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'bvh8': {'memory': 10460492, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-align16': {'memory': 10014732, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-idx-align16': {'memory': 9887372, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-idx': {'memory': 9855532, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-u16-align16': {'memory': 10205772, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-u16-idx-align16': {'memory': 10205772, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-u16-idx': {'memory': 10173932, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8-u16': {'memory': 10173932, 'nodes': {'primitives': 262267, 'interiors': 3980}}, 'cl-bvh8': {'memory': 9982892, 'nodes': {'primitives': 262267, 'interiors': 3980}}}, 'power-plant': {'eq-align16': {'memory': 501442024, 'nodes': {'primitives': 12759246, 'nodes': 2631823}}, 'eq': {'memory': 490914732, 'nodes': {'primitives': 12759246, 'nodes': 2631823}}, 'pbrt-align16': {'memory': 543551192, 'nodes': {'primitives': 12759246, 'nodes': 2631823}}, 'pbrt-q16-soaos': {'memory': 501442024, 'nodes': {'primitives': 12759246, 'q_aabbs': 2631823, 'nodes': 2631823}}, 'pbrt-q16': {'memory': 501442024, 'nodes': {'primitives': 12759246, 'nodes': 2631823}}, 'pbrt-soaos-align16': {'memory': 585660360, 'nodes': {'primitives': 12759246, 'aabbs': 2631823, 'nodes': 2631823}}, 'pbrt-soaos': {'memory': 543551192, 'nodes': {'primitives': 12759246, 'aabbs': 2631823, 'nodes': 2631823}}, 'pbrt': {'memory': 543551192, 'nodes': {'primitives': 12759246, 'nodes': 2631823}}, 'ptr': {'memory': 575133068, 'nodes': {'primitives': 12759246, 'nodes': 115800212}}, 'bvh8-align16': {'memory': 509361144, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'bvh8': {'memory': 509361144, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-align16': {'memory': 487473768, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-idx-align16': {'memory': 481220232, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-idx': {'memory': 479656848, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-u16-align16': {'memory': 496854072, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-u16-idx-align16': {'memory': 496854072, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-u16-idx': {'memory': 495290688, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8-u16': {'memory': 495290688, 'nodes': {'primitives': 12759246, 'interiors': 195423}}, 'cl-bvh8': {'memory': 485910384, 'nodes': {'primitives': 12759246, 'interiors': 195423}}}}
    embree_memory_utilization = {'lucy': {'embree-bvh8i': {'memory': 1184531218, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 2022605586, 'nodes': {'aabbs': 1}}}, 'sheep': {'embree-bvh8i': {'memory': 85039513, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 170290839, 'nodes': {'aabbs': 1}}}, 'san-miguel-x35-y22-z47': {'embree-bvh8i': {'memory': 374447538, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 675464347, 'nodes': {'aabbs': 1}}}, 'hairball': {'embree-bvh8i': {'memory': 99324264,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 191216222, 'nodes': {'aabbs': 1}}}, 'white-oak': {'embree-bvh8i': {'memory': 1371537, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 2521825, 'nodes': {'aabbs': 1}}}, 'sponza': {'embree-bvh8i': {'memory': 10135535, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 17728274, 'nodes': {'aabbs': 1}}}, 'power-plant': {'embree-bvh8i': {'memory': 458312646, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 833827635, 'nodes': {'aabbs': 1}}}}
    for model, layouts in embree_memory_utilization.items():
        assert model in memory_utilization
        memory_utilization[model].update(layouts)

    trace_data = process_trace_data(
        raw_data, mean_strategy=mean_strategy, ray_count_range=None)
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
            print(f"  [no memory data found]")

    layout_groups = {
        'bvh8': ['bvh8', 'bvh8-align16', 'bvh-q8',
                 'bvh-q16', 'bvh8-q8-align16', 'bvh8-q8-ci-align16',
                 'bvh8-q8-ci', 'bvh8-q16-align16', 'bvh8-q16-ci-align16',
                 'bvh8-q16-ci',
                 ],
        'bvh2': ['sg-eq', 'pbrt', 'pbrt-align16', 'sg-eq-align16',
                 'ptr', 'pbrt-soaos-align16', 'pbrt-soaos',
                 'pbrt-q16', 'pbrt-q16-soaos',
                 ],
        'embree': ['embree-bvh8i', 'embree-bvh8v'],
    }
    ray_count_range = None
    label_dominated_points = True
    plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                          filename, machine_type, mean_strategy, ray_count_range, label_dominated_points)
