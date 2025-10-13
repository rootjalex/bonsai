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


def process_trace_data(raw_data, ray_count_range=None, mean_type='geomean'):
    processed_data = defaultdict(lambda: defaultdict(dict))
    R = (0, sys.maxsize) if ray_count_range is None else ray_count_range

    for model in raw_data:
        for layout in raw_data[model]:
            times = []
            for ray_count in raw_data[model][layout]:
                if not (ray_count >= R[0] and ray_count <= R[1]):
                    continue
                values = raw_data[model][layout][ray_count]
                avg_value = calculate_average(values)
                if avg_value > 0:
                    times.append(avg_value)
            assert times
            if mean_type == 'geomean':
                result = math.exp(sum(math.log(t) for t in times) / len(times))
            else:
                result = sum(times) / len(times)
            processed_data[model][layout] = result

    return processed_data


def process_trace_data_geomean(raw_data):
    return process_trace_data(raw_data, ray_count_range=None, mean_type='geomean')


def plot_pareto_frontiers_geomean(processed_data, memory_data, layout_groups, output_path, machine_type, mean_type='geomean'):
    return plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count_range=None, mean_type=mean_type)


def plot_pareto_frontiers(processed_data, memory_data, layout_groups, output_path, machine_type, ray_count_range=None, mean_type='geomean', label_dominated_points=True):
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
        x = points[:, 0]
        y = points[:, 1]

        memory_values = x / 1024 / 1024
        memory_unit = 'MB'
        time_values = y
        time_unit = 'ms'

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
            group_time = time_values[group_mask]
            if label_dominated_points and np.any(~is_pareto):
                ax.scatter(group_memory[~is_pareto], group_time[~is_pareto],
                           c='lightgray', s=80, alpha=0.5,
                           marker='o', edgecolors='gray', linewidth=1, zorder=1)

            ax.scatter(group_memory[is_pareto], group_time[is_pareto],
                       c=group_color, s=180, alpha=0.9,
                       edgecolors='black', linewidth=2.5,
                       marker=group_marker, label=group_name, zorder=3)

            pareto_indices = np.where(is_pareto)[0]
            if len(pareto_indices) > 1:
                pareto_sorted = sorted(
                    pareto_indices, key=lambda i: group_memory[i])
                pareto_x = [group_memory[i] for i in pareto_sorted]
                pareto_y = [group_time[i] for i in pareto_sorted]
                ax.plot(pareto_x, pareto_y, color=group_color,
                        linestyle=group_linestyle, alpha=0.8, linewidth=3, zorder=2)

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
        ax.set_xlabel(
            f'Memory Utilization ({memory_unit})', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'Trace Time ({time_unit})',
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

    title = 'Trace Time vs Memory Utilization'
    if ray_count_range is not None:
        title += f'\n{mean_type} ({ray_count_range[0]:,} - {ray_count_range[1]:,})'
    else:
        title += f'\n{mean_type} (all)'
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
            results_dir, f'{name}-{mean_type}{log2(ray_count_range[0])}-{log2(ray_count_range[1])}.pdf')
    else:
        output_file = os.path.join(
            results_dir, f'{name}-{mean_type}.pdf')

    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Pareto frontier plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pareto_plotter.py <data_file> [mean_type]")
        print("  mean_type: 'arithmetic' (default) or 'geomean'")
        sys.exit(1)

    filename = sys.argv[1]
    mean_type = sys.argv[2] if len(sys.argv) > 2 else 'arithmetic'

    if mean_type not in ['geomean', 'arithmetic']:
        print(
            f"Error: mean_type must be 'geomean' or 'arithmetic', got '{mean_type}'")
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

    raw_data, machine_type = parse_trace_scaling_data(data_text)
    raw_data, machine_type = parse_trace_scaling_data(data_text)
    memory_utilization = {}
    embree_mu = {'lucy': {'embree-bvh8i': {'memory': 1184515489, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 2022579372, 'nodes': {'aabbs': 1}}}, 'sheep': {'embree-bvh8i': {'memory': 85013299, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 170263576, 'nodes': {'aabbs': 1}}}, 'san-miguel-x35-y22-z47': {'embree-bvh8i': {'memory': 374480044, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 675439181, 'nodes': {'aabbs': 1}}}, 'hairball': {'embree-bvh8i': {'memory': 99327410,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 191183716, 'nodes': {'aabbs': 1}}}, 'white-oak': {'embree-bvh8i': {'memory': 1367343, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 2521825, 'nodes': {'aabbs': 1}}}, 'sponza': {'embree-bvh8i': {'memory': 10135535, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 17718837, 'nodes': {'aabbs': 1}}}, 'power-plant': {'embree-bvh8i': {'memory': 458345152, 'nodes': {'aabbs': 1}}, 'embree-bvh8v': {'memory': 833888452, 'nodes': {'aabbs': 1}}}}

    for model, layouts in embree_mu.items():
        assert model in memory_utilization
        memory_utilization[model].update(layouts)

    trace_data = process_trace_data(
        raw_data, ray_count_range=None, mean_type=mean_type)
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

    layout_groups = {
        'bvh8': ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16',
                 'cl-bvh8-idx', 'cl-bvh8-idx-align16',
                 ],
        'bvh2': ['eq', 'pbrt', 'pbrt-align16', 'eq-align16',
                 'ptr', 'soaos-align16', 'soaos',
                 ],
        'embree': ['embree-bvh8i', 'embree-bvh8v'],
    }

    plot_pareto_frontiers_geomean(trace_data, memory_utilization, layout_groups,
                                  filename, machine_type, mean_type)

    for R in [(2**18, 2**22)]:
        if machine_type != "cpu":
            continue
        trace_data = process_trace_data(raw_data, R, mean_type)
        plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                              filename, machine_type, R, mean_type)
    for R in [(2**22, 2**25)]:
        if machine_type != "cuda":
            continue
        trace_data = process_trace_data(raw_data, R, mean_type)
        plot_pareto_frontiers(trace_data, memory_utilization, layout_groups,
                              filename, machine_type, R, mean_type)
