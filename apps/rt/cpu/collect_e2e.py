import matplotlib.pyplot as plt
import numpy as np
import re
import math
import os
from collections import defaultdict


def parse_benchmark_data(data_text):
    lines = [line.strip()
             for line in data_text.strip().split('\n') if line.strip()]

    ray_count = None
    if lines and lines[0].isdigit():
        ray_count = int(lines[0])
        data_text = '\n'.join(data_text.strip().split('\n')[1:])

    sections = data_text.strip().split('---')
    parsed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    machine_type = None
    all_layouts = set()
    hit_ratios = {}

    for section in sections:
        if not section.strip():
            continue

        lines = [line.strip()
                 for line in section.strip().split('\n') if line.strip()]
        model_name = lines[0]

        i = 1
        while i < len(lines):
            if ',' in lines[i]:
                config_parts = [part.strip() for part in lines[i].split(',')]
                if machine_type is None and len(config_parts) >= 2:
                    machine_type = config_parts[1]
                layout = config_parts[2]
                all_layouts.add(layout)

                i += 1
                while i < len(lines) and ':' in lines[i]:
                    key, value = lines[i].split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if 'ms' in value:
                        numeric_value = int(re.findall(r'\d+', value)[0])
                    else:
                        numeric_value = int(value)

                    parsed_data[model_name][layout][key].append(numeric_value)

                    if key == 'hits' and model_name not in hit_ratios and ray_count is not None:
                        hit_ratios[model_name] = numeric_value / ray_count

                    i += 1
            else:
                i += 1

    layouts = sorted(list(all_layouts))
    return parsed_data, ray_count, machine_type, layouts, hit_ratios


def calculate_average(values, method='arithmetic'):
    if len(values) <= 4:
        filtered_values = values
    else:
        sorted_values = sorted(values)
        filtered_values = sorted_values[2:-2]

    if method == 'geometric':
        if any(v <= 0 for v in filtered_values):
            return 0
        return math.exp(sum(math.log(v) for v in filtered_values) / len(filtered_values))
    else:
        return sum(filtered_values) / len(filtered_values)


def process_data(raw_data, method='arithmetic'):
    processed_data = {}
    for model in raw_data:
        processed_data[model] = {}
        for layout in raw_data[model]:
            processed_data[model][layout] = {}
            for key in raw_data[model][layout]:
                values = raw_data[model][layout][key]
                processed_data[model][layout][key] = calculate_average(
                    values, method)
    return processed_data


def format_ray_count(ray_count):
    assert ray_count is not None
    assert ray_count > 0, ray_count
    log10_val = math.log10(ray_count)
    if abs(log10_val - round(log10_val)) < 1e-10:
        exponent = int(round(log10_val))
        return f"10^{exponent}"

    if (ray_count & (ray_count - 1)) == 0:
        exponent = int(math.log2(ray_count))
        return f"2^{exponent}"

    return f"{ray_count:,}"


def validate_hits_consistency(data):
    print("Validating hits consistency:")
    print("-" * 40)

    for model in data:
        hits_values = []
        for layout in data[model]:
            hits = data[model][layout]['hits']
            hits_values.append(hits)

        if len(set(hits_values)) == 1:
            print(f"✓ {model}: All layouts have {hits_values[0]:.1f} hits")
        else:
            print(
                f"⚠ {model}: Inconsistent hits across layouts: {[f'{h:.1f}' for h in hits_values]}")
    print()


def create_plots(data, ray_count, machine_type, layouts, hit_ratios, input_path, method='arithmetic'):
    models = list(data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#8B5A3C', '#4A90E2']

    results_dir = os.path.join(os.path.dirname(input_path))
    os.makedirs(results_dir, exist_ok=True)

    plt.style.use('default')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ray_display = format_ray_count(ray_count)
    method_str = 'geomean' if method == 'geometric' else 'arithmetic mean'
    title = f'closest hit ray tracing ({method_str})'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    x_pos = np.arange(len(models))
    assert len(layouts) > 0, layouts
    width = 0.8 / len(layouts)

    model_labels = []
    for model in models:
        model_labels.append(f'{model}')

    ax1 = axes[0]
    for i, layout in enumerate(layouts):
        values = [data[model][layout]['canonical tree'] for model in models]
        bars = ax1.bar(x_pos + i*width, values, width,
                       label=layout.upper(), color=colors[i % len(colors)], alpha=0.8)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{value:.1f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Canonical Tree Build Time')
    ax1.set_xticks(x_pos + width * (len(layouts) - 1) / 2)
    ax1.set_xticklabels(model_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, layout in enumerate(layouts):
        values = [data[model][layout]['specialized tree'] for model in models]
        bars = ax2.bar(x_pos + i*width, values, width,
                       label=layout.upper(), color=colors[i % len(colors)], alpha=0.8)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.annotate(f'{value:.1f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Specialized Tree Build Time')
    ax2.set_xticks(x_pos + width * (len(layouts) - 1) / 2)
    ax2.set_xticklabels(model_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    model_labels = []
    for model in models:
        assert (model in hit_ratios), model
        hit_percentage = hit_ratios[model] * 100
        model_labels.append(f'{model} ({hit_percentage:.1f}%)')
    ax3 = axes[2]
    for i, layout in enumerate(layouts):
        values = [data[model][layout]['trace time'] for model in models]
        bars = ax3.bar(x_pos + i*width, values, width,
                       label=layout.upper(), color=colors[i % len(colors)], alpha=0.8)

    ax3.set_xlabel('Model')
    ax3.set_ylabel('Time (ms)')
    trace_title = f'Trace Time ({ray_display} rays)'
    ax3.set_title(trace_title)
    ax3.set_xticks(x_pos + width * (len(layouts) - 1) / 2)
    ax3.set_xticklabels(model_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    all_trace_zero = all(data[model][layout]['trace time'] == 0
                         for model in models for layout in layouts)
    if all_trace_zero:
        ax3.text(0.5, 0.5, 'All trace times are 0ms',
                 transform=ax3.transAxes, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    plt.tight_layout()
    method_suffix = 'geomean' if method == 'geometric' else 'arithmetic'
    output_path = os.path.join(
        results_dir, f'chrt_{ray_count}_{method_suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()


def print_summary_table(data, layouts, method='arithmetic'):
    method_str = 'geomean' if method == 'geometric' else 'arithmetic mean'
    print(f"Summary Table ({method_str}):")
    print("=" * 80)

    header = f"{'Model':<12} {'Layout':<8} {'Canonical':<12} {'Specialized':<12} {'Trace':<10}"
    print(header)
    print("-" * 80)

    for model in data:
        for i, layout in enumerate(layouts):
            canonical = data[model][layout]['canonical tree']
            specialized = data[model][layout]['specialized tree']
            trace = data[model][layout]['trace time']

            model_name = model if i == 0 else ""
            row = f"{model_name:<12} {layout.upper():<8} {canonical:<12.1f} {specialized:<12.1f} {trace:<10.1f}"
            print(row)

        if model != list(data.keys())[-1]:
            print("-" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: python benchmark_parser.py <benchmark_data_file> [arithmetic|geometric]")
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
            benchmark_data = file.read()
        print(f"Successfully loaded data from: {filename}\n")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{filename}': {e}")
        sys.exit(1)

    raw_data, ray_count, machine_type, layouts, hit_ratio = parse_benchmark_data(
        benchmark_data)
    processed_data = process_data(raw_data, method)

    validate_hits_consistency(processed_data)
    print_summary_table(processed_data, layouts, method)
    print()
    create_plots(processed_data, ray_count, machine_type,
                 layouts, hit_ratio, filename, method)
