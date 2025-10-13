import sys
import os
from collections import defaultdict
import numpy as np

DEFAULT_LAYOUT_GROUPS = {
    'bvh8': ['bvh8', 'bvh8-align16', 'cl-bvh8', 'cl-bvh8-align16',
             'cl-bvh8-idx', 'cl-bvh8-idx-align16',
             # 'cl-cw-bvh8-idx', 'cl-cw-bvh8-idx-align16',
             ],
    'bvh2': ['eq', 'pbrt', 'pbrt-align16', 'eq-align16',
             'ptr', 'soaos-align16', 'soaos',
             ],
    'embree': ['embree-bvh4', 'embree-bvh8'],
}


def parse_benchmark_file(filename):
    """Parse benchmark file and organize data by model, machine, layout, and size."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with open(filename, 'r') as f:
        lines = f.readlines()

    current_model = None
    current_config = None  # "machine, layout"
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a config line (starts with "rt,")
        if line.startswith('rt,'):
            parts = line.split(',')
            if len(parts) >= 3:
                # Format: rt, machine, layout
                machine = parts[1].strip()
                layout = parts[2].strip()
                current_config = f"{machine}, {layout}"
            i += 1
            continue

        if line.isdigit() and current_model and current_config:
            size = int(line)
            if i + 1 < len(lines) and 'hits' in lines[i + 1]:
                hits_line = lines[i + 1].strip()
                hits = int(hits_line.split(':')[1].strip())
                results[current_model][current_config][size].append(hits)
            i += 1
        elif line and not line.startswith('rt,') and 'hits' not in line and 'trace time' not in line:
            current_model = line
            i += 1
        else:
            i += 1

    return results


def get_layout_from_config(config):
    parts = config.split(',')
    if len(parts) >= 2:
        return parts[1].strip()
    return config


def filter_by_layout_group(results, layout_groups):
    valid_layouts = set()
    for group_layouts in layout_groups.values():
        valid_layouts.update(group_layouts)

    filtered = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model in results:
        for config in results[model]:
            layout = get_layout_from_config(config)
            if layout in valid_layouts:
                filtered[model][config] = results[model][config]

    return filtered


def analyze_differences(results):
    for model in sorted(results.keys()):
        print(f"\n{'='*80}")
        print(f"Model: {model}")
        print('='*80)

        ray_counts = set()
        for config in results[model].keys():
            ray_counts.update(results[model][config].keys())
        ray_counts = sorted(ray_counts)

        for ray_count in ray_counts:
            ptr_avg = None
            for config in results[model].keys():
                layout = get_layout_from_config(config)
                if layout == 'ptr' and ray_count in results[model][config]:
                    hits_list = results[model][config][ray_count]
                    if hits_list:
                        ptr_avg = np.mean(hits_list)
                        break

            if ptr_avg is None:
                print(f"\n  Ray count {ray_count}: No ptr baseline found")
                continue

            print(
                f"\n  Ray count {ray_count} (ptr baseline: {int(ptr_avg)} hits):")
            has_differences = False
            for config in sorted(results[model].keys()):
                if ray_count in results[model][config]:
                    hits_list = results[model][config][ray_count]
                    if hits_list:
                        avg = np.mean(hits_list)
                        diff = avg - ptr_avg
                        layout = get_layout_from_config(config)
                        if abs(diff) > 0.5:
                            has_differences = True
                            sign = '+' if diff > 0 else ''
                            print(
                                f"    {layout:30} {sign}{int(diff):6} hits (avg: {int(avg)})")

            if not has_differences:
                print(f"    All layouts have the same hits as ptr")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_hits.py <hits_file> [layout_group]")
        print("\nAvailable layout groups:")
        for group_name in DEFAULT_LAYOUT_GROUPS.keys():
            print(f"  - {group_name}")
        print("\nIf no layout_group is specified, all layouts will be used.")
        sys.exit(1)

    filename = sys.argv[1]
    if len(sys.argv) >= 3:
        group_name = sys.argv[2]
        if group_name in DEFAULT_LAYOUT_GROUPS:
            layout_groups = {group_name: DEFAULT_LAYOUT_GROUPS[group_name]}
            print(f"Using layout group: {group_name}")
        else:
            print(f"Error: Unknown layout group '{group_name}'")
            print("Available groups:", ", ".join(DEFAULT_LAYOUT_GROUPS.keys()))
            sys.exit(1)
    else:
        layout_groups = DEFAULT_LAYOUT_GROUPS
        print("Using all layout groups")

    results = parse_benchmark_file(filename)
    filtered_results = filter_by_layout_group(results, layout_groups)
    analyze_differences(filtered_results)
