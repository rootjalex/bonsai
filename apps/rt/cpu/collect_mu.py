import matplotlib.pyplot as plt
import numpy as np
import re
import math
import os
from collections import defaultdict
import sys


def parse_layout_memory_and_nodes(data_text):
    """Parse layout data to extract memory utilization per model."""
    lines = data_text.strip().split('\n')
    data = defaultdict(lambda: defaultdict(lambda: {'memory': 0, 'nodes': {}}))

    current_model = None
    current_layout = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == '---':
            i += 1
            continue

        # Model name
        if (',' not in line and ':' not in line and not line.isdigit()
                and not line.startswith(';;') and not line.startswith('./')):
            current_model = line
            current_layout = None
            i += 1
            continue

        # Configuration line
        if line.startswith('rt, '):
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 3:
                current_layout = parts[2]
                if current_model and current_layout:
                    data[current_model][current_layout] = {
                        'memory': 0, 'nodes': {}}
            i += 1
            continue

        # Node definition line
        if line.startswith(';;') and current_model and current_layout:
            match = re.match(r';;\s*(\S+):\s*(\d+),(\d+)', line)
            if match:
                node_type = match.group(1)
                size = int(match.group(2))
                count = int(match.group(3))

                data[current_model][current_layout]['nodes'][node_type] = count
                # Add to memory utilization
                data[current_model][current_layout]['memory'] += size * count
            i += 1
            continue

        i += 1

    # For 'ptr' layouts, add node memory based on pbrt node count
    for model in data:
        if 'pbrt' in data[model] and 'ptr' in data[model]:
            pbrt_node_count = data[model]['pbrt']['nodes'].get('nodes', 0)
            pointer_node_memory = pbrt_node_count * 44

            # Add to ptr layout's memory
            data[model]['ptr']['memory'] += pointer_node_memory

            # Store node memory in ptr layout's nodes dictionary as 'nodes'
            data[model]['ptr']['nodes']['nodes'] = pointer_node_memory

    # Convert to regular dict
    result = {}
    for model in data:
        result[model] = {}
        for layout in data[model]:
            if data[model][layout]['memory'] > 0:
                result[model][layout] = {
                    'memory': data[model][layout]['memory'],
                    'nodes': dict(data[model][layout]['nodes'])
                }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pareto_plotter.py <data_file> [ray_count]")
        sys.exit(1)

    filename = sys.argv[1]
    ray_count = int(sys.argv[2]) if len(sys.argv) > 2 else None

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

    # Parse data
    print(parse_layout_memory_and_nodes(data_text))
