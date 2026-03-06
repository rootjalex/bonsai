#!/usr/bin/env python3
"""Find layouts that are not pareto-optimal in any evaluation context."""

import csv
from collections import defaultdict

# Ray counts we evaluate across: 2^18, 2^19, 2^20, 2^21, 2^22, 2^23
RAY_COUNTS = [262144, 524288, 1048576, 2097152, 4194304, 8388608]


def is_dominated(point, other_points):
    """Check if point is dominated by any other point.

    A point (latency, memory) is dominated if there exists another point that is
    <= in both dimensions and strictly < in at least one dimension.
    """
    latency, memory = point
    for other_latency, other_memory in other_points:
        if other_latency <= latency and other_memory <= memory:
            if other_latency < latency or other_memory < memory:
                return True
    return False


def compute_latency(times_by_ray_count):
    """Compute weighted average latency (time per ray) across ray counts.

    For each ray_count:
    - Drop 2 lowest and 2 highest times
    - Average the remaining times
    - Compute latency = avg_time / ray_count

    Then compute weighted average of latencies across ray counts.
    """
    total_weighted_latency = 0.0
    total_weight = 0

    for ray_count in RAY_COUNTS:
        times = times_by_ray_count.get(ray_count, [])
        if len(times) < 5:
            # Not enough samples to drop 2 lowest and 2 highest
            continue

        # Sort and drop 2 lowest and 2 highest
        sorted_times = sorted(times)
        trimmed_times = sorted_times[2:-2]

        if not trimmed_times:
            continue

        avg_time = sum(trimmed_times) / len(trimmed_times)
        latency = avg_time / ray_count

        total_weighted_latency += ray_count * latency
        total_weight += ray_count

    if total_weight == 0:
        return float('inf')

    return total_weighted_latency / total_weight


def main():
    # Read CSV data
    data = []
    with open('rt-results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Group by context (app, machine, scene, ray_type) - ray_count is aggregated
    # For each (context, layout, ray_count), collect all trace times
    # key: (context, layout) -> {ray_count: [times]}
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    memory_data = defaultdict(lambda: defaultdict(lambda: float('inf')))

    for row in data:
        layout = row['layout']
        if layout.startswith('embree'):
            continue
        context = (row['app'], row['machine'], row['scene'], row['ray_type'])
        ray_count = int(row['ray_count'])
        trace_time = float(row['trace_time-ms'])
        memory = float(row['total-memory-b'])

        raw_data[context][layout][ray_count].append(trace_time)
        memory_data[context][layout] = min(memory_data[context][layout], memory)

    # Compute latency for each (context, layout)
    context_layouts = defaultdict(dict)

    for context, layouts in raw_data.items():
        for layout, times_by_ray_count in layouts.items():
            latency = compute_latency(times_by_ray_count)
            memory = memory_data[context][layout]
            context_layouts[context][layout] = {'latency': latency, 'memory': memory}

    # For each layout, check if it's pareto-optimal in any context
    all_layouts = set()
    pareto_optimal_layouts = {}  # layout -> example context where it's pareto-optimal

    for context, layouts in context_layouts.items():
        all_layouts.update(layouts.keys())

        # Get all points for this context
        all_points = [(layouts[l]['latency'], layouts[l]['memory']) for l in layouts]

        for layout, values in layouts.items():
            point = (values['latency'], values['memory'])
            other_points = [p for p in all_points if p != point]

            # A layout is pareto-optimal if it's not dominated
            if not is_dominated(point, other_points):
                if layout not in pareto_optimal_layouts:
                    pareto_optimal_layouts[layout] = context

    # Layouts that are NOT pareto-optimal in ANY context
    non_pareto_layouts = all_layouts - set(pareto_optimal_layouts.keys())

    print("Layouts that are NOT pareto-optimal in any context:")
    for layout in sorted(non_pareto_layouts):
        print(f"  {layout}")

    print(f"\nTotal: {len(non_pareto_layouts)} out of {len(all_layouts)} layouts")

    print("\nPareto-optimal layouts (with example context):")
    for layout in sorted(pareto_optimal_layouts.keys()):
        ctx = pareto_optimal_layouts[layout]
        print(f"  {layout}: {ctx}")


if __name__ == '__main__':
    main()
