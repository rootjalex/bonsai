import os
import re
import sys
from typing import List, Tuple, Dict, Optional
import memory
import layout_grouping


# triangle counts
MODEL_TO_TRIANGLES = {
    'lucy': 28055728,
    'sheep': 2967664,
    'san-miguel-x35-y22-z47': 9832536,
    'hairball': 2880000,
    'white-oak': 36760,
    'sponza': 262267,
    'power-plant': 12759246,
}
TRIANGLE_BYTES = 36  # bonsai sizeof(Triangle)


def parse_filename(filename: str) -> Tuple[str, str]:
    """
        x86-camera.txt -> (x86, primary)
        cuda-secondary.txt -> (cuda, secondary)

    returns `(machine, ray_type)`
    """
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    parts = name_without_ext.split('-', 1)
    if len(parts) != 2:
        raise ValueError(
            f"Filename {basename} must be in format <machine>-<ray_type>.txt")

    machine, ray_type = parts

    # Convert 'camera' to 'primary' for sake of uniformity.
    if ray_type == 'camera':
        ray_type = 'primary'

    return machine, ray_type


def parse_benchmark_data(
    filename: str,
    layout_group: Dict[str, List[str]],
    memory_util: Optional[Dict[str, Dict[str, float]]] = None,
    output_file: Optional[str] = None
) -> List[Tuple]:
    """Don't be lazy, read the code bud."""
    machine, ray_type = parse_filename(filename)

    with open(filename, 'r') as f:
        data = f.read()

    results = []

    scenes = data.strip().split('---')

    layout_to_group = {}
    for group, layouts in layout_group.items():
        for layout in layouts:
            layout_to_group[layout] = group

    if memory_util is None:
        memory_util = {}

    for scene_block in scenes:
        scene_block = scene_block.strip()
        if not scene_block:
            continue

        lines = scene_block.split('\n')
        scene_name = lines[0].strip()

        current_layout = None
        application = None
        current_ray_count = None
        current_hits = None

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            if line.startswith('rt,'):
                parts = line.split(',')
                if len(parts) >= 3:
                    application = parts[0].strip()
                    current_layout = parts[2].strip()
                current_ray_count = None
                current_hits = None
                continue

            if line.isdigit() and current_layout:
                current_ray_count = int(line)
                continue

            hits_match = re.match(r'hits\s*:\s*(\d+)', line)
            if hits_match and current_ray_count is not None:
                current_hits = int(hits_match.group(1))
                continue

            trace_match = re.match(r'trace time\s*:\s*(\d+)\s*ms', line)
            if trace_match and current_hits is not None and current_ray_count is not None:
                trace_time = int(trace_match.group(1))
                group = layout_to_group.get(current_layout)

                total_memory = None
                if scene_name in memory_util and current_layout in memory_util[scene_name]:
                    total_memory = memory_util[scene_name][current_layout]['memory']

                results.append((
                    application,
                    machine,
                    scene_name,
                    current_layout,
                    current_ray_count,
                    current_hits,
                    trace_time,
                    group,
                    ray_type,
                    total_memory,
                    # This isn't totally accurate for Embree, which may compress triangles further.
                    total_memory -
                    (MODEL_TO_TRIANGLES[scene_name] * TRIANGLE_BYTES)

                ))
                assert all(x is not None for x in results[-1]), results[-1]

    # Use provided output file or default to input directory
    if output_file is None:
        results_dir = os.path.dirname(
            filename) if os.path.dirname(filename) else '.'
        name = os.path.splitext(os.path.basename(filename))[0]
        output_file = os.path.join(results_dir, f'{name}.csv')

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(output_file)

    with open(output_file, 'a') as f:
        # Write header only if file doesn't exist
        if not file_exists:
            f.write(
                "app,machine,scene,layout,ray_count,hits,trace_time-ms,group,ray_type,total-memory-b,bvh-memory-b\n")
        for data in results:
            to_csv = ','.join(str(x) for x in data)
            f.write(f"{to_csv}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python hygiene.py <benchmark_data_file> <output_csv_file>")
        sys.exit(1)

    layout_groups = layout_grouping.retrieve_layout_groups()
    memory_utilization = memory.retrieve_memory_utilization()
    parse_benchmark_data(sys.argv[1], layout_groups,
                         memory_utilization, sys.argv[2])
