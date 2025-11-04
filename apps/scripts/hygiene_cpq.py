import os
import re
import sys
import argparse
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


def parse_cpq_benchmark_data(
    filename: str,
    machine: str,
    layout_group: Dict[str, List[str]],
    memory_util: Optional[Dict[str, Dict[str, float]]] = None,
    output_file: Optional[str] = None
) -> List[Tuple]:
    """Parse walk on spheres (CPQ) benchmark data."""

    with open(filename, 'r') as f:
        lines = f.readlines()

    results = []

    layout_to_group = {}
    for group, layouts in layout_group.items():
        for layout in layouts:
            layout_to_group[layout] = group

    if memory_util is None:
        memory_util = {}

    query_count = 2**20  # Fixed query count for cuda is 2^20
    i = 0
    current_scene = None
    current_layout = None

    while i < len(lines):
        line = lines[i].strip()

        # Check for section separator
        if line == '---':
            current_scene = None
            current_layout = None
            i += 1
            continue

        # Check for empty lines
        if not line:
            i += 1
            continue

        # Check if this is a scene name (standalone line without commas)
        if ',' not in line and not line.startswith('wos'):
            current_scene = line
            print(f"\nProcessing scene: {current_scene}")
            i += 1
            continue

        # Check if this is a "wos, machine, layout" line
        if line.startswith('wos,'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 3:
                _, _, current_layout = parts
                print(f"  Layout: {current_layout}")
            i += 1
            continue

        # Check if this is a measurement line: "layout, scene, time"
        if ',' in line and current_scene and current_layout:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 3:
                layout, scene, time_ms = parts
                if layout == current_layout and scene == current_scene:
                    try:
                        time_value = int(time_ms)
                        group = layout_to_group.get(layout)

                        # Get memory utilization for the scene
                        total_memory = None
                        bvh_memory = None
                        if scene in memory_util and layout in memory_util[scene]:
                            total_memory = memory_util[scene][layout]['memory']
                            if scene in MODEL_TO_TRIANGLES:
                                bvh_memory = total_memory - \
                                    (MODEL_TO_TRIANGLES[scene]
                                     * TRIANGLE_BYTES)

                        results.append((
                            "cpq",
                            machine,
                            scene,
                            layout,
                            query_count,
                            time_value,
                            "NA",  # fcpw-cpq-time-ms
                            "NA",  # build-time-ms
                            "NA",  # fcpw-build-time-ms
                            group,
                            total_memory,
                            bvh_memory
                        ))
                    except ValueError:
                        pass  # Skip lines where time is not a valid integer

        i += 1

    # Use provided output file or default
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
                "app,machine,scene,layout,query-count,cpq-time-ms,fcpw-cpq-time-ms,"
                "build-time-ms,fcpw-build-time-ms,group,total-memory-b,bvh-memory-b\n"
            )
        for data in results:
            to_csv = ','.join(str(x) for x in data)
            f.write(f"{to_csv}\n")

    print(f"\nTotal: Parsed {len(results)} measurements")
    return results


def parse_benchmark_data(
    filename: str,
    machine: str,
    layout_group: Dict[str, List[str]],
    memory_util: Optional[Dict[str, Dict[str, float]]] = None,
    output_file: Optional[str] = None
) -> List[Tuple]:
    """Parse closest point query benchmark data."""

    with open(filename, 'r') as f:
        lines = f.readlines()

    results = []

    layout_to_group = {}
    for group, layouts in layout_group.items():
        for layout in layouts:
            layout_to_group[layout] = group

    if memory_util is None:
        memory_util = {}

    i = 0
    block_count = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for header lines that start with "---"
        if line.startswith('---') and line.endswith('---'):
            block_count += 1
            # Parse header: "--- lucy - bvh8-q8-ci - 10000 ---"
            header = line.replace('---', '').strip()

            print(f"\nBlock {block_count}: {line}")

            # Split by ' - ' (with spaces) to handle layouts with dashes
            header_parts = [p.strip() for p in header.split(' - ')]

            if len(header_parts) != 3:
                print(
                    f"  Skipping - header has {len(header_parts)} parts, expected 3")
                i += 1
                continue

            scene, layout, query_count = header_parts
            print(
                f"  Scene: {scene}, Layout: {layout}, Query count: {query_count}")

            i += 1
            measurement_count = 0

            # Parse measurements until we hit another --- or end of file
            while i < len(lines):
                line = lines[i].strip()

                # Check if we hit next section
                if line.startswith('---') and line.endswith('---'):
                    break

                if not line:
                    i += 1
                    continue

                # Skip the "scene, query_count" lines
                if ',' in line and not line.startswith('['):
                    i += 1
                    continue

                # Look for start of a measurement (fcpw tree construction comes first)
                fcpw_build_match = re.match(
                    r'\[fcpw\]\s+tree construction\s*:\s*(\d+)\s*ms', line)
                if fcpw_build_match:
                    fcpw_build = int(fcpw_build_match.group(1))

                    # Next line should be bonsai tree construction
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    bonsai_build_match = re.match(
                        r'\[bonsai\]\s+tree construction\s*:\s*(\d+)\s*ms', line)
                    if not bonsai_build_match:
                        continue
                    bonsai_build = int(bonsai_build_match.group(1))

                    # Next line should be fcpw closest point query
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    fcpw_cpq_match = re.match(
                        r'\[fcpw\]\s+closest point query\s*:\s*(\d+)\s*ms', line)
                    if not fcpw_cpq_match:
                        continue
                    fcpw_cpq = int(fcpw_cpq_match.group(1))

                    # Next line should be bonsai closest point query
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    bonsai_cpq_match = re.match(
                        r'\[bonsai\]\s+closest point query\s*:\s*(\d+)\s*ms', line)
                    if not bonsai_cpq_match:
                        continue
                    bonsai_cpq = int(bonsai_cpq_match.group(1))

                    # We have a complete measurement!
                    measurement_count += 1
                    group = layout_to_group.get(layout)

                    # Get memory utilization for the scene
                    total_memory = None
                    bvh_memory = None
                    if scene in memory_util and layout in memory_util[scene]:
                        total_memory = memory_util[scene][layout]['memory']
                        if scene in MODEL_TO_TRIANGLES:
                            bvh_memory = total_memory - \
                                (MODEL_TO_TRIANGLES[scene] * TRIANGLE_BYTES)

                    results.append((
                        "cpq",
                        machine,
                        scene,
                        layout,
                        query_count,
                        bonsai_cpq,
                        fcpw_cpq,
                        bonsai_build,
                        fcpw_build,
                        group,
                        total_memory,
                        bvh_memory
                    ))

                i += 1

            print(f"  Found {measurement_count} measurements in this block")
            continue

        i += 1

    # Use provided output file or default
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
                "app,machine,scene,layout,query-count,cpq-time-ms,fcpw-cpq-time-ms,"
                "build-time-ms,fcpw-build-time-ms,group,total-memory-b,bvh-memory-b\n"
            )
        for data in results:
            to_csv = ','.join(str(x) for x in data)
            f.write(f"{to_csv}\n")

    print(f"\nTotal: Parsed {len(results)} measurements")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse benchmark data')
    parser.add_argument('input_file', help='Input benchmark data file')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--machine', required=True, choices=['x86', 'cuda', 'arm'],
                        help='Machine type: x86, cuda, or arm')

    args = parser.parse_args()

    layout_groups = layout_grouping.retrieve_layout_groups()
    memory_utilization = memory.retrieve_memory_utilization()

    if args.machine == 'cuda':
        parse_cpq_benchmark_data(
            args.input_file,
            args.machine,
            layout_groups,
            memory_utilization,
            args.output_file
        )
    else:
        parse_benchmark_data(
            args.input_file,
            args.machine,
            layout_groups,
            memory_utilization,
            args.output_file
        )
