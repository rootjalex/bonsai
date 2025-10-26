import os
import re
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import layout_grouping


# triangle counts
MODEL_TO_TRIANGLES = {
    'hairball': 2880000,
    'dragon': 871414,
    'hairball_60_70_10': 2880000,
    'dragon_60_70_10': 871414,
}
TRIANGLE_BYTES = 36  # bonsai sizeof(Triangle)


def parse_benchmark_data(
    filename: str,
    machine: str,
    layout_group: Dict[str, List[str]],
    memory_util: Optional[Dict[str, Dict[str, float]]] = None,
    output_file: Optional[str] = None
) -> List[Tuple]:
    """Parse collision detection benchmark data."""

    with open(filename, 'r') as f:
        lines = f.readlines()

    results = []

    layout_to_group = {}
    for group, layouts in layout_group.items():
        for layout in layouts:
            layout_to_group[layout] = group

    i = 0
    block_count = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for header lines that start with "---"
        if line.startswith('---') and line.endswith('---'):
            block_count += 1
            # Parse header: "--- hairball - hairball_60_70_10 - fcl ---"
            # or "--- hairball - hairball_60_70_10 - pbrt-q16 ---"
            header = line.replace('---', '').strip()

            print(f"\nBlock {block_count}: {line}")

            # Split by ' - ' (with spaces) to handle layouts with dashes
            header_parts = [p.strip() for p in header.split(' - ')]

            if len(header_parts) != 3:
                print(
                    f"  Skipping - header has {len(header_parts)} parts, expected 3")
                i += 1
                continue

            scene1, scene2, layout = header_parts
            print(f"  Scene1: {scene1}, Scene2: {scene2}, Layout: {layout}")

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

                # Look for start of a measurement (bonsai tree construction)
                bonsai_build_match = re.match(
                    r'\[bonsai\]\s+tree construction\s*:\s*(\d+)\s*ms', line)
                if bonsai_build_match:
                    bonsai_build = int(bonsai_build_match.group(1))

                    # Next line should be bonsai collision detection
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    bonsai_collision_match = re.match(
                        r'\[bonsai\]\s+collision detection\s*:\s*(\d+)\s*ms', line)
                    if not bonsai_collision_match:
                        continue
                    bonsai_collision = int(bonsai_collision_match.group(1))

                    # Next line should be fcl tree construction
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    fcl_build_match = re.match(
                        r'\[fcl\]\s+tree construction\s*:\s*(\d+)\s*ms', line)
                    if not fcl_build_match:
                        continue
                    fcl_build = int(fcl_build_match.group(1))

                    # Next line should be fcl collision detection
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    fcl_collision_match = re.match(
                        r'\[fcl\]\s+collision detection\s*:\s*(\d+)\s*ms', line)
                    if not fcl_collision_match:
                        continue
                    fcl_collision = int(fcl_collision_match.group(1))

                    # Next line should be collision count
                    i += 1
                    if i >= len(lines):
                        break
                    line = lines[i].strip()
                    collision_count_match = re.match(
                        r'collision count:\s*(\d+)', line)
                    if not collision_count_match:
                        continue
                    collision_count = int(collision_count_match.group(1))

                    # We have a complete measurement!
                    measurement_count += 1
                    group = layout_to_group.get(layout)

                    # Calculate triangle memory for both scenes
                    triangle_memory = 0
                    if scene1 in MODEL_TO_TRIANGLES:
                        triangle_memory += MODEL_TO_TRIANGLES[scene1] * \
                            TRIANGLE_BYTES
                    if scene2 in MODEL_TO_TRIANGLES:
                        triangle_memory += MODEL_TO_TRIANGLES[scene2] * \
                            TRIANGLE_BYTES

                    results.append((
                        "cd",
                        machine,
                        scene1,
                        scene2,
                        layout,
                        collision_count,
                        bonsai_collision,
                        fcl_collision,
                        bonsai_build,
                        fcl_build,
                        group,
                        triangle_memory
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
                "app,machine,scene1,scene2,layout,collisions,collision-time-ms,fcl-collision-time-ms,"
                "build-time-ms,fcl-build-time-ms,group,triangle-memory-b\n"
            )
        for data in results:
            to_csv = ','.join(str(x) for x in data)
            f.write(f"{to_csv}\n")

    print(f"\nTotal: Parsed {len(results)} measurements")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parse collision detection benchmark data')
    parser.add_argument('input_file', help='Input benchmark data file')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--machine', required=True, choices=['x86', 'cuda', 'arm'],
                        help='Machine type: x86, cuda, or arm')

    args = parser.parse_args()

    layout_groups = layout_grouping.retrieve_layout_groups()

    parse_benchmark_data(
        args.input_file,
        args.machine,
        layout_groups,
        {},  # TODO(cgyurgyik): get memory utilization for CD.
        args.output_file
    )
