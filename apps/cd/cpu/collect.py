import re
import sys
from collections import defaultdict


def collect_by_layout(data: str, current_layout: str):
    """Collect benchmark data organized by layout."""
    lines = [line.strip() for line in data.strip().split('\n')]

    # Dictionary to store data per layout
    # Structure: {layout: {'fcl_tree': [], 'fcl_collision': [], 'bonsai_tree': [], 'bonsai_collision': []}}
    layout_data = defaultdict(lambda: {
        'fcl_tree': [],
        'fcl_collision': [],
        'bonsai_tree': [],
        'bonsai_collision': []
    })

    for line in lines:
        # Check if this line contains a layout identifier
        # Adjust this pattern based on your actual log format
        # Common patterns: "Layout: X", "[layout=X]", etc.
        layout_match = re.search(
            r'(?:Layout|layout)[:\s=]+(\w+)', line, re.IGNORECASE)
        if layout_match:
            current_layout = layout_match.group(1)
            continue

        # Skip if we don't have a current layout
        if current_layout is None:
            continue

        # Collect FCL measurements
        if '[fcl]' in line:
            if 'tree construction' in line:
                match = re.search(r': (\d+) ms', line)
                if match:
                    layout_data[current_layout]['fcl_tree'].append(
                        int(match.group(1)))
            elif 'collision detection' in line:
                match = re.search(r': (\d+) ms', line)
                if match:
                    layout_data[current_layout]['fcl_collision'].append(
                        int(match.group(1)))

        # Collect Bonsai measurements
        elif '[bonsai]' in line:
            if 'tree construction' in line:
                match = re.search(r': (\d+) ms', line)
                if match:
                    layout_data[current_layout]['bonsai_tree'].append(
                        int(match.group(1)))
            elif 'collision detection' in line:
                match = re.search(r': (\d+) ms', line)
                if match:
                    layout_data[current_layout]['bonsai_collision'].append(
                        int(match.group(1)))

    return layout_data


def analyze_layout(layout_name, data):
    """Analyze and print statistics for a single layout."""
    print(f"\n{'='*70}")
    print(f"LAYOUT: {layout_name}")
    print(f"{'='*70}")

    fcl_tree = data['fcl_tree']
    fcl_collision = data['fcl_collision']
    bonsai_tree = data['bonsai_tree']
    bonsai_collision = data['bonsai_collision']

    print("\nTotal measurements:")
    print(f"  FCL tree construction:       {len(fcl_tree)}")
    print(f"  FCL collision detection:     {len(fcl_collision)}")
    print(f"  Bonsai tree construction:    {len(bonsai_tree)}")
    print(f"  Bonsai collision detection:  {len(bonsai_collision)}")

    # Check if we have data
    if not fcl_tree or not fcl_collision or not bonsai_tree or not bonsai_collision:
        print("\n  ⚠️  WARNING: Missing data for this layout")
        return

    # Calculate averages
    avg_fcl_tree = sum(fcl_tree) / len(fcl_tree)
    avg_fcl_collision = sum(fcl_collision) / len(fcl_collision)
    avg_bonsai_tree = sum(bonsai_tree) / len(bonsai_tree)
    avg_bonsai_collision = sum(bonsai_collision) / len(bonsai_collision)

    print(f"\nAverages:")
    print(f"  FCL tree construction:       {avg_fcl_tree:.1f} ms")
    print(f"  FCL collision detection:     {avg_fcl_collision:.1f} ms")
    print(f"  Bonsai tree construction:    {avg_bonsai_tree:.1f} ms")
    print(f"  Bonsai collision detection:  {avg_bonsai_collision:.1f} ms")

    # Calculate ratios
    tree_ratio = avg_bonsai_tree / avg_fcl_tree
    collision_ratio = avg_bonsai_collision / avg_fcl_collision

    print(f"\nBonsai vs FCL:")
    print(f"  Tree construction ratio:     {tree_ratio:.3f}x")
    print(f"  Collision detection ratio:   {collision_ratio:.3f}x")

    # Calculate speedup/slowdown with color indicators
    print(f"\nPerformance:")
    if tree_ratio < 1.0:
        print(
            f"  Tree construction:           ✓ Bonsai is {1/tree_ratio:.2f}x faster")
    else:
        print(
            f"  Tree construction:           ✗ Bonsai is {tree_ratio:.2f}x slower")

    if collision_ratio < 1.0:
        print(
            f"  Collision detection:         ✓ Bonsai is {1/collision_ratio:.2f}x faster")
    else:
        print(
            f"  Collision detection:         ✗ Bonsai is {collision_ratio:.2f}x slower")


def print_summary(layout_data):
    """Print overall summary across all layouts."""
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL LAYOUTS")
    print(f"{'='*70}\n")

    print(f"{'Layout':<20} {'Tree Ratio':>15} {'Collision Ratio':>15} {'Overall':>15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")

    for layout, data in sorted(layout_data.items()):
        fcl_tree = data['fcl_tree']
        fcl_collision = data['fcl_collision']
        bonsai_tree = data['bonsai_tree']
        bonsai_collision = data['bonsai_collision']

        if not fcl_tree or not fcl_collision or not bonsai_tree or not bonsai_collision:
            print(f"{layout:<20} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
            continue

        avg_fcl_tree = sum(fcl_tree) / len(fcl_tree)
        avg_fcl_collision = sum(fcl_collision) / len(fcl_collision)
        avg_bonsai_tree = sum(bonsai_tree) / len(bonsai_tree)
        avg_bonsai_collision = sum(bonsai_collision) / len(bonsai_collision)

        tree_ratio = avg_bonsai_tree / avg_fcl_tree
        collision_ratio = avg_bonsai_collision / avg_fcl_collision
        overall_ratio = (tree_ratio + collision_ratio) / 2

        print(
            f"{layout:<20} {tree_ratio:>14.3f}x {collision_ratio:>14.3f}x {overall_ratio:>14.3f}x")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        with open(filename, 'r') as file:
            content = file.read()
            from pathlib import Path
            layout = Path(filename).stem
            layout_data = collect_by_layout(content, str(layout))

            if not layout_data:
                print("No layout data found in file.")
                print("Make sure your log file contains layout identifiers.")
                sys.exit(1)

            # Analyze each layout
            for layout, data in sorted(layout_data.items()):
                analyze_layout(layout, data)

            # Print summary
            if len(layout_data) > 1:
                print_summary(layout_data)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
