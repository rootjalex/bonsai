import re
import sys


def parse_embree_output(text):
    results = []
    lines = text.split('\n')

    current_object = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this looks like an object name (single word/identifier, not a separator)
        if (line and
            re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', line) and
            line != '---' and
            i + 1 < len(lines) and
                lines[i + 1].strip().startswith('rt, embree,')):
            current_object = line

        # Check if line starts with "rt, embree, "
        if line.startswith('rt, embree,'):
            layout = line.split('rt, embree,')[1].strip()

            # Search forward for the "total : used" line
            for j in range(i + 1, min(i + 200, len(lines))):
                check_line = lines[j].strip()
                if check_line.startswith('total') and 'used' in check_line:
                    bytes_match = re.search(
                        r'total\s*:\s*used\s*=\s*([\d.]+)\s*MB', check_line)
                    if bytes_match:
                        size_mb = float(bytes_match.group(1))
                        size_bytes = int(size_mb * 1024 * 1024)

                        results.append({
                            'object': current_object,
                            'layout': layout,
                            'bytes': size_bytes
                        })
                        break

        i += 1

    return results


def main():
    # Read from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    results = parse_embree_output(text)

    # Group by object
    objects = {}
    for r in results:
        obj = r['object']
        if obj not in objects:
            objects[obj] = []
        objects[obj].append(r)

    # Output grouped by object
    for obj, layouts in objects.items():
        print(f"object: {obj}")
        for layout in layouts:
            print(f"rt, embree, {layout['layout']}")
            print(f";; aabbs: 1, {layout['bytes']}")
        print()


if __name__ == '__main__':
    main()
