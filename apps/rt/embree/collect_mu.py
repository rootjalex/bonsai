import re
import sys
from collections import defaultdict


def parse_embree_output(text):
    """Parse Embree output to extract memory usage per object and layout."""
    lines = text.split('\n')
    data = defaultdict(lambda: defaultdict(lambda: {'memory': 0, 'nodes': {}}))

    current_object = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line == '---':
            i += 1
            continue

        # Check if this looks like an object name (single word/identifier, not a separator)
        if (line and
            re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', line) and
            line != '---' and
            i + 1 < len(lines) and
                lines[i + 1].strip().startswith('rt, embree,')):
            current_object = line
            i += 1
            continue

        # Check if line starts with "rt, embree, "
        if line.startswith('rt, embree,'):
            layout = line.split('rt, embree,')[1].strip()

            if current_object and layout:
                data[current_object][layout] = {'memory': 0, 'nodes': {}}

            # Search forward for the "total : used" line
            for j in range(i + 1, min(i + 200, len(lines))):
                check_line = lines[j].strip()
                if check_line.startswith('total') and 'used' in check_line:
                    bytes_match = re.search(
                        r'total\s*:\s*used\s*=\s*([\d.]+)\s*MB', check_line)
                    if bytes_match:
                        size_mb = float(bytes_match.group(1))
                        size_bytes = int(size_mb * 1024 * 1024)

                        if current_object and layout:
                            data[current_object][layout]['memory'] = size_bytes
                            data[current_object][layout]['nodes']['aabbs'] = 1
                        break

        i += 1

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <embree_output_file>")
        sys.exit(1)

    embree_file = sys.argv[1]

    try:
        with open(embree_file, 'r') as f:
            embree_text = f.read()
        print(f"Successfully loaded data from: {embree_file}\n")
    except FileNotFoundError:
        print(f"Error: File '{embree_file}' not found.")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file '{embree_file}': {e}")
        sys.exit(1)

    # Parse embree output
    result = parse_embree_output(embree_text)

    # Print the dictionary
    print(result)


if __name__ == '__main__':
    main()
