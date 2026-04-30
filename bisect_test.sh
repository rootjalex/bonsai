#!/usr/bin/env bash
set -e

# Go to the test directory
cd apps/rtiow/cpu

# Run the test (adjust if build step is needed)
./test.sh > /dev/null 2>&1

IMG="rtiow-cpu-image.ppm"

# Basic sanity check
if [ ! -f "$IMG" ]; then
  echo "Image not produced"
  exit 125   # tell git bisect to skip
fi

# Skip header (PPM header is usually 3 lines)
# Then check if any non-zero pixel exists
if tail -n +4 "$IMG" | grep -q '[1-9]'; then
  # Found non-zero pixel → GOOD
  exit 0
else
  # All zeros → BAD
  exit 1
fi
