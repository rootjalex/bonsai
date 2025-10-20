#!/bin/bash
set -euo pipefail

# Maximum exponent (passed as first argument)
if [ $# -lt 1 ]; then
    echo "Usage: $0 <max_exponent>"
    exit 1
fi

MAX_EXP=$1

# Loop from 8 to MAX_EXP
for EXP in $(seq 8 "$MAX_EXP"); do
    VALUE=$((2**EXP))
    echo "Running test.sh with VALUE=$VALUE"
    ./apps/queries/joins/salary/test.sh "$VALUE" &> apps/queries/joins/salary/tmp.txt
    grep -E "bonsai_salary|DuckDB:" apps/queries/joins/salary/tmp.txt
    rm apps/queries/joins/salary/tmp.txt
done
