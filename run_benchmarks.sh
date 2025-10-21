#!/bin/bash

set -e

echo "Running 1D"
./apps/queries/range/test.sh &> range1d_sah.txt
echo "Running count-1D"
./apps/queries/count-1D/test.sh &> count1d_sah.txt
echo "Running 2D"
./apps/queries/2D/test.sh &> dim2_sah.txt
echo "Running dist Joins"
./apps/queries/joins/dist2d/test.sh &> dist_join.txt
echo "Running salary join"
./apps/queries/joins/salary/test.sh &> salary_join.txt
