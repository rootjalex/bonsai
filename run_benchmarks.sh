# #!/bin/bash

# set -e

# echo "Running RT"
# ./apps/queries/rt/test.sh &> results/rt_benchmark.txt
# echo "Running CPQ"
# ./apps/queries/cp/test.sh &> results/cp_benchmark.txt
# echo "Running CD"
# ./apps/cd/cpu/fcl/test.sh &> results/cd_benchmark.txt

# echo "Running 1D"
# ./apps/queries/range/test.sh &> results/range1d_sah.txt
# echo "Running count-1D"
# ./apps/queries/count-1D/test.sh &> results/count1d_sah.txt
# echo "Running 2D"
# ./apps/queries/2D/test.sh &> results/dim2_sah.txt
# echo "Running dist Joins"
# ./apps/queries/joins/dist2d/test.sh &> results/dist_join.txt
# echo "Running salary join"
# ./apps/queries/joins/salary/run_tests.sh 24 &> results/salary_join.txt

#!/bin/bash

set +e  # don't exit on error

run_step() {
    local name="$1"
    local cmd="$2"
    local outfile="$3"

    echo "Running $name"
    eval "$cmd" &> "$outfile"
    local status=$?

    if [ $status -ne 0 ]; then
        echo "FAILED: $name failed with exit code $status"
    else
        echo "COMPLETED: $name completed successfully"
    fi
}

run_step "RT" "./apps/queries/rt/test.sh" "results/rt_benchmark.txt"
run_step "CPQ" "./apps/queries/cp/test.sh" "results/cp_benchmark.txt"
run_step "CD" "./apps/cd/cpu/fcl/test.sh" "results/cd_benchmark.txt"

run_step "1D" "./apps/queries/range/test.sh" "results/range1d_sah.txt"
run_step "count-1D" "./apps/queries/count-1D/test.sh" "results/count1d_sah.txt"
run_step "2D" "./apps/queries/2D/test.sh" "results/dim2_sah.txt"
run_step "dist Joins" "./apps/queries/joins/dist2d/test.sh" "results/dist_join.txt"
run_step "salary join" "./apps/queries/joins/salary/run_tests.sh 24" "results/salary_join.txt"

echo "All benchmarks attempted."
