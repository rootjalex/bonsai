#!/bin/bash 

num_runs=${1:-10}

set -e

# Enable this to be run from either root or 
# the directory where this script exists.
if [[ "$(pwd)" == */apps/rtiow/cpu ]]; then
  cd ../../..
fi

PREFIX="apps/rtiow/cpu"

# Compile
cmake --build build --config Debug -j
./build/compiler -i $PREFIX/main.bonsai -o $PREFIX/main.bir
./build/compiler -i $PREFIX/main.bonsai -b llvm -o $PREFIX/main.ll
./build/compiler -i $PREFIX/main.bonsai -b cpp -o $PREFIX/main
# clang++ -g -std=c++20 -O3 \
#   -stdlib=libc++ \
#   -g -O0 -fsanitize=address,undefined \
#   -L$CONDA_PREFIX/lib -I$CONDA_PREFIX/include/c++/v1 -lc++ -lc++abi \
#   $PREFIX/main_hook.cpp $PREFIX/main.o \
#    -o $PREFIX/bonsai.out

# # Run
# time LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./$PREFIX/bonsai.out $PREFIX/rtiow-cpu-image.ppm

cd $PREFIX
clang++   -g   -O3   -fsanitize=address,undefined   -fno-omit-frame-pointer -L$CONDA_PREFIX/lib -I$CONDA_PREFIX/include/c++/v1 -lc++ -lc++abi  -c main_hook.cpp   -o main_hook.o
/scratch/ajroot/_bonsai/deps/llvm-install/bin/llc -O3   -filetype=obj   -relocation-model=pic   -mtriple=x86_64-unknown-linux-gnu   main.ll   -o utils.o
clang++ -g -O3 -fsanitize=address,undefined -fno-omit-frame-pointer         main_hook.o utils.o -lm  -L$CONDA_PREFIX/lib -I$CONDA_PREFIX/include/c++/v1 -lc++ -lc++abi  -o main.out         -shared-libasan


echo "done build"
exit
# time ./main.out output.ppm

# GTIME=$(which gtime)

times=()

for ((i = 1; i <= num_runs; i++)); do
  echo "Run #$i"

  start=$(date +%s.%N)
  ./main.out output.ppm > /dev/null
  end=$(date +%s.%N)
  
  t=$(echo "$end - $start" | bc -l)

  echo "Time: $t"
  times+=($t)
done

min=${times[0]}
max=${times[0]}
sum=0

for t in "${times[@]}"; do
  # Convert to floating point
  (( $(echo "$t < $min" | bc -l) )) && min=$t
  (( $(echo "$t > $max" | bc -l) )) && max=$t
  sum=$(echo "$sum + $t" | bc -l)
done

avg=$(echo "$sum / ${#times[@]}" | bc -l)

echo
echo "Min time: $min seconds"
echo "Max time: $max seconds"
echo "Avg time: $avg seconds"

# Clean up
# rm $PREFIX/main.bir
# rm $PREFIX/main.ll
# rm $PREFIX/main.o
# rm $PREFIX/bonsai.out
# rm -r $PREFIX/bonsai.out.dSYM

exit 0
