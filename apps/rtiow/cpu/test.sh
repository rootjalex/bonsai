set -ex

cd ../../..
cmake --build build --config Debug -j
./build/compiler -i apps/rtiow/cpu/main.bonsai -o apps/rtiow/cpu/main.bir
./build/compiler -i apps/rtiow/cpu/main.bonsai -b llvm -o apps/rtiow/cpu/main.ll
./build/compiler -i apps/rtiow/cpu/main.bonsai -b cpp -o apps/rtiow/cpu/main
cd apps/rtiow/cpu
clang++ -g -std=c++20 -O3 main_hook.cpp main.o -o bonsai.out
time ./bonsai.out image.ppm
