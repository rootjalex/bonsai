set -ex

cd ../..
cmake --build build --config Debug -j
./build/compiler -i apps/rtiow/main.bonsai -o apps/rtiow/main.bir
./build/compiler -i apps/rtiow/main.bonsai -b llvm -o apps/rtiow/main.ll
./build/compiler -i apps/rtiow/main.bonsai -b cpp -o apps/rtiow/main
cd apps/rtiow
clang++ -g -std=c++20 -O3 main_hook.cpp main.o -o bonsai.out
time ./bonsai.out image.ppm
