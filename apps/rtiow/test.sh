set -ex

cd ../..
cmake --build build --config Debug -j
cd apps/rtiow
../../build/compiler -i main.bonsai -b cpp -o main
../../build/compiler -i main.bonsai -b llvm -o main.ll
clang++ -std=c++20 -O3 main_hook.cpp main.o -o main_runner
time ./compare 32768 4 > image.ppm
time ./main_runner 32768 4 > bimage.ppm