set -ex

cd ../..
cmake --build build --config Debug -j
cd apps/rtiow
../../build/compiler -i main.bonsai -b cpp -o main
../../build/compiler -i main.bonsai -b llvm -o main.ll
clang++ -std=c++20 -O3 main_hook.cpp main.o -o bonsai.out
time ./bonsai.out 800 > bimage.ppm
clang++ -std=c++20 -O3 compare.cpp -o compare.out
time ./compare.out 800 > image.ppm
