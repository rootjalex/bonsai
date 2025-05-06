cd ../..
cmake --build build --config Debug -j
./build/compiler -i apps/rtiow/trimain.bonsai -o apps/rtiow/trimain.bir
./build/compiler -i apps/rtiow/trimain.bonsai -b llvm -o apps/rtiow/trimain.ll
./build/compiler -i apps/rtiow/trimain.bonsai -b cpp -o apps/rtiow/trimain
cd apps/rtiow
clang++ -std=c++20 -O3 trimain.cpp trimain.o -o trimain.out
time ./trimain.out image.ppm