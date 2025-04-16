#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"

#include <string>

namespace bonsai {
namespace codegen {

// Converts program to a C++ header with struct and function declarations, and
// its respective .o file, compiled from the LLVM backend.
// For example, if your output filename is "foo" and your entry point is
// "main.cpp", then run the following commands with Clang:
//
//   clang++ -c main.cpp -o main.o  # build main.cpp
//   clang++ main.o foo.o -o main   # link "foo.o"
//   ./main                         # run it
void to_cpp(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen
} // namespace bonsai
