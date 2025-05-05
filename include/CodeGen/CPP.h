#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"

#include <string>

namespace bonsai {
namespace codegen {

// Converts program to a C++ header with exported struct and function
// declarations, and its respective .o file, compiled from the LLVM backend. For
// example, if your output filename is `foo` and your entry point is `main.cpp`
// (which should `#include "foo.h"`), then run the following commands (Clang):
//
//   ./bonsai -i <input> -o foo -b cpp  # lower bonsai IR to header + object
//   clang++ -c main.cpp -o main.o      # build main.cpp
//   clang++ main.o foo.o -o main       # link `foo.o`
//   ./main                             # run it
//
// If no output filename is passed, it will just print the Bonsai header
// declarations and LLVM Module to standard I/O (mostly for testing/debugging
// purposes).
void to_cpp(const ir::Program &program, const CompilerOptions &options);

// Emits the appropriate C++ type for a given bonsai type.
void emit_type(std::ostream &ss, ir::Type type);

} // namespace codegen
} // namespace bonsai
