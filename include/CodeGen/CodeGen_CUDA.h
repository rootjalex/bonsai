#pragma once

#include "CompilerOptions.h"
#include "IR/Frame.h"
#include "IR/Function.h"
#include "IR/Printer.h"
#include "IR/Program.h"
#include "IR/Visitor.h"
#include "Scope.h"

#include <memory>

namespace bonsai {
namespace codegen {

// Generates CUDA from a bonsai program with the respective compiler options.
// If an output file is provided, then the emitted CUDA program is written
// there. Otherwise it is printed to standard I/O.
void to_cuda(const ir::Program &program, const CompilerOptions &options);

} // namespace codegen

struct CodeGen_CUDA : public ir::Printer {
    CodeGen_CUDA(std::ostream &os) : ir::Printer(os) {}
};

} //  namespace bonsai
