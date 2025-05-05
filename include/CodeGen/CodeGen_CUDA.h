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

class CodeGen_CUDA : public ir::Printer {
  public:
    explicit CodeGen_CUDA(std::ostream &os) : ir::Printer(os), os(os) {}

    virtual void visit(const ir::LetStmt *) override;
    virtual void visit(const ir::Return *) override;

    void print(const ir::Program &program);
    void print(const ir::Function &function);

  private:
    void increment_indent() { set_indent(get_indent().indent + 1); }
    void decrement_indent() { set_indent(get_indent().indent - 1); }
    std::ostream &os;
};

} //  namespace bonsai
