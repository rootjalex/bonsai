#include "CodeGen/CodeGen_LLVM.h"
#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"
#include "IR/TypeEnforcement.h"

#include <iostream>

using namespace bonsai::ir;
using namespace bonsai;

void test_broadcast() {
    Type f32 = Float_t::make(32);
    Type f32x3 = Vector_t::make(f32, 3);
    global_disable_type_enforcement();
    Expr _1 = IntImm::make(Type(), 1);
    Expr a = Var::make(f32x3, "a");
    Expr ap1 = BinOp::make(BinOp::Div, _1, a);
    std::cout << ap1 << std::endl;
}

int main(void) { test_broadcast(); }
