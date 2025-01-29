#include "CodeGen/CodeGen_LLVM.h"
#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

#include <iostream>

using namespace bonsai::ir;
using namespace bonsai;

void test_example() {
    Type f32 = Float_t::make(32);
    Expr _1 = FloatImm::make(f32, 1.0);
    Expr a = Var::make(f32, "a");
    Expr ap1 = BinOp::make(BinOp::Add, a, _1);
    std::cout << ap1 << std::endl;
}

void test_example2() {
    // Expected to fail
    Type f32 = Float_t::make(32);
    try {
        Expr _1 = IntImm::make(f32, 1);
        std::cerr << "Failed!\n";
    } catch (const std::runtime_error &e) {
        std::cout << "Successfully caught: " << e.what();
    }
}

int main(void) {
    test_example();
    test_example2();
}
