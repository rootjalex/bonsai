#include "CodeGen/CodeGen_LLVM.h"
#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

using namespace bonsai::ir;
using namespace bonsai;

void test_example() {
    Type f32 = Float_t::make(32);
    Expr _1 = FloatImm::make(f32, 1.0);
    Expr a = Var::make(f32, "a");
    Expr ap1 = BinOp::make(BinOp::Add, a, _1);
    Expr _2 = FloatImm::make(f32, 2.0);
    Expr expr = BinOp::make(BinOp::Add, ap1, _2);
    std::cout << expr << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_expr_function(expr);
}

void test_example2() {
    Type i16 = Int_t::make(16);
    Expr _1 = IntImm::make(i16, 1);
    Expr a = Var::make(i16, "a");
    Expr expr = BinOp::make(BinOp::Add, a, _1);
    Expr _2 = IntImm::make(i16, 2);
    expr = BinOp::make(BinOp::Mul, expr, _2);
    std::cout << expr << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_expr_function(expr);
}

void test_example3() {
    Type i16 = Int_t::make(16);
    Expr _1 = IntImm::make(i16, 1);
    Expr a = Var::make(i16, "a");
    Expr expr = BinOp::make(BinOp::Add, a, _1);
    Expr _2 = IntImm::make(i16, 2);
    expr = BinOp::make(BinOp::Add, expr, _2);
    std::cout << expr << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_expr_function(expr);
}

int main(void) {
    test_example();
    test_example2();
    test_example3();
}
