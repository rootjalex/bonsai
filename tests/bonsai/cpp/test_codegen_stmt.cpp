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
    Stmt stmt = Return::make(expr);
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

void test_example2() {
    Type f32 = Float_t::make(32);
    Stmt stmt = Sequence::make({
        Store::make("_a", Expr(), Var::make(f32, "a")),
        // TODO: fix this!! bring back SSA
        // LetStmt::make("b", Var::make(f32, "a"), Return::make(Var::make(f32,
        // "b"))),
        LetStmt::make(WriteLoc("b", f32), Var::make(f32, "a")),
        Return::make(Var::make(f32, "b")),
    });
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

void test_example3() {
    Type f32 = Float_t::make(32);
    Expr a = Var::make(f32, "a");
    Expr b = Var::make(f32, "b");
    Stmt stmt = IfElse::make(BinOp::make(BinOp::Lt, a, b), Return::make(a),
                             Return::make(b));
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

void test_example4() {
    constexpr int WIDTH = 16;
    Type f32 = Float_t::make(32);
    Type f32x4 = Vector_t::make(f32, WIDTH);
    Expr a = Var::make(f32x4, "a");
    // TODO: operator overloading
    Expr _1x = Broadcast::make(WIDTH, FloatImm::make(f32, 1.0f));
    for (const auto op : {VectorReduce::Add, VectorReduce::Mul,
                          VectorReduce::Min, VectorReduce::Max}) {
        Expr b = BinOp::make(BinOp::Add, a, _1x);
        b = VectorReduce::make(op, b);
        Stmt stmt = Return::make(b);
        std::cout << stmt << std::endl;
        // CodeGen_LLVM codegen;
        // codegen.print_stmt_function(stmt);
    }
}

void test_example5() {
    constexpr int WIDTH = 16;
    Type f32 = Float_t::make(32);
    Type f32x4 = Vector_t::make(f32, WIDTH);
    Expr a = Var::make(f32x4, "a");
    // TODO: operator overloading
    Expr _1x = Broadcast::make(WIDTH, FloatImm::make(f32, 1.0f));
    Expr b = BinOp::make(BinOp::Add, a, _1x);
    for (const int stride : {1, 2, 4, 8}) {
        Expr idx = Ramp::make(0, stride, WIDTH);
        Stmt stmt = Sequence::make({
            Store::make("buffer", idx, b),
            Return::make(b),
        });
        std::cout << stmt << std::endl;
        // CodeGen_LLVM codegen;
        // codegen.print_stmt_function(stmt);
    }
}

int main(void) {
    test_example();
    test_example2();
    test_example3();
    test_example4();
    test_example5();
}
