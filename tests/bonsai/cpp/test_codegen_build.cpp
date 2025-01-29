#include "CodeGen/CodeGen_LLVM.h"
#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

using namespace bonsai::ir;
using namespace bonsai;

void test_example() {
    // TODO: how to make recursive types?
    // Probably need StructPtr_t with just a name
    Type f32 = Float_t::make(32);
    Type i32 = Int_t::make(32);
    Type point_with_id = Struct_t::make(
        "point_with_id", {{"id", i32}, {"x", f32}, {"y", f32}, {"z", f32}});
    Expr x = Var::make(f32, "x");
    Expr y = Var::make(f32, "y");
    Expr z = Var::make(f32, "z");
    Expr i0 = IntImm::make(i32, 0);
    Expr value = Build::make(point_with_id, {i0, x, y, z});
    Stmt stmt = Return::make(value);
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

void test_example1() {
    Type f32 = Float_t::make(32);
    Type f32x3 = Vector_t::make(f32, 3);
    Expr x = Var::make(f32, "x");
    Expr y = Var::make(f32, "y");
    Expr z = Var::make(f32, "z");
    Expr value = Build::make(f32x3, {x, y, z});
    Stmt stmt = Return::make(value);
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

void test_example2() {
    // TODO: how to make recursive types?
    // Probably need StructPtr_t with just a name
    Type f32 = Float_t::make(32);
    Type i32 = Int_t::make(32);
    Type point_with_id = Struct_t::make(
        "point_with_id", {{"x", f32}, {"y", f32}, {"z", f32}, {"id", i32}});
    Expr x = Var::make(f32, "x");
    Expr y = Var::make(f32, "y");
    Expr z = Var::make(f32, "z");
    Expr i0 = IntImm::make(i32, 0);
    Expr value = Build::make(point_with_id, {x, y, z, i0});
    Stmt stmt = Return::make(value);
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

void test_example3() {
    Type f32 = Float_t::make(32);
    Type i32 = Int_t::make(32);
    Type point_with_id = Struct_t::make(
        "point_with_id", {{"x", f32}, {"y", f32}, {"z", f32}, {"id", i32}});
    Expr x = Var::make(f32, "x");
    Expr y = Var::make(f32, "y");
    Expr z = Var::make(f32, "z");
    Expr i0 = IntImm::make(i32, 0);
    Expr value = Build::make(point_with_id, {x, y, z, i0});
    Stmt stmt = Return::make(Access::make("x", value));
    std::cout << stmt << std::endl;
    // CodeGen_LLVM codegen;
    // codegen.print_stmt_function(stmt);
}

int main(void) {
    test_example();
    test_example1();
    test_example2();
    test_example3();
}
