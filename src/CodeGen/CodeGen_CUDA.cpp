#include "CodeGen/CodeGen_CUDA.h"

#include "CodeGen/CPP.h"
#include "IR/Analysis.h"
#include "IR/Expr.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"
#include "Lower/Intrinsics.h"
#include "Lower/TopologicalOrder.h"

#include "Utils.h"

#include <fstream>
#include <sstream>

namespace bonsai {
namespace codegen {
void to_cuda(const ir::Program &program, const CompilerOptions &options) {
    if (options.output_file.empty()) {
        CodeGen_CUDA codegen(std::cout);
        codegen.print(program);
        return;
    }
    std::ofstream os(options.output_file);
    internal_assert(os.is_open()) << "failed to open: " << options.output_file;
    CodeGen_CUDA codegen(os);
    codegen.print(program);
    os.close();
}
} // namespace codegen

using namespace ir;

namespace {

// Returns whether this requires an allocation.
// TODO(cgyurgyik): What about structs? Right now, I'm assuming these are stack
// allocated...
// TODO(cgyurgyik): constant sized arrays of a "small size" probably can be
// stack allocated.
bool requires_allocation(ir::Type type) { return type.is<Array_t, Set_t>(); }

std::string vector_lane_to_field(uint32_t lane) {
    switch (lane) {
    case 0:
        return "x";
    case 1:
        return "y";
    case 2:
        return "z";
    case 3:
        return "w";
    default:
        internal_error << "unexpected vector lane: " << lane;
    }
}

// Returns the appropriate prefix for builtin CUDA vector types.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
std::string vector_prefix(Type element_type) {
    if (element_type.is<Bool_t>()) {
        // TODO(cgyurgyik): this is a home-grown bool vector for now. There are
        // likely better alternatives.
        return "bool";
    }
    if (element_type.is<Int_t, UInt_t>()) {
        const bool is_unsigned = element_type.is<UInt_t>();
        switch (element_type.bits()) {
        case 64:
            return std::string(is_unsigned ? "u" : "") + "longlong";
        case 32:
            return std::string(is_unsigned ? "u" : "") + "int";
        case 16:
            return std::string(is_unsigned ? "u" : "") + "short";
        case 8:
            return std::string(is_unsigned ? "u" : "") + "char";
        default:
            break;
        }
    }
    if (const auto *float_t = element_type.as<Float_t>();
        float_t && float_t->is_ieee754()) {
        switch (float_t->bits()) {
        case 64:
            return "double";
        case 32:
            return "float";
        case 16:
            return "half";
        default:
            break;
        }
    }
    internal_error << "[unimplemented] vector prefix for element type: "
                   << element_type;
}

} // namespace

void CodeGen_CUDA::visit(const Int_t *node) { codegen::emit_type(os, node); }

void CodeGen_CUDA::visit(const UInt_t *node) { codegen::emit_type(os, node); }

void CodeGen_CUDA::visit(const Float_t *node) {
    if (node->is_ieee754()) {
        switch (node->bits()) {
        case 16:
            os << "__half";
            return;
        case 32:
            os << "float";
            return;
        case 64:
            os << "double";
            return;
        default:
            break;
        }
    }
    internal_error << "[unimplemented] float type codegen on CUDA: "
                   << Type(node);
}

void CodeGen_CUDA::visit(const Array_t *node) {
    node->etype.accept(this);
    os << '*';
}

void CodeGen_CUDA::visit(const Struct_t *node) {
    if (!is_declaration) {
        os << node->name;
        return;
    }
    os << get_indent();
    os << "struct" << ' ' << node->name << ' ' << '{' << '\n';
    // TODO: alignment or packing?
    ScopedValue<bool> _(is_declaration, false);
    increment();
    for (const auto &[name, type] : node->fields) {
        // TODO: handle constant-sized arrays?
        os << get_indent();
        const auto *array_t = type.as<Array_t>();
        (array_t == nullptr ? type : array_t->etype).accept(this);
        if (array_t) {
            os << '*';
        }
        os << ' ' << name;
        if (const auto &it = node->defaults.find(name);
            it != node->defaults.cend()) {
            os << " = ";
            print_no_parens(it->second);
        }
        os << ';';
        if (array_t) {
            os << ' ' << '/' << '/' << ' ' << "of size" << ' ';
            array_t->size.accept(this);
        }
        os << '\n';
    }
    decrement();
    os << get_indent() << '}' << ';' << '\n';
}

void CodeGen_CUDA::visit(const Vector_t *node) {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
    internal_assert(1 <= node->lanes && node->lanes <= 4)
        << "[unimplemented] vector size: " << node->lanes
        << " in CUDA codegen, " << Type(node);
    os << vector_prefix(node->etype) << node->lanes;
}

void CodeGen_CUDA::visit(const Ptr_t *node) {
    os << "const" << ' ';
    // Unlike the Bonsai printer, we cannot print () in argument parameters.
    node->etype.accept(this);
    os << "*";
}

void CodeGen_CUDA::visit(const FloatImm *node) {
    // TODO(cgyurgyik): Do we want *everything* to be printed as a double?
    // The `f` suffix does not compile in CUDA.
    os << node->value;
}

void CodeGen_CUDA::visit(const VecImm *node) {
    const std::vector<Expr> &vs = node->values;
    if (vs.empty()) {
        os << '{' << '}';
    }
    size_t lanes = vs.size();
    Type etype = vs.front().type();
    os << "make" << '_' << vector_prefix(std::move(etype)) << lanes << '(';
    // TODO(cgyurgyik): Use the constant codegen from AJ's C++ fix.
    print_expr_list(vs);
    os << ')';
}

void CodeGen_CUDA::visit(const Infinity *node) {
    // TODO(cgyurgyik): Assumes implementation-defined version of infinity.
    os << "INFINITY";
}

void CodeGen_CUDA::visit(const Cast *node) {
    // TODO(cgyurgyik): Is this what cast really means in Bonsai?
    os << '(';
    node->type.accept(this);
    os << ')';
    node->value.accept(this);
}

void CodeGen_CUDA::visit(const Broadcast *node) {
    os << "make" << '_' << vector_prefix(node->value.type()) << node->lanes;
    os << '(';
    node->value.accept(this);
    os << ')';
}

void CodeGen_CUDA::visit(const VectorReduce *node) {
    switch (node->op) {
    case VectorReduce::OpType::Add: {
        os << "sum" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    case VectorReduce::OpType::Mul: {
        os << "mul" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    case VectorReduce::OpType::Idxmax: {
        os << "idxmax" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    case VectorReduce::OpType::Idxmin: {
        os << "idxmin" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    case VectorReduce::OpType::Min: {
        os << "min" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    case VectorReduce::OpType::Max: {
        os << "max" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    default:
        internal_error << "[unimplemented] VectorReduce CUDA codegen: "
                       << Expr(node);
    }
}

void CodeGen_CUDA::visit(const VectorShuffle *node) {
    // This assumes shuffling within a single thread, and defaults to a naive
    // implementation in Bonsai's runtime/CUDA/math.h
    os << "shuffle" << '(';
    node->value.accept(this);
    os << ',' << ' ';
    print_expr_list(node->idxs);
    os << ')';
}

void CodeGen_CUDA::visit(const Ramp *node) {
    internal_error << "[unimplemented] Ramp CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Build *node) {
    node->type.accept(this);
    os << '{';
    for (size_t i = 0, n = node->values.size(); i < n; i++) {
        if (i != 0) {
            os << ',' << ' ';
        }
        print_no_parens(node->values[i]);
    }
    os << '}';
}

void CodeGen_CUDA::visit(const Select *node) {
    const auto *vector_t = node->type.as<ir::Vector_t>();
    if (node->cond.type().is<Vector_t>()) {
        // Perform element wise select.
        internal_assert(vector_t) << Type(node->type);
        ir::Type element_type = vector_t->etype;
        // b: vector[bool, 2] = ...;
        // s: vector[i32, 2] = select(b, v0, v1);
        // ->
        // int2 s = make_int2(b.x ? v0.x : v1.x, b.y ? v0.y : v1.y)
        int64_t lanes = vector_t->lanes;
        os << "make" << '_' << vector_prefix(element_type) << lanes << '(';
        for (int i = 0; i < lanes; ++i) {
            Expr c = Access::make(vector_lane_to_field(i), node->cond);
            Expr t = Access::make(vector_lane_to_field(i), node->tvalue);
            Expr f = Access::make(vector_lane_to_field(i), node->fvalue);
            Select::make(c, t, f).accept(this);
            if (i + 1 == lanes) {
                continue;
            }
            os << ',';
        }
        os << ')';
        return;
    }
    open();
    node->cond.accept(this);
    os << " ? ";
    node->tvalue.accept(this);
    os << " : ";
    node->fvalue.accept(this);
    close();
}

void CodeGen_CUDA::visit(const ir::Intrinsic *node) {
    switch (node->op) {
    // https://developer.download.nvidia.com/cg/dot.html#:~:text=Reference%20Implementation,b.z%20%2B%20a.w*b.w%3B%20%7D
    case ir::Intrinsic::OpType::dot: {
        os << "dot" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html
    case ir::Intrinsic::OpType::sqrt: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << "sqrt" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::min: {
        os << "min" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::max: {
        os << "max" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::cos: {
        os << "cos" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::sin: {
        os << "sin" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::tan: {
        os << "tan" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::pow: {
        os << "pow" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::norm: {
        os << "length" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::abs: {
        os << "abs" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::cross: {
        os << "cross" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::rand: {
        // TODO(cgyurgyik): I don't think this will work on device, need to use
        // cuRAND (https://docs.nvidia.com/cuda/curand/index.html).
        os << "rand" << '(' << ')';
        return;
    }
    case ir::Intrinsic::OpType::fma: {
        // TODO(cgyurgyik): This intrinsic (and perhaps others) has different
        // names for different types, e.g.,
        // {fmaf : f32, fma : f64, __nv_fp128_fma: f128}
        // We need to check the type and choose the right variant.
        os << "fma" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    default:
        internal_error << "[unimplemented] Intrinsic CUDA codegen: "
                       << Expr(node);
    }
}

void CodeGen_CUDA::visit(const ir::Access *node) {
    ir::Expr value = node->value;
    value.accept(this);
    // TODO(cgyurgyik): This is wrong... we still pass structs by address. We
    // need to distinguish stack allocated structs and those passed by argument.
    // I think the right answer is to keep track of function parameters.
    // (Similar solution/case for Visitor::Deref).
    os << (requires_allocation(value.type()) ? "->" : ".");
    os << node->field;
}

void CodeGen_CUDA::visit(const Deref *node) {
    if (node->type.is<Struct_t>() || requires_allocation(node->type)) {
        os << '(' << '*';
        node->expr.accept(this);
        os << ')';
        return;
    }
    node->expr.accept(this);
}

void CodeGen_CUDA::visit(const ir::LetStmt *node) {
    os << get_indent();
    if (!node->loc.type.is<ir::Vector_t>()) {
        // TODO(bonsai/#149): Add `const` arithmetic operation overloads.
        os << "const" << ' ';
    }
    node->loc.type.accept(this);
    os << ' ' << node->loc.base << ' ' << '=' << ' ';
    node->value.accept(this);
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const Allocate *node) {
    // TODO(ajr): if this is a launched kernel, this cannot be an array
    // allocation. Otherwise, this should probably cuda malloc for arrays.
    ir::Type type = node->loc.type;
    const std::string &base = node->loc.base;
    os << get_indent();
    if (!requires_allocation(type)) {
        if (!node->mutating) {
            type.accept(this);
            os << ' ';
        }
        // Simple types do not need allocations.
        os << base << ' ' << '=' << ' ';
        node->value.accept(this);
        os << ';' << '\n';
        return;
    }

    if (!node->mutating) {
        if (const auto *array_t = type.as<Array_t>()) {
            type.accept(this);
            os << '*' << ' ' << base << ';' << '\n';
            // TODO(cgyurgyik): Check status of the CUDA malloc.
            os << get_indent() << "(void)" << "cudaMalloc" << '(';
            os << '(' << "void" << '*' << '*' << ')' << '&' << base << ',';
            array_t->size.accept(this);
            os << ' ' << '*' << ' ' << "sizeof" << '(';
            array_t->etype.accept(this);
            os << ')' << ')' << ';' << '\n';
            return;
        }
    }
    internal_error << "[unimplemented] Allocate CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const Store *node) {
    // TODO(ajr): write to output location.
    internal_error << "[unimplemented] Store CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const Accumulate *node) {
    const WriteLoc &current = node->loc;
    ir::Expr update = node->value;
    os << get_indent() << current.base << ' ';
    switch (node->op) {
    case Accumulate::OpType::Add:
        os << '+';
        break;
    case Accumulate::OpType::Sub:
        os << '-';
        break;
    case Accumulate::OpType::Mul:
        os << '*';
        break;
    case Accumulate::OpType::Argmax: {
    case Accumulate::OpType::Argmin:
        // acc = select(curr.first <=> update.first, curr, update)
        os << '=' << ' ';
        Expr cv = Var::make(current.type, current.base);
        // TODO(cgyurgyik): Gross...
        Expr cfirst = Access::make("_field0", cv);
        Expr ufirst = Access::make("_field0", update);
        ir::Expr cond = node->op == Accumulate::OpType::Argmax
                            ? cfirst > ufirst
                            : cfirst < ufirst;
        ir::Select::make(std::move(cond), std::move(cv), std::move(update))
            .accept(this);
        os << ';' << '\n';
        return;
    }
    }
    os << '=' << ' ';
    update.accept(this);
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const ir::Return *node) {
    os << get_indent() << "return";
    if (ir::Expr value = node->value; value.defined()) {
        os << ' ';
        value.accept(this);
    }
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const ir::CallStmt *node) {
    node->func.accept(this);
    os << '(';
    print_expr_list(node->args);
    os << ')';
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const Print *node) {
    // TODO(cgyurgyik): CUDA enables printing through `printf`. I
    // imagine (though have not verified) this is going to use the same
    // format specifiers as C printf, so we can just refactor the LLVM
    // version and use it here.
    internal_error << "[unimplemented] Print CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const IfElse *node) {
    os << get_indent() << "if" << ' ' << '(';
    print_no_parens(node->cond);
    os << ')' << ' ' << '{' << '\n';
    increment();
    node->then_body.accept(this);
    decrement();
    os << get_indent() << '}';
    if (node->else_body.defined()) {
        os << ' ' << "else" << ' ' << '{' << '\n';
        increment();
        node->else_body.accept(this);
        decrement();
        os << get_indent() << "}";
    }
    os << "\n";
}

void CodeGen_CUDA::visit(const DoWhile *node) {
    os << get_indent() << "do" << ' ' << '{';
    node->body.accept(this);
    os << get_indent() << '}' << ' ' << "while" << ' ' << '(';
    print_no_parens(node->cond);
    os << ')' << '\n';
}

void CodeGen_CUDA::visit(const Label *node) {
    os << '/' << '/' << node->name << '\n';
    node->body.accept(this);
}

void CodeGen_CUDA::visit(const ForAll *node) {
    const ForAll::Slice &slice = node->slice;
    os << get_indent() << "for" << ' ' << '(';
    Type iterator_type = slice.begin.type();
    iterator_type.accept(this);
    os << ' ' << node->index << ' ' << '=' << ' ';
    slice.begin.accept(this);
    os << ';' << ' ' << node->index << ' ' << '<' << ' ';
    slice.end.accept(this);
    os << ';' << ' ' << node->index << ' ' << '+' << '=' << ' ';
    slice.stride.accept(this);
    os << ')' << ' ' << '{' << '\n';
    increment();
    node->body.accept(this);
    decrement();
    os << get_indent() << '}' << '\n';
}

void CodeGen_CUDA::visit(const Continue *node) { os << "continue" << ';'; }

void CodeGen_CUDA::visit(const Launch *node) {
    internal_error << "[unimplemented] Launch CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::emit_prologue() {
    // Half (16-bit, IEEE-754) floating point.
    os << '#' << "include" << ' ' << "<cuda_fp16.h>" << '\n';
    // CUDA intrinsics
    os << '#' << "include" << ' ' << "<math.h>" << '\n';
    // C++ fixed width integral types
    os << '#' << "include" << ' ' << "<cstdint>" << '\n';
    // Overload arithmetic operators and intrinsics for vectorized math.
    // TODO(cgyurgyik): assumes the compiler is run from the root
    // directory. There is some way to make this work with <>, `-I`
    // passed to the compiler.
    os << '#' << "include" << ' ' << "\"runtime/CUDA/math.h\"" << '\n';
    os << '\n';
}

void CodeGen_CUDA::print(const Program &program) {
    emit_prologue();
    is_declaration = true;
    std::set<Type> visited;
    // TODO(cgyurgyik): Program types should be defined before their use; CUDA
    // mandates it.
    for (const auto &[_, type] : program.types) {
        if (!type.is<Struct_t>()) {
            // This is just an alias of an non-aggregate type, e.g.,
            // element Float = f32;
            continue;
        }
        const auto &[it, inserted] = visited.insert(type);
        if (!inserted) {
            // This is just an alias to another declared struct, e.g.,
            // element E { x: i32; }
            // element F = E; // <--
            continue;
        }
        type.accept(this);
        os << '\n';
    }
    is_declaration = false;

    // CUDA requires functions to be declared before uses.
    const std::vector<std::string> topological_order =
        lower::func_topological_order(program.funcs,
                                      /*undef_calls=*/false);
    for (int i = 0, e = topological_order.size(); i < e; ++i) {
        const std::string &name = topological_order[i];
        const auto &it = program.funcs.find(name);
        internal_assert(it != program.funcs.end());
        const auto &func = it->second;
        if (func == nullptr) {
            // Minimize aborts when printing, since we use printing to
            // debug.
            os << get_indent() << "[NULL FUNCTION]" << '\n';
            continue;
        }
        print(*func);
        os << '\n';
        if (i + 1 == e) {
            continue;
        }
        os << '\n';
    }
}

void CodeGen_CUDA::print(const Function &function) {
    os << get_indent();
    function.ret_type.accept(this);
    os << ' ' << function.name << '(';
    for (int i = 0, e = function.args.size(); i < e; ++i) {
        const Function::Argument &arg = function.args[i];
        arg.type.accept(this);
        os << ' ' << arg.name;
        if (ir::Expr value = arg.default_value; value.defined()) {
            os << '=';
            value.accept(this);
        }
        if (i + 1 == e) {
            continue;
        }
        os << ',' << ' ';
    }
    os << ')' << ' ' << '{' << '\n';
    increment();
    function.body.accept(this);
    decrement();
    os << get_indent() << '}';
}

} //  namespace bonsai
