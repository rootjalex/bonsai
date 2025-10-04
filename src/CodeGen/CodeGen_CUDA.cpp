#include "CodeGen/CodeGen_CUDA.h"

#include "CodeGen/CPP.h"
#include "IR/Analysis.h"
#include "IR/Argument.h"
#include "IR/Expr.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Stmt.h"
#include "IR/Type.h"
#include "Lower/Intrinsics.h"
#include "Lower/Random.h"
#include "Lower/TopologicalOrder.h"
#include "Opt/Simplify.h"

#include "Utils.h"

#include <fstream>
#include <sstream>

namespace bonsai {
namespace codegen {
void to_cuda(const ir::Program &program, const CompilerOptions &options) {
    if (options.output_file.empty()) {
        CodeGen_CUDA codegen(std::cout, program);
        codegen.print(program);
        return;
    }
    std::ofstream os(options.output_file);
    internal_assert(os.is_open()) << "failed to open: " << options.output_file;
    CodeGen_CUDA codegen(os, program);
    codegen.print(program);
    os.close();
}

} // namespace codegen

using namespace ir;

namespace {

std::set<std::string> gather_pointer_children(const ir::Type &type) {
    const auto *struct_t = type.as<ir::Struct_t>();
    if (struct_t == nullptr) {
        return {};
    }
    std::set<std::string> pointers;
    for (const auto &[name, child] : struct_t->fields) {
        const auto *ptr_t = child.as<ir::Ptr_t>();
        if (ptr_t == nullptr) {
            continue;
        }
        const auto *ref_t = ptr_t->etype.as<ir::Ref_t>();
        if (ref_t == nullptr) {
            continue;
        }
        pointers.insert(ref_t->name);
    }
    return pointers;
}

std::string recursive_malloc() {
    return R"(
std::function<void(Node**, Node*)> cudaMallocAndCopyToDeviceRecursive = [&](Node** device_node_ptr, Node* host_node) {
    if (!host_node) {
        *device_node_ptr = nullptr;
        return;
    }

    Node* d_node;
    cudaMalloc((void**)&d_node, sizeof(Node));
    cudaMemcpy(d_node, host_node, sizeof(Node), cudaMemcpyHostToDevice);

    if (host_node->nprims == 0) {
        Arm_Interior* arm = reinterpret_cast<Arm_Interior*>(
            host_node->split0on_nprims.data());

        Node* d_left = nullptr;
        Node* d_right = nullptr;

        if (arm->left) {
            cudaMallocAndCopyToDeviceRecursive(&d_left, arm->left);
        }
        if (arm->right) {
            cudaMallocAndCopyToDeviceRecursive(&d_right, arm->right);
        }

        Node temp_node = *host_node;
        Arm_Interior* temp_arm = reinterpret_cast<Arm_Interior*>(
            temp_node.split0on_nprims.data());
        temp_arm->left = d_left;
        temp_arm->right = d_right;

        cudaMemcpy(d_node, &temp_node, sizeof(Node),
                   cudaMemcpyHostToDevice);
    }

    *device_node_ptr = d_node;
};

cudaMallocAndCopyToDeviceRecursive(&__node, (*triangles).node);)";
}

std::string compilerfy_name(std::string name) {
    if (name.starts_with("d_")) {
        return name;
    }
    return "__" + name;
}

// Capitalizes the first letter of `name`.
void capitalize_first(std::string &name) {
    if (!name.empty() && std::isalpha(name.front())) {
        name.front() = std::toupper(name.front());
    }
}

// Returns whether this type contains a field with a memory address.
bool has_address(Type type) {
    if (type.is<Ptr_t, Array_t>()) {
        return true;
    }
    if (type.is<Vector_t>()) {
        return has_address(type.element_of());
    }
    if (const auto *struct_t = type.as<Struct_t>()) {
        return std::any_of(
            struct_t->fields.begin(), struct_t->fields.end(),
            [](const TypedVar &v) { return has_address(v.type); });
    }
    if (const auto *tuple_t = type.as<Tuple_t>()) {
        return std::any_of(tuple_t->etypes.begin(), tuple_t->etypes.end(),
                           [](const ir::Type &t) { return has_address(t); });
    }
    return false;
}

std::vector<TypedVar> get_immediate_addressed_children(Type type) {
    std::vector<TypedVar> children;
    internal_assert(type.is<Struct_t>()) << "[unimplemented] " << type;
    const auto *struct_t = type.as<Struct_t>();
    for (const auto &[name, type] : struct_t->fields) {
        if (type.is<Array_t>()) {
            children.push_back(TypedVar(name, type));
            continue;
        }
        if (type.is<Ptr_t>()) {
            internal_assert(type.element_of().is<Struct_t>());
            children.push_back(TypedVar(name, type.element_of()));
        }
    }
    return children;
}

// Returns whether this is a context type created during parallelization.
// The contexts already allocate everything to device; we want to avoid also
// needing to allocate the context pointer on device, so we just copy it.
bool is_context_type(Type t) {
    const auto *s = t.as<Struct_t>();
    if (s == nullptr) {
        return false;
    }
    return s->name.starts_with("_ctx");
}

// Returns the relevant CUDA version of each intrinsic, e.g.,
// cuda_intrinsic("sin", f32) -> "sinf"
std::string cuda_intrinsic(std::string intrinsic, Type type) {
    const auto *float_t = type.as<ir::Float_t>();
    if (float_t == nullptr) {
        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INT.html
        return intrinsic;
    }
    internal_assert(float_t->is_ieee754()) << type;
    switch (float_t->bits()) {
    case 128:
        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html
        return "__nv_fp128_" + intrinsic;
    case 64:
        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html
        return intrinsic;
    case 32:
        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html
        if (std::vector<std::string> is =
                {
                    "abs",
                    "dim",
                    "divide",
                    "max",
                    "min",
                    "mod",
                    "rexp",
                };
            std::find(is.cbegin(), is.cend(), intrinsic) != is.cend()) {
            return "f" + intrinsic + "f";
        }
        return intrinsic + "f";
    case 16:
        // https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__FUNCTIONS.html
        return "h2" + intrinsic;
    default:
        internal_error << "unimplemented: " << type;
    }
}

std::string bonsai_scalar_type_to_cpp(Type type) {
    if (const auto *struct_t = type.as<Struct_t>()) {
        std::string name = struct_t->name;
        capitalize_first(name);
        return name;
    }
    if (const auto *index_t = type.as<Index_t>()) {
        return "size_t";
    }
    if (const auto *float_t = type.as<Float_t>()) {
        internal_assert(float_t->is_ieee754());
        switch (float_t->bits()) {
        case 64:
            return "double";
        case 32:
            return "float";
        case 16:
            return "__half";
        default:
            break;
        }
    }
    if (const auto *int_t = type.as<Int_t>()) {
        switch (int_t->bits) {
        case 64:
            return "int64_t";
        case 32:
            return "int32_t";
        case 16:
            return "int16_t";
        case 8:
            return "int8_t";
        default:
            break;
        }
    }
    if (const auto *uint_t = type.as<UInt_t>()) {
        switch (uint_t->bits) {
        case 64:
            return "uint64_t";
        case 32:
            return "uint32_t";
        case 16:
            return "uint16_t";
        case 8:
            return "uint8_t";
        default:
            break;
        }
    }
    internal_error << "[unimplemented]: " << type;
}

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
        // TODO(cgyurgyik): this is a home-grown bool vector for now. There
        // are likely better alternatives.
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
    // Treat arrays as 1-dimensional...
    ir::Type etype = node->etype;
    while (etype.is<Array_t>()) {
        etype = etype.as<Array_t>()->etype;
    }
    etype.accept(this);
    os << '*';
}

void CodeGen_CUDA::visit(const BVH_t *node) {
    for (const auto &variant : node->variants) {
        // Forward declare
        os << "struct " << variant.name() << ";\n";
    }
    os << "using " << node->name << " = ";
    os << "std::variant<";
    bool first = true;
    for (const auto &variant : node->variants) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << variant.name();
    }
    os << ">;\n";
    return;
}

void CodeGen_CUDA::visit(const Struct_t *node) {
    std::string name = node->name;
    capitalize_first(name);
    if (!is_declaration) {
        os << name;
        return;
    }
    std::set<std::string> pointers = gather_pointer_children(ir::Type(node));
    for (const std::string &pointer : pointers) {
        std::string name = pointer;
        capitalize_first(name);
        os << get_indent() << "struct " << name << ";\n";
    }
    os << get_indent();
    os << "struct" << ' ';
    if (std::optional<int64_t> alignment = node->alignment) {
        os << "alignas(" << *alignment << ") ";
    }
    os << name << ' ' << '{' << '\n';
    ScopedValue<bool> _(is_declaration, false);
    increment();
    for (const auto &[name, type] : node->fields) {
        os << get_indent();
        const auto *array_t = type.as<Array_t>();
        if (array_t && array_t->size.defined() && is_const(array_t->size)) {
            os << "cuda::std::array<";
            array_t->etype.accept(this);
            os << ", ";
            array_t->size.accept(this);
            os << ">";
            os << ' ' << name << ";\n";
            continue;
        }
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
        if (type.is_scalar()) {
            if (uint64_t b = type.bits(); b < 8 || !is_power_of_two(b)) {
                os << " : " << b;
            }
        }
        os << ';';
        os << '\n';
    }
    decrement();
    os << get_indent() << '}';
    if (node->is_packed()) {
        os << ' ' << "__attribute__((packed))";
    }
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const Vector_t *node) {
    if (const auto *vector_t = node->etype.as<Vector_t>()) {
        os << "cuda::std::array<";
        internal_assert(!vector_t->etype.is<Vector_t>())
            << "[unimplemented]: " << ir::Type(node);
        os << vector_prefix(vector_t->etype) << vector_t->lanes;
        os << ", " << std::to_string(node->lanes) + ">";
        return;
    }
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
    if (1 <= node->lanes && node->lanes <= 4) {
        os << vector_prefix(node->etype) << node->lanes;
        return;
    }
    os << "cuda::std::array<" << bonsai_scalar_type_to_cpp(node->etype) << ", ";
    os << node->lanes << ">";
}

void CodeGen_CUDA::visit(const Ptr_t *node) {
    // TODO(cgyurgyik): This isn't constant for an argument return type.
    // os << "const" << ' ' << "__restrict__" << ' ';
    // Unlike the Bonsai printer, we cannot print () in argument parameters.
    Type etype = node->etype;
    if (is_context_type(etype)) {
        etype.accept(this);
        return;
    }
    etype.accept(this);
    if (etype.is<Ref_t>()) {
        return;
    }
    os << "*";
}

void CodeGen_CUDA::visit(const Ref_t *node) {
    std::string name = node->name;
    capitalize_first(name);
    os << name << "*";
}

void CodeGen_CUDA::visit(const Rand_State_t *node) { os << "curandState"; }

void CodeGen_CUDA::visit(const DynArray_t *node) {
    // TODO(cgyurgyik): this probably has some performance implications. Can we
    // always infer whether this is a {host, device}_vector?
    os << "thrust::universal_vector<";
    node->etype.accept(this);
    os << ">";
}

void CodeGen_CUDA::visit(const ir::Option_t *node) {
    os << "cuda::std::optional<";
    node->etype.accept(this);
    os << ">";
}

void CodeGen_CUDA::visit(const ir::Tuple_t *node) {
    os << "cuda::std::tuple<";
    print_type_list(node->etypes);
    os << ">";
}

void CodeGen_CUDA::visit(const Index_t *node) {
    os << bonsai_scalar_type_to_cpp(node);
}

void CodeGen_CUDA::visit(const FloatImm *node) {
    if (node->type.bits() == 64) {
        // The default type for floating point literals.
        os << node->value;
        return;
    }
    if (is_const_zero(node) || is_const_one(node)) {
        // Avoid unnecessary noise.
        os << node->value;
        return;
    }
    os << '(';
    os << bonsai_scalar_type_to_cpp(node->type);
    os << ')' << node->value;
}

void CodeGen_CUDA::visit(const VecImm *node) {
    const std::vector<Expr> &vs = node->values;
    if (vs.empty()) {
        os << '{' << '}';
        return;
    }
    size_t lanes = vs.size();
    Type etype = vs.front().type();
    os << "make" << '_' << vector_prefix(std::move(etype)) << lanes << '(';
    print_expr_list(vs);
    os << ')';
}

void CodeGen_CUDA::visit(const StringImm *node) {
    print_string_imm(os, node->value);
}

void CodeGen_CUDA::visit(const Infinity *node) { os << "INFINITY"; }

void CodeGen_CUDA::visit(const Cast *node) {
    ir::Expr value = node->value;
    ir::Type type = node->type;
    if (value.type().is<Vector_t>() && type.is<Vector_t>()) {
        internal_assert(node->mode == Cast::Mode::Convert)
            << "unimplemented: vector reinterpret cast: " << Expr(node);
        // A special-cased vec[T] -> vec[U] cast.
        const auto *v = value.type().as<Vector_t>();
        const auto *t = type.as<Vector_t>();
        internal_assert(v->lanes == t->lanes);
        os << "make" << '_' << vector_prefix(t->etype) << v->lanes << '(';
        for (int i = 0, e = v->lanes; i < e; ++i) {
            value.accept(this);
            os << '.' << vector_lane_to_field(i);
            if (i + 1 == e) {
                continue;
            }
            os << ',' << ' ';
        }
        os << ')';
        return;
    }

    switch (node->mode) {
    case Cast::Mode::Convert:
        if (node->value.type().is<ir::Option_t>()) {
            if (node->type.is<ir::Bool_t>()) {
                print_no_parens(node->value);
                os << ".has_value()";
                return;
            }
            os << "*";
            print_no_parens(node->value);
            return;
        }
        if (!type.is<ir::Option_t>()) {
            // No need to cast for Option_t. this should be readily inferred.
            os << '(';
            node->type.accept(this);
            os << ')';
        }
        value.accept(this);
        return;
    case Cast::Mode::Reinterpret:
        if (node->type.is<Ptr_t, Array_t>()) {
            os << "reinterpret_cast";
        } else {
            // See "runtime/CUDA/helpers.h" for what this function is doing.
            os << "bonsai_reinterpret";
        }
        os << '<';
        node->type.accept(this);
        os << '>' << '(';
        value.accept(this);
        os << ')';
        return;
    }
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
    case VectorReduce::OpType::And: {
        os << "reduce_and" << '(';
        node->value.accept(this);
        os << ')';
        return;
    }
    case VectorReduce::OpType::Or: {
        os << "reduce_or" << '(';
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
    // implementation in Bonsai's runtime/CUDA/helpers.h
    os << "shuffle" << '(';
    node->value.accept(this);
    os << ',' << ' ' << '{';
    print_expr_list(node->idxs);
    os << '}' << ')';
}

void CodeGen_CUDA::visit(const Ramp *node) {
    internal_error << "[unimplemented] Ramp CUDA codegen: " << Expr(node);
}

void CodeGen_CUDA::visit(const Build *node) {
    if (const auto *option_t = node->type.as<ir::Option_t>();
        option_t && node->values.empty()) {
        os << "cuda::std::nullopt";
        return;
    }
    node->type.accept(this);
    os << '{';
    for (size_t i = 0, n = node->values.size(); i < n; i++) {
        if (i != 0) {
            os << ',' << ' ';
        }
        if (is_context_type(node->type)) {
            if (const auto *d = node->values[i].as<Deref>()) {
                print_no_parens(d->expr);
                continue;
            }
        }
        print_no_parens(node->values[i]);
    }
    os << '}';
}

void CodeGen_CUDA::visit(const Deref *node) {
    Expr pointee = node->expr;
    internal_assert(pointee.type().is<Ptr_t>());
    if (is_context_type(pointee.type().element_of())) {
        pointee.accept(this);
        return;
    }
    if (const auto *p = pointee.type().as<Ptr_t>()) {
        if (p->etype.is<Array_t>()) {
            pointee.accept(this);
            return;
        }
    }
    os << '(' << '*';
    pointee.accept(this);
    os << ')';
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
    case ir::Intrinsic::OpType::dot: {
        // https://developer.download.nvidia.com/cg/dot.html#:~:text=Reference%20Implementation,b.z%20%2B%20a.w*b.w%3B%20%7D
        os << "dot" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::sqrt: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << cuda_intrinsic("sqrt", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::min: {
        ir::Type element_type = node->args.front().type();
        os << cuda_intrinsic("min", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::max: {
        ir::Type element_type = node->args.front().type();
        os << cuda_intrinsic("max", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::cos: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << cuda_intrinsic("cos", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::sin: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << cuda_intrinsic("sin", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::tan: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << cuda_intrinsic("tan", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::pow: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << cuda_intrinsic("pow", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::abs: {
        ir::Type element_type = node->args.front().type();
        os << cuda_intrinsic("abs", element_type) << '(';
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
    case ir::Intrinsic::OpType::cross: {
        os << "cross" << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::rand: {
        // This always produces a random float in (0, 1].
        internal_assert(node->args.empty())
            << "TODO: support vector rand generation on CUDA: " << Expr(node);
        const auto *float_t = node->type.as<Float_t>();
        internal_assert(float_t && float_t->is_ieee754()) << node->type;
        if (on_device) {
            switch (float_t->bits()) {
            case 64:
                os << "curand_uniform_double(" << lower::rng_state_name << ")";
                return;
            case 32:
                os << "curand_uniform(" << lower::rng_state_name << ")";
                return;
            default:
                internal_error << "unsupported on-device random type: "
                               << node->type;
            }
        }
        os << "random" << '<' << bonsai_scalar_type_to_cpp(node->type) << '>'
           << '(' << ')';
        return;
    }
    case ir::Intrinsic::OpType::fma: {
        ir::Type element_type = node->args.front().type();
        internal_assert(element_type.is<ir::Float_t>()) << element_type;
        os << cuda_intrinsic("fma", element_type) << '(';
        print_expr_list(node->args);
        os << ')';
        return;
    }
    case ir::Intrinsic::OpType::argmax: {
        os << "argmax(";
        print_expr_list(node->args);
        os << ")";
        return;
    }
    case ir::Intrinsic::OpType::fadd_rd:
    case ir::Intrinsic::OpType::fadd_ru:
    case ir::Intrinsic::OpType::fsub_rd:
    case ir::Intrinsic::OpType::fsub_ru:
    case ir::Intrinsic::OpType::fdiv_rd:
    case ir::Intrinsic::OpType::fdiv_ru:
    case ir::Intrinsic::OpType::fmul_rd:
    case ir::Intrinsic::OpType::fmul_ru:
    case ir::Intrinsic::OpType::frcp_rd:
    case ir::Intrinsic::OpType::frcp_ru: {
        if (on_device) {
            os << "__" << to_string(node->op) << "(";
            print_expr_list(node->args);
            os << ")";
            return;
        }
    }
        [[fallthrough]];
    case ir::Intrinsic::OpType::floorf:
    case ir::Intrinsic::OpType::roundf:
    case ir::Intrinsic::OpType::ceilf: {
        os << to_string(node->op) << "(";
        print_expr_list(node->args);
        os << ")";
        return;
    }
    default:
        internal_error << "[unimplemented] Intrinsic CUDA codegen: "
                       << Expr(node);
    }
}

void CodeGen_CUDA::visit(const ir::Access *node) {
    node->value.accept(this);
    os << ".";
    os << node->field;
}

void CodeGen_CUDA::visit(const ir::Extract *node) {
    // TODO(cgyurgyik): CUDA builtin vector types cannot extract by index, and
    // we cannot overload this. The best solution probably is to create our own
    // vector types that match the alignment of CUDA builtin vector types, but
    // enable us to overload operators.
    std::optional<uint32_t> index = get_constant_value(node->idx);

    if (const auto *vector_t = node->vec.type().as<Vector_t>();
        vector_t && vector_t->lanes <= 4 && index.has_value() &&
        !vector_t->etype.is_iterable()) {
        Access::make(vector_lane_to_field(*index), node->vec).accept(this);
        return;
    }
    if (const auto &tuple_t = node->vec.type().as<Tuple_t>()) {
        os << "cuda::std::get<";
        internal_assert(index.has_value()) << node->idx;
        os << *index << ">(";
        print_no_parens(node->vec);
        os << ")";
        return;
    }

    node->vec.accept(this);
    os << "[";
    node->idx.accept(this);
    os << "]";
}

void CodeGen_CUDA::visit(const ir::Slice *node) {
    os << "slice";
    const bool is_constant_bounds =
        is_const(node->begin) && is_const(node->end);
    os << (is_constant_bounds ? "<" : "(");
    node->begin.accept(this);
    os << ", ";
    node->end.accept(this);
    os << (is_constant_bounds ? ">(" : ",");
    node->value.accept(this);
    os << ")";
}

void CodeGen_CUDA::visit(const AtomicAdd *node) {
    os << "atomicAdd(";
    node->ptr.accept(this);
    os << ", ";
    node->value.accept(this);
    os << ")";
}

void CodeGen_CUDA::visit(const ir::LetStmt *node) {
    os << get_indent();
    if (!node->loc.type.is<ir::Vector_t>()) {
        // TODO(bonsai/#149): Add `const` arithmetic operation overloads.
        // TODO(cgyurgyik): Need to revisit this and fix const qualifier
        // issues.
        //  os << "const" << ' ';
    }
    node->loc.type.accept(this);
    os << ' ' << node->loc.base() << ' ' << '=' << ' ';
    node->value.accept(this);
    os << ';' << '\n';
}

void CodeGen_CUDA::free_host_memory() {
    if (!device_allocated.empty()) {
        std::vector<Stmt> frees;
        for (auto &[name, type] : device_allocated) {
            ir::Expr e = Var::make(type, compilerfy_name(name));
            if (!has_address(type)) {
                continue; // TODO(cgyurgyik): fix
            }
            frees.push_back(Free::make(std::move(e)));
        }
        device_allocated.clear();
        for (const Stmt &free : frees) {
            free.accept(this);
        }
    }
}

// TODO(cgyurgyik): Verify this is coming from device memory.
void CodeGen_CUDA::visit(const Free *node) {
    free_host_memory();
    os << get_indent() << "cudaFree" << '(';
    ir::Expr value = node->value;
    if (const auto *d = value.as<Deref>(); d && d->expr.type().is<Ptr_t>()) {
        d->expr.accept(this);
    } else {
        value.accept(this);
    }
    os << ')' << ';' << '\n';
}

void CodeGen_CUDA::visit(const Var *node) {
    os << node->name;
    if (is_build && node->name == "node" && node->type.is<Ref_t>()) {
        // TODO(cgyurgyik): another ugly hack, we use std::variant's
        // `holds_alternative`/`get` so we need to avoid the case:
        //
        //   if (holds_alernative<T>(*node)) {
        //     const T& node = get<T>(*node); <-- duplicate `node` name
        //   }
        os << "_";
    }
}

void CodeGen_CUDA::emit_to_device(const Allocate *node) {
    Expr value = node->value;
    if (value.type().is<ir::Ptr_t>()) {
        value = Deref::make(value);
    }
    const std::string &base = node->loc.base();
    Type type = node->loc.type;
    if (type.is<Ptr_t>()) {
        type = type.element_of();
        internal_assert(type.is<Struct_t>());
    }
    if (type.is<Array_t>() || !has_address(type)) {
        emit_to_device(base, type, value);
        return;
    }

    std::vector<TypedVar> types = get_immediate_addressed_children(type);
    // copy the children to the device...
    for (const auto &[name, type] : types) {
        emit_to_device(name, type, Access::make(name, value),
                       /*parent=*/value);
        device_allocated.push_back(TypedVar(name, type));
    }
    // ...and then hook them back up.
    internal_assert(base.starts_with("d_")) << base;
    std::string original = base.substr(2, base.size());
    std::string copy = "h_" + original;
    // Make a shallow copy for non-pointer members.

    os << get_indent();
    type.accept(this);
    os << ' ' << copy << ' ';
    os << '=' << ' ';
    if (value.is<ir::Deref>()) {
        os << '*';
    }
    os << original << ';' << '\n';
    // Then copy all the recently device-allocated members.
    for (const auto &[name, type] : types) {
        os << get_indent() << copy << '.' << name << ' ';
        os << '=' << ' ' << compilerfy_name(name) << ';' << '\n';
    }
    // Finally, emit the base struct.
    emit_to_device(base, type, Var::make(type, copy));
}

void CodeGen_CUDA::emit_to_device(std::string base, ir::Type type,
                                  ir::Expr value,
                                  std::optional<ir::Expr> parent) {
    if (const auto *array_t = type.as<Array_t>()) {
        emit_to_device(base, array_t, value, parent);
        return;
    }
    const auto *struct_t = type.as<Struct_t>();
    emit_to_device(base, struct_t, value);
}

void CodeGen_CUDA::emit_to_device(std::string base, const Struct_t *struct_t,
                                  Expr value) {
    os << get_indent();
    struct_t->accept(this);
    os << '*' << ' ' << compilerfy_name(base) << ';' << '\n';
    if (struct_t->name == "node") {
        // TODO(cgyurgyik): ultra hack to get cuda `ptr` to work.
        os << recursive_malloc() << "\n";
        return;
    }
    os << get_indent() << "cudaMallocAndCopyToDevice" << '(';
    os << '(' << "void" << '*' << '*' << ')' << '&' << compilerfy_name(base)
       << ',' << ' ';
    internal_assert(value.defined())
        << "allocation to device expects a value (what is copied)";
    if (!value.type().is<Ptr_t>()) {
        os << '&';
    }
    value.accept(this);
    os << ',' << ' ' << "sizeof" << '(';
    struct_t->accept(this);
    os << ')' << ')' << ';' << '\n';
}

void CodeGen_CUDA::emit_to_device(std::string base, const Array_t *array_t,
                                  Expr value, std::optional<ir::Expr> parent) {
    internal_assert(!has_address(array_t->etype))
        << "[unimplemented] array with pointers: " << array_t;
    os << get_indent();
    array_t->accept(this);
    os << ' ' << compilerfy_name(base) << ';' << '\n';
    os << get_indent() << "cudaMallocAndCopyToDevice" << '(';
    os << '(' << "void" << '*' << '*' << ')' << '&' << compilerfy_name(base)
       << ',' << ' ';
    internal_assert(value.defined())
        << "allocation to device expects a value (what is copied)";
    value.accept(this);
    os << ',' << ' ';

    if (parent.has_value()) {
        // The size needs to be correctly accessed from the struct.
        parent->accept(this);
        os << '.';
    }
    if (array_t->size.defined()) {
        array_t->size.accept(this);
    }
    os << ' ' << '*' << ' ' << "sizeof" << '(';
    array_t->etype.accept(this);
    os << ')' << ')' << ';' << '\n';
}

void CodeGen_CUDA::visit(const Allocate *node) {
    ir::Type type = node->loc.type;
    const std::string &b = node->loc.base();

    switch (node->memory) {
    case Allocate::Memory::Stack: {
        os << get_indent();
        if (const auto *vector_t = type.as<Vector_t>()) {
            type->accept(this);
            os << ' ' << b;
            if (node->value.defined()) {
                os << ' ' << '=' << ' ';
                node->value.accept(this);
            }
            os << ';' << '\n';
            return;
        }
        if (type.is<ir::Ptr_t>()) {
            type->accept(this);
            os << ' ' << b;
            os << " = new ";
            type.element_of().accept(this);
            os << "();\n";
            return;
        }
        if (const auto *array_type = type.as<Array_t>()) {
            // <type> <name>[<size>];
            array_type->etype.accept(this);
            os << ' ' << b << '[';
            Expr size = array_type->size;
            internal_assert(is_const(size))
                << "expected constant array size, received: " << size;
            size.accept(this);
            os << ']' << ';' << '\n';
            return;
        }
        type.accept(this);
        if (is_context_type(type)) {
            os << ' ' << b << ' ' << '=' << ' ';
            internal_assert(node->value.defined())
                << "undefined value for CUDA stack allocation: " << Stmt(node);
            node->value.accept(this);
            os << ';' << '\n';
            return;
        }
        os << ' ' << b;
        if (!node->value.defined() && type.is_scalar()) {
            os << ' ' << '=' << ' ';
            os << make_zero(type);
        } else if (node->value.defined()) {
            os << ' ' << '=' << ' ';
            node->value.accept(this);
        }
        os << ';' << '\n';
        return;
    } break;
    case Allocate::Memory::Heap: {
        os << get_indent();
        if (const auto *array_t = type.as<Array_t>()) {
            if (is_build) {
                type.accept(this);
                os << ' ' << b << ' ' << '=' << ' ';
                os << "reinterpret_cast<";
                type.accept(this);
                os << ">(";
                os << "malloc(";
                os << "sizeof(";
                type.element_of().accept(this);
                os << ") * ";
                type.size().accept(this);
                os << "));\n";
                return;
            }
            type.accept(this);
            os << ' ' << b << ';' << '\n';
            // TODO(cgyurgyik): check the status of the CUDA malloc.
            os << get_indent() << "(void)" << "cudaMalloc" << '(';
            os << '(' << "void" << '*' << '*' << ')' << '&' << b << ',' << ' ';
            array_t->size.accept(this);
            os << ' ' << '*' << ' ' << "sizeof" << '(';
            array_t->etype.accept(this);
            os << ')' << ')' << ';' << '\n';
            return;
        }
        if (const auto *dyn_array_t = type.as<DynArray_t>()) {
            type.accept(this);
            os << ' ' << b << ';' << '\n';
            if (dyn_array_t->capacity.defined()) {
                os << get_indent();
                os << b << ".reserve(";
                dyn_array_t->capacity.accept(this);
                os << ");\n";
            }
            return;
        }
    } break;
    case Allocate::Memory::Device: {
        emit_to_device(node);
        return;
    } break;
    case Allocate::Memory::Host: {
        os << get_indent();
        if (const auto *array_t = type.as<Array_t>()) {
            type.accept(this);
            os << ' ' << b << ';' << '\n';
            os << get_indent() << "mallocAndCopyFromDevice" << '(';
            os << '(' << "void" << '*' << '*' << ')' << '&' << b << ',' << ' ';
            internal_assert(node->value.defined())
                << "allocation to device expects a value (what is copied)";
            node->value.accept(this);
            os << ',' << ' ';
            array_t->size.accept(this);
            os << ' ' << '*' << ' ' << "sizeof" << '(';
            array_t->etype.accept(this);
            os << ')' << ')' << ';' << '\n';
            return;
        }
    } break;
    case Allocate::Memory::Global: {
        // do nothing; globals are handled in the program prologue.
        return;
    }
    }
    internal_error << "[unimplemented] Allocate CUDA codegen: " << Stmt(node);
}

void CodeGen_CUDA::visit(const Store *node) {
    os << get_indent();
    print_loc(os, node->loc, /*is_assignment=*/true);
    os << " = ";
    node->value.accept(this);
    os << ";\n";
}

void CodeGen_CUDA::visit(const Accumulate *node) {
    const WriteLoc &current = node->loc;
    ir::Expr update = node->value;
    os << get_indent();
    if (program.globals.contains(current.to_expr())) {
        // __managed__ memory must be atomically accumulated.
        internal_assert(node->op == Accumulate::OpType::Add);
        ir::AtomicAdd::make(ir::PtrTo::make(current.to_expr()), update)
            .accept(this);
        os << ";\n";
        return;
    }

    print_loc(os, current, /*is_assignment=*/true);
    os << ' ';
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
        // curr arg{min|max}= update;
        // ->
        // curr = arg{min|max}(curr, update);
        os << '=' << ' ';
        os << "arg" << (node->op == Accumulate::OpType::Argmax ? "max" : "min")
           << '(';
        Var::make(current.type, current.base()).accept(this);
        os << ',' << ' ';
        update.accept(this);
        os << ')';
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
    // TODO(cgyurgyik): kernel calls must be configured <<<X,Y>>>
    os << get_indent();
    node->func.accept(this);
    os << '(';
    print_expr_list(node->args);
    os << ')';
    os << ';' << '\n';
}

void CodeGen_CUDA::visit(const Print *node) {
    // TODO(cgyurgyik): Extend support; borrow from LLVM's printf.
    std::vector<Expr> args = node->args;
    os << get_indent() << "printf" << '(';
    if (Expr value = args.front();
        args.size() == 1 && value.type().is_scalar()) {
        std::string specifier = get_specifier(value.type());
        os << '\"' << specifier << "\\n" << '\"' << ',' << ' ';
        value.accept(this);
        os << ')' << ';' << '\n';
        return;
    }
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
    os << get_indent() << "do" << ' ' << '{' << '\n';
    increment();
    node->body.accept(this);
    decrement();
    os << get_indent() << '}' << ' ' << "while" << ' ' << '(';
    print_no_parens(node->cond);
    os << ')' << ';' << '\n';
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

void CodeGen_CUDA::visit(const Continue *node) {
    os << get_indent() << "continue" << ';' << '\n';
}

void CodeGen_CUDA::visit(const Break *node) {
    os << get_indent() << "break" << ';' << '\n';
}

void CodeGen_CUDA::visit(const Launch *node) {
    os << get_indent() << node->func;
    os << '<' << '<' << '<';
    ir::Expr n = node->n;
    // TODO(cgyurgyik): This number, 512 was chosen arbitrarily. The full
    // block size (1024) was causing resource launch errors.
    Expr block_size = make_const(n.type(), 128);
    opt::Simplify::simplify((n + (block_size - 1)) / block_size).accept(this);
    os << ',' << ' ';
    block_size.accept(this);
    os << '>' << '>' << '>';
    os << '(';
    print_expr_list(node->args);
    os << ')' << ';' << '\n';
    os << get_indent() << "cudaDeviceSynchronize" << '(' << ')' << ';' << '\n';
}

void CodeGen_CUDA::visit(const Match *node) {
    increment();
    for (size_t i = 0, e = node->arms.size(); i < e; i++) {
        const auto &[variant, body] = node->arms[i];
        const auto *struct_t = variant.struct_type.as<Struct_t>();
        os << get_indent();
        if (i > 0) {
            os << "else ";
        }
        os << "if (std::holds_alternative<" << struct_t->name << ">";
        os << "(*";
        node->loc.accept(this);
        os << "_";
        os << ")) {\n";
        increment();
        os << get_indent() << "const " << struct_t->name << "& ";
        node->loc.accept(this);
        os << " = " << "std::get<" << struct_t->name << ">";
        os << "(*";
        node->loc.accept(this);
        os << "_";
        os << ");\n";
        body.accept(this);
        decrement();
        os << get_indent() << "}\n";
    }
    decrement();
}

void CodeGen_CUDA::visit(const AppendStmt *node) {
    os << get_indent();
    node->loc.to_expr().accept(this);
    os << ".push_back(";
    print_no_parens(node->value);
    os << ");\n";
}

void CodeGen_CUDA::emit_prologue() {
    // Overload arithmetic operators and intrinsics for vectorized math.
    // Requires: `-Iruntime/CUDA` to work.
    os << '#' << "include" << ' ' << "\"helpers.h\"" << '\n';
    os << '\n';
    // CUDA headers
    os << '#' << "include" << ' ' << "<cuda/std/array>" << '\n';
    os << '#' << "include" << ' ' << "<thrust/universal_vector.h>" << '\n';
    os << '#' << "include" << ' ' << "<cuda/std/optional>" << '\n';
    os << '#' << "include" << ' ' << "<cuda/std/tuple>" << '\n';
    os << '\n';
    // STL headers
    os << '#' << "include" << ' ' << "<stdint.h>" << '\n';
    os << '#' << "include" << ' ' << "<variant>" << '\n';
    os << '\n';
}

void CodeGen_CUDA::setup_kernel_rng(const Function &function) {
    internal_assert(function.is_kernel())
        << "CUDA rng can only run on device, received:\n"
        << function;
    // We need to print the thread index (first statement) before cuRAND
    // state setup since it is used as a seed.
    const auto *seq = function.body.as<Sequence>();
    internal_assert(seq) << "unexpected kernel with non-sequence body: "
                         << function.body;
    // Whether we've seen this exit condition.
    bool thread_within_bounds_visited = false;
    // Whether the thread index (TID) has been initialized. This is
    // absolutely necessary before we visit the exit condition and ensures
    // the compiler will never haphazardly produce incorrect code.
    bool tid_seen = false;
    for (int i = 0, e = seq->stmts.size(); i < e; ++i) {
        seq->stmts[i].accept(this);
        constexpr char TID[] = "tid";
        if (const auto *seed = seq->stmts[i].as<LetStmt>();
            !tid_seen && seed && seed->loc.base() == TID) {
            tid_seen = true;
        }
        // TODO(cgyurgyik): We can future-proof this even more by ensuring
        // the condition contains the correct variable (which won't be TID,
        // so isn't necessarily straight forward). However, this is a good
        // first step.
        if (thread_within_bounds_visited || !tid_seen) {
            continue;
        }
        // We'd like to only initialize this state if the thread id is
        // within bounds. This is usually done by some statement of the
        // form:
        //  `if (tid >= C) { return; }`
        const auto *ifelse = seq->stmts[i].as<IfElse>();
        if (ifelse == nullptr) {
            continue;
        }
        const auto *r = ifelse->then_body.as<Return>();
        if (r == nullptr || r->value.defined()) {
            continue;
        }
        thread_within_bounds_visited = true;
        os << get_indent() << "curandState " << lower::rng_state_name << ";\n";
        os << get_indent() << "curand_init(" << TID << ", 0, 0, &"
           << lower::rng_state_name << ");\n";
    }
    internal_assert(thread_within_bounds_visited)
        << "the thread id is never verified to be within bounds for "
           "kernel: "
        << function;
}

void CodeGen_CUDA::print(const Expr &expr) { expr.accept(this); }

void CodeGen_CUDA::print(const Program &program) {
    emit_prologue();
    is_declaration = true;
    for (const ir::Expr &global : program.globals) {
        internal_assert(global.type().is_scalar())
            << "[unimplemented] non-scalar globals: `" << global << "`";
        os << get_indent() << "inline __managed__ ";
        global.type().accept(this);
        os << " ";
        global.accept(this);
        os << " = ";
        make_zero(global.type()).accept(this);
        os << ";\n";
    }
    std::set<Type> visited;

    if (auto it = program.schedules.find(ir::Target::Host);
        it != program.schedules.end()) {
        for (const auto &[name, type] : it->second.tree_types) {
            type.accept(this);
            os << '\n';
        }
    }

    const ir::TypeMap &types = program.types;
    std::vector<std::string> types_topological =
        lower::type_topological_order(types);
    for (const std::string &name : types_topological) {
        auto tit = types.find(name);
        internal_assert(tit != types.end());
        const Type &type = tit->second;
        if (!type.is<Struct_t, BVH_t>()) {
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
    std::set<std::string> hosts = lower::find_host_functions(program.funcs);
    std::set<std::string> devices =
        lower::find_device_functions(program.funcs, hosts);
    for (int i = 0, e = topological_order.size(); i < e; ++i) {
        const std::string &name = topological_order[i];
        const auto &it = program.funcs.find(name);
        internal_assert(it != program.funcs.end());
        const auto &func = it->second;
        ScopedValue<bool> _(on_device, devices.contains(func->name));
        if (func == nullptr) {
            // Minimize aborts when printing, since we use printing to
            // debug.
            os << get_indent() << "[NULL FUNCTION]" << '\n';
            continue;
        }
        if (func->is_kernel()) {
            os << "__global__" << ' ';
            internal_assert(func->ret_type.is<Void_t>())
                << "bonsai kernels must have a void return type, received: "
                << func->ret_type;
        } else {
            if (devices.contains(func->name)) {
                os << "__device__" << ' ';
            }
            if (hosts.contains(func->name)) {
                os << "__host__" << ' ';
            }
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
    is_build =
        function.name.starts_with("rec_") || function.name.starts_with("build");
    function.ret_type.accept(this);
    os << ' ' << function.name << '(';
    for (int i = 0, e = function.args.size(); i < e; ++i) {
        const Argument &arg = function.args[i];
        arg.type.accept(this);
        if (arg.type.is<ir::DynArray_t>()) {
            os << '&';
        }
        os << ' ' << arg.name;
        if (is_build && arg.name == "node" && arg.type.is<Ref_t>()) {
            os << '_';
        }
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
    if (function.must_setup_rng()) {
        setup_kernel_rng(function);
    } else {
        function.body.accept(this);
    }
    decrement();
    os << get_indent() << '}';
    is_build = false;
}

void CodeGen_CUDA::print_loc(std::ostream &os, const ir::WriteLoc &loc,
                             bool is_assignment) {
    std::string ss;
    const bool should_deref = loc.base_type().is<ir::Ptr_t>() &&
                              !is_context_type(loc.base_type().element_of());
    if (should_deref) {
        ss += "(*";
    }
    ss += loc.base();
    if (should_deref) {
        ss += ")";
    }
    bool is_pointer = false;
    for (const auto &value : loc.accesses) {
        if (std::holds_alternative<std::string>(value)) {
            if (is_pointer) {
                ss += "->";
                is_pointer = false;
            } else {
                ss += ".";
            }
            ss += std::get<std::string>(value);
            continue;
        }
        if (std::holds_alternative<ir::Expr>(value)) {
            ir::Expr idx = std::get<ir::Expr>(value);
            if (loc.base_type().is<ir::Vector_t>() && is_const(idx)) {
                ss += ".";
                ss += vector_lane_to_field(*get_constant_value(idx));
                continue;
            }
            ss += "[";
            std::stringstream stream;
            CodeGen_CUDA printer(stream, program);
            printer.print(idx);
            ss += stream.str();
            ss += "]";
            continue;
        }
        if (std::holds_alternative<ir::WriteLoc::Cast>(value)) {
            auto cast = std::get<ir::WriteLoc::Cast>(value);
            std::string b;
            if (cast.mode == ir::Cast::Mode::Reinterpret) {
                b += "reinterpret";
                b += '_';
            }
            b += "cast<";
            std::stringstream t;
            codegen::emit_type(t, cast.type);
            b += t.str();
            b += ">(";
            if (is_assignment) {
                b += "&"; // *must* take the address for assignment.
                is_pointer = true;
            }
            b += ss;
            b += ")";
            ss = std::move(b);

            continue;
        }
        internal_error << "unexpected WriteLoc access type";
    }
    os << ss;
}

} //  namespace bonsai
