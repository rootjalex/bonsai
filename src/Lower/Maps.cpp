#include "Lower/Maps.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Frame.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Opt/Simplify.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

using namespace ir;

namespace {
// TODO: make configurable?
// Used to compute offset into arrays.
static const Type index_t = UInt_t::make(32);

int64_t acounter = 0; // unique identifier for allocations.
std::string unique_alloc_name() {
    return "_alloc" + std::to_string(acounter++);
}
std::string unique_loop_name(const size_t depth) {
    return "_i" + std::to_string(depth);
}

struct BuildMapArgs {
    std::string result;
    Type base_type;
    // TODO: store scheduling information here, it's static
    // across compiling foralls.
};

// TODO: make sure this works for i32, i64, u64, etc...
Expr stride_index(Expr base, Expr stride, Expr offset) {
    // return cast(index_t, base) * cast(index_t, stride) + cast(index_t,
    // offset);
    return base * stride + offset;
}

// Does not insert allocations, is used to recursively build forall loops.
// `idx` is the index to the current `start` of the output
Stmt build_traversal_helper(const Expr &func, const Expr &array,
                            const size_t depth, const Expr &out_idx,
                            const BuildMapArgs &args, FuncMap &funcs) {
    std::string loop_idx = unique_loop_name(depth); // TODO: always?

    const Array_t *array_t = array.type().as<Array_t>();
    internal_assert(array_t);

    // TODO(ajr): apply scheduling to the for loop.
    Expr end = array_t->size;
    internal_assert(end.defined() && end.type().defined());
    Expr begin = make_zero(end.type());
    Expr stride = make_one(end.type());

    Expr index = Var::make(end.type(), loop_idx);
    Expr load = Extract::make(array, index);

    // TODO: support nested stores instead.
    Expr nested_idx = stride_index(out_idx, end, index);

    Stmt loop_body;

    // Now apply func to loaded value.
    internal_assert(func.type().is_func());
    if (const Lambda *l = func.as<Lambda>()) {
        internal_assert(l->args.size() == 1)
            << "[unimplemented] map fusion for tuple splitting:" << func
            << " called with " << load;
        Expr expr = replace(l->args[0].name, load, l->value);
        // Try to fuse a nested map.
        if (auto map = as_map(expr)) {
            Expr next_func = map->a, next_array = map->b;
            loop_body = build_traversal_helper(next_func, next_array, depth + 1,
                                               nested_idx, args, funcs);
        } else {
            WriteLoc loc(args.result, args.base_type);
            loc.add_index_access(nested_idx);
            loop_body = Assign::make(loc, expr, /*mutating=*/true);
        }
    } else {
        Expr expr = Call::make(func, {load});
        WriteLoc loc(args.result, args.base_type);
        loc.add_index_access(nested_idx);
        loop_body = Assign::make(loc, expr, /*mutating=*/true);
    }

    return ForAll::make(
        loop_idx,
        ForAll::Slice{std::move(begin), std::move(end), std::move(stride)},
        std::move(loop_body));
}

// Outermost call, inserts an allocation of the output size.
Stmt build_traversal(const SetOp *map_expr, FuncMap &funcs) {
    Type alloc_type = flatten_array_type(map_expr->type);

    // TODO(ajr): set `args` by schedule.
    BuildMapArgs args;
    std::string alloc_name = unique_alloc_name();
    args.result = alloc_name;
    args.base_type = alloc_type;

    Stmt alloc = Assign::make(WriteLoc(alloc_name, alloc_type),
                              Build::make(alloc_type), /*mutating=*/false);
    static const Expr zero = make_zero(index_t);
    Stmt body = build_traversal_helper(map_expr->a, map_expr->b, /*depth=*/0,
                                       zero, args, funcs);

    Expr ret_var = cast(map_expr->type, Var::make(alloc_type, alloc_name));
    Stmt return_var = Return::make(ret_var);

    // TODO: flatten sequence?
    return Sequence::make(
        {std::move(alloc), std::move(body), std::move(return_var)});
}

struct LowerMapsImpl : public Mutator {
    FuncMap &funcs;

    LowerMapsImpl(FuncMap &funcs) : funcs(funcs) {}

    // For unique func names
    size_t counter = 0;

    std::string new_func_name() {
        return "_traverse_array" + std::to_string(counter++);
    }

    Expr visit(const SetOp *node) override {
        if (node->op == SetOp::map && node->type.is<Array_t>()) {
            return build_func(node);
        }
        return Mutator::visit(node);
    }

    // Returns a call to the func.
    // Inserts the built func into new_funcs
    // TODO(ajr): this is stolen from Lower/Trees.cpp, merge duplicate code.
    Expr build_func(const SetOp *node) {
        const std::string func = new_func_name();
        const auto free_vars = gather_free_vars(node);

        Stmt body = build_traversal(node, funcs);
        internal_assert(body.defined());

        std::vector<Function::Argument> func_args;
        std::transform(free_vars.cbegin(), free_vars.cend(),
                       std::back_inserter(func_args), [&](const auto &var) {
                           return Function::Argument(var.name, var.type);
                       });

        Type ret_type = node->type;
        ;
        auto f = std::make_shared<Function>(
            func, std::move(func_args), std::move(ret_type), std::move(body),
            Function::InterfaceList{},
            std::vector<Function::Attribute>{}); // TODO: inline?
        Type call_type = f->call_type();

        const auto [_, inserted] = funcs.try_emplace(func, std::move(f));
        internal_assert(inserted)
            << "Function: " << func << " already exists in program funcs\n";

        std::vector<Expr> call_args;
        std::transform(free_vars.begin(), free_vars.end(),
                       std::back_inserter(call_args),
                       [&](auto &var) -> Expr { return var; });

        return Call::make(Var::make(std::move(call_type), func), call_args);
    }
};

} // namespace

Program LowerMaps::run(Program program) const {
    // TODO(ajr): get info from the schedule.

    LowerMapsImpl lower(program.funcs);

    for (auto &[_, func] : program.funcs) {
        func->body = lower.mutate(std::move(func->body));
    }

    return program;
}

} // namespace lower
} // namespace bonsai
