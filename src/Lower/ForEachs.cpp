#include "Lower/ForEachs.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace lower {

namespace {

bool is_range_call(const ir::Expr &expr) {
    if (!expr.type().is<ir::Array_t>()) {
        return false;
    }
    const ir::Call *call = expr.as<ir::Call>();
    if (call == nullptr) {
        return false;
    }
    const ir::Var *callee = call->func.as<ir::Var>();
    if (callee == nullptr) {
        return false;
    }
    return callee->name == "range";
}

ir::Expr get_range_offset(const ir::Expr &expr) {
    internal_assert(expr.type().is<ir::Array_t>());
    const ir::Call *call = expr.as<ir::Call>();
    internal_assert(call);
    const ir::Var *callee = call->func.as<ir::Var>();
    internal_assert(callee && callee->name == "range");
    internal_assert(call->args.size() == 3);
    return call->args[1];
}

ir::Expr get_range_iterable(const ir::Expr &expr) {
    internal_assert(expr.type().is<ir::Array_t>());
    const ir::Call *call = expr.as<ir::Call>();
    internal_assert(call);
    const ir::Var *callee = call->func.as<ir::Var>();
    internal_assert(callee && callee->name == "range");
    internal_assert(call->args.size() == 3);
    return call->args[0];
}

// Lowers for-each loops to for-all loops.
struct LowerToForAll : public ir::Mutator {
    uint64_t lcounter = 0; // unique identifier for iterator variables.

    std::string unique_idx_name() {
        return "?idx" + std::to_string(lcounter++);
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        internal_assert(!contains<ir::Yield>(node->body));

        // TODO: should generate a For loop if not parallelizable...

        ir::Expr iterable = node->iter;

        const ir::Array_t *array_type = iterable.type().as<ir::Array_t>();
        internal_assert(array_type);

        ir::Expr end = array_type->size;
        ir::Expr begin = make_zero(end.type());
        ir::Expr stride = make_one(end.type());

        std::string idx_name = unique_idx_name();

        ir::Expr idx = ir::Var::make(end.type(), idx_name);

        ir::Expr load;
        if (is_range_call(iterable)) {
            idx = get_range_offset(iterable) + idx;
            iterable = get_range_iterable(iterable);
        }

        load = ir::Extract::make(iterable, idx);
        // `var = iterable[idx]`
        ir::Stmt do_load = ir::LetStmt::make(
            ir::WriteLoc(node->name, iterable.type().element_of()),
            std::move(load));

        ir::Stmt body = mutate(node->body);

        ir::ForAll::Slice slice{std::move(begin), std::move(end),
                                std::move(stride)};

        return ir::ForAll::make(idx_name, std::move(do_load), std::move(slice),
                                std::move(body));
    }
};

} // namespace

ir::FuncMap LowerForEachs::run(ir::FuncMap funcs) const {
    LowerToForAll convert_fa;
    for (auto &[_, f] : funcs) {
        f->body = convert_fa.mutate(f->body);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
