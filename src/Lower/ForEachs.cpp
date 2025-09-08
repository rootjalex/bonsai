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
    const ir::Generator *call = expr.as<ir::Generator>();
    if (call == nullptr) {
        return false;
    }
    return call->op == ir::Generator::range;
}

ir::Expr get_range_offset(const ir::Expr &expr) {
    internal_assert(expr.type().is<ir::Array_t>());
    const ir::Generator *call = expr.as<ir::Generator>();
    internal_assert(call);
    internal_assert(call->op == ir::Generator::range);
    internal_assert(call->args.size() == 3);
    return call->args[1];
}

ir::Expr get_range_iterable(const ir::Expr &expr) {
    internal_assert(expr.type().is<ir::Array_t>());
    const ir::Generator *call = expr.as<ir::Generator>();
    internal_assert(call);
    internal_assert(call->op == ir::Generator::range);
    internal_assert(call->args.size() == 3);
    return call->args[0];
}

// Lowers for-each loops to for-all loops.
struct LowerToForAll : public ir::Mutator {
    uint64_t lcounter = 0; // unique identifier for iterator variables.

    std::map<std::string, ir::Expr> repls;

    std::string unique_idx_name() {
        return "_idx" + std::to_string(lcounter++);
    }

    ir::Expr visit(const ir::Var *node) override {
        if (auto iter = repls.find(node->name); iter != repls.end()) {
            return iter->second;
        }
        return node;
    }

    ir::Expr get_n(const ir::Type &type) {
        if (const auto *array_t = type.as<ir::Array_t>()) {
            return array_t->size;
        } else if (const auto *vector_t = type.as<ir::Vector_t>()) {
            return vector_t->lanes;
        }
        internal_error << "Unknown iterable type: " << type;
    }

    ir::Stmt visit(const ir::ForEach *node) override {
        ir::Expr iterable = node->iter;

        ir::Expr end = get_n(iterable.type());
        internal_assert(end.defined()) << ir::Stmt(node);
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
        auto [_, inserted] = repls.try_emplace(node->name, load);
        internal_assert(inserted)
            << "Lowering ForEach encountered duplicate variable: "
            << node->name;

        ir::Stmt body = mutate(node->body);

        repls.erase(node->name);

        ir::ForAll::Slice slice{std::move(begin), std::move(end),
                                std::move(stride)};

        return ir::ForAll::make(idx_name, std::move(slice), std::move(body));
    }
};

} // namespace

ir::FuncMap LowerForEachs::run(ir::FuncMap funcs,
                               const CompilerOptions &options) const {
    LowerToForAll convert_fa;
    for (auto &[_, f] : funcs) {
        f->body = convert_fa.mutate(f->body);
    }
    return funcs;
}

} // namespace lower
} // namespace bonsai
