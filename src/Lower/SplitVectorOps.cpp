#include "Lower/SplitVectorOps.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

std::string get_split_name(const std::string &name, size_t lane) {
    return "_" + name + "_e" + std::to_string(lane);
}

std::string get_split_func_name(const std::string &fname) {
    return "__extract_" + fname;
}

std::vector<std::string> get_vector_list(const std::string &name,
                                         const Vector_t *vector_t) {
    std::vector<std::string> fields(vector_t->lanes);
    for (size_t i = 0; i < vector_t->lanes; i++) {
        fields[i] = get_split_name(name, i);
    }
    return fields;
}

struct SplitVectorOpsImpl : public Mutator {
    std::vector<Expr> exprs;

    // Allocations in this function, that have been rewritten.
    std::set<std::string> safe_write_locs;

    // This should always be true except for during returns
    // and writes to external allocations.
    bool split_var_reads = true;

    std::vector<Expr> get_exprs() {
        std::vector<Expr> ret = std::move(exprs);
        exprs.clear();
        return ret;
    }

    // Default behavior of these are fine.
    // Expr visit(const IntImm *node) override;
    // Expr visit(const UIntImm *) override;
    // Expr visit(const FloatImm *) override;
    // Expr visit(const BoolImm *) override;
    // Expr visit(const Infinity *) override;

    Expr visit(const VecImm *node) override {
        exprs = node->values;
        return Expr();
    }

    Expr visit(const Var *node) override {
        if (!node->type.is<Vector_t>()) {
            return node;
        }

        exprs.resize(node->type.lanes());
        Type etype = node->type.element_of();
        for (size_t i = 0; i < node->type.lanes(); i++) {
            exprs[i] = Var::make(etype, get_split_name(node->name, i));
        }
        if (!split_var_reads) {
            // This is in some return value or external
            // write. Needs to be returned to vector type.
            return Build::make(node->type, get_exprs());
        }
        return Expr();
    }

    Expr visit(const BinOp *node) override {
        Expr a = mutate(node->a);
        std::vector<Expr> a_exprs = get_exprs();
        Expr b = mutate(node->b);
        std::vector<Expr> b_exprs = get_exprs();
        internal_assert(a.defined() == b.defined())
            << "Splitting vector ops has weird behavior on binop: "
            << Expr(node) << " -> " << a << " and " << b;
        internal_assert(a_exprs.size() == b_exprs.size())
            << "Splitting vector ops has weird behavior on binop: "
            << Expr(node) << " -> " << a_exprs.size() << " versus "
            << b_exprs.size();
        internal_assert(a.defined() ^ !a_exprs.empty())
            << "Splitting value: " << node->a << " gave weird behavior: " << a
            << " but " << a_exprs.size() << " elems split.";
        internal_assert(b.defined() ^ !b_exprs.empty())
            << "Splitting value: " << node->b << " gave weird behavior: " << b
            << " but " << b_exprs.size() << " elems split.";
        if (a.defined()) {
            if (a.same_as(node->a) && b.same_as(node->b)) {
                return node;
            }
            return BinOp::make(node->op, std::move(a), std::move(b));
        }

        const size_t n = a_exprs.size();
        exprs.resize(n);
        for (size_t i = 0; i < n; i++) {
            exprs[i] = BinOp::make(node->op, std::move(a_exprs[i]),
                                   std::move(b_exprs[i]));
        }
        return Expr();
    }

    Expr visit(const UnOp *node) override {
        Expr a = mutate(node->a);
        std::vector<Expr> a_exprs = get_exprs();

        internal_assert(a.defined() ^ !a_exprs.empty())
            << "Splitting value: " << node->a << " gave weird behavior: " << a
            << " but " << a_exprs.size() << " elems split.";

        if (a.defined()) {
            if (a.same_as(node->a)) {
                return node;
            }
            return UnOp::make(node->op, a);
        }

        const size_t n = a_exprs.size();
        exprs.resize(n);
        for (size_t i = 0; i < n; i++) {
            exprs[i] = UnOp::make(node->op, std::move(a_exprs[i]));
        }
        return Expr();
    }

    Expr visit(const Select *node) override {
        Expr cond = mutate(node->cond);
        std::vector<Expr> cond_exprs = get_exprs();

        Expr t = mutate(node->tvalue);
        std::vector<Expr> t_exprs = get_exprs();

        Expr f = mutate(node->fvalue);
        std::vector<Expr> f_exprs = get_exprs();

        internal_assert(cond.defined() ^ !cond_exprs.empty())
            << "Select cond gave inconsistent split: " << node->cond << " -> "
            << cond << " with " << cond_exprs.size();
        internal_assert(t.defined() ^ !t_exprs.empty())
            << "Select true_value gave inconsistent split: " << node->tvalue
            << " -> " << t << " with " << t_exprs.size();
        internal_assert(f.defined() ^ !f_exprs.empty())
            << "Select false_value gave inconsistent split: " << node->fvalue
            << " -> " << f << " with " << f_exprs.size();

        internal_assert(f.defined() == t.defined())
            << "Splitting vector ops has weird behavior on select: "
            << Expr(node) << " -> " << t << " and " << f;
        internal_assert(t_exprs.size() == f_exprs.size())
            << "Splitting vector ops has weird behavior on select: "
            << Expr(node) << " -> " << t_exprs.size() << " versus "
            << f_exprs.size();

        if (cond.defined()) {
            if (cond.same_as(node->cond) && t.same_as(node->tvalue) &&
                f.same_as(node->fvalue)) {
                return node;
            }
            if (t.defined()) {
                return Select::make(cond, t, f);
            }
            // The condition was not a vector, but the options were.
            const size_t n = t_exprs.size();
            exprs.resize(n);
            for (size_t i = 0; i < n; i++) {
                exprs[i] = Select::make(cond, t_exprs[i], f_exprs[i]);
            }
            return Expr();
        }

        internal_assert(t_exprs.size() == cond_exprs.size())
            << "Splitting vector ops has weird behavior on select: "
            << Expr(node) << " -> " << t_exprs.size() << " versus "
            << cond_exprs.size();

        const size_t n = t_exprs.size();
        exprs.resize(n);
        for (size_t i = 0; i < n; i++) {
            exprs[i] =
                Select::make(std::move(cond_exprs[i]), std::move((t_exprs[i])),
                             std::move(f_exprs[i]));
        }
        return Expr();
    }

    Expr visit(const Cast *node) override {
        Expr a = mutate(node->value);
        std::vector<Expr> a_exprs = get_exprs();

        internal_assert(a.defined() ^ !a_exprs.empty())
            << "Cast operand gave inconsistent split.";

        if (a.defined()) {
            internal_assert(!node->type.is<Vector_t>());
            if (a.same_as(node->value))
                return node;
            return Cast::make(node->type, a);
        }

        internal_assert(node->type.is<Vector_t>());
        exprs.resize(a_exprs.size());
        for (size_t i = 0; i < a_exprs.size(); i++) {
            exprs[i] =
                Cast::make(node->type.element_of(), std::move(a_exprs[i]));
        }
        return Expr();
    }

    Expr visit(const Broadcast *node) override {
        Expr a = mutate(node->value);
        internal_assert(a.defined())
            << "Mutation of a broadcasted primitive broke something: "
            << node->value << " -> " << a;
        exprs.resize(node->lanes);
        for (size_t i = 0; i < node->lanes; i++) {
            exprs[i] = a;
        }
        return Expr();
    }
    Expr visit(const VectorReduce *node) override {
        if (!node->value.type().is_vector()) {
            // Could be array reduction?
            return Mutator::visit(node);
        }
        Expr value = mutate(node->value);
        internal_assert(!value.defined())
            << "Vector splitting expected mutated expression: " << node->value
            << " mapped to " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << "Vector splitting expected mutated expression: " << node->value;
        std::function<Expr(Expr, Expr)> make;
        switch (node->op) {
        case VectorReduce::Add: {
            make = [](Expr a, Expr b) {
                return BinOp::make(BinOp::Add, std::move(a), std::move(b));
            };
            break;
        }
        case VectorReduce::And: {
            make = [](Expr a, Expr b) {
                return BinOp::make(BinOp::BwAnd, std::move(a), std::move(b));
            };
            break;
        }
        case VectorReduce::Max: {
            make = [](Expr a, Expr b) {
                return Intrinsic::make(Intrinsic::max,
                                       {std::move(a), std::move(b)});
            };
            break;
        }
        case VectorReduce::Min: {
            make = [](Expr a, Expr b) {
                return Intrinsic::make(Intrinsic::min,
                                       {std::move(a), std::move(b)});
            };
            break;
        }
        case VectorReduce::Mul: {
            make = [](Expr a, Expr b) {
                return BinOp::make(BinOp::Mul, std::move(a), std::move(b));
            };
            break;
        }
        case VectorReduce::Or: {
            make = [](Expr a, Expr b) {
                return BinOp::make(BinOp::BwOr, std::move(a), std::move(b));
            };
            break;
        }
        default: {
            // TODO: IdxMax and IdxMin!
            internal_error << "[unimplemented] vector splitting of: "
                           << Expr(node);
        }
        }
        Expr expr = v_exprs.front();
        for (size_t i = 1; i < v_exprs.size(); i++) {
            expr = make(std::move(expr), std::move(v_exprs[i]));
        }
        return expr;
    }

    Expr make_select_chain(Expr idx, std::vector<Expr> values) {
        if (auto cidx = get_constant_value(idx); cidx.has_value()) {
            internal_assert(*cidx < values.size())
                << "Bad select chain lowering: " << " lane " << *cidx
                << " requested on vector of size: " << values.size();
            return values[*cidx];
        } else {
            // Lower to chain of selects.
            internal_assert(values.size() >= 2);
            Expr chain = values.back();
            for (int32_t i = values.size() - 2; i >= 0; i--) {
                chain = Select::make(idx == i, values[i], std::move(chain));
            }
            return chain;
        }
    }

    Expr visit(const VectorShuffle *node) override {
        Expr value = mutate(node->value);
        internal_assert(!value.defined())
            << "Vector splitting expected mutated expression: " << node->value
            << " mapped to " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << "Vector splitting expected mutated expression: " << node->value;
        exprs.resize(node->idxs.size());
        for (size_t i = 0; i < node->idxs.size(); i++) {
            exprs[i] = make_select_chain(node->idxs[i], v_exprs);
        }
        return Expr();
    }

    Expr visit(const Extract *node) override {
        // Tuple lowering should have happened by now.
        internal_assert(node->vec.type().is_vector());
        Expr value = mutate(node->vec);
        internal_assert(!value.defined())
            << "Vector splitting expected mutated expression: " << node->vec
            << " mapped to " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << "Vector splitting expected mutated expression: " << node->vec;
        Expr idx = mutate(node->idx);
        internal_assert(idx.defined())
            << "Vector splitting expected non-mutated expression: " << node->idx
            << " mapped to " << idx;
        return make_select_chain(std::move(idx), std::move(v_exprs));
    }

    Expr visit(const Build *node) override {
        if (node->type.is_vector()) {
            internal_assert(!node->values.empty());
            internal_assert(node->values.size() == node->type.lanes());
            exprs.resize(node->values.size());
            for (size_t i = 0; i < node->values.size(); i++) {
                Expr e = mutate(node->values[i]);
                internal_assert(e.defined()) << node->values[i];
                exprs[i] = std::move(e);
            }
            return Expr();
        } else {
            std::vector<Expr> values(node->values.size());
            for (size_t i = 0; i < node->values.size(); i++) {
                Expr e = mutate(node->values[i]);
                internal_assert(e.defined()) << node->values[i];
                values[i] = std::move(e);
            }
            return Build::make(node->type, std::move(values));
        }
    }

    Expr visit(const Access *node) override {
        if (!node->type.is_vector()) {
            return Mutator::visit(node);
        }
        // Load a vector field from a struct.
        // Assume, for now, we can't rewrite the
        // struct type. That might change later.
        Expr value = mutate(node->value);
        internal_assert(value.same_as(node->value))
            << "How did the Access value get changed? " << Expr(node)
            << " split base into " << value;
        internal_assert(exprs.empty());
        exprs.resize(node->type.lanes());
        for (size_t i = 0; i < node->type.lanes(); i++) {
            exprs[i] = Extract::make(node, i);
        }
        return Expr();
    }

    Expr visit(const Intrinsic *node) override {
        const size_t num_args = node->args.size();
        std::vector<Expr> rebuilt_args(num_args);
        std::vector<std::vector<Expr>> split_args(num_args);

        bool needs_split = false;
        size_t lanes = 0;

        // Mutate and collect splits
        for (size_t i = 0; i < num_args; ++i) {
            Expr arg = mutate(node->args[i]);
            std::vector<Expr> splits = get_exprs();

            internal_assert(arg.defined() ^ !splits.empty())
                << "Splitting value: " << node->args[i] << " at index " << i
                << " of intrinsics call: " << Expr(node)
                << " gave inconsistent result: defined = " << arg.defined()
                << ", split size = " << splits.size();

            if (arg.defined()) {
                rebuilt_args[i] = arg;
            } else {
                internal_assert((i == 0) || needs_split);
                needs_split = true;
                internal_assert((i == 0) || lanes == splits.size());
                lanes = splits.size();
                split_args[i] = std::move(splits);
            }
        }

        if (!needs_split) {
            if (std::equal(rebuilt_args.begin(), rebuilt_args.end(),
                           node->args.begin(),
                           [](const Expr &a, const Expr &b) {
                               return a.same_as(b);
                           })) {
                return node;
            } else {
                return Intrinsic::make(node->op, std::move(rebuilt_args));
            }
        }

        // Rebuild lane-wise
        exprs.resize(lanes);
        for (size_t lane = 0; lane < lanes; ++lane) {
            std::vector<Expr> lane_args(num_args);
            for (size_t i = 0; i < num_args; ++i) {
                lane_args[i] = split_args[i].empty() ? rebuilt_args[i]
                                                     : split_args[i][lane];
            }
            exprs[lane] = Intrinsic::make(node->op, std::move(lane_args));
        }

        return Expr();
    }

    Expr visit(const Call *node) override {
        std::vector<Expr> args(node->args.size());
        bool any_changed = false;
        // TODO(ajr): support vector arguments.
        // ScopedValue<bool> _(split_var_reads, false); ?
        for (size_t i = 0; i < node->args.size(); i++) {
            internal_assert(!node->args[i].type().is_vector())
                << "TODO(ajr): support vector arguments: " << Expr(node);
            args[i] = mutate(node->args[i]);
            internal_assert(args[i].defined())
                << node->args[i] << " got destroyed";
            any_changed = any_changed || !args[i].same_as(node->args[i]);
        }
        if (!any_changed) {
            return node;
        }
        return Call::make(node->func, std::move(args));
    }

    // Don't support (for now?).
    RESTRICT_MUTATOR(Expr, Ramp);
    RESTRICT_MUTATOR(Expr, Unwrap);
    RESTRICT_MUTATOR(Expr, Generator);
    RESTRICT_MUTATOR(Expr, Lambda);
    RESTRICT_MUTATOR(Expr, GeomOp);
    RESTRICT_MUTATOR(Expr, SetOp);
    RESTRICT_MUTATOR(Expr, Instantiate);
    RESTRICT_MUTATOR(Expr, PtrTo);
    RESTRICT_MUTATOR(Expr, Deref);

    // Stmts
    // TODO(ajr)
    RESTRICT_MUTATOR(Stmt, CallStmt);
    RESTRICT_MUTATOR(Stmt, Print);

    Stmt visit(const Return *node) override {
        ScopedValue<bool> _(split_var_reads, false);
        Expr value = mutate(node->value);
        internal_assert(value.defined());
        if (value.same_as(node->value)) {
            return node;
        }
        return Return::make(std::move(value));
    }

    Stmt visit(const LetStmt *node) override {
        Expr value = mutate(node->value);
        if (value.same_as(node->value)) {
            internal_assert(!value.type().is_vector());
            return node;
        } else if (value.defined() && !value.type().is_vector()) {
            return LetStmt::make(node->loc, std::move(value));
        }

        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << node->value << " split into nothing";
        // A vector LetStmt needs to be broken into lanes x LetStmts
        std::vector<Stmt> lets(v_exprs.size());
        Type etype = node->loc.base_type.element_of();
        for (size_t i = 0; i < v_exprs.size(); i++) {
            lets[i] = LetStmt::make(
                WriteLoc(get_split_name(node->loc.base, i), etype),
                std::move(v_exprs[i]));
        }

        return Sequence::make(std::move(lets));
    }

    // Default behaviors are fine.
    // RESTRICT_MUTATOR(Stmt, IfElse);
    // RESTRICT_MUTATOR(Stmt, DoWhile);

    Stmt visit(const Sequence *node) override {
        bool changed = false;
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());

        auto flatten = [&](const Stmt &stmt) {
            Stmt mut = mutate(stmt);
            changed = changed || !mut.same_as(stmt);
            internal_assert(mut.defined()) << stmt;
            if (const ir::Sequence *seq = mut.as<ir::Sequence>()) {
                stmts.insert(stmts.end(), seq->stmts.begin(), seq->stmts.end());
                changed = true;
            } else {
                stmts.emplace_back(std::move(mut));
            }
        };

        for (const auto &stmt : node->stmts) {
            flatten(stmt);
        }

        if (!changed) {
            return node;
        }
        internal_assert(!stmts.empty());
        return ir::Sequence::make(std::move(stmts));
    }

    Stmt visit(const Allocate *node) override {
        Expr value = mutate(node->value);
        if (value.same_as(node->value)) {
            internal_assert(!value.type().is_vector());
            return node;
        } else if (value.defined() && !value.type().is_vector()) {
            return Allocate::make(node->loc, std::move(value), node->memory);
        }

        internal_assert(node->loc.accesses.empty()) << Stmt(node);

        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << node->value << " split into nothing";
        // A vector Allocate needs to be broken into lanes x Allocates
        std::vector<Stmt> allocs(v_exprs.size());
        Type etype = node->loc.base_type.element_of();
        for (size_t i = 0; i < v_exprs.size(); i++) {
            allocs[i] = Allocate::make(
                WriteLoc(get_split_name(node->loc.base, i), etype),
                std::move(v_exprs[i]), node->memory);
        }

        safe_write_locs.insert(node->loc.base);
        return Sequence::make(std::move(allocs));
    }

    Stmt visit(const Store *node) override {
        // TODO(ajr): if we supported masks here, we might have to mutate it?
        internal_assert(!node->mask.defined());
        Expr value = mutate(node->value);
        if (value.same_as(node->value)) {
            internal_assert(!value.type().is_vector());
            return node;
        } else if (value.defined() && !value.type().is_vector()) {
            return Store::make(node->loc, std::move(value),
                               /*mask*/ node->mask);
        }

        internal_assert(node->loc.accesses.empty()) << "TODO:" << Stmt(node);

        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << node->value << " split into nothing";
        // A vector Store needs to be broken into lanes x Stores
        std::vector<Stmt> stores(v_exprs.size());
        Type etype = node->loc.base_type.element_of();
        internal_assert(!node->mask.defined());
        for (size_t i = 0; i < v_exprs.size(); i++) {
            stores[i] =
                Store::make(WriteLoc(get_split_name(node->loc.base, i), etype),
                            std::move(v_exprs[i]), /*mask=*/Expr());
        }

        safe_write_locs.insert(node->loc.base);
        return Sequence::make(std::move(stores));
    }

    Stmt visit(const Accumulate *node) override {
        // e.g. vec.x *= 4; (should be canonicalized to vec[0] *= 4).
        internal_assert(node->loc.accesses.empty())
            << "TODO: split with accesses: " << Stmt(node);
        internal_assert(node->op != Accumulate::Argmin &&
                        node->op != Accumulate::Argmax)
            << "TODO: split with accesses: " << Stmt(node);

        Expr value = mutate(node->value);

        if (!node->loc.base_type.is_vector()) {
            if (value.same_as(node->value)) {
                return node;
            }
            internal_assert(!value.type().is_vector()) << value;
            return Accumulate::make(node->loc, node->op, std::move(value));
        }

        if (value.defined()) {
            // Broadcast accumulate
            internal_assert(!value.type().is_vector()) << value;
            const size_t lanes = node->loc.base_type.lanes();
            std::vector<Stmt> accs(lanes);
            Type etype = node->loc.base_type.element_of();
            for (size_t i = 0; i < lanes; i++) {
                accs[i] = Accumulate::make(
                    WriteLoc(get_split_name(node->loc.base, i), etype),
                    node->op, value);
            }
            return Sequence::make(std::move(accs));
        }

        internal_assert(!value.defined())
            << "Vector operation should have been split: " << node->value
            << " -> " << value;
        std::vector<Expr> v_exprs = get_exprs();
        internal_assert(!v_exprs.empty())
            << node->value << " split into nothing";

        // A vector Accumulate needs to be broken into lanes x Accumulates
        std::vector<Stmt> accs(v_exprs.size());
        Type etype = node->loc.base_type.element_of();
        for (size_t i = 0; i < v_exprs.size(); i++) {
            accs[i] = Accumulate::make(
                WriteLoc(get_split_name(node->loc.base, i), etype), node->op,
                std::move(v_exprs[i]));
        }
        return Sequence::make(std::move(accs));
    }

    RESTRICT_MUTATOR(Stmt, Label);
    RESTRICT_MUTATOR(Stmt, RecLoop);
    RESTRICT_MUTATOR(Stmt, Match);
    RESTRICT_MUTATOR(Stmt, Yield);
    RESTRICT_MUTATOR(Stmt, Scan);
    RESTRICT_MUTATOR(Stmt, YieldFrom);
    RESTRICT_MUTATOR(Stmt, ForAll);
    RESTRICT_MUTATOR(Stmt, ForEach);
    RESTRICT_MUTATOR(Stmt, Continue);
    RESTRICT_MUTATOR(Stmt, Launch);
};

} // namespace

std::shared_ptr<Function> split_vector_ops(const Function &func) {
    SplitVectorOpsImpl lowerer;
    std::vector<Function::Argument> args;

    for (const auto &arg : func.args) {
        const Vector_t *vector_t = arg.type.as<Vector_t>();
        if (!vector_t) {
            internal_assert(!contains<Vector_t>(arg.type))
                << arg.name << " : " << arg.type;
            args.push_back(arg);
            continue;
        }
        internal_assert(!arg.default_value.defined());
        auto vlist = get_vector_list(arg.name, vector_t);
        for (auto &field : vlist) {
            args.emplace_back(std::move(field), vector_t->etype,
                              /*default_value=*/Expr(), arg.mutating);
        }
    }

    Stmt body = lowerer.mutate(func.body);

    // TODO(ajr): could get rid of structs entirely with RtoP!!
    Type ret_type = func.ret_type;

    return std::make_shared<Function>(
        get_split_func_name(func.name), std::move(args), std::move(ret_type),
        std::move(body), func.interfaces, func.attributes);
}

} // namespace lower
} // namespace bonsai
