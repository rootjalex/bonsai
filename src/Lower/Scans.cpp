#include "Lower/Scans.h"
#include "Lower/Trees.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <optional>
#include <unordered_set>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

std::string scan_func_name(const std::vector<TypedVar> &args) {
    std::string func_name = "_scan";
    for (const auto &arg : args) {
        if (const BVH_t *bvh_t = arg.type.as<BVH_t>()) {
            func_name += "_" + bvh_t->name;
        }
    }
    return func_name;
}

// The accumulate that combines a subtree's contribution into a reduction.
Accumulate::OpType accumulate_op_for(AggOp::OpType op) {
    switch (op) {
    case AggOp::sum:
        return Accumulate::Add;
    case AggOp::prod:
        return Accumulate::Mul;
    default:
        internal_error << "Cannot scan a subtree with reduction: "
                       << to_string(op);
    }
}

// A scan traverses a whole subtree. By default it unions the elements into a
// set; when the scan carries a reduction it folds them into an accumulator
// instead, applying the scan's map function to each element first.
// Substitute an element into a unary lambda.
Expr apply_unary_lambda(const Expr &lambda_expr, const Expr &value) {
    const Lambda *lambda = lambda_expr.as<Lambda>();
    internal_assert(lambda && lambda->args.size() == 1)
        << "Scan map is not a unary lambda: " << lambda_expr;
    return replace(lambda->args[0].name, value, lambda->value);
}

// A scan walks a whole subtree. The two kinds differ in where the elements go,
// and that decides the shape of the function built for them.
//
// A reducing scan folds each element into an accumulator its caller owns, so
// it takes that accumulator as a mutating argument and returns nothing.
//
// A set-union scan has no accumulator to fold into: its elements are the
// traversal's own output. It therefore takes only the trees and *yields*,
// exactly as the traversal it was lifted out of does, and returns a set.
// LowerDynamicSets gives every set-returning function a backing allocation
// and turns its yields into appends, so this one needs no special handling
// there, and -- unlike an accumulator -- a yield needs nothing to exist yet.
std::shared_ptr<Function>
build_scan_func(const std::vector<TypedVar> &trees,
                const std::optional<TypedVar> &acc,
                const std::string &func_name,
                const std::optional<AggOp::OpType> &op, const Expr &map_func) {
    internal_assert(op.has_value() == acc.has_value())
        << "A reducing scan accumulates and a set-union scan yields: "
        << func_name;
    // TODO: support product scans!
    internal_assert(trees.size() == 1)
        << "[unimplemented] scanning a product of trees";
    const BVH_t *bvh_t = trees.front().type.as<BVH_t>();
    internal_assert(bvh_t) << trees.front().type;

    std::vector<ir::Argument> f_args;
    f_args.reserve(trees.size() + acc.has_value());
    for (const TypedVar &tree : trees) {
        f_args.emplace_back(tree.name, tree.type);
    }
    Type ret_type = Set_t::make(bvh_t->primitive);
    if (acc.has_value()) {
        // The accumulator is written through, and is what the scan produces.
        f_args.emplace_back(acc->name, acc->type, /*default_value=*/Expr(),
                            /*mutating=*/true);
        ret_type = Void_t::make();
    }

    std::shared_ptr<Function> func = std::make_shared<Function>(
        func_name, std::move(f_args), std::move(ret_type), Stmt(),
        Function::InterfaceList{}, std::vector<Function::Attribute>{});

    struct ScansToCalls : public Mutator {
        std::shared_ptr<Function> func;
        bool reduces;
        WriteLoc write_loc;
        std::optional<AggOp::OpType> op;
        Expr map_func;

        // The parameters deliberately do not share the members' names. The
        // body reads `func` after the mem-initializer has moved from it, so a
        // parameter named `func` would shadow the member and be null there.
        ScansToCalls(std::shared_ptr<Function> scan_func,
                     std::optional<AggOp::OpType> agg_op, Expr map)
            : func(std::move(scan_func)), reduces(agg_op.has_value()),
              op(std::move(agg_op)), map_func(std::move(map)) {
            if (reduces) {
                write_loc =
                    WriteLoc(func->args.back().name, func->args.back().type);
            }
        }

        size_t counter = 0;

        // Contribute one element: yield it, or fold it into the accumulator.
        Stmt contribute(const Expr &value) {
            if (!reduces) {
                return Yield::make(value);
            }
            Expr mapped = map_func.defined()
                              ? apply_unary_lambda(map_func, value)
                              : value;
            return Accumulate::make(write_loc, accumulate_op_for(*op),
                                    std::move(mapped));
        }

        // Contribute every element of an iterable. Yielding a whole iterable
        // is what Iterate means, and LowerDynamicSets turns it into a single
        // append; a reduction has to fold the elements in one at a time.
        Stmt contribute_each(const Expr &values) {
            if (!reduces) {
                return Iterate::make(values);
            }
            {
                // Summing the same constant over every element is that
                // constant times how many there are, which the runtime
                // answers directly rather than by iterating.
                if (*op == AggOp::sum && map_func.defined()) {
                    const Lambda *lambda = map_func.as<Lambda>();
                    internal_assert(lambda && lambda->args.size() == 1);
                    if (is_const(lambda->value)) {
                        Expr n = cast_to(write_loc.base_type(), count(values));
                        Expr total = is_const_one(lambda->value)
                                         ? std::move(n)
                                         : Expr(lambda->value) * std::move(n);
                        return Accumulate::make(write_loc, Accumulate::Add,
                                                std::move(total));
                    }
                }
            }
            std::string name = "_elem" + std::to_string(counter++);
            Expr element = Var::make(values.type().element_of(), name);
            return ForEach::make(std::move(name), values, contribute(element));
        }

        // The arguments of a recursive call on one child.
        std::vector<Expr> call_args_for(const Expr &child) const {
            std::vector<Expr> call_args(func->args.size(), Expr());
            if (const Tuple_t *tuple_t = child.type().as<Tuple_t>()) {
                internal_assert(tuple_t->etypes.size() + reduces ==
                                func->args.size());
                for (size_t i = 0; i < tuple_t->etypes.size(); i++) {
                    call_args[i] = Extract::make(child, i);
                }
            } else {
                call_args[0] = child;
            }
            if (reduces) {
                // Hand the same accumulator down to the subtree.
                call_args.back() =
                    Var::make(func->args.back().type, func->args.back().name);
            }
            return call_args;
        }

        Stmt visit(const Scan *node) override {
            Expr fexpr = Var::make(func->call_type(), func->name);
            std::vector<Stmt> stmts;
            for (const Expr &child : break_tuple(node->value)) {
                std::vector<Expr> call_args = call_args_for(child);
                if (reduces) {
                    stmts.push_back(
                        CallStmt::make(fexpr, std::move(call_args)));
                    continue;
                }
                // The subtree returns a set; everything in it is ours.
                stmts.push_back(
                    Iterate::make(Call::make(fexpr, std::move(call_args))));
            }
            return Sequence::make(std::move(stmts));
        }

        Stmt visit(const Iterate *node) override {
            return contribute_each(node->value);
        }

        Stmt visit(const Yield *node) override {
            return contribute(node->value);
        }

        RESTRICT_MUTATOR(Stmt, YieldFrom);
    };

    Stmt match_body = build_base_scan(trees.front().name, bvh_t);
    func->body = ScansToCalls(func, op, map_func).mutate(match_body);
    return func;
}

struct LowerScansImpl : public Mutator {
    FuncMap new_funcs;

    std::map<std::string, Type> bvh_types;
    std::vector<TypedVar> args;

    Expr visit(const Var *node) override {
        if (const auto *bvh_t = node->type.as<BVH_t>()) {
            bvh_types[bvh_t->name] = node->type;
        }
        return ir::Mutator::visit(node);
    }

    // Scan functions built so far, so that two scans of the same trees with
    // the same reduction and map share one function.
    struct BuiltScan {
        std::string base;
        std::optional<AggOp::OpType> op;
        Expr map_func;
        std::string name;
    };
    std::vector<BuiltScan> built;

    // The trees this scan walks: the traversal's tree arguments.
    std::vector<TypedVar> scan_trees() const {
        std::vector<TypedVar> trees;
        for (const auto &arg : args) {
            if (arg.type.is<BVH_t>()) {
                trees.push_back(arg);
            }
        }
        return trees;
    }

    // The accumulator a reducing scan folds into. A set-union scan has none:
    // it yields, and its elements are the traversal's own output.
    std::optional<TypedVar> scan_accumulator(const Scan *node) const {
        if (!node->op.has_value()) {
            return std::nullopt;
        }
        return TypedVar{node->loc.base(), node->loc.base_type()};
    }

    Expr get_or_build_callable(const Scan *node,
                               const std::vector<TypedVar> &trees,
                               const std::optional<TypedVar> &acc) {
        const std::string base = scan_func_name(trees);
        for (const auto &b : built) {
            const bool same_map =
                b.map_func.defined() == node->func.defined() &&
                (!b.map_func.defined() || equals(b.map_func, node->func));
            if (b.base == base && b.op == node->op && same_map) {
                const auto &f = new_funcs.at(b.name);
                return Var::make(f->call_type(), f->name);
            }
        }
        // Need to build this func.
        std::string name = base;
        if (node->op.has_value()) {
            name += "_" + to_string(*node->op);
        }
        if (new_funcs.contains(name)) {
            name += "_" + std::to_string(built.size());
        }
        auto func = build_scan_func(trees, acc, name, node->op, node->func);
        Expr ret = Var::make(func->call_type(), func->name);
        new_funcs[name] = std::move(func);
        built.push_back({base, node->op, node->func, std::move(name)});
        return ret;
    }

    Stmt visit(const RecLoop *node) override {
        internal_assert(args.empty()) << Stmt(node);
        // A RecLoop carries full ir::Arguments; the scan lowering only needs
        // each one's name and type.
        args.reserve(node->args.size());
        for (const auto &arg : node->args) {
            args.push_back(TypedVar{arg.name, arg.type});
        }
        std::vector<TypedVar> free_vars = gather_free_vars(node->body);
        // Add non-duplicating free_vars.
        std::unordered_set<std::string> arg_names;
        for (const auto &arg : args) {
            arg_names.insert(arg.name);
        }

        for (const auto &var : free_vars) {
            if (arg_names.insert(var.name).second) {
                args.push_back(var);
            }
        }

        Stmt body = mutate(node->body);

        args.clear();

        if (body.same_as(node->body)) {
            return node;
        }
        return RecLoop::make(node->args, std::move(body));
    }

    // TODO: note that this does not work for product scans yet!
    Stmt visit(const Scan *node) override {
        internal_assert(!args.empty() && args.front().type.is<BVH_t>())
            << args.front();

        const std::vector<TypedVar> trees = scan_trees();
        const std::optional<TypedVar> acc = scan_accumulator(node);
        Expr callable = get_or_build_callable(node, trees, acc);

        std::vector<Stmt> stmts;
        for (const Expr &id : break_tuple(node->value)) {
            std::vector<Expr> call_args;
            if (const Tuple_t *tuple_t = id.type().as<Tuple_t>()) {
                internal_assert(tuple_t->etypes.size() < args.size());
                for (size_t i = 0; i < tuple_t->etypes.size(); i++) {
                    call_args.push_back(Extract::make(id, i));
                }
            } else {
                call_args.push_back(id);
            }
            if (acc.has_value()) {
                call_args.push_back(Var::make(acc->type, acc->name));
                stmts.push_back(CallStmt::make(callable, std::move(call_args)));
                continue;
            }
            // The scan returns a set, and the traversal yields all of it.
            stmts.push_back(
                Iterate::make(Call::make(callable, std::move(call_args))));
        }
        return Sequence::make(std::move(stmts));
    }
};

} // namespace

ir::FuncMap LowerScans::run(ir::FuncMap funcs,
                            const CompilerOptions &options) const {
    LowerScansImpl lowerer;
    for (const auto &[name, func] : funcs) {
        // lowerer.args = func->typedvar_argtypes();
        lowerer.bvh_types.clear();
        func->body = lowerer.mutate(func->body);
    }

    for (auto &[name, func] : lowerer.new_funcs) {
        auto [_, inserted] = funcs.try_emplace(name, std::move(func));
        internal_assert(inserted)
            << "Failed to insert recursive lowering: " << name;
    }

    return funcs;
}

} // namespace lower
} // namespace bonsai
