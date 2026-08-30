#include "Lower/Scans.h"
#include "Lower/Trees.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
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

std::shared_ptr<Function>
build_scan_func(const std::vector<TypedVar> &args, const std::string &func_name,
                const std::optional<AggOp::OpType> &op, const Expr &map_func) {
    std::vector<ir::Argument> f_args(args.size());
    for (size_t i = 0, e = args.size(); i < e; i++) {
        f_args[i].name = args[i].name;
        f_args[i].type = args[i].type;
        // The output -- a set to append to, or an accumulator to fold into --
        // is the last argument and is written through.
        f_args[i].mutating =
            args[i].type.is<Set_t>() || (op.has_value() && i + 1 == e);
    }
    std::shared_ptr<Function> func = std::make_shared<Function>(
        func_name, std::move(f_args), Void_t::make(), Stmt(),
        Function::InterfaceList{}, std::vector<Function::Attribute>{});

    struct ScansToCalls : public Mutator {
        std::shared_ptr<Function> func;
        Expr write_expr;
        WriteLoc write_loc;
        std::optional<AggOp::OpType> op;
        Expr map_func;

        // The parameters deliberately do not share the members' names. The
        // body reads `func` after the mem-initializer has moved from it, so a
        // parameter named `func` would shadow the member and be null there.
        ScansToCalls(std::shared_ptr<Function> scan_func,
                     std::optional<AggOp::OpType> agg_op, Expr map)
            : func(std::move(scan_func)), op(std::move(agg_op)),
              map_func(std::move(map)) {
            write_expr =
                Var::make(func->args.back().type, func->args.back().name);
            write_loc =
                WriteLoc(func->args.back().name, func->args.back().type);
        }

        size_t counter = 0;

        // Contribute one element: append it to the output set, or fold it
        // into the accumulator.
        Stmt contribute(const Expr &value) {
            if (!op.has_value()) {
                return AppendStmt::make(write_loc, value);
            }
            Expr mapped = map_func.defined()
                              ? apply_unary_lambda(map_func, value)
                              : value;
            return Accumulate::make(write_loc, accumulate_op_for(*op),
                                    std::move(mapped));
        }

        // Appending an iterable adds all of its elements at once, but a
        // reduction has to fold them in one at a time.
        Stmt contribute_each(const Expr &values) {
            if (!op.has_value()) {
                return AppendStmt::make(write_loc, values);
            }
            // Summing the same constant over every element is that constant
            // times how many there are, which the runtime answers directly
            // rather than by iterating.
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
            std::string name = "_elem" + std::to_string(counter++);
            Expr element = Var::make(values.type().element_of(), name);
            return ForEach::make(std::move(name), values, contribute(element));
        }

        Stmt visit(const Scan *node) override {
            // return YieldFrom::make(node->value);

            std::vector<Expr> call_args(func->args.size());
            auto ids = break_tuple(node->value);
            std::vector<Stmt> stmts;
            stmts.reserve(ids.size());

            Expr fexpr = Var::make(func->call_type(), func->name);

            for (const auto &id : ids) {
                std::vector<Expr> call_args(func->args.size(), Expr());
                if (const Tuple_t *tuple_t = id.type().as<Tuple_t>()) {
                    internal_assert(tuple_t->etypes.size() + 1 ==
                                    func->args.size());
                    for (size_t i = 0; i < tuple_t->etypes.size(); i++) {
                        call_args[i] = Extract::make(id, i);
                    }
                } else {
                    call_args[0] = id;
                }
                // Put the write location in the call.
                call_args.back() = write_expr;
                stmts.push_back(CallStmt::make(fexpr, std::move(call_args)));
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

    // TODO: support product scans!
    internal_assert(args.size() == 2)
        << "[unimplemented] scanning a product of trees";
    const BVH_t *bvh_t0 = args.front().type.as<BVH_t>();
    internal_assert(bvh_t0);
    // write argument
    internal_assert(op.has_value() || args.back().type.is<Set_t>());

    Stmt match_body = build_base_scan(args.front().name, bvh_t0);
    // Need to rewrite scans in ^ to recursive calls.
    // And Yields to Appends

    auto tree_args = args;
    tree_args.pop_back(); // lose write loc

    func->body = ScansToCalls(func, op, map_func).mutate(match_body);
    // func->body = RecLoop::make(std::move(tree_args), std::move(func->body));

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

    // The trees to traverse, and the location to write into. A set-union scan
    // writes into the enclosing output set; a reducing scan writes into its
    // own accumulator.
    std::pair<std::vector<TypedVar>, TypedVar>
    scan_arguments(const Scan *node) const {
        std::vector<TypedVar> trees;
        for (const auto &arg : args) {
            if (arg.type.is<BVH_t>()) {
                trees.push_back(arg);
            }
        }
        if (node->op.has_value()) {
            return {std::move(trees),
                    TypedVar{node->loc.base(), node->loc.base_type()}};
        }
        for (const auto &arg : args) {
            if (arg.type.is<Set_t>()) {
                return {std::move(trees), arg};
            }
        }
        internal_error << "No output for scan: " << Stmt(node);
    }

    Expr get_or_build_callable(const Scan *node,
                               const std::vector<TypedVar> &scan_args) {
        const std::string base = scan_func_name(scan_args);
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
        auto func = build_scan_func(scan_args, name, node->op, node->func);
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
        auto ids = break_tuple(node->value);
        std::vector<Stmt> stmts;
        stmts.reserve(ids.size());

        internal_assert(!args.empty() && args.front().type.is<BVH_t>())
            << args.front();

        auto [trees, output] = scan_arguments(node);
        std::vector<TypedVar> scan_args = trees;
        scan_args.push_back(output);
        Expr callable = get_or_build_callable(node, scan_args);

        // Make n scan calls.
        for (const auto &id : ids) {
            std::vector<Expr> call_args;
            if (const Tuple_t *tuple_t = id.type().as<Tuple_t>()) {
                internal_assert(tuple_t->etypes.size() < args.size());
                for (size_t i = 0; i < tuple_t->etypes.size(); i++) {
                    call_args.push_back(Extract::make(id, i));
                }
            } else {
                call_args.push_back(id);
            }
            call_args.push_back(Var::make(output.type, output.name));
            stmts.push_back(CallStmt::make(callable, std::move(call_args)));
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
