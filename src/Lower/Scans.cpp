#include "Lower/Scans.h"
#include "Lower/Trees.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"

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

// TODO: support scan augmentations!
std::shared_ptr<Function> build_scan_func(const std::vector<TypedVar> &args) {
    std::string func_name = scan_func_name(args);
    std::vector<Function::Argument> f_args(args.size());
    for (size_t i = 0, e = args.size(); i < e; i++) {
        f_args[i].name = args[i].name;
        f_args[i].type = args[i].type;
        f_args[i].mutating = args[i].type.is<Set_t>();
    }
    std::shared_ptr<Function> func = std::make_shared<Function>(
        func_name, std::move(f_args), Void_t::make(), Stmt(),
        Function::InterfaceList{}, std::vector<Function::Attribute>{});

    struct ScansToCalls : public Mutator {
        std::shared_ptr<Function> func;
        Expr write_expr;
        WriteLoc write_loc;

        ScansToCalls(std::shared_ptr<Function> _func) : func(std::move(_func)) {
            write_expr =
                Var::make(func->args.back().type, func->args.back().name);
            write_loc =
                WriteLoc(func->args.back().name, func->args.back().type);
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
            return Append::make(write_loc, node->value);
        }

        Stmt visit(const Yield *node) override {
            return Append::make(write_loc, node->value);
        }

        RESTRICT_MUTATOR(Stmt, YieldFrom);
    };

    // TODO: support product scans!
    internal_assert(args.size() == 2);
    const BVH_t *bvh_t0 = args.front().type.as<BVH_t>();
    internal_assert(bvh_t0);
    // write argument
    internal_assert(args.back().type.is<Set_t>());

    Stmt match_body = build_base_scan(args.front().name, bvh_t0);
    // Need to rewrite scans in ^ to recursive calls.
    // And Yields to Appends

    auto tree_args = args;
    tree_args.pop_back(); // lose write loc

    func->body = ScansToCalls(func).mutate(match_body);
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

    Expr get_or_build_callable() {
        std::vector<TypedVar> scan_args;
        for (const auto &arg : args) {
            if (arg.type.is<BVH_t>() || arg.type.is<Set_t>()) {
                scan_args.push_back(arg);
            }
        }
        std::string name = scan_func_name(scan_args);
        if (const auto &iter = new_funcs.find(name); iter != new_funcs.cend()) {
            return Var::make(iter->second->call_type(), iter->second->name);
        }
        // Need to build this func.
        auto func = build_scan_func(scan_args);
        Expr ret = Var::make(func->call_type(), func->name);
        new_funcs[name] = std::move(func);
        return ret;
    }

    Stmt visit(const RecLoop *node) override {
        internal_assert(args.empty()) << Stmt(node);
        args = node->args;
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
        std::string bvh_name = args.front().type.as<BVH_t>()->name;

        Expr callable = get_or_build_callable();

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
            call_args.push_back(args.back());
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
