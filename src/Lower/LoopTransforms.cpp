#include "Lower/LoopTransforms.h"

#include "Opt/Parallelize.h"

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

using namespace ir;

Stmt split_loop(Stmt stmt, const std::string &loop_idx,
                const std::string &outer, const std::string &inner,
                const Expr &factor, const bool generate_tail, FuncMap &funcs) {
    struct SplitLoop : public Mutator {
        const std::string &loop_idx;
        const std::string &outer;
        const std::string &inner;
        const Expr &factor;
        const bool generate_tail;
        FuncMap &funcs;

        SplitLoop(const std::string &loop_idx, const std::string &outer,
                  const std::string &inner, const Expr &factor,
                  const bool generate_tail, FuncMap &funcs)
            : loop_idx(loop_idx), outer(outer), inner(inner), factor(factor),
              generate_tail(generate_tail), funcs(funcs) {}

        Stmt visit(const ForAll *node) override {
            if (node->index != loop_idx) {
                return Mutator::visit(node);
            }
            internal_assert(!generate_tail)
                << "[unimplemented] split with tail strategy\n";
            internal_assert(is_const_one(node->slice.stride));

            Type index_t = node->slice.end.type();
            Expr zero = make_zero(index_t);
            Expr one = make_one(index_t);
            Expr io = Var::make(index_t, outer);
            Expr ii = Var::make(index_t, inner);
            // io increments by `factor`
            Expr i = io + ii;
            Stmt body = replace({{loop_idx, i}}, node->body);

            // ii in [0, factor)
            body = ForAll::make(inner, ForAll::Slice{zero, factor, one},
                                std::move(body));
            // io in [start, end, stride = factor)
            return ForAll::make(
                outer,
                ForAll::Slice{node->slice.begin, node->slice.end, factor},
                std::move(body));
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                // TODO(ajr): hope to God it's impossible to have self-recursion
                // in these.
                if (var->name.starts_with("_traverse_array")) {
                    funcs[var->name]->body =
                        split_loop(std::move(funcs[var->name]->body), loop_idx,
                                   outer, inner, factor, generate_tail, funcs);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    SplitLoop splitter(loop_idx, outer, inner, factor, generate_tail, funcs);
    return splitter.mutate(std::move(stmt));
}

Stmt rewrite_yieldfroms(Stmt body, WriteLoc count_loc, Expr count_var,
                        WriteLoc queue_loc, Type queue_etype) {
    struct RewriteYieldFroms : public Mutator {
        WriteLoc count_loc;
        Expr count_var;
        WriteLoc queue_loc;
        Type queue_etype;

        RewriteYieldFroms(WriteLoc count_loc, Expr count_var,
                          WriteLoc queue_loc, Type queue_etype)
            : count_loc(std::move(count_loc)), count_var(std::move(count_var)),
              queue_loc(std::move(queue_loc)),
              queue_etype(std::move(queue_etype)) {}

        Stmt visit(const YieldFrom *node) override {
            // TODO(ajr): handle sorting here?
            auto ids = break_tuple(node->value);
            // TODO(ajr): handle if the tuple must be compressed.

            // TODO(ajr): make sure LLVM does a compressed write.
            std::vector<Stmt> stmts;
            stmts.reserve(ids.size() + 1);

            for (size_t i = 0; i < ids.size(); i++) {
                Expr value = ids[i];
                if (!equals(value.type(), queue_etype)) {
                    internal_assert(value.type().is<Tuple_t>() &&
                                    value.is<Build>())
                        << "Loopify cannot build " << queue_etype
                        << " from recursion on : " << value
                        << " of type: " << value.type();
                    const Build *build = value.as<Build>();
                    internal_assert(build);
                    value = Build::make(queue_etype, build->values);
                }
                // Write to queue at current count.
                WriteLoc current_queue_loc = queue_loc;
                current_queue_loc.add_index_access(count_var + i);

                Stmt write_queue =
                    Store::make(current_queue_loc, std::move(value));
                stmts.emplace_back(std::move(write_queue));
            }

            // Increment counter.
            // TODO(ajr): This is not correct with child-bv trees.
            Stmt inc_counter =
                Accumulate::make(count_loc, Accumulate::Add,
                                 make_const(count_var.type(), ids.size()));

            stmts.push_back(std::move(inc_counter));
            return Sequence::make(std::move(stmts));
        }
    };

    RewriteYieldFroms rewriter(std::move(count_loc), std::move(count_var),
                               std::move(queue_loc), std::move(queue_etype));
    return rewriter.mutate(std::move(body));
}

size_t unique_counter = 0;
size_t get_unique_counter() { return unique_counter++; }
std::string unique_struct_name(size_t counter) {
    return "_queue_struct" + std::to_string(counter);
}
std::string unique_count_name(size_t counter) {
    return "_queue_count" + std::to_string(counter);
}
std::string unique_top_name(size_t counter) {
    return "_queue_top" + std::to_string(counter);
}
std::string unique_queue_name(size_t counter) {
    return "_queue" + std::to_string(counter++);
}

// Returns whether this is a tail call to itself.
template <typename T>
bool is_tail_call(const T *node, const Function &f) {
    const auto *v = node->func.template as<Var>();
    internal_assert(v);
    return v->name == f.name;
}

// Verifies this function is appropriate for tail call optimization, and throws
// an error otherwise.
void verify_valid_tail_recursion(const Stmt &body, const Function &function) {
    struct Checker : public Visitor {
        Checker(const Function &function) : function(function) {}

        void visit(const ir::Call *node) override {
            // Calls to this function must only exist when return'ing.
            internal_assert(!is_tail_call(node, function))
                << "unexpected non-return call to same function in tail "
                   "recursion of "
                << function.name;
            Visitor::visit(node);
        }

        void visit(const ir::CallStmt *node) override {
            // Calls to this function must only exist when return'ing.
            internal_assert(!is_tail_call(node, function))
                << "unexpected non-return call to same function in tail "
                   "recursion of "
                << function.name;
            Visitor::visit(node);
        }

        void visit(const ir::Return *node) override {
            const Expr &value = node->value;
            internal_assert(value.defined())
                << "unexpected no return value: " << Stmt(node);
            if (const auto *call = value.as<Call>()) {
                internal_assert(is_tail_call(call, function))
                    << "unexpected call to another function: " << Expr(call)
                    << " in tail recursion of: " << function.name;
            } else {
                node->value.accept(this);
            }
        }

      private:
        const Function &function;
    };

    std::vector<Function::Argument> args = function.args;
    for (int i = 0, e = args.size(); i < e; i++) {
        const Function::Argument &farg = function.args[i];
        // Right now we conservatively assume that tail recursion does
        // not have mutating arguments.
        internal_assert(!farg.mutating)
            << "unexpected mutable argument in tail recursion: " << function;
    }
    Checker checker(function);
    body.accept(&checker);
}

// Rewrites tail recursion to a do-while loop to avoid stack overflow (i.e.,
// tail call optimization). This is done by saving each variable to the stack
// and using these during iteration. For example, func acc(n: i32) {
//   if (n <= 0) { return 1; }
//   foo(n);
//   return acc(n - 1);
// }
//
// ->
//
// func acc(n: i32) {
//   S_n = n;
//   do {
//     if (S_n <= 0) { return 1; }
//     foo(S_n);
//     S_n = S_n - 1;
//     continue;
//   } while (true);
// }
Stmt handle_tail_recursion(Stmt body, const Function &function) {
    // Not all arguments of a function need to be stack allocated when
    // transforming tail recursion, e.g., if we are just passing back the
    // same unmodified argument. This pass determines which ones do.
    struct RequiresStackAllocation : public Visitor {
        RequiresStackAllocation(const Function &function)
            : function(function) {}

        void visit(const ir::Call *node) override {
            const auto *var = node->func.as<Var>();
            if (var == nullptr) {
                return;
            }
            if (var->name != function.name) {
                return;
            }
            std::vector<Expr> args = node->args;
            for (int i = 0, e = args.size(); i < e; i++) {
                const Function::Argument &farg = function.args[i];
                const auto *v = args[i].as<Var>();
                if (v == nullptr || v->name != farg.name) {
                    arguments.insert(farg.name);
                }
            }
        }

        const Function &function;
        std::set<std::string> arguments;
    };

    struct TailRecursionToImperative : public Mutator {
        TailRecursionToImperative(const Function &function,
                                  const std::set<std::string> &requires_stack)
            : function(function), requires_stack(requires_stack) {}
        Stmt visit(const Sequence *node) override {
            // This should only be performed on the top-level sequence.
            if (!entry) {
                return Mutator::visit(node);
            }
            entry = false;
            std::vector<Stmt> stmts;
            // Save state variables locally on the stack.
            for (const ir::Function::Argument &arg : function.args) {
                if (!requires_stack.contains(arg.name)) {
                    continue;
                }
                std::string old_name = arg.name;
                std::string new_name = "S_" + old_name;
                WriteLoc location(new_name, arg.type);
                old_to_new[old_name] = new_name;
                state_variables.push_back(new_name);
                Expr value = Var::make(arg.type, old_name);
                stmts.push_back(Allocate::make(location, std::move(value),
                                               Allocate::Memory::Stack));
            }
            // Place the rest of the body in a DoWhile.
            std::vector<Stmt> loop;
            std::transform(node->stmts.begin(), node->stmts.end(),
                           std::back_inserter(loop),
                           [&](const Stmt &stmt) { return mutate(stmt); });
            Stmt do_while =
                DoWhile::make(Sequence::make(loop), ir::BoolImm::make(true));

            // Then add the loop.
            stmts.push_back(std::move(do_while));
            return Sequence::make(std::move(stmts));
        }

        Expr visit(const Var *node) override {
            // Replace variable references with the new state variables.
            auto it = old_to_new.find(node->name);
            if (it == old_to_new.end()) {
                return Mutator::visit(node);
            }
            return Var::make(node->type, it->second);
        }

        Expr visit(const Call *node) override {
            const auto *var = node->func.as<Var>();
            if (var && var->name == function.name) {
                // TODO(cgyurgyik): Overly restrictive, we can relax this.
                // It just requires making statements inside visits to
                // expressions, which is additional complexity (similar to
                // renaming in DCE).
                internal_error << "Loopify of tail-recursion requires all "
                                  "calls to be returned, received: "
                               << Expr(node);
            }
            return Mutator::visit(node);
        }

        Stmt visit(const Return *node) override {
            if (const auto *call = node->value.as<Call>()) {
                if (const auto *var = call->func.as<Var>();
                    var && var->name == function.name) {
                    std::vector<Stmt> statements;
                    std::vector<Expr> args = call->args;
                    for (int i = 0, e = args.size(); i < e; ++i) {
                        if (!requires_stack.contains(function.args[i].name)) {
                            continue;
                        }
                        WriteLoc loc(state_variables[i], args[i].type());
                        statements.push_back(Store::make(loc, mutate(args[i])));
                    }
                    statements.push_back(Continue::make());
                    return Sequence::make(std::move(statements));
                }
            }
            Expr value = mutate(node->value);
            if (value.same_as(node->value)) {
                return node;
            }
            return Return::make(std::move(value));
        }

      private:
        bool entry = true;
        // The function being transformed.
        const Function &function;
        const std::set<std::string> &requires_stack;
        // A mapping from the old argument name to the new state variable.
        std::unordered_map<std::string, std::string> old_to_new;
        // An ordered list of state variables.
        std::vector<std::string> state_variables;
    };

    RequiresStackAllocation visit(function);
    body.accept(&visit);
    TailRecursionToImperative lower(function, visit.arguments);
    return lower.mutate(std::move(body));
}

Stmt loopify(std::string name, Stmt stmt, std::optional<Expr> queue_size,
             FuncMap &funcs) {
    struct LoopifyImpl : public Mutator {
        std::optional<Expr> queue_size;
        FuncMap &funcs;

        bool in_recloop = false;

        LoopifyImpl(std::optional<Expr> queue_size, FuncMap &funcs)
            : queue_size(std::move(queue_size)), funcs(funcs) {}

        Stmt visit(const RecLoop *node) override {
            const size_t unique_id = get_unique_counter();

            Type queue_etype;
            if (node->args.size() == 1) {
                queue_etype = node->args[0].type;
            } else {
                // TODO: pack?
                constexpr auto P = Struct_t::Attribute::packed;
                // TODO: need to add this to program.types
                queue_etype = ir::Struct_t::make(unique_struct_name(unique_id),
                                                 node->args, {P});
                internal_error
                    << "Need to add packed queue_etype to program.types"
                    << queue_etype;
            }

            std::vector<Stmt> stmts;

            // TODO(ajr): stack-top optimization?
            // TODO(ajr): confirm that LLVM always turns this into phi
            // node
            std::string count_name = unique_count_name(unique_id);
            Type count_type = queue_size->type();
            WriteLoc count_loc(count_name, count_type);
            Expr count_var = Var::make(count_type, count_name);
            stmts.push_back(Allocate::make(count_loc, make_one(count_type),
                                           Allocate::Memory::Stack));

            std::string top_name = unique_top_name(unique_id);
            std::string queue_name = unique_queue_name(unique_id);
            Type queue_type = Array_t::make(queue_etype, *queue_size);
            WriteLoc queue_loc(queue_name, queue_type);
            stmts.push_back(Allocate::make(queue_loc, Allocate::Memory::Stack));
            Expr queue_var = Var::make(queue_type, queue_name);
            WriteLoc queue_top = queue_loc;
            queue_top.add_index_access(make_zero(count_type));
            stmts.push_back(Store::make(queue_top, make_zero(queue_etype)));

            std::vector<Stmt> loop_body;
            loop_body.reserve(node->args.size() + 3);

            // count -= 1
            loop_body.push_back(Accumulate::make(count_loc, Accumulate::Sub,
                                                 make_one(count_type)));

            // Read the top element
            // args = extract from queue[count]
            if (node->args.size() == 1) {
                Expr top_expr = Extract::make(queue_var, count_var);
                WriteLoc arg_loc(node->args[0].name, node->args[0].type);
                loop_body.push_back(
                    LetStmt::make(std::move(arg_loc), std::move(top_expr)));
            } else {
                WriteLoc top_loc(top_name, queue_etype);
                stmts.push_back(LetStmt::make(
                    top_loc, Extract::make(queue_var, count_var)));
                Expr top_expr = Var::make(queue_etype, top_name);
                for (const auto &arg : node->args) {
                    WriteLoc arg_loc(arg.name, arg.type);
                    loop_body.push_back(LetStmt::make(
                        std::move(arg_loc), Access::make(arg.name, top_expr)));
                }
            }

            // Turn YieldFroms into queue writes.
            loop_body.push_back(rewrite_yieldfroms(
                node->body, count_loc, count_var, queue_loc, queue_etype));

            Expr quit_cond = count_var != make_zero(count_type);

            stmts.push_back(
                DoWhile::make(Sequence::make(loop_body), std::move(quit_cond)));

            return Sequence::make(stmts);
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                std::string name = var->name;
                // TODO(ajr): hope to God it's impossible to have
                // self-recursion in these.
                if (name.starts_with("_traverse_tree")) {
                    funcs[name]->body = loopify(
                        name, std::move(funcs[name]->body), queue_size, funcs);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    if (!queue_size.has_value()) {
        auto it = funcs.find(name);
        internal_assert(it != funcs.end()) << name;
        const Function &func = *it->second;
        verify_valid_tail_recursion(stmt, func);
        return handle_tail_recursion(std::move(stmt), func);
    }

    LoopifyImpl rewriter(std::move(queue_size), funcs);
    return rewriter.mutate(std::move(stmt));
}

} // namespace

ir::Program LoopTransforms::run(ir::Program program,
                                const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }

    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    ir::TransformMap transforms =
        std::move(program.schedules[ir::Target::Host].func_transforms);

    if (transforms.empty()) {
        return program;
    }

    // TODO: support names like triangles.data as
    // loop IDs
    auto get_name = [](const Location &loc) {
        internal_assert(loc.names.size() == 1)
            << "[unimplemented] nested location names in transforms.";
        // Loops are always labelled with a "_" in lowering.
        return "_" + loc.names.back();
    };

    for (const auto &[name, ts] : transforms) {
        auto fiter = program.funcs.find(name);
        internal_assert(fiter != program.funcs.end());

        auto &func = fiter->second;

        Stmt body = std::move(func->body);
        for (const auto &t : ts) {
            std::visit(Overloaded{[&](const Loopify &l) {
                                      body =
                                          loopify(name, std::move(body),
                                                  l.queue_size, program.funcs);
                                  },
                                  [&](const Sort &sort) {
                                      // no-op, should have been handled in
                                      // Lower/Sorts.cpp
                                  },
                                  [&](const Split &split) {
                                      std::string i = get_name(split.i);
                                      std::string io = get_name(split.io);
                                      std::string ii = get_name(split.ii);
                                      body = split_loop(std::move(body), i, io,
                                                        ii, split.factor,
                                                        split.generate_tail,
                                                        program.funcs);
                                  },
                                  [&](const Parallelize &par) {
                                      std::string i = get_name(par.i);
                                      body = opt::parallelize_forall(
                                          i, std::move(body), program, options);
                                  }},
                       t);
        }
        func->body = std::move(body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
