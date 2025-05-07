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
            // TODO(ajr): yieldfrom should be rewritten into yieldfromlist
            Expr value;
            if (!equals(node->value.type(), queue_etype)) {
                internal_assert(node->value.type().is<Tuple_t>() &&
                                node->value.is<Build>())
                    << "Loopify cannot build " << queue_etype
                    << " from recursion on : " << node->value
                    << " of type: " << node->value.type();
                const Build *build = node->value.as<Build>();
                internal_assert(build);
                value = Build::make(queue_etype, build->values);
            } else {
                value = node->value;
            }

            // Write to queue at current count.
            WriteLoc current_queue_loc = queue_loc;
            current_queue_loc.add_index_access(count_var);
            Stmt write_queue = Assign::make(current_queue_loc, std::move(value),
                                            /*mutating=*/true);
            // Increment counter.
            Stmt inc_counter = Accumulate::make(count_loc, Accumulate::Add,
                                                make_one(count_var.type()));
            return Sequence::make(
                {std::move(write_queue), std::move(inc_counter)});
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

Stmt loopify(Stmt stmt, std::optional<Expr> queue_size, FuncMap &funcs) {
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
            stmts.push_back(Assign::make(count_loc, make_one(count_type),
                                         /*mutating=*/false));

            // TODO(ajr): this should be a stack allocation for constant-sized
            // Exprs! For now, we don't free. that's really bad.
            std::string top_name = unique_top_name(unique_id);
            std::string queue_name = unique_queue_name(unique_id);
            Type queue_type = Array_t::make(queue_etype, *queue_size);
            WriteLoc queue_loc(queue_name, queue_type);
            stmts.push_back(Assign::make(queue_loc, Build::make(queue_type),
                                         /*mutating=*/false));
            Expr queue_var = Var::make(queue_type, queue_name);
            WriteLoc queue_top = queue_loc;
            queue_top.add_index_access(make_zero(count_type));
            stmts.push_back(Assign::make(queue_top, make_zero(queue_etype),
                                         /*mutating=*/true));

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
                // TODO(ajr): hope to God it's impossible to have self-recursion
                // in these.
                if (var->name.starts_with("_traverse_tree")) {
                    funcs[var->name]->body = loopify(
                        std::move(funcs[var->name]->body), queue_size, funcs);
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    internal_assert(queue_size.has_value())
        << "[unimplemented] recursion to iteration for tail-recursion: "
        << stmt;

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
                                          loopify(std::move(body), l.queue_size,
                                                  program.funcs);
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
                                          i, std::move(body), program.funcs,
                                          program.types);
                                  }},
                       t);
        }
        func->body = std::move(body);
    }

    return program;
}

} // namespace lower
} // namespace bonsai
