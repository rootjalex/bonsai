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
            std::visit(Overloaded{[&](const Split &split) {
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
