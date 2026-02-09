#include "SSA/Convert.h"

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <iostream>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

// Finds first Yield or Return block via DFS.
// Does not recurse into parfors (skips to continuation)
std::shared_ptr<Block> GetTerminatorBlock(const std::shared_ptr<Function> &func,
                                          const std::string &origin) {
    std::map<std::string, std::shared_ptr<Block>> bmap;

    for (const auto &block : func->blocks) {
        bmap[block->name] = block;
    }

    std::map<std::string, std::shared_ptr<Block>> visited;

    std::function<std::shared_ptr<Block>(const std::string &)> dfs =
        [&](const std::string &bname) -> std::shared_ptr<Block> {
        if (visited.contains(bname)) {
            return visited[bname];
        }
        internal_assert(bmap.contains(bname)) << bname;

        auto block = bmap[bname];

        return std::visit(
            overloads{
                [&](const std::monostate &m) -> std::shared_ptr<Block> {
                    func->dump(std::cout);
                    internal_error
                        << "GetTerminatorBlock called on unfinished block: "
                        << bname;
                },
                [&](const Terminator::Jump &j) -> std::shared_ptr<Block> {
                    return dfs(j.name);
                },
                [&](const Terminator::Dispatch &d) -> std::shared_ptr<Block> {
                    internal_assert(!d.targets.empty()) << bname;
                    std::shared_ptr<Block> first = dfs(d.targets[0].name);
                    for (size_t i = 1; i < d.targets.size(); i++) {
                        std::shared_ptr<Block> next = dfs(d.targets[i].name);
                        internal_assert(first == next)
                            << "Diverging CF in terminator of: " << bname;
                    }
                    return first;
                },
                [&](const Terminator::Return &r) -> std::shared_ptr<Block> {
                    return block;
                },
                [&](const Terminator::ParFor &p) -> std::shared_ptr<Block> {
                    return dfs(p.cont.name);
                },
                [&](const Terminator::Yield &y) -> std::shared_ptr<Block> {
                    return block;
                },
                [&](const Terminator::Call &c) -> std::shared_ptr<Block> {
                    return dfs(c.cont.name);
                },
            },
            block->terminator.data);
    };

    return dfs(origin);
}

} // namespace

void split(FuncMap &funcs, std::string func, std::string idx, int factor,
           std::string outer, std::string inner, bool exact) {
    internal_assert(funcs.contains(func)) << func;
    auto f = funcs[func];

    std::vector<std::shared_ptr<Block>> blocks;

    for (auto &block : f->blocks) {
        blocks.push_back(block);
        if (!std::holds_alternative<Terminator::ParFor>(
                block->terminator.data)) {
            
            continue;
        }
        Terminator::ParFor parfor =
            std::get<Terminator::ParFor>(block->terminator.data);
        if (parfor.index != idx) {
            continue;
        }

        internal_assert(exact) << "TODO: handle guards inside split()";

        // parfor i in start:end:stride body(i) cont()
        // ->
        // parfor outer in start:end:factor inner_loop(outer) cont()
        // block inner_loop(o):
        //  parfor inner in o:o+factor:stride body(i) inner_cont()
        // block inner_cont(): yield

        Type itype = parfor.start->get_type();
        auto split_factor = std::make_shared<Value>(Constant{itype, factor});

        // TODO: truly unique name generation?
        std::shared_ptr<Block> outer_yield = std::make_shared<Block>();
        outer_yield->name = parfor.body.name + "_yield_" + outer;
        outer_yield->terminator.data = Terminator::Yield{};
        outer_yield->owner = f;

        std::shared_ptr<Block> inner_loop = std::make_shared<Block>();
        inner_loop->name = parfor.body.name + "_split_" + outer;
        inner_loop->owner = f; // This *MUST* exist before make_instruction
        Argument outer_arg{itype, outer};
        auto v_outer_arg = std::make_shared<Value>(outer_arg);
        auto inner_end = inner_loop->make_instruction(
            itype, Instruction::Op::Add, {v_outer_arg, split_factor});

        internal_assert(std::holds_alternative<Constant>(parfor.stride->data))
            << "TODO: handle non-Constant strides in split mining";

        inner_loop->terminator.data = Terminator::ParFor{
            inner,         v_outer_arg, inner_end,
            parfor.stride, parfor.body, Terminator::Jump{outer_yield->name}};
        inner_loop->args.push_back(outer_arg);
        for (const auto &arg : parfor.body.args) {
            auto as_arg = arg->get_argument();
            if (as_arg.has_value()) {
                inner_loop->args.push_back(*as_arg);
            }
        }

        // TODO: lookups? or are those only necessary in construction?
        inner_loop->preds = {block};
        outer_yield->preds = {inner_loop};

        blocks.push_back(inner_loop);
        blocks.push_back(outer_yield);

        block->terminator.data = Terminator::ParFor{
            outer,
            parfor.start,
            parfor.end,
            split_factor,
            Terminator::Jump{inner_loop->name, parfor.body.args},
            parfor.cont};

    }
    if (blocks.size() == f->blocks.size()) {
        internal_error << "Did not find loop: " << idx << " in function: " << func;
    }
    f->blocks = std::move(blocks);
}

void defer(FuncMap &funcs, std::string func, std::string qname,
           std::string owner, std::string storage, ir::Expr size,
           std::vector<Cursor> cursors) {}

} // namespace ssa
} // namespace ir
} // namespace bonsai
