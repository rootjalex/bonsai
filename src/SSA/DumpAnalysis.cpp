#include "SSA/DumpAnalysis.h"

#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
#include "SSA/Convert.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Utils.h"

#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

void print_list(std::ostream &os, const std::vector<std::string> &names) {
    os << "[";
    for (size_t i = 0; i < names.size(); i++) {
        if (i != 0) {
            os << ", ";
        }
        os << names[i];
    }
    os << "]";
}

template <typename Blocks>
void print_blocks(std::ostream &os, const Cfg &cfg, const Blocks &blocks) {
    std::vector<std::string> names;
    for (BlockId b : blocks) {
        names.push_back(cfg.name(b));
    }
    print_list(os, names);
}

void print_edges(std::ostream &os, const Cfg &cfg,
                 const std::vector<Edge> &edges) {
    os << "[";
    bool first = true;
    for (const auto &[from, to] : edges) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << cfg.name(from) << "->" << cfg.name(to);
    }
    os << "]";
}

// Dumps every control-flow analysis for the subgraph rooted at `entry`.
//
// Rooting at a ParFor's body block yields exactly that loop body: Yield has no
// successors, so the traversal stops at the end of the body. This is the same
// region `vectorize()` transforms, and within it the body block is (correctly)
// always executed, which is not true when the ParFor is viewed from the
// enclosing function as a two-way branch.
void dump_region(std::ostream &os, const std::string &indent,
                 const Function &func, const std::string &entry) {
    const Cfg cfg(func, entry);
    const DomTree dom = compute_dominator_tree(cfg);
    const DomTree pdom = compute_post_dominator_tree(cfg);
    const LoopForest loops = compute_loop_forest(cfg, dom);
    const ControlDependence cdep = compute_control_dependence(cfg, pdom);
    const BlockIndex index = compute_block_index(cfg, dom, loops);

    os << indent << "rpo: ";
    print_blocks(os, cfg, cfg.rpo);
    os << "\n";

    // Print in block-index order: this is the order partial linearization
    // visits blocks in, and the order whose compactness it depends on.
    for (size_t i = 0; i < index.order.size(); i++) {
        const BlockId b = index.order[i];
        os << indent << "[" << i << "] " << cfg.name(b) << "\n";

        os << indent << "  succs: ";
        print_blocks(os, cfg, cfg.succs[b]);
        os << "\n";
        os << indent << "  preds: ";
        print_blocks(os, cfg, cfg.preds[b]);
        os << "\n";
        os << indent << "  idom: "
           << (dom.contains(b) ? cfg.name(dom.idom[b]) : "<none>") << "\n";
        os << indent << "  ipdom: ";
        if (!pdom.contains(b)) {
            os << "<none>";
        } else if (pdom.idom[b] == pdom.root) {
            os << virtual_exit();
        } else {
            os << cfg.name(pdom.idom[b]);
        }
        os << "\n";
        os << indent << "  cdep: ";
        print_edges(os, cfg, cdep[b]);
        os << "\n";

        // A block with no control dependences executes whenever the region
        // does, so it can never need an execution mask no matter which
        // branches turn out to be divergent.
        os << indent << "  always-executed: " << (cdep[b].empty() ? "yes" : "no")
           << "\n";
    }

    if (loops.empty()) {
        os << indent << "loops: none\n";
        return;
    }
    os << indent << "loops:\n";
    for (const Loop &loop : loops.loops()) {
        os << indent << "  header " << cfg.name(loop.header);
        if (loop.parent != NO_BLOCK) {
            os << " (nested in " << cfg.name(loop.parent) << ")";
        }
        os << "\n";
        os << indent << "    latches: ";
        print_blocks(os, cfg, loop.latches);
        os << "\n";
        os << indent << "    blocks: ";
        print_blocks(os, cfg, loop.blocks);
        os << "\n";
        os << indent << "    exits: ";
        print_edges(os, cfg, loop.exits);
        os << "\n";
    }
}

// Dumps the uniform/varying classification of the region rooted at `entry`,
// per block, so it can be read against the SSA dump above it.
void dump_divergence(std::ostream &os, const std::string &indent,
                     const ssa::Function &func, const std::string &entry,
                     const std::string &index) {
    const Divergence div = analyze_divergence(func, entry, {index});
    const Cfg cfg(func, entry);

    for (BlockId b : cfg.rpo) {
        const Block &block = cfg[b];
        const std::string &name = block.name;
        os << indent << name << ": "
           << (div.masked.count(name) ? "masked" : "unmasked");
        if (div.branches.count(name)) {
            os << ", divergent branch";
        }
        os << "\n";

        std::vector<std::string> varying;
        for (const auto &arg : block.args) {
            if (div.args.count({name, arg.name})) {
                varying.push_back(arg.name);
            }
        }
        for (const auto &instr : block.instrs) {
            if (!div.instrs.count(instr.get())) {
                continue;
            }
            // A side-effecting instruction has no name; print it whole, since
            // a varying store is what will become a scatter.
            if (instr->name.empty()) {
                std::ostringstream text;
                instr->dump(text);
                varying.push_back(text.str());
            } else {
                varying.push_back(instr->name);
            }
        }
        os << indent << "  varying: ";
        print_list(os, varying);
        os << "\n";
    }
}

void dump(std::ostream &os, const std::string &fname, ssa::Function &func) {
    internal_assert(!func.blocks.empty()) << fname << " has no blocks";

    // Promote first: this is what `vectorize()` does, and the divergence of a
    // mutable local is only visible once it is a value rather than memory.
    os << "promoted " << promote_allocas(func, func.blocks[0]->name)
       << " allocation(s) in " << fname << "\n";

    os << "function " << fname << ":\n";
    func.dump(os);
    dump_region(os, "  ", func, func.blocks[0]->name);

    // Then each ParFor body region, which is the unit `vectorize()` works on.
    for (const auto &block : func.blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor == nullptr) {
            continue;
        }
        os << "  parfor " << parfor->index << " body region ("
           << parfor->body.name << "):\n";
        dump_region(os, "    ", func, parfor->body.name);
        os << "    divergence (varying: " << parfor->index << "):\n";
        dump_divergence(os, "      ", func, parfor->body.name, parfor->index);
    }
}

} // namespace

ir::Program DumpSSAAnalysis::run(ir::Program program,
                                 const CompilerOptions &options) const {
    FuncMap fmap;
    for (const auto &[name, func] : program.funcs) {
        fmap[name] = build(func);
    }

    // Loopify before dumping, since it is what puts a back edge in a function
    // at all: without it there is no loop for these analyses to report, and
    // the loop forest of a traversal is exactly what a reader comes here for.
    // Nothing else in the schedule changes the CFG at this level.
    if (const auto it = program.schedules.find(ir::Target::Host);
        it != program.schedules.end()) {
        for (const auto &[name, ts] : it->second.func_transforms) {
            if (!fmap.contains(name)) {
                continue;
            }
            for (const auto &t : ts) {
                if (const auto *l = std::get_if<ir::Loopify>(&t)) {
                    int size = 0;
                    if (l->queue_size.has_value()) {
                        const auto n =
                            get_constant_value<int64_t>(*l->queue_size);
                        internal_assert(n.has_value() && *n > 0)
                            << "loopify(" << *l->queue_size << ") on " << name
                            << " needs a constant, positive stack depth";
                        size = int(*n);
                    }
                    loopify(fmap, name, size);
                }
            }
        }
    }

    // std::map iteration is ordered, so the dump is deterministic.
    for (const auto &[name, func] : fmap) {
        dump(std::cout, name, *func);
    }
    return program;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
