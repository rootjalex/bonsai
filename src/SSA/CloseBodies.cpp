#include "SSA/CloseBodies.h"

#include "SSA/Analysis.h"

#include "Error.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

// Every value a block's terminator reads, its jumps' arguments included: what
// the block hands on is as much a use as what it computes with.
void terminator_operands(const Block &block,
                         vector<shared_ptr<Value>> &operands) {
    const auto jump = [&](const Terminator::Jump &j) {
        for (const auto &a : j.args) {
            operands.push_back(a);
        }
    };
    std::visit(overloads{
                   [](const std::monostate &) {},
                   jump,
                   [&](const Terminator::Dispatch &d) {
                       operands.push_back(d.cond);
                       for (const auto &target : d.targets) {
                           jump(target);
                       }
                   },
                   [&](const Terminator::Return &r) {
                       if (r.value) {
                           operands.push_back(r.value);
                       }
                   },
                   [&](const Terminator::ParFor &p) {
                       operands.push_back(p.start);
                       operands.push_back(p.end);
                       operands.push_back(p.stride);
                       jump(p.body);
                       jump(p.cont);
                   },
                   [](const Terminator::Yield &) {},
                   [&](const Terminator::Call &c) {
                       jump(c.call);
                       jump(c.cont);
                   },
                   [&](const Terminator::MultiCall &c) {
                       jump(c.call);
                       jump(c.cont);
                       for (const auto &vs : c.varying) {
                           for (const auto &v : vs) {
                               operands.push_back(v);
                           }
                       }
                       for (const auto &k : c.keys) {
                           operands.push_back(k);
                       }
               }},
               block.terminator.data);
}

// Block::get_value finds what a block defines through its `lookups`, the
// by-name index of its arguments and instructions. A rewrite that makes an
// instruction directly rather than through make_instruction leaves it out of
// the index, and a lookup for it would then thread the name in from the
// predecessors as though the block did not define it. So the index is
// completed first.
void index_definitions(Function &func) {
    for (const auto &block : func.blocks) {
        for (const auto &arg : block->args) {
            block->lookups.emplace(arg.name, std::make_shared<Value>(arg));
        }
        for (const auto &instr : block->instrs) {
            block->lookups.emplace(instr->name, std::make_shared<Value>(instr));
        }
    }
}

} // namespace

size_t close_parfor_bodies(Function &func) {
    index_definitions(func);
    // Threading a value in walks the body head's predecessors
    // (Block::get_value), so they have to be right, and right means the
    // live graph's: a block a rewrite left behind that nothing reaches any
    // more still names its old targets, and a predecessor list rebuilt from
    // every block would count it -- which made a call continuation of an
    // imported function appear to have two predecessors, and the relooper
    // refuse it. Dropping the dead blocks first is what makes the rebuilt
    // lists true.
    remove_unreachable_blocks(func);

    // The loops, gathered first: threading a value in adds arguments to
    // blocks and to jumps, not blocks to the function, so the list stays
    // good while they are worked through.
    vector<const Terminator::ParFor *> loops;
    for (const auto &block : func.blocks) {
        if (const auto *p =
                std::get_if<Terminator::ParFor>(&block->terminator.data)) {
            loops.push_back(p);
        }
    }

    size_t threaded = 0;
    for (const Terminator::ParFor *p : loops) {
        // The body: everything the body's first block reaches, which ends at
        // the yields and takes in any loop nested inside.
        const Cfg region(func, p->body.name);
        set<string> inside;
        for (const auto &block : region.blocks()) {
            for (const auto &arg : block->args) {
                inside.insert(arg.name);
            }
            for (const auto &instr : block->instrs) {
                inside.insert(instr->name);
            }
        }

        // What the body uses and does not define, each once, in the order
        // first met.
        vector<std::pair<string, Type>> free;
        set<string> seen;
        const auto note = [&](const shared_ptr<Value> &v) {
            std::visit(overloads{
                           [](const Constant &) {},
                           [&](const Argument &a) {
                               if (inside.count(a.name) == 0 &&
                                   seen.insert(a.name).second) {
                                   free.emplace_back(a.name, a.type);
                               }
                           },
                           [&](const shared_ptr<Instruction> &i) {
                               if (inside.count(i->name) == 0 &&
                                   seen.insert(i->name).second) {
                                   free.emplace_back(i->name, i->type);
                               }
                           }},
                       v->data);
        };
        for (const auto &block : region.blocks()) {
            for (const auto &instr : block->instrs) {
                for (const auto &operand : instr->operands) {
                    note(operand);
                }
            }
            vector<shared_ptr<Value>> operands;
            terminator_operands(*block, operands);
            for (const auto &operand : operands) {
                note(operand);
            }
        }

        // Each becomes an argument of the body's first block, which asks its
        // predecessor -- the loop -- for the value, and the loop's edge into
        // the body passes it (Block::get_value, through the ParFor case).
        const shared_ptr<Block> &head = region.block(region.id(p->body.name));
        for (const auto &[name, type] : free) {
            head->get_value(name, type);
            threaded++;
        }
    }
    return threaded;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
