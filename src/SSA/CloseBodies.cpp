#include "SSA/CloseBodies.h"

#include "SSA/Analysis.h"

#include "IR/Analysis.h"

#include "Error.h"

#include <functional>
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

// The values a type names: an array whose length is not a constant is
// `f32[n]` for the value `n`, and the code generator finds `n` by name where
// it has to know the length -- allocating the array, copying it whole,
// asking its size -- so a body that does one of those uses `n` as surely as
// one that computes with it, and a kernel made of the body has to be handed
// it. Through pointers and arrays, not into a struct: a layout struct's
// array fields are sized by its own fields, which are not values of the
// function (and are filtered out by name below in any case).
void values_in_type(const Type &type,
                    const std::function<void(const string &, const Type &)> &note) {
    if (const auto *a = type.as<Array_t>()) {
        if (a->size.defined()) {
            for (const TypedVar &var : gather_free_vars(a->size)) {
                note(var.name, var.type);
            }
        }
        values_in_type(a->etype, note);
    } else if (const auto *d = type.as<DynArray_t>()) {
        if (d->capacity.defined()) {
            for (const TypedVar &var : gather_free_vars(d->capacity)) {
                note(var.name, var.type);
            }
        }
        values_in_type(d->etype, note);
    } else if (const auto *p = type.as<Ptr_t>()) {
        values_in_type(p->etype, note);
    }
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
    // Every name the function defines, for telling a value named in a type
    // from a layout's field name, which no block defines.
    set<string> defined;
    for (const auto &block : func.blocks) {
        for (const auto &arg : block->args) {
            defined.insert(arg.name);
        }
        for (const auto &instr : block->instrs) {
            defined.insert(instr->name);
        }
    }
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
        // first met: the values its instructions and terminators read, and
        // the values named by the types the body has to know the length of
        // (see values_in_type) -- what it allocates, what it asks the size
        // of, and an array it stores whole. Not the types of what it merely
        // reads through: a kernel indexing `fs : i32[n]` never evaluates
        // `n`, and handing it `n` would be a parameter nothing reads.
        vector<std::pair<string, Type>> free;
        set<string> seen;
        const auto note_name = [&](const string &name, const Type &type) {
            if (inside.count(name) == 0 && seen.insert(name).second) {
                internal_assert(type.defined())
                    << "the body of the loop over " << p->index << " names `"
                    << name << "` in a type, and the name has no type of its own";
                free.emplace_back(name, type);
            }
        };
        const auto note = [&](const shared_ptr<Value> &v) {
            std::visit(overloads{
                           [](const Constant &) {},
                           [&](const Argument &a) { note_name(a.name, a.type); },
                           [&](const shared_ptr<Instruction> &i) {
                               note_name(i->name, i->type);
                           }},
                       v->data);
        };
        const auto note_in_type = [&](const Type &type) {
            values_in_type(type, [&](const string &name, const Type &t) {
                if (defined.count(name) != 0) {
                    note_name(name, t);
                }
            });
        };
        for (const auto &block : region.blocks()) {
            for (const auto &instr : block->instrs) {
                for (const auto &operand : instr->operands) {
                    note(operand);
                }
                if (instr->op == Instruction::Op::Alloca ||
                    instr->op == Instruction::Op::Alloc) {
                    note_in_type(instr->type);
                }
                if (instr->queried_type.defined()) {
                    note_in_type(instr->queried_type);
                }
                if (instr->op == Instruction::Op::Store &&
                    instr->operands.size() >= 2) {
                    note_in_type(instr->operands[1]->get_type());
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
