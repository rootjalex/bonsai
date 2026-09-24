#include "SSA/ReorderLoops.h"

#include "SSA/Analysis.h"
#include "SSA/Definitions.h"
#include "SSA/LoopArithmetic.h"
#include "SSA/Reach.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"
#include "SSA/Simplify.h"
#include "SSA/Storage.h"

#include "Error.h"
#include "Utils.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

// The block whose terminator is the parfor named `index`, or null.
shared_ptr<Block> loop_block(const Function &f, const string &index) {
    shared_ptr<Block> found;
    for (const auto &block : f.blocks) {
        const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (p != nullptr && p->index == index) {
            internal_assert(!found)
                << "Two loops named " << index << " in " << f.blocks.front()->name;
            found = block;
        }
    }
    return found;
}

// Whether an instruction may be computed again somewhere else in the program
// with the same result: pure, and not a read of memory, which a store
// between the two places could change (see SSA/ReorderLoops.h).
bool recomputable(const Instruction &in) {
    return pure(in) && !reads_memory(in);
}

// Whether an instruction may fault or trap when run on values it would not
// otherwise have been run on: an integer division by anything but a
// constant that is not zero. What keeps a value from being moved before the
// loop that computed it, since the loop may have had no iterations.
bool traps(const Instruction &in) {
    if (in.op != Instruction::Op::Div && in.op != Instruction::Op::Mod) {
        return false;
    }
    if (in.type.is_float()) {
        return false;
    }
    const auto divisor = in.operands.size() == 2 ? as_int(in.operands[1]) : std::nullopt;
    return !(divisor.has_value() && *divisor != 0);
}

// What becomes of an instruction of the prologue (SSA/ReorderLoops.h).
enum class Fate {
    Hoist,     // the same on every iteration and safe to run once: before
    Recompute, // pure arithmetic: again wherever it is used
    Expand,    // a value made some other way: once per iteration, in an array
    Effect,    // no value read across the cut: in the prologue loop
};

// Whether the value `of` is read anywhere but in `block`'s own instructions:
// by another block, or by `block`'s terminator, which is where the inner
// loop's body and continuation take what they read of the prologue.
bool read_outside(Function &func, const Block &block, const Instruction *of) {
    bool found = false;
    const auto is_it = [&](shared_ptr<Value> &v) {
        const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
        found = found || (held != nullptr && held->get() == of);
    };
    for (const auto &other : func.blocks) {
        if (other.get() == &block) {
            for_each_value(other->terminator, is_it);
        } else {
            for (const auto &instr : other->instrs) {
                for (auto &operand : instr->operands) {
                    is_it(operand);
                }
            }
            for_each_value(other->terminator, is_it);
        }
    }
    return found;
}

// A block made by this rewrite, owned by `f`.
shared_ptr<Block> fresh_block(const shared_ptr<Function> &f, const string &name) {
    for (const auto &block : f->blocks) {
        internal_assert(block->name != name)
            << "reorder: a block named " << name << " already exists";
    }
    auto block = std::make_shared<Block>();
    block->name = name;
    block->owner = f;
    return block;
}

} // namespace

void reorder(FuncMap &funcs, string func, string inner, string outer) {
    internal_assert(funcs.contains(func))
        << "reorder applied to unknown func: " << func;
    auto f = funcs[func];
    const string what = "reorder(" + inner + ", " + outer + ") on " + func;

    // The two loops, and which is inside which today. `inner` is to be the
    // inner loop, so it is the outer one now: `p` of the pixel nest, `s` the
    // sample loop inside it.
    const shared_ptr<Block> hp = loop_block(*f, inner);
    const shared_ptr<Block> hs = loop_block(*f, outer);
    internal_assert(hp) << what << ": no parfor named " << inner
                        << ". Only a parfor can be reordered -- a sequential "
                           "loop has an order to keep.";
    internal_assert(hs) << what << ": no parfor named " << outer
                        << ". Only a parfor can be reordered -- a sequential "
                           "loop has an order to keep.";
    // The loops as they are: read through references while the prologue is
    // rewritten below, which threads values into the outer body and so adds
    // to its loop's arguments, and copied once that is done.
    auto &P_live = std::get<Terminator::ParFor>(hp->terminator.data);
    auto &S_live = std::get<Terminator::ParFor>(hs->terminator.data);
    if (Cfg(*f, S_live.body.name).contains(hp->name)) {
        internal_error << what << ": " << inner << " is already inside " << outer
                       << ", so the directive asks for the order the loops "
                       << "have. A schedule that changes nothing is a mistake "
                       << "somewhere; write the loops in the order wanted, "
                       << "innermost first, as Halide's reorder does.";
    }
    internal_assert(Cfg(*f, P_live.body.name).contains(hs->name))
        << what << ": " << outer << " is not inside " << inner
        << ", so the two are not a nest to interchange.";

    const BlockMap blocks = make_block_map(*f);
    const shared_ptr<Block> B0 = blocks.at(P_live.body.name);
    // The prologue is one block, ending at the inner loop. A prologue with
    // control flow of its own -- the inner loop under a condition, or after a
    // call, which ends a block in this form -- would have to be replicated
    // as a guard inside the new inner body, and is not guessed at.
    internal_assert(B0.get() == hs.get())
        << what << ": the body of " << inner << " does not run " << outer
        << " directly (" << B0->name << " ends with something other than the "
        << "loop). What comes between them has control flow of its own -- a "
        << "branch, or a call to a function that is not inlined -- which this "
        << "interchange does not replicate into the inner body; write the loop "
        << "to be outer as the whole of the body, or the branch or call as a "
        << "loop of its own before it.";
    // A value the prologue refers to by name alone -- specialize() renames
    // the parameter's uses to the copy's name, and the copy's definition is
    // an instruction of the prologue -- is made to refer to the instruction,
    // so that what is done to the instruction below is done to every use.
    {
        const auto canonical = [&](shared_ptr<Value> &v) {
            const auto *a = std::get_if<Argument>(&v->data);
            if (a == nullptr) {
                return;
            }
            const auto it = B0->lookups.find(a->name);
            if (it != B0->lookups.end() &&
                std::holds_alternative<shared_ptr<Instruction>>(it->second->data)) {
                v = it->second;
            }
        };
        for (const auto &instr : B0->instrs) {
            for (auto &operand : instr->operands) {
                canonical(operand);
            }
        }
        for_each_value(B0->terminator, canonical);
    }
    const shared_ptr<Block> Bs0 = blocks.at(S_live.body.name);
    const shared_ptr<Block> Cs = blocks.at(S_live.cont.name);
    const shared_ptr<Block> CP = blocks.at(P_live.cont.name);

    // A block inside the inner loop, or in the epilogue, may refer to an
    // instruction of the prologue directly rather than through the arguments
    // that thread it in -- a block may use any value that dominates it, and
    // specialize() leaves such uses. After the interchange the prologue no
    // longer dominates either, so every such use is made to go through the
    // arguments first: what the inner loop's body and the continuation take
    // is then the complete account of what they read of the prologue.
    {
        set<const Block *> body_blocks;
        for (const auto &region : {Cfg(*f, Bs0->name), Cfg(*f, Cs->name)}) {
            for (const auto &block : region.blocks()) {
                body_blocks.insert(block.get());
            }
        }
        const auto through_args = [&](const shared_ptr<Block> &X, shared_ptr<Value> &v) {
            const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
            if (held != nullptr && (*held)->owner.lock().get() == B0.get()) {
                v = reach(X, v);
            }
        };
        for (const auto &X : f->blocks) {
            if (!body_blocks.contains(X.get())) {
                continue;
            }
            for (const auto &instr : X->instrs) {
                for (auto &operand : instr->operands) {
                    through_args(X, operand);
                }
            }
            for_each_value(X->terminator, [&](shared_ptr<Value> &v) { through_args(X, v); });
        }
    }
    internal_assert(!B0->args.empty() && !Bs0->args.empty())
        << what << ": a parfor body takes its index first";
    // The index as the body knows it: the loop's name may carry a variant's
    // suffix the argument does not (SSA/Specialize.cpp).
    const Argument p_arg = B0->args.front();
    const Argument s_arg = Bs0->args.front();
    const Type ptype = P_live.start->get_type();
    const shared_ptr<Value> p_start = P_live.start;
    const shared_ptr<Value> p_end = P_live.end;
    const shared_ptr<Value> p_stride = P_live.stride;

    // Two bound loops swap places only if the hardware nests that way round.
    if (P_live.binding.has_value() && S_live.binding.has_value()) {
        internal_assert(may_nest(*S_live.binding, *P_live.binding))
            << what << ": " << inner << " is bound to "
            << to_string(*P_live.binding) << " and " << outer << " to "
            << to_string(*S_live.binding) << ", and a "
            << to_string(*P_live.binding) << " does not run inside a "
            << to_string(*S_live.binding) << ". Reorder before binding, or "
            << "bind the loops the other way round.";
    }

    // The inner loop's range may not come from the outer index: a triangular
    // nest has no rectangle to interchange without new bounds. What else it
    // may be worked out from is settled once the prologue is classified.
    for (const auto &bound : {S_live.start, S_live.end, S_live.stride}) {
        const auto *a = std::get_if<Argument>(&bound->data);
        internal_assert(a == nullptr || a->name != p_arg.name)
            << what << ": the range of " << outer << " is " << inner
            << "'s index -- a triangular nest -- and there is no rectangle to "
            << "interchange without rewriting the bounds, which is not built.";
    }

    // The trip count of the outer loop, in its header, for the arrays that
    // expand a value per iteration; made only when one is needed.
    shared_ptr<Value> trip;
    const auto trip_of_p = [&]() {
        if (!trip) {
            trip = trip_count(*hp, ptype, p_start, p_end, p_stride);
        }
        return trip;
    };
    const auto expansion_array = [&](const string &name, const Type &etype,
                                     bool scratch) {
        auto array = make_alloca(*f, hp, Array_t::make(etype, as_expr(trip_of_p())),
                                 name + "!expanded");
        std::get<shared_ptr<Instruction>>(array->data)->scratch = scratch;
        return array;
    };

    // The region the outer loop runs, and what in it writes what: for each
    // slot of the function (an allocation) and each pointer parameter, the
    // instructions that write it -- a store or an accumulate through an
    // address rooted at it, a push of it, a store of its address as a value,
    // or handing it to a callee's `mut` parameter; a callee's other
    // parameters are read-only, as the language has them. A read of memory
    // nothing in the region writes is the same on every iteration. A writer
    // recorded as null is one that is not a whole store of the slot in the
    // prologue, which is what keeps a slot from being hoisted below.
    Definitions defs(*f, /*lenient=*/true);
    set<string> in_p;
    struct Root {
        const Instruction *slot = nullptr; // an allocation of the function
        string param;                       // or a parameter of it, by name
        bool whole = true;                  // the base itself, not an address into it
    };
    const string entry_name = f->blocks.front()->name;
    const bool explain = std::getenv("BONSAI_EXPLAIN_REORDER") != nullptr;
    // A merge of addresses -- a block argument its predecessors pass
    // different values -- is rooted where every incoming address is, when
    // they agree; each is followed in its predecessor.
    std::function<Root(string, shared_ptr<Value>, int)> root_of_impl =
        [&](string block, shared_ptr<Value> v, int depth) -> Root {
        Root root;
        for (size_t hops = 0; hops < 64; hops++) {
            const Definition d = defs.of(block, v);
            if (!d.value) {
                return {};
            }
            if (const auto *a = std::get_if<Argument>(&d.value->data)) {
                if (d.block == entry_name) {
                    root.param = a->name;
                    return root;
                }
                if (depth > 8) {
                    return {};
                }
                const auto merge = blocks.find(d.block);
                if (merge == blocks.end()) {
                    return {};
                }
                size_t k = merge->second->args.size();
                for (size_t i = 0; i < merge->second->args.size(); i++) {
                    if (merge->second->args[i].name == a->name) {
                        k = i;
                    }
                }
                if (k == merge->second->args.size()) {
                    return {};
                }
                std::optional<Root> agreed;
                for (const auto &weak : merge->second->preds) {
                    const shared_ptr<Block> pred = weak.lock();
                    const shared_ptr<Value> in = pred ? passed_to(*pred, *merge->second, k)
                                                      : nullptr;
                    if (!in) {
                        return {}; // an index, a call's value: no address
                    }
                    const Root r = root_of_impl(pred->name, in, depth + 1);
                    if (r.slot == nullptr && r.param.empty()) {
                        return {};
                    }
                    if (!agreed.has_value()) {
                        agreed = r;
                    } else if (agreed->slot != r.slot || agreed->param != r.param) {
                        return {};
                    }
                    agreed->whole = agreed->whole && r.whole;
                }
                if (!agreed.has_value()) {
                    return {};
                }
                agreed->whole = agreed->whole && root.whole;
                return *agreed;
            }
            const auto *held = std::get_if<shared_ptr<Instruction>>(&d.value->data);
            if (held == nullptr) {
                return {};
            }
            const Instruction &in = **held;
            switch (in.op) {
            case Instruction::Op::Alloca:
            case Instruction::Op::Alloc:
                root.slot = &in;
                return root;
            case Instruction::Op::GEP:
            case Instruction::Op::FieldPtr:
                root.whole = false;
                [[fallthrough]];
            case Instruction::Op::Set:
            case Instruction::Op::Cast:
            case Instruction::Op::Reinterpret: {
                if (in.operands.empty() || !in.owner.lock()) {
                    return {};
                }
                block = in.owner.lock()->name;
                v = in.operands[0];
                continue;
            }
            default:
                return {};
            }
        }
        return {};
    };
    const auto root_of = [&](const string &block, const shared_ptr<Value> &v) {
        return root_of_impl(block, v, 0);
    };
    map<const Instruction *, vector<const Instruction *>> slot_writers;
    set<string> written_params;
    bool unknown_write = false;
    const auto note_write = [&](const string &block, const shared_ptr<Value> &addr,
                                const Instruction *whole_store) {
        const Root r = root_of(block, addr);
        if (r.slot != nullptr) {
            slot_writers[r.slot].push_back(r.whole ? whole_store : nullptr);
        } else if (!r.param.empty()) {
            written_params.insert(r.param);
        } else {
            if (explain && !unknown_write) {
                std::cerr << "; " << what << ": a write in " << block
                          << " goes through an address rooted at nothing known (";
                addr->dump(std::cerr);
                std::cerr << "), so no read of memory in the prologue is taken "
                             "as invariant\n";
            }
            unknown_write = true;
        }
    };
    // Every allocation a value holds an address into -- the value itself, or
    // an aggregate made of such -- which a store of the value lets escape.
    std::function<void(const string &, const shared_ptr<Value> &)> escapes =
        [&](const string &block, const shared_ptr<Value> &v) {
            if (!v->get_type().carries_reference()) {
                return;
            }
            const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
            if (held != nullptr && ((*held)->op == Instruction::Op::MakeStruct ||
                                    (*held)->op == Instruction::Op::Select)) {
                const string home = (*held)->owner.lock() ? (*held)->owner.lock()->name : block;
                for (const auto &operand : (*held)->operands) {
                    escapes(home, operand);
                }
                return;
            }
            note_write(block, v, nullptr);
        };
    {
        const Cfg region(*f, B0->name);
        for (const auto &block : region.blocks()) {
            in_p.insert(block->name);
            for (const auto &instr : block->instrs) {
                switch (instr->op) {
                case Instruction::Op::Store:
                    if (instr->operands.size() >= 2) {
                        note_write(block->name, instr->operands[0],
                                   block == B0 ? instr.get() : nullptr);
                        escapes(block->name, instr->operands[1]);
                    }
                    break;
                case Instruction::Op::AccAdd:
                case Instruction::Op::AccMul:
                case Instruction::Op::AccSub:
                case Instruction::Op::AccMin:
                case Instruction::Op::AccMax:
                case Instruction::Op::AccArgmin:
                case Instruction::Op::AccArgmax:
                case Instruction::Op::AtomicAdd:
                    if (!instr->operands.empty()) {
                        note_write(block->name, instr->operands[0], nullptr);
                    }
                    break;
                case Instruction::Op::Push:
                    for (const auto &operand : instr->operands) {
                        escapes(block->name, operand);
                    }
                    break;
                default:
                    break;
                }
            }
            // Handed to a callee: written when the callee's parameter is
            // `mut`, read otherwise. A jump's argument only carries the
            // address to another block of the region, looked at in turn.
            const auto handed = [&](const Terminator::Jump &call) {
                const auto callee = funcs.find(call.name);
                for (size_t k = 0; k < call.args.size(); k++) {
                    const auto &arg = call.args[k];
                    if (!arg->get_type().carries_reference()) {
                        continue;
                    }
                    bool mutating = true; // an unknown callee may write
                    if (callee != funcs.end() && !callee->second->blocks.empty()) {
                        const auto &params = callee->second->blocks.front()->args;
                        mutating = k >= params.size() || params[k].mutating;
                    }
                    if (mutating) {
                        escapes(block->name, arg);
                    }
                }
            };
            if (const auto *c = std::get_if<Terminator::Call>(&block->terminator.data)) {
                handed(c->call);
            } else if (const auto *m = std::get_if<Terminator::MultiCall>(
                           &block->terminator.data)) {
                handed(m->call);
            }
        }
    }

    // What becomes of each instruction of the prologue, by what it is rather
    // than by whether it varies with the index. Three things are the same on
    // every iteration and safe to run once, before the loop, and are hoisted:
    // a read of memory nothing in the region writes -- a slot of the function
    // (the inliner's result slot for a value computed before the loop, the
    // sample count a match on the sampler settled) or a read-only parameter
    // (the scene); pure arithmetic that cannot trap on such values, on the
    // header's values and on constants; and a `mut` local of the prologue
    // whose every write is a whole store of such a value in the prologue and
    // whose address escapes to nothing that writes -- which is then one slot
    // before the loop rather than one per iteration (specialize()'s copy of
    // the integrator, whose address every kernel is handed). Everything else
    // is recomputed where it is used if it is pure arithmetic, computed once
    // per iteration in the prologue loop and expanded if it makes a value
    // some other way, and left in the prologue loop if it is an effect --
    // LLVM hoists what it can out of the loops the copies end up in, under
    // its own rules about what may run speculatively. To a fixed point, since
    // a hoisted slot makes its loads invariant and those may make more so.
    map<const Instruction *, Fate> fate;
    const auto invariant_operand = [&](const shared_ptr<Value> &v) {
        if (std::holds_alternative<Constant>(v->data)) {
            return true;
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            return a->name != p_arg.name; // the header's values, not the index
        }
        const auto it = fate.find(std::get<shared_ptr<Instruction>>(v->data).get());
        return it == fate.end() || it->second == Fate::Hoist;
    };
    // Whether a read of memory in the prologue reads what was there before
    // the loop, on every iteration.
    const auto invariant_read = [&](const Instruction &in) {
        if (in.operands.empty() || unknown_write) {
            return false;
        }
        // The address itself has to be the same on every iteration: an
        // element's index, an address computed from the base.
        if (!std::all_of(in.operands.begin(), in.operands.end(), invariant_operand)) {
            return false;
        }
        const Root r = root_of(B0->name, in.operands[0]);
        if (r.slot != nullptr) {
            const shared_ptr<Block> home = r.slot->owner.lock();
            const bool of_prologue = home && home.get() == B0.get();
            const bool hoisted =
                of_prologue && fate.contains(r.slot) && fate.at(r.slot) == Fate::Hoist;
            if (!hoisted && (!home || in_p.contains(home->name))) {
                return false; // a slot of the iteration
            }
            const auto writers = slot_writers.find(r.slot);
            if (writers == slot_writers.end()) {
                return true;
            }
            // A hoisted slot's writers are the hoisted whole stores, which
            // precede the read in the prologue's order as they did.
            return hoisted && std::all_of(writers->second.begin(), writers->second.end(),
                                          [&](const Instruction *w) {
                                              return w != nullptr && fate.contains(w) &&
                                                     fate.at(w) == Fate::Hoist;
                                          });
        }
        if (!r.param.empty()) {
            for (const Argument &param : f->blocks.front()->args) {
                if (param.name == r.param) {
                    return !param.mutating &&
                           (param.type.is<Ptr_t>() || param.type.is_reference()) &&
                           !written_params.contains(r.param);
                }
            }
        }
        return false;
    };
    for (bool changed = true; changed;) {
        changed = false;
        const auto assign = [&](const Instruction *in, Fate now) {
            const auto it = fate.find(in);
            if (it == fate.end() || it->second != now) {
                fate[in] = now;
                changed = true;
            }
        };
        for (const auto &instr : B0->instrs) {
            const Instruction *in = instr.get();
            const bool decided = fate.contains(in) && fate.at(in) == Fate::Hoist &&
                                 (in->op == Instruction::Op::Alloca ||
                                  in->op == Instruction::Op::Alloc ||
                                  in->op == Instruction::Op::Store);
            if (decided) {
                continue; // a hoisted slot and its stores, settled below
            }
            if (in->op == Instruction::Op::Alloca || in->op == Instruction::Op::Alloc) {
                assign(in, Fate::Effect);
            } else if (reads_memory(*in) && in->op != Instruction::Op::LoadField) {
                assign(in, invariant_read(*in) ? Fate::Hoist
                                               : (in->name.empty() ? Fate::Effect : Fate::Expand));
            } else if (recomputable(*in)) {
                const bool lift = !traps(*in) &&
                                  std::all_of(in->operands.begin(), in->operands.end(),
                                              invariant_operand);
                assign(in, lift ? Fate::Hoist : Fate::Recompute);
            } else if (!in->name.empty()) {
                assign(in, Fate::Expand);
            } else {
                assign(in, Fate::Effect);
            }
        }
        // A slot of the prologue whose writes are whole stores of invariant
        // values in the prologue, and nothing else, is hoisted with them.
        for (const auto &instr : B0->instrs) {
            const Instruction *in = instr.get();
            if ((in->op != Instruction::Op::Alloca && in->op != Instruction::Op::Alloc) ||
                fate.at(in) == Fate::Hoist || in->type.is_reference() || unknown_write) {
                continue;
            }
            const auto writers = slot_writers.find(in);
            bool hoistable = true;
            if (writers != slot_writers.end()) {
                for (const Instruction *w : writers->second) {
                    const shared_ptr<Block> home = w ? w->owner.lock() : nullptr;
                    if (w == nullptr || w->op != Instruction::Op::Store ||
                        !home || home.get() != B0.get() || w->operands.size() < 2 ||
                        !invariant_operand(w->operands[1])) {
                        hoistable = false;
                        break;
                    }
                }
            }
            if (!hoistable) {
                continue;
            }
            assign(in, Fate::Hoist);
            if (writers != slot_writers.end()) {
                for (const Instruction *w : writers->second) {
                    assign(w, Fate::Hoist);
                }
            }
        }
    }
    if (explain) {
        for (const auto &instr : B0->instrs) {
            const char *name = nullptr;
            switch (fate.at(instr.get())) {
            case Fate::Hoist:
                name = "hoisted";
                break;
            case Fate::Recompute:
                name = "recomputed";
                break;
            case Fate::Expand:
                name = "expanded";
                break;
            case Fate::Effect:
                name = "prologue";
                break;
            }
            std::cerr << "; " << what << ": " << name << ": ";
            instr->dump(std::cerr);
            std::cerr << "\n";
        }
    }

    // A `mut` local of the outer iteration that the inner loop or the
    // epilogue uses, and that was not hoisted, becomes one slot per
    // iteration: its allocation is replaced, in place, by the address of the
    // slot, which is index arithmetic and is recomputed wherever the local
    // is used. A local only the prologue uses stays a local of the prologue
    // loop.
    {
        vector<shared_ptr<Instruction>> before = B0->instrs;
        for (const auto &instr : before) {
            if ((instr->op != Instruction::Op::Alloca &&
                 instr->op != Instruction::Op::Alloc) ||
                fate.at(instr.get()) == Fate::Hoist) {
                continue;
            }
            if (!read_outside(*f, *B0, instr.get())) {
                continue;
            }
            internal_assert(!instr->type.is_reference())
                << what << ": " << instr->name << " is an array local of "
                << inner << "'s iteration that " << outer << " reads. One slot "
                << "per iteration of " << inner << " would be an array of "
                << "arrays, which this form has no handle for; declare it "
                << "inside " << outer << ", or as an array over " << inner
                << "'s range before the nest.";
            const Type etype = instr->type.as<Ptr_t>()->etype;
            auto array = expansion_array(instr->name, etype, /*scratch=*/false);
            // The slot's address, where the allocation was: what the local
            // is from here on.
            size_t at = 0;
            while (at < B0->instrs.size() && B0->instrs[at] != instr) {
                at++;
            }
            internal_assert(at < B0->instrs.size());
            auto index = std::make_shared<Value>(p_arg);
            const auto place = [&](Instruction::Op op,
                                   vector<shared_ptr<Value>> operands,
                                   const Type &type) {
                auto made = std::make_shared<Instruction>(
                    f->get_unique_name(), type, op, std::move(operands), B0);
                B0->instrs.insert(B0->instrs.begin() + at++, made);
                auto value = std::make_shared<Value>(made);
                B0->lookups[made->name] = value;
                fate[made.get()] = Fate::Recompute;
                return value;
            };
            if (!(as_int(p_start).has_value() && *as_int(p_start) == 0)) {
                index = place(Instruction::Op::Sub, {index, reach(B0, p_start)}, ptype);
            }
            if (!(as_int(p_stride).has_value() && *as_int(p_stride) == 1)) {
                index = place(Instruction::Op::Div, {index, reach(B0, p_stride)}, ptype);
            }
            auto slot = place(Instruction::Op::GEP, {reach(B0, array), index},
                              instr->type);
            fate.erase(instr.get());
            replace_uses(*f, instr.get(), slot);
        }
    }

    // The loops as they are now that the prologue's locals are slots: the
    // values the inner loop's body and continuation take, and what the outer
    // loop passes its body.
    const Terminator::ParFor P = P_live;
    const Terminator::ParFor S = S_live;

    // The outer loop's values as its header knows them, by the name the body
    // takes them under: a parfor hands its body the index and then its jump's
    // arguments, one per body argument after the first.
    map<string, shared_ptr<Value>> header_value;
    internal_assert(P.body.args.size() + 1 == B0->args.size())
        << what << ": " << B0->name << " takes " << B0->args.size()
        << " arguments and its loop passes " << P.body.args.size();
    for (size_t k = 1; k < B0->args.size(); k++) {
        header_value[B0->args[k].name] = P.body.args[k - 1];
    }

    // The inner loop's range has to be there before the outer loop starts.
    // A constant or a value the outer loop's header passed its body is; so
    // is what the prologue works out from those and from the slots above by
    // arithmetic that cannot trap, which is hoisted with them. Anything
    // else -- a read that may see the loop's own writes, a division that may
    // be by zero, a draw -- would run once, whether or not the loop has any
    // iterations, what the program ran once per iteration; it is refused
    // rather than speculated.
    {
        set<const Instruction *> for_bounds;
        std::function<void(const shared_ptr<Value> &)> walk =
            [&](const shared_ptr<Value> &v) {
                const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
                if (held == nullptr || !fate.contains(held->get()) ||
                    !for_bounds.insert(held->get()).second) {
                    return;
                }
                for (const auto &operand : (*held)->operands) {
                    walk(operand);
                }
            };
        for (const auto &bound : {S.start, S.end, S.stride}) {
            walk(bound);
        }
        // In the prologue's order, so that an instruction's operands are
        // settled before it is.
        for (const auto &instr : B0->instrs) {
            if (!for_bounds.contains(instr.get()) || fate.at(instr.get()) == Fate::Hoist) {
                continue;
            }
            bool hoistable = fate.at(instr.get()) == Fate::Recompute && !traps(*instr);
            for (const auto &operand : instr->operands) {
                const auto *held = std::get_if<shared_ptr<Instruction>>(&operand->data);
                if (held != nullptr && fate.contains(held->get()) &&
                    fate.at(held->get()) != Fate::Hoist) {
                    hoistable = false;
                }
            }
            internal_assert(hoistable)
                << what << ": the range of " << outer << " is worked out inside "
                << inner << "'s body from " << instr->name << " = "
                << op_name(instr->op) << ", which "
                << (fate.at(instr.get()) == Fate::Recompute
                        ? "may trap"
                        : "may differ from one iteration to the next")
                << ", so it cannot be computed once before " << inner
                << " as the interchanged nest needs. Compute the range before "
                << "the loop.";
            fate[instr.get()] = Fate::Hoist;
        }
    }

    // Which prologue values each side of the cut reads: the inner body
    // through its loop's captures, the epilogue through the continuation's
    // arguments, the prologue loop through its effects and the values it
    // expands. The arithmetic is followed through; an expanded value is a
    // leaf, read back rather than recomputed.
    const auto closure = [&](const vector<shared_ptr<Value>> &roots) {
        set<const Instruction *> found;
        std::function<void(const shared_ptr<Value> &)> walk =
            [&](const shared_ptr<Value> &v) {
                const auto *held = std::get_if<shared_ptr<Instruction>>(&v->data);
                if (held == nullptr) {
                    return;
                }
                const Instruction *in = held->get();
                const auto at = fate.find(in);
                if (at == fate.end() || !found.insert(in).second) {
                    return;
                }
                if (at->second != Fate::Recompute) {
                    return;
                }
                for (const auto &operand : (*held)->operands) {
                    walk(operand);
                }
            };
        for (const auto &v : roots) {
            walk(v);
        }
        return found;
    };
    const set<const Instruction *> needed_body = closure(S.body.args);
    const set<const Instruction *> needed_epi = closure(S.cont.args);
    vector<shared_ptr<Value>> pro_roots;
    for (const auto &instr : B0->instrs) {
        if (fate.at(instr.get()) != Fate::Recompute) {
            for (const auto &operand : instr->operands) {
                pro_roots.push_back(operand);
            }
        }
    }
    const set<const Instruction *> needed_pro = closure(pro_roots);
    // The values expanded: made some other way than arithmetic, and read
    // across the cut.
    set<const Instruction *> expanded;
    for (const auto &instr : B0->instrs) {
        if (fate.at(instr.get()) == Fate::Expand &&
            (needed_body.contains(instr.get()) || needed_epi.contains(instr.get()))) {
            expanded.insert(instr.get());
        }
    }
    // What stays in the prologue loop: the effects, everything made some
    // other way than arithmetic (an unread random draw still steps the
    // generator), and the arithmetic they read.
    vector<shared_ptr<Instruction>> stays;
    for (const auto &instr : B0->instrs) {
        const Fate ft = fate.at(instr.get());
        if (ft == Fate::Effect || ft == Fate::Expand ||
            (ft == Fate::Recompute && needed_pro.contains(instr.get()))) {
            stays.push_back(instr);
        }
    }
    const bool prologue_loop = !stays.empty();
    const bool epilogue_loop =
        !(Cs->instrs.empty() && Cs->args.empty() &&
          std::holds_alternative<Terminator::Yield>(Cs->terminator.data));

    // The hoisted instructions go to the outer loop's header, which runs once
    // before everything the rewrite makes; their operands become the header's
    // values of the same names.
    for (const auto &instr : B0->instrs) {
        if (fate.at(instr.get()) != Fate::Hoist) {
            continue;
        }
        for (auto &operand : instr->operands) {
            if (const auto *a = std::get_if<Argument>(&operand->data)) {
                const auto hv = header_value.find(a->name);
                internal_assert(hv != header_value.end())
                    << what << ": " << instr->name << " reads " << a->name
                    << ", which " << B0->name << " does not take";
                operand = reach(hp, hv->second);
            }
        }
        if (!instr->name.empty()) {
            B0->lookups.erase(instr->name);
        }
        instr->owner = hp;
        hp->instrs.push_back(instr);
        if (!instr->name.empty()) {
            hp->lookups[instr->name] = std::make_shared<Value>(instr);
        }
    }
    // The arrays of the expanded values, and the stores into them at the end
    // of the prologue loop's iteration.
    map<const Instruction *, shared_ptr<Value>> arrays;
    for (const Instruction *in : expanded) {
        arrays[in] = expansion_array(in->name, in->type, /*scratch=*/true);
    }
    // The prologue loop's body is what stays, with the names of what was
    // dropped or moved forgotten.
    {
        set<const Instruction *> staying;
        for (const auto &instr : stays) {
            staying.insert(instr.get());
        }
        for (const auto &instr : B0->instrs) {
            if (!staying.contains(instr.get()) &&
                fate.at(instr.get()) != Fate::Hoist) {
                B0->lookups.erase(instr->name);
            }
        }
        B0->instrs = stays;
    }
    if (prologue_loop) {
        // What stays reads the hoisted values through the loop's edge now.
        for (const auto &instr : B0->instrs) {
            for (auto &operand : instr->operands) {
                const auto *held = std::get_if<shared_ptr<Instruction>>(&operand->data);
                if (held != nullptr && fate.contains(held->get()) &&
                    fate.at(held->get()) == Fate::Hoist) {
                    operand = reach(B0, operand);
                }
            }
        }
        shared_ptr<Value> index;
        for (const auto &instr : stays) {
            if (!expanded.contains(instr.get())) {
                continue;
            }
            if (!index) {
                index = normalized_index(*B0, ptype, std::make_shared<Value>(p_arg),
                                         reach(B0, P.start), reach(B0, P.stride));
            }
            auto slot = B0->make_instruction(
                Ptr_t::make(instr->type), Instruction::Op::GEP,
                {reach(B0, arrays.at(instr.get())), index});
            B0->make_side_effect(Instruction::Op::Store,
                                 {slot, std::make_shared<Value>(instr)});
        }
    }

    // The new blocks. `HS` heads the new outer loop; `LP` is its body and
    // heads the new inner loop; `B1` is the inner body, which recomputes what
    // the old body read of the prologue and jumps into the old inner body;
    // `Y` ends the outer iteration. With an epilogue, `HE` heads the epilogue
    // loop and `E0` is its body, jumping into the old continuation.
    const string stem = B0->name + "!reorder";
    const shared_ptr<Block> HS = prologue_loop ? fresh_block(f, stem + "_" + outer) : hp;
    const shared_ptr<Block> LP = fresh_block(f, stem + "_" + inner);
    const shared_ptr<Block> B1 = fresh_block(f, stem + "_body");
    const shared_ptr<Block> Y = fresh_block(f, stem + "_yield");
    const shared_ptr<Block> HE = epilogue_loop ? fresh_block(f, stem + "_epilogue") : nullptr;
    const shared_ptr<Block> E0 = epilogue_loop ? fresh_block(f, stem + "_epilogue_body") : nullptr;

    auto v_s = LP->add_argument(s_arg);
    auto v_p1 = B1->add_argument(p_arg);
    shared_ptr<Value> v_p_epi = E0 ? E0->add_argument(p_arg) : nullptr;

    // Terminators first, then the blocks join the function, then the values
    // they read are threaded in along the edges just made.
    Y->terminator.data = Terminator::Yield{};
    LP->terminator.data = Terminator::ParFor{P.index, nullptr, nullptr, nullptr,
                                             Terminator::Jump{B1->name, {}},
                                             Terminator::Jump{Y->name, {}},
                                             P.binding};
    HS->terminator.data = Terminator::ParFor{
        S.index, nullptr, nullptr, nullptr, Terminator::Jump{LP->name, {}},
        Terminator::Jump{epilogue_loop ? HE->name : CP->name, {}}, S.binding};
    if (prologue_loop) {
        // In place: threading values into the prologue's body above added to
        // this loop's arguments, which a copy taken earlier would not have.
        auto &prologue = std::get<Terminator::ParFor>(hp->terminator.data);
        prologue.index = P.index + "!prologue";
        prologue.cont = Terminator::Jump{HS->name, {}};
        B0->terminator.data = Terminator::Yield{};
    }
    if (epilogue_loop) {
        HE->terminator.data = Terminator::ParFor{
            P.index + "!epilogue", nullptr, nullptr, nullptr,
            Terminator::Jump{E0->name, {}}, Terminator::Jump{CP->name, {}},
            P.binding};
        E0->terminator.data = Terminator::Jump{Cs->name, {}};
    }
    B1->terminator.data = Terminator::Jump{Bs0->name, {}};

    for (const auto &block : {HS, LP, B1, Y, HE, E0}) {
        if (block && block != hp) {
            f->blocks.push_back(block);
        }
    }
    // The old prologue block, when nothing stays in it, and the old inner
    // continuation, when it only ended the outer body, are unreachable now
    // and go before anything walks the predecessors.
    remove_unreachable_blocks(*f);

    // The bounds: the outer loop's as the header block knows them, the inner
    // loop's threaded from it.
    {
        auto &hs_loop = std::get<Terminator::ParFor>(HS->terminator.data);
        const auto at_hs = [&](const shared_ptr<Value> &bound) {
            if (const auto *a = std::get_if<Argument>(&bound->data)) {
                const auto hv = header_value.find(a->name);
                if (hv != header_value.end()) {
                    return reach(HS, hv->second);
                }
            }
            return reach(HS, bound);
        };
        hs_loop.start = at_hs(S.start);
        hs_loop.end = at_hs(S.end);
        hs_loop.stride = at_hs(S.stride);
    }
    {
        auto &lp_loop = std::get<Terminator::ParFor>(LP->terminator.data);
        lp_loop.start = reach(LP, P.start);
        lp_loop.end = reach(LP, P.end);
        lp_loop.stride = reach(LP, P.stride);
    }
    if (epilogue_loop) {
        auto &he_loop = std::get<Terminator::ParFor>(HE->terminator.data);
        he_loop.start = reach(HE, P.start);
        he_loop.end = reach(HE, P.end);
        he_loop.stride = reach(HE, P.stride);
    }

    // A prologue value as a new block refers to it: the index is the block's
    // own; a header value is threaded in, and so is a hoisted instruction; a
    // recomputed instruction is copied, once per block, its operands first;
    // an expanded one is read back from its array at the block's iteration
    // number. The first copy of a dropped instruction keeps the program's
    // name for it.
    map<const Block *, map<const Instruction *, shared_ptr<Value>>> copies;
    map<const Block *, shared_ptr<Value>> normalized;
    // Two instructions of one name in one function are one value to this
    // form, so a name is reused only once it names nothing anywhere.
    set<string> names_taken;
    for (const auto &block : f->blocks) {
        for (const auto &instr : block->instrs) {
            names_taken.insert(instr->name);
        }
    }
    std::function<shared_ptr<Value>(const shared_ptr<Block> &, const shared_ptr<Value> &,
                                    const shared_ptr<Value> &)>
        carry = [&](const shared_ptr<Block> &T, const shared_ptr<Value> &v,
                    const shared_ptr<Value> &p_here) -> shared_ptr<Value> {
        if (std::holds_alternative<Constant>(v->data)) {
            return v;
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            if (a->name == p_arg.name) {
                return p_here;
            }
            const auto hv = header_value.find(a->name);
            return reach(T, hv != header_value.end() ? hv->second : v);
        }
        const auto &instr = std::get<shared_ptr<Instruction>>(v->data);
        const auto ft = fate.find(instr.get());
        if (ft == fate.end()) {
            return reach(T, v); // not the prologue's: defined above the nest
        }
        auto &made = copies[T.get()];
        if (const auto done = made.find(instr.get()); done != made.end()) {
            return done->second;
        }
        switch (ft->second) {
        case Fate::Hoist:
            return made[instr.get()] = reach(T, v);
        case Fate::Recompute: {
            vector<shared_ptr<Value>> operands;
            for (const auto &operand : instr->operands) {
                operands.push_back(carry(T, operand, p_here));
            }
            const bool keep_name = names_taken.insert(instr->name).second;
            auto copy = std::make_shared<Instruction>(
                keep_name ? instr->name : f->get_unique_name(), instr->type,
                instr->op, std::move(operands), T);
            copy->queried_type = instr->queried_type;
            copy->intrinsic = instr->intrinsic;
            copy->reduce = instr->reduce;
            copy->shuffle = instr->shuffle;
            copy->atomic = instr->atomic;
            copy->compact = instr->compact;
            copy->scratch = instr->scratch;
            T->instrs.push_back(copy);
            auto value = std::make_shared<Value>(copy);
            T->lookups[copy->name] = value;
            return made[instr.get()] = value;
        }
        case Fate::Expand: {
            internal_assert(arrays.contains(instr.get()));
            auto &index = normalized[T.get()];
            if (!index) {
                index = normalized_index(*T, ptype, p_here, reach(T, P.start),
                                         reach(T, P.stride));
            }
            auto slot = T->make_instruction(Ptr_t::make(instr->type),
                                            Instruction::Op::GEP,
                                            {reach(T, arrays.at(instr.get())), index});
            return made[instr.get()] =
                       T->make_instruction(instr->type, Instruction::Op::Load, {slot});
        }
        case Fate::Effect:
            break;
        }
        internal_error << what << ": " << op_name(instr->op) << " is read as a value";
        return nullptr;
    };

    // Into the old inner body: its index, then what its loop passed it.
    {
        vector<shared_ptr<Value>> args = {reach(B1, v_s)};
        for (const auto &capture : S.body.args) {
            args.push_back(carry(B1, capture, v_p1));
        }
        std::get<Terminator::Jump>(B1->terminator.data).args = std::move(args);
    }
    // Into the old continuation, from the epilogue loop's body.
    if (epilogue_loop) {
        vector<shared_ptr<Value>> args;
        for (const auto &passed : S.cont.args) {
            args.push_back(carry(E0, passed, v_p_epi));
        }
        std::get<Terminator::Jump>(E0->terminator.data).args = std::move(args);
    }
    // Onwards after the nest: what the outer loop's header passed on, as the
    // block that now leaves the nest knows it.
    {
        const shared_ptr<Block> leaving = epilogue_loop ? HE : HS;
        auto &loop = std::get<Terminator::ParFor>(leaving->terminator.data);
        vector<shared_ptr<Value>> args;
        for (const auto &passed : P.cont.args) {
            args.push_back(reach(leaving, passed));
        }
        loop.cont.args = std::move(args);
    }

    refresh_preds(*f);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
