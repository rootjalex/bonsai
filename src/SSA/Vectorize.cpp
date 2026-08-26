#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
#include "SSA/CloneFunction.h"
#include "SSA/Linearize.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// The lanes of the gang, i.e. the extent of the loop being vectorized. The
// schedule is expected to have split the loop down to the gang width already
// (see the comment on vectorize() below), so this must come out exact.
uint32_t gang_width(const Terminator::ParFor &parfor) {
    auto constant = [](const shared_ptr<Value> &v) -> int64_t {
        const auto *c = std::get_if<Constant>(&v->data);
        internal_assert(c) << "vectorize() needs constant loop bounds";
        return std::visit(overloads{
                              [](int64_t i) { return i; },
                              [](uint64_t u) { return int64_t(u); },
                              [](bool b) -> int64_t {
                                  internal_error << "boolean loop bound";
                                  return 0;
                              },
                              [](double d) -> int64_t {
                                  internal_error << "floating point loop bound";
                                  return 0;
                              },
                              [](const string &s) -> int64_t {
                                  internal_error << "symbolic loop bound: " << s;
                                  return 0;
                              },
                          },
                          c->data);
    };

    const int64_t start = constant(parfor.start);
    const int64_t end = constant(parfor.end);
    const int64_t stride = constant(parfor.stride);
    internal_assert(stride > 0 && end > start)
        << "vectorize() needs a non-empty, forward loop";
    internal_assert((end - start) % stride == 0)
        << "vectorize() needs the loop extent to divide by its stride; "
        << "split() the loop to the gang width first";
    return uint32_t((end - start) / stride);
}

// The vector form of a value that a lane holds one of.
Type widen(const Type &type, uint32_t lanes) {
    internal_assert(!type.is_vector())
        << "TODO: vectorize a value that is already a vector: " << type;
    return Vector_t::make(type, lanes);
}

// The operands of `op` that hold a value each lane has its own copy of, as
// opposed to structural operands: the aggregate an index is applied to, the
// address a store writes through, a lane count, a field number.
//
// Widening only ever touches the first kind. Whether the second kind can be
// varying at all is a per-op question -- a varying address is a
// gather/scatter, a varying field number is not expressible -- and is checked
// where the operand is used.
vector<size_t> value_operands(const Instruction &instr) {
    const size_t n = instr.operands.size();
    switch (instr.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eq:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Lt:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Sub: {
        vector<size_t> all(n);
        for (size_t i = 0; i < n; i++) {
            all[i] = i;
        }
        return all;
    }

    // An index into an aggregate, or through a pointer: the aggregate stays
    // as it is and the index is what goes per-lane.
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::GEP:
        return {1};

    // The address is structural; the value written is per-lane, and so is the
    // execution mask a predicated store carries as a third operand.
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::Store:
        return n > 2 ? vector<size_t>{1, 2} : vector<size_t>{1};

    case Instruction::Op::Eps:
    case Instruction::Op::Inf:
        return {};

    // Already one value per lane, built that way: its base and stride are
    // uniform scalars and must stay that way.
    case Instruction::Op::Ramp:
        return {};

    default:
        instr.dump(std::cerr);
        internal_error << "TODO: vectorize the operation above.";
        return {};
    }
}

// Rewrites every reference to the value named `name` in `block` -- operands,
// jump arguments, and the values a terminator uses directly -- to `value`.
void substitute(Block &block, const string &name,
                const shared_ptr<Value> &value) {
    auto replace = [&](shared_ptr<Value> &v) {
        if (v && std::holds_alternative<Argument>(v->data) &&
            std::get<Argument>(v->data).name == name) {
            v = value;
        }
    };

    for (const auto &instr : block.instrs) {
        for (auto &operand : instr->operands) {
            replace(operand);
        }
    }

    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) {
                       for (auto &a : j.args) {
                           replace(a);
                       }
                   },
                   [&](Terminator::Dispatch &d) {
                       replace(d.cond);
                       for (auto &t : d.targets) {
                           for (auto &a : t.args) {
                               replace(a);
                           }
                       }
                   },
                   [&](Terminator::Return &r) { replace(r.value); },
                   [&](Terminator::ParFor &p) {
                       replace(p.start);
                       replace(p.end);
                       replace(p.stride);
                       for (auto &a : p.body.args) {
                           replace(a);
                       }
                       for (auto &a : p.cont.args) {
                           replace(a);
                       }
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       for (auto &a : c.call.args) {
                           replace(a);
                       }
                       for (auto &a : c.cont.args) {
                           replace(a);
                       }
                   },
               },
               block.terminator.data);

    const auto it = block.lookups.find(name);
    if (it != block.lookups.end()) {
        it->second = value;
    }
}

// Widens a block argument in place. A reference to it is a copy of the same
// Argument rather than a pointer to one, so every copy has to be retyped --
// and after linearization those copies are spread across the region, since
// folding the branches let values be used directly instead of being threaded
// through arguments.
void widen_argument(const BlockMap &blocks, const set<string> &region,
                    const string &owner, const string &name, uint32_t lanes) {
    for (auto &arg : blocks.at(owner)->args) {
        if (arg.name == name) {
            arg.type = widen(arg.type, lanes);
        }
    }

    auto retype = [&](const shared_ptr<Value> &v) {
        if (v && std::holds_alternative<Argument>(v->data)) {
            Argument &a = std::get<Argument>(v->data);
            if (a.name == name && !a.type.is_vector()) {
                a.type = widen(a.type, lanes);
            }
        }
    };

    for (const string &block_name : region) {
        Block &block = *blocks.at(block_name);
        for (const auto &instr : block.instrs) {
            for (const auto &operand : instr->operands) {
                retype(operand);
            }
        }
        std::visit(overloads{
                       [&](std::monostate &) {},
                       [&](Terminator::Jump &t) {
                           for (auto &a : t.args) {
                               retype(a);
                           }
                       },
                       [&](Terminator::Dispatch &t) {
                           retype(t.cond);
                           for (auto &target : t.targets) {
                               for (auto &a : target.args) {
                                   retype(a);
                               }
                           }
                       },
                       [&](Terminator::Return &t) { retype(t.value); },
                       [&](Terminator::ParFor &t) {
                           for (auto &a : t.body.args) {
                               retype(a);
                           }
                           for (auto &a : t.cont.args) {
                               retype(a);
                           }
                       },
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &t) {
                           for (auto &a : t.call.args) {
                               retype(a);
                           }
                           for (auto &a : t.cont.args) {
                               retype(a);
                           }
                       },
                   },
                   block.terminator.data);
        const auto it = block.lookups.find(name);
        if (it != block.lookups.end()) {
            retype(it->second);
        }
    }
}

// Turns every varying value in the region into one value per lane: its type
// becomes a vector, and the uniform values it is combined with are broadcast
// to match. Blocks are visited in execution order, so an instruction's
// operands have already been widened by the time it is reached.
void widen_region(Function &func, const string &entry, const Divergence &div,
                  uint32_t lanes) {
    const BlockMap blocks = make_block_map(func);
    const AdjacencyMap all_succs = compute_successors(func);
    const set<string> region = reachable_from(entry, all_succs);

    AdjacencyMap succs;
    for (const string &name : region) {
        succs[name];
        for (const string &s : all_succs.at(name)) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }

    for (const string &name : reverse_postorder(entry, succs)) {
        Block &block = *blocks.at(name);

        for (const Argument &arg : block.args) {
            if (div.args.count({name, arg.name})) {
                widen_argument(blocks, region, name, arg.name, lanes);
            }
        }

        vector<shared_ptr<Instruction>> widened;
        for (const auto &instr : block.instrs) {
            if (!div.instrs.count(instr.get())) {
                widened.push_back(instr);
                continue;
            }

            // Every lane's copy of a value has to be there for the ones that
            // vary to be combined with it, so uniform operands are broadcast
            // up to the gang width.
            for (const size_t k : value_operands(*instr)) {
                shared_ptr<Value> &operand = instr->operands[k];
                if (operand->get_type().is_vector()) {
                    continue;
                }
                auto count = std::make_shared<Value>(
                    Constant{UInt_t::make(32), uint64_t(lanes)});
                auto bc = std::make_shared<Instruction>(
                    func.get_unique_name(), widen(operand->get_type(), lanes),
                    Instruction::Op::Bc,
                    vector<shared_ptr<Value>>{operand, count},
                    block.shared_from_this());
                widened.push_back(bc);
                operand = std::make_shared<Value>(bc);
            }

            // A store has no result, and an instruction built per-lane in the
            // first place (the ramp) is already the right type; everything
            // else now produces one value per lane.
            if (instr->type.defined() && !instr->name.empty() &&
                !instr->type.is_vector()) {
                instr->type = widen(instr->type, lanes);
            }
            widened.push_back(instr);
        }
        block.instrs = std::move(widened);
    }
}

// A callee, specialized for how a gang calls it: which of its parameters
// arrive as vectors, and whether the call is made under a mask.
struct VariantKey {
    string callee;
    vector<bool> varying;
    bool masked = false;
    uint32_t lanes = 0;

    bool operator<(const VariantKey &o) const {
        return std::tie(callee, varying, masked, lanes) <
               std::tie(o.callee, o.varying, o.masked, o.lanes);
    }
};

// The name a variant is generated under. A function called both ways ends up
// with two of these, and a scalar caller keeps calling the original.
string variant_name(const VariantKey &key) {
    string name = key.callee + "$gang" + std::to_string(key.lanes);
    for (size_t i = 0; i < key.varying.size(); i++) {
        if (key.varying[i]) {
            name += "_v" + std::to_string(i);
        }
    }
    return key.masked ? name + "$masked" : name;
}

// Builds the variant: a copy of the callee whose varying parameters are
// vectors, vectorized the same way a ParFor body is.
//
// A masked variant takes the caller's execution mask as a final parameter and
// runs entirely under it, so its stores write only the lanes the caller had
// enabled. An unmasked variant has no such parameter and no such masking,
// which is why it is worth having both: a call that is always executed should
// not pay for predication (ispc section 5.7 passes the mask the same way, and
// only for functions that need it).
shared_ptr<Function> specialize(FuncMap &funcs, const VariantKey &key,
                                const string &name) {
    const auto original = funcs.find(key.callee);
    internal_assert(original != funcs.end())
        << "Cannot vectorize a call to unknown function: " << key.callee;

    auto variant = clone_function(*original->second);
    Block &entry_block = *variant->blocks.front();

    // The entry block carries the function's name (see FunctionBuilder in
    // SSA/Convert.cpp), and that is the name the generated code is emitted
    // under -- so without renaming it the variant would collide with the
    // function it was specialized from.
    const string old_entry = entry_block.name;
    entry_block.name = name;
    for (const auto &block : variant->blocks) {
        std::visit(overloads{
                       [&](std::monostate &) {},
                       [&](Terminator::Jump &j) {
                           if (j.name == old_entry) {
                               j.name = name;
                           }
                       },
                       [&](Terminator::Dispatch &d) {
                           for (auto &target : d.targets) {
                               if (target.name == old_entry) {
                                   target.name = name;
                               }
                           }
                       },
                       [&](Terminator::Return &) {},
                       [&](Terminator::ParFor &) {},
                       [&](Terminator::Yield &) {},
                       [&](Terminator::Call &c) {
                           if (c.cont.name == old_entry) {
                               c.cont.name = name;
                           }
                       },
                   },
                   block->terminator.data);
    }
    internal_assert(entry_block.args.size() == key.varying.size())
        << "Call to " << key.callee << " passes " << key.varying.size()
        << " arguments to a function taking " << entry_block.args.size();

    // The varying parameters become one value per lane. They are seeded into
    // the divergence analysis by name, the same way a loop index is.
    set<string> varying_names;
    for (size_t i = 0; i < key.varying.size(); i++) {
        if (key.varying[i]) {
            varying_names.insert(entry_block.args[i].name);
        }
    }

    shared_ptr<Value> mask;
    if (key.masked) {
        // Declared scalar and widened below with everything else varying, so
        // that the masks computed inside the function have the same shape.
        const Argument mask_arg{Bool_t::make(), "!mask"};
        entry_block.args.push_back(mask_arg);
        mask = std::make_shared<Value>(mask_arg);
        entry_block.lookups[mask_arg.name] = mask;
        varying_names.insert(mask_arg.name);
    }

    const string entry = entry_block.name;
    promote_allocas(*variant, entry);

    // Linearization folds a region down to a single path, so the function
    // needs a single exit for that path to end at.
    unify_returns(*variant);

    linearize(*variant, entry,
              analyze_divergence(*variant, entry, varying_names), mask);

    const Divergence div = analyze_divergence(*variant, entry, varying_names);
    internal_assert(div.branches.empty())
        << "Linearization left a divergent branch in " << name;
    widen_region(*variant, entry, div, key.lanes);

    // The return type follows whatever the exit ends up carrying, which is a
    // vector if the returned value turned out to be varying.
    for (const auto &block : variant->blocks) {
        if (const auto *ret =
                std::get_if<Terminator::Return>(&block->terminator.data);
            ret != nullptr && ret->value) {
            variant->ret_type = ret->value->get_type();
        }
    }

    funcs[name] = variant;
    return variant;
}

} // namespace

void vectorize(FuncMap &funcs, std::string func, std::string idx) {
    internal_assert(funcs.contains(func))
        << "vectorize applied to unknown func:" << func;
    auto f = funcs[func];

    // The loop to vectorize, and the region its body spans.
    shared_ptr<Block> loop;
    for (const auto &block : f->blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->index == idx) {
            internal_assert(!loop)
                << "Two loops named " << idx << " in " << func;
            loop = block;
        }
    }
    internal_assert(loop) << "Did not find loop: " << idx
                          << " in function: " << func;

    const Terminator::ParFor parfor =
        std::get<Terminator::ParFor>(loop->terminator.data);
    const uint32_t lanes = gang_width(parfor);

    // A mutable local written by one lane is a value that differs between
    // lanes, not memory traffic: promote before anything else looks at the
    // body (see SSA/PromoteAllocas.h).
    promote_allocas(*f, f->blocks[0]->name);

    const string entry = parfor.body.name;

    // Which calls are made under a mask has to be settled before the branches
    // are folded away, since folding them is what makes a conditional call
    // unconditional in the control flow.
    const Divergence before = analyze_divergence(*f, entry, {idx});
    set<string> conditional_calls;
    {
        const BlockMap blocks = make_block_map(f);
        for (const string &name : reachable_from(entry, compute_successors(*f))) {
            if (std::holds_alternative<Terminator::Call>(
                    blocks.at(name)->terminator.data) &&
                before.masked.count(name)) {
                conditional_calls.insert(name);
            }
        }
    }

    // Fold away the branches the lanes disagree about, so that what is left
    // is control flow every lane follows together, with masks standing in for
    // the branches that were folded (see SSA/Linearize.h).
    const BlockMasks masks = linearize(*f, entry, before);

    const BlockMap blocks = make_block_map(f);
    const AdjacencyMap all_succs = compute_successors(*f);
    const set<string> region = reachable_from(entry, all_succs);

    internal_assert(parfor.cont.args.empty())
        << "TODO: thread the continuation arguments of " << idx
        << " through its body";

    AdjacencyMap succs;
    for (const string &name : region) {
        succs[name];
        for (const string &s : all_succs.at(name)) {
            if (region.count(s)) {
                succs[name].push_back(s);
            }
        }
    }

    // The lane indices, which is what the loop index becomes. An index of
    // this shape is what later makes an access to a[i] a dense vector load
    // rather than a gather.
    Block &body = *blocks.at(entry);
    auto ramp = std::make_shared<Instruction>(
        f->get_unique_name(), widen(parfor.start->get_type(), lanes),
        Instruction::Op::Ramp,
        vector<shared_ptr<Value>>{parfor.start, parfor.stride}, body.shared_from_this());
    body.instrs.insert(body.instrs.begin(), ramp);
    auto ramp_value = std::make_shared<Value>(ramp);

    internal_assert(!body.args.empty() && body.args[0].name == idx)
        << "ParFor body " << entry << " does not take " << idx
        << " as its first argument";
    body.args.erase(body.args.begin());

    // Everywhere, not just in the body block: linearization drops the block
    // arguments that used to thread the index onwards, so blocks further
    // along the region refer to the index directly.
    for (const string &name : region) {
        substitute(*blocks.at(name), idx, ramp_value);
    }

    // Re-run the analysis now that the region is linearized and the index is
    // the ramp: the masks and blends linearization introduced have to be
    // classified too, and the index is no longer a block argument to seed on.
    const Divergence div =
        analyze_divergence(*f, entry, {}, {ramp.get()});
    internal_assert(div.branches.empty())
        << "Linearization left a divergent branch in " << *div.branches.begin();

    // Point each call at a variant of its callee that takes the arguments in
    // the shape this gang has them in.
    //
    // A call that the lanes make unconditionally needs no mask: every lane is
    // executing it, so the callee can store freely. A call inside folded
    // control flow does need one, and gets the mask of the block it sits in
    // as an extra argument. Both variants can exist at once, for a function
    // called both ways.
    map<VariantKey, string> variants;
    for (const string &name : region) {
        auto block = blocks.at(name);
        auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call == nullptr) {
            continue;
        }

        VariantKey key;
        key.callee = call->call.name;
        key.lanes = lanes;
        key.masked = conditional_calls.count(name) > 0;
        // Which arguments vary comes from the analysis, not from their types:
        // the call sites are rewritten before the region is widened, so
        // nothing has a vector type yet.
        for (const auto &arg : call->call.args) {
            key.varying.push_back(div.is_varying(name, *arg));
        }

        // Nothing to specialize: every argument arrives as it would from a
        // scalar caller and no mask is needed, so the original function is
        // already the right one to call.
        const bool needs_variant =
            key.masked || std::any_of(key.varying.begin(), key.varying.end(),
                                      [](bool v) { return v; });
        if (!needs_variant) {
            continue;
        }

        const auto cached = variants.find(key);
        string name_of_variant;
        if (cached != variants.end()) {
            name_of_variant = cached->second;
        } else {
            name_of_variant = variant_name(key);
            specialize(funcs, key, name_of_variant);
            variants[key] = name_of_variant;
        }
        call->call.name = name_of_variant;

        if (key.masked) {
            // The mask the call site runs under, which linearization computed
            // when it folded the branch that made the call conditional.
            const auto mask = masks.find(name);
            internal_assert(mask != masks.end())
                << "Call in " << name << " is conditional but its block has "
                << "no mask";
            call->call.args.push_back(mask->second);
        }
    }

    widen_region(*f, entry, div, lanes);

    // With the body vectorized, the loop is gone: it runs exactly once, so
    // its header falls straight into the body and the body's Yield falls
    // through to what followed the loop.
    loop->terminator.data = Terminator::Jump{parfor.body.name, parfor.body.args};

    for (const string &name : region) {
        auto block = blocks.at(name);
        if (!std::holds_alternative<Terminator::Yield>(block->terminator.data)) {
            continue;
        }
        block->terminator.data = Terminator::Jump{parfor.cont.name};

        auto cont = blocks.at(parfor.cont.name);
        std::erase_if(cont->preds, [&](const std::weak_ptr<Block> &p) {
            const auto ptr = p.lock();
            return ptr && ptr->name == loop->name;
        });
        cont->preds.push_back(block);
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
