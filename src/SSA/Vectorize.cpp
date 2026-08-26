#include "SSA/Analysis.h"
#include "SSA/AnalyzeDivergence.h"
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

    // The address is structural; the value written is per-lane.
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::Store:
        return {1};

    case Instruction::Op::Eps:
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

// Widens a block argument in place. Every reference to it -- in this block's
// operands, and in the argument list itself -- is the same Argument by name,
// so the type has to be updated in all of them.
void widen_argument(Block &block, const string &name, uint32_t lanes) {
    for (auto &arg : block.args) {
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
    for (const auto &instr : block.instrs) {
        for (const auto &operand : instr->operands) {
            retype(operand);
        }
    }
    const auto it = block.lookups.find(name);
    if (it != block.lookups.end()) {
        retype(it->second);
    }
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

    const BlockMap blocks = make_block_map(f);
    const AdjacencyMap all_succs = compute_successors(*f);
    const string entry = parfor.body.name;
    const set<string> region = reachable_from(entry, all_succs);

    const Divergence div = analyze_divergence(*f, entry, {idx});

    for (const string &name : region) {
        const Block &block = *blocks.at(name);
        internal_assert(!div.branches.count(name))
            << "TODO: partial linearization of the divergent branch in "
            << name;
        internal_assert(
            !std::holds_alternative<Terminator::Call>(block.terminator.data))
            << "TODO: masked clones for the call in " << name;
    }
    internal_assert(div.masked.empty())
        << "TODO: execution masks; " << *div.masked.begin() << " is masked";
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
    substitute(body, idx, ramp_value);

    // Widen every varying value, in the order the blocks execute so that an
    // instruction's operands are already widened when it is reached.
    for (const string &name : reverse_postorder(entry, succs)) {
        Block &block = *blocks.at(name);

        for (const Argument &arg : block.args) {
            if (div.args.count({name, arg.name})) {
                widen_argument(block, arg.name, lanes);
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
                    f->get_unique_name(), widen(operand->get_type(), lanes),
                    Instruction::Op::Bc,
                    vector<shared_ptr<Value>>{operand, count},
                    block.shared_from_this());
                widened.push_back(bc);
                operand = std::make_shared<Value>(bc);
            }

            // A store has no result; everything else now produces one value
            // per lane.
            if (instr->type.defined() && !instr->name.empty()) {
                instr->type = widen(instr->type, lanes);
            }
            widened.push_back(instr);
        }
        block.instrs = std::move(widened);
    }

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
