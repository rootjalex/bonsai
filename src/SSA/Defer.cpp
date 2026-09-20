#include "SSA/Defer.h"

#include "SSA/Analysis.h"
#include "SSA/SSA.h"
#include "SSA/Simplify.h"

#include "IR/Expr.h"
#include "IR/Type.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

//===--------------------------------------------------------------------===//
// Values
//===--------------------------------------------------------------------===//

const Type &u32() {
    static const Type t = UInt_t::make(32);
    return t;
}

shared_ptr<Value> constant_u32(uint64_t v) {
    return std::make_shared<Value>(Constant{u32(), v});
}

shared_ptr<Value> constant_bool(bool b) {
    return std::make_shared<Value>(Constant{Bool_t::make(), b});
}

// A constant of the index type `type` -- an unsigned index makes an unsigned
// constant, which is the arm its type is read back out of (see split()).
shared_ptr<Value> index_constant(const Type &type, int64_t v) {
    return type.is_uint()
               ? std::make_shared<Value>(Constant{type, uint64_t(v)})
               : std::make_shared<Value>(Constant{type, v});
}

optional<uint64_t> constant_of(const shared_ptr<Value> &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return *i >= 0 ? optional<uint64_t>(uint64_t(*i)) : std::nullopt;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u;
    }
    return std::nullopt;
}

bool is_undef(const shared_ptr<Value> &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    return c != nullptr && std::holds_alternative<Undefined>(c->data);
}

// `v` as `block` may refer to it. A block refers only to its own
// instructions and arguments in this form; a value defined elsewhere is
// threaded in as an argument along every path from its definition
// (Block::get_value), and this is where that is asked for.
shared_ptr<Value> reach(const shared_ptr<Block> &block,
                        const shared_ptr<Value> &v) {
    return std::visit(
        overloads{
            [&](const Constant &) { return v; },
            [&](const Argument &a) -> shared_ptr<Value> {
                if (const auto it = block->lookups.find(a.name);
                    it != block->lookups.end()) {
                    return it->second;
                }
                return block->get_value(a.name, a.type);
            },
            [&](const shared_ptr<Instruction> &i) -> shared_ptr<Value> {
                if (i->owner.lock().get() == block.get()) {
                    return v;
                }
                return block->get_value(i->name, i->type);
            },
        },
        v->data);
}

vector<shared_ptr<Value>> reach_all(const shared_ptr<Block> &block,
                                    const vector<shared_ptr<Value>> &vs) {
    vector<shared_ptr<Value>> out;
    out.reserve(vs.size());
    for (const auto &v : vs) {
        out.push_back(reach(block, v));
    }
    return out;
}

// An instruction placed at `at` in `block`'s list rather than appended, and
// entered in the block's lookups so that later blocks can ask for it by name.
shared_ptr<Value> insert_instruction(Function &func,
                                     const shared_ptr<Block> &block, size_t at,
                                     Type type, Instruction::Op op,
                                     vector<shared_ptr<Value>> operands) {
    auto instr = std::make_shared<Instruction>(
        func.get_unique_name(), std::move(type), op, std::move(operands), block);
    block->instrs.insert(block->instrs.begin() + at, instr);
    auto value = std::make_shared<Value>(instr);
    block->lookups[instr->name] = value;
    return value;
}

void insert_side_effect(const shared_ptr<Block> &block, size_t at,
                        Instruction::Op op, vector<shared_ptr<Value>> operands,
                        bool atomic = false) {
    auto instr = std::make_shared<Instruction>(op, std::move(operands), block);
    instr->atomic = atomic;
    block->instrs.insert(block->instrs.begin() + at, instr);
}

// Storage for a value of `type`, appended to `block`: what a `mut` local is
// in this form -- an Alloca whose value is the pointer to it, or the array
// itself for an array (see the Allocate visitor in SSA/Convert.cpp). Named
// `name` when one is given, which has to be a name the function does not
// use, so that the generated code says what the storage is.
shared_ptr<Value> make_alloca(Function &func, const shared_ptr<Block> &block,
                              const Type &type, const string &name = "") {
    if (!name.empty()) {
        for (const auto &other : func.blocks) {
            internal_assert(!other->lookups.contains(name))
                << func.blocks.front()->name << " already has a value named "
                << name << ", which a queue's storage would be called";
        }
    }
    auto instr = std::make_shared<Instruction>(
        name.empty() ? func.get_unique_name() : name,
        type.is_reference() ? type : Ptr_t::make(type), Instruction::Op::Alloca,
        vector<shared_ptr<Value>>{}, block);
    block->instrs.push_back(instr);
    auto value = std::make_shared<Value>(instr);
    block->lookups[instr->name] = value;
    return value;
}

// Changes the type of the argument `name` of `block`, wherever the block
// holds a copy of it: its parameter list, its lookups, and every value in it
// that names the argument. An Argument is copied by value into a Value, so
// there is no one place to change it (see widen_argument in Vectorize.cpp).
void retype_argument(Block &block, const string &name, const Type &type) {
    for (Argument &arg : block.args) {
        if (arg.name == name) {
            arg.type = type;
        }
    }
    const auto retype = [&](const shared_ptr<Value> &v) {
        if (v && std::holds_alternative<Argument>(v->data)) {
            Argument &a = std::get<Argument>(v->data);
            if (a.name == name) {
                a.type = type;
            }
        }
    };
    if (const auto it = block.lookups.find(name); it != block.lookups.end()) {
        retype(it->second);
    }
    for (const auto &instr : block.instrs) {
        for (const auto &operand : instr->operands) {
            retype(operand);
        }
    }
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) {
                       for (auto &a : j.args) {
                           retype(a);
                       }
                   },
                   [&](Terminator::Dispatch &d) {
                       retype(d.cond);
                       for (auto &t : d.targets) {
                           for (auto &a : t.args) {
                               retype(a);
                           }
                       }
                   },
                   [&](Terminator::Return &r) { retype(r.value); },
                   [&](Terminator::ParFor &p) {
                       retype(p.start);
                       retype(p.end);
                       retype(p.stride);
                       for (auto &a : p.body.args) {
                           retype(a);
                       }
                       for (auto &a : p.cont.args) {
                           retype(a);
                       }
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       for (auto &a : c.call.args) {
                           retype(a);
                       }
                       for (auto &a : c.cont.args) {
                           retype(a);
                       }
                   },
                   [&](Terminator::MultiCall &c) {
                       for (auto &a : c.call.args) {
                           retype(a);
                       }
                       for (auto &a : c.cont.args) {
                           retype(a);
                       }
                       for (auto &vs : c.varying) {
                           for (auto &a : vs) {
                               retype(a);
                           }
                       }
                   },
               },
               block.terminator.data);
}

//===--------------------------------------------------------------------===//
// Where a value comes from
//===--------------------------------------------------------------------===//

// Where a value a block refers to is actually defined.
//
// A block's arguments are how this form carries a value from where it is
// computed to where it is used, one argument per block on the way, so the
// argument `weight` of some block deep in a loop body says nothing about
// where `weight` was computed. Following the arguments back through the
// jumps that pass them finds out: the definition is the instruction, the
// function parameter, or the block whose predecessors disagree about the
// value -- a merge, or a loop index, or a call's result -- which is then the
// definition itself. Constants are defined nowhere in particular.
struct Definition {
    shared_ptr<Value> value; // the value as its defining block refers to it
    string block;            // its block; empty for a constant
};

bool same_definition(const Definition &a, const Definition &b) {
    return a.value && b.value && same_value(*a.value, *b.value) &&
           a.block == b.block;
}

class Definitions {
  public:
    explicit Definitions(const Function &func)
        : func(func), bmap(make_block_map(func)) {}

    Definition of(const string &block, const shared_ptr<Value> &v) {
        return std::visit(
            overloads{
                [&](const Constant &) { return Definition{v, ""}; },
                [&](const shared_ptr<Instruction> &i) {
                    const auto owner = i->owner.lock();
                    internal_assert(owner) << "instruction " << i->name
                                           << " outlived its block";
                    return Definition{v, owner->name};
                },
                [&](const Argument &a) { return of_argument(block, a); },
            },
            v->data);
    }

  private:
    Definition of_argument(const string &block_name, const Argument &a) {
        const auto key = std::make_pair(block_name, a.name);
        if (const auto it = memo.find(key); it != memo.end()) {
            return it->second;
        }
        const auto biter = bmap.find(block_name);
        internal_assert(biter != bmap.end())
            << block_name << " is not a block of " << func.blocks[0]->name;
        const shared_ptr<Block> &block = biter->second;
        const auto self = std::make_shared<Value>(a);
        const Definition here{self, block_name};

        // The function's parameters are the entry block's arguments.
        if (block.get() == func.blocks.front().get()) {
            return memo[key] = here;
        }
        size_t k = block->args.size();
        for (size_t i = 0; i < block->args.size(); i++) {
            if (block->args[i].name == a.name) {
                k = i;
                break;
            }
        }
        internal_assert(k < block->args.size())
            << a.name << " is referred to in " << block_name
            << " but is neither one of its arguments nor a parameter";

        // While this argument is being resolved it is on the stack, so a
        // predecessor that leads back here is a loop: the argument is
        // carried around the loop and defined by the header, which is here.
        if (!visiting.insert(key).second) {
            return here;
        }

        optional<Definition> found;
        bool merge = false;
        for (const auto &weak : block->preds) {
            const auto pred = weak.lock();
            internal_assert(pred) << "predecessor of " << block_name << " died";
            const shared_ptr<Value> passed = passed_to(*pred, *block, k);
            if (!passed) {
                // Defined by the edge itself: a loop index, a call's result.
                merge = true;
                break;
            }
            const Definition d = of(pred->name, passed);
            if (found.has_value() && !same_definition(*found, d)) {
                merge = true;
                break;
            }
            found = d;
        }
        visiting.erase(key);

        if (merge || !found.has_value()) {
            return memo[key] = here;
        }
        return memo[key] = *found;
    }

    // The value `pred` passes to `block`'s argument `k` along its edge there,
    // or null when the edge defines that argument itself.
    static shared_ptr<Value> passed_to(const Block &pred, const Block &block,
                                       size_t k) {
        shared_ptr<Value> result;
        std::visit(
            overloads{
                [&](const std::monostate &) {},
                [&](const Terminator::Jump &j) {
                    if (j.name == block.name && k < j.args.size()) {
                        result = j.args[k];
                    }
                },
                [&](const Terminator::Dispatch &d) {
                    for (const auto &t : d.targets) {
                        if (t.name == block.name && k < t.args.size()) {
                            result = t.args[k];
                        }
                    }
                },
                [&](const Terminator::Return &) {},
                [&](const Terminator::ParFor &p) {
                    if (p.body.name == block.name) {
                        // The body's first argument is the index, which the
                        // loop defines.
                        if (k >= 1 && k - 1 < p.body.args.size()) {
                            result = p.body.args[k - 1];
                        }
                    } else if (p.cont.name == block.name &&
                               k < p.cont.args.size()) {
                        result = p.cont.args[k];
                    }
                },
                [&](const Terminator::Yield &) {},
                [&](const Terminator::Call &c) {
                    if (c.cont.name != block.name) {
                        return;
                    }
                    // A kept result is the continuation's first argument, and
                    // the call defines it.
                    const size_t offset = c.drop ? 0 : 1;
                    if (k >= offset && k - offset < c.cont.args.size()) {
                        result = c.cont.args[k - offset];
                    }
                },
                [&](const Terminator::MultiCall &c) {
                    if (c.cont.name == block.name && k < c.cont.args.size()) {
                        result = c.cont.args[k];
                    }
                },
            },
            pred.terminator.data);
        return result;
    }

    const Function &func;
    BlockMap bmap;
    map<std::pair<string, string>, Definition> memo;
    set<std::pair<string, string>> visiting;
};

// Whether an operation computes its result from its operands alone, so that
// a copy of it elsewhere, given the same operands, gives the same value. A
// read of memory is not one -- except through a pointer nothing writes,
// which the caller checks for Load and ExtractIdx.
bool recomputable(const Instruction &in) {
    switch (in.op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::Inf:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Load:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Reduce:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Shuffle:
    case Instruction::Op::SizeOf:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
        return true;
    case Instruction::Op::Intrinsic:
        return in.intrinsic != ir::Intrinsic::rand;
    default:
        return false;
    }
}

// The mutable local a definition is, if it is one: an allocation whose value
// is the pointer to it.
const Instruction *as_local(const Definition &d) {
    if (!d.value || d.block.empty()) {
        return nullptr;
    }
    const auto *i = std::get_if<shared_ptr<Instruction>>(&d.value->data);
    if (i == nullptr) {
        return nullptr;
    }
    const Instruction *in = i->get();
    const bool allocation = in->op == Instruction::Op::Alloca ||
                            in->op == Instruction::Op::Alloc;
    return allocation && in->type.is<Ptr_t>() ? in : nullptr;
}

//===--------------------------------------------------------------------===//
// The call graph
//===--------------------------------------------------------------------===//

set<string> callees_of(const Function &f) {
    set<string> out;
    for (const auto &block : f.blocks) {
        if (const auto *call = block->terminator.callee()) {
            out.insert(call->name);
        }
    }
    return out;
}

// Every function reachable from `start` by calls, `start` excluded unless it
// is reached again.
set<string> reachable_functions(const FuncMap &funcs, const string &start) {
    set<string> seen;
    vector<string> work{start};
    bool first = true;
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (!first && !seen.insert(name).second) {
            continue;
        }
        first = false;
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue;
        }
        for (const string &callee : callees_of(*it->second)) {
            work.push_back(callee);
        }
    }
    return seen;
}

// A call one function makes to another: the block whose terminator it is.
struct CallSite {
    string caller;
    shared_ptr<Block> block;
    Terminator::Call *call() const {
        return std::get_if<Terminator::Call>(&block->terminator.data);
    }
};

// Whether the call ending `block` is in tail position: its continuation does
// nothing but return what the call returned, or return nothing after a call
// whose result is dropped. The same test loopify() applies.
bool is_tail_call(const Terminator::Call &call, const BlockMap &bmap) {
    const auto it = bmap.find(call.cont.name);
    internal_assert(it != bmap.end()) << call.cont.name;
    const Block &cont = *it->second;
    const auto *returns = std::get_if<Terminator::Return>(&cont.terminator.data);
    if (returns == nullptr || !cont.instrs.empty() || !call.cont.args.empty()) {
        return false;
    }
    if (call.drop) {
        return cont.args.empty() && returns->value == nullptr;
    }
    return cont.args.size() == 1 && returns->value != nullptr &&
           std::holds_alternative<Argument>(returns->value->data) &&
           std::get<Argument>(returns->value->data).name == cont.args[0].name;
}

//===--------------------------------------------------------------------===//
// Copying a continuation
//===--------------------------------------------------------------------===//

// A copy of the blocks `region` of `func`, renamed with `suffix`, for placing
// the rest of a producer's iteration inside the drain. Instructions get fresh
// names, and so does every argument that carried one of the old names on,
// so that the copy and the original can live in one function; a name the
// region refers to but does not define -- a value from before the call,
// threaded in -- keeps its name, since that is what its definition is called.
// Jumps to blocks outside the region are left pointing where they were, for
// the caller to redirect. Predecessors are not set; the caller rebuilds them.
map<string, shared_ptr<Block>>
clone_region(Function &func, const vector<shared_ptr<Block>> &region,
             const string &suffix) {
    // The names defined inside the region: instructions, and the arguments
    // that carry them.
    map<string, string> renamed;
    map<const Instruction *, shared_ptr<Instruction>> instrs;
    map<string, shared_ptr<Block>> copies;

    for (const auto &block : region) {
        auto copy = std::make_shared<Block>();
        copy->name = block->name + suffix;
        copy->provenance = block->provenance;
        copy->owner = block->owner;
        for (const auto &instr : block->instrs) {
            // A Set is a program's name for a value (`let weight = ...`),
            // which the relooper binds under that name; it keeps it, and the
            // copy's scope keeps it apart from the original's.
            const bool keeps_name =
                instr->name.empty() || instr->op == Instruction::Op::Set;
            const string fresh =
                keeps_name ? instr->name : func.get_unique_name();
            if (!keeps_name) {
                renamed[instr->name] = fresh;
            }
            auto instr_copy = std::make_shared<Instruction>(
                fresh, instr->type, instr->op, vector<shared_ptr<Value>>{},
                copy);
            instr_copy->queried_type = instr->queried_type;
            instr_copy->intrinsic = instr->intrinsic;
            instr_copy->reduce = instr->reduce;
            instr_copy->shuffle = instr->shuffle;
            instr_copy->atomic = instr->atomic;
            instrs[instr.get()] = instr_copy;
            copy->instrs.push_back(instr_copy);
        }
        copies[block->name] = copy;
    }

    const auto rename = [&](const string &name) -> const string & {
        const auto it = renamed.find(name);
        return it == renamed.end() ? name : it->second;
    };
    const auto clone_value =
        [&](const shared_ptr<Value> &v) -> shared_ptr<Value> {
        if (!v) {
            return nullptr;
        }
        return std::visit(
            overloads{
                [&](const Constant &) { return std::make_shared<Value>(*v); },
                [&](const Argument &a) {
                    Argument copy = a;
                    copy.name = rename(a.name);
                    return std::make_shared<Value>(copy);
                },
                [&](const shared_ptr<Instruction> &i) {
                    const auto it = instrs.find(i.get());
                    if (it == instrs.end()) {
                        // Defined outside the region; refers to the same
                        // instruction, which the caller threads in.
                        return std::make_shared<Value>(*v);
                    }
                    return std::make_shared<Value>(it->second);
                },
            },
            v->data);
    };
    const auto clone_values = [&](const vector<shared_ptr<Value>> &vs) {
        vector<shared_ptr<Value>> out;
        out.reserve(vs.size());
        for (const auto &v : vs) {
            out.push_back(clone_value(v));
        }
        return out;
    };
    const auto clone_jump = [&](const Terminator::Jump &j) {
        const auto target = copies.find(j.name);
        return Terminator::Jump{target == copies.end() ? j.name
                                                       : target->second->name,
                                clone_values(j.args)};
    };

    for (const auto &block : region) {
        auto copy = copies.at(block->name);
        for (const Argument &arg : block->args) {
            Argument a = arg;
            a.name = rename(arg.name);
            copy->args.push_back(a);
        }
        for (size_t i = 0; i < block->instrs.size(); i++) {
            copy->instrs[i]->operands =
                clone_values(block->instrs[i]->operands);
        }
        for (const auto &[name, value] : block->lookups) {
            copy->lookups[rename(name)] = clone_value(value);
        }
        copy->terminator.data = std::visit(
            overloads{
                [&](const std::monostate &m) -> decltype(Terminator::data) {
                    return m;
                },
                [&](const Terminator::Jump &j) -> decltype(Terminator::data) {
                    return clone_jump(j);
                },
                [&](const Terminator::Dispatch &d)
                    -> decltype(Terminator::data) {
                    Terminator::Dispatch out;
                    out.cond = clone_value(d.cond);
                    for (const auto &t : d.targets) {
                        out.targets.push_back(clone_jump(t));
                    }
                    return out;
                },
                [&](const Terminator::Return &r) -> decltype(Terminator::data) {
                    return Terminator::Return{clone_value(r.value)};
                },
                [&](const Terminator::ParFor &p) -> decltype(Terminator::data) {
                    Terminator::ParFor out = p;
                    out.start = clone_value(p.start);
                    out.end = clone_value(p.end);
                    out.stride = clone_value(p.stride);
                    out.body = clone_jump(p.body);
                    out.cont = clone_jump(p.cont);
                    return out;
                },
                [&](const Terminator::Yield &y) -> decltype(Terminator::data) {
                    return y;
                },
                [&](const Terminator::Call &c) -> decltype(Terminator::data) {
                    Terminator::Call out;
                    out.call = Terminator::Jump{c.call.name,
                                                clone_values(c.call.args)};
                    out.cont = clone_jump(c.cont);
                    out.drop = c.drop;
                    return out;
                },
                [&](const Terminator::MultiCall &c)
                    -> decltype(Terminator::data) {
                    Terminator::MultiCall out;
                    out.call = Terminator::Jump{c.call.name,
                                                clone_values(c.call.args)};
                    out.cont = clone_jump(c.cont);
                    out.varying_at = c.varying_at;
                    for (const auto &vs : c.varying) {
                        out.varying.push_back(clone_values(vs));
                    }
                    out.keys = clone_values(c.keys);
                    out.drop = c.drop;
                    return out;
                },
            },
            block->terminator.data);
    }
    return copies;
}

// The blocks reachable from `entry` inside `func` along intraprocedural
// edges, in block-name order.
vector<shared_ptr<Block>> region_from(const Function &func,
                                      const string &entry) {
    const Cfg cfg(func, entry);
    vector<shared_ptr<Block>> out;
    for (BlockId b : cfg.rpo) {
        out.push_back(cfg.block(b));
    }
    return out;
}

// Whether `type` is, or holds, an address: a pointer or an array handle.
bool holds_address(const Type &type) {
    if (type.is<Ptr_t>() || type.is_reference()) {
        return true;
    }
    if (const auto *s = type.as<Struct_t>()) {
        return std::any_of(s->fields.begin(), s->fields.end(),
                           [](const TypedVar &f) {
                               return holds_address(f.type);
                           });
    }
    if (const auto *t = type.as<Tuple_t>()) {
        return std::any_of(t->etypes.begin(), t->etypes.end(), holds_address);
    }
    return false;
}

// An SSA value as an expression a type may carry: the size of the entry
// array is one, and it is a run-time value.
Expr as_expr(const shared_ptr<Value> &v) {
    return std::visit(
        overloads{
            [&](const Constant &c) -> Expr {
                if (const auto *u = std::get_if<uint64_t>(&c.data)) {
                    return UIntImm::make(c.type, *u);
                }
                if (const auto *i = std::get_if<int64_t>(&c.data)) {
                    return IntImm::make(c.type, *i);
                }
                internal_error << "a queue's size is not an integer";
                return Expr();
            },
            [&](const Argument &a) -> Expr { return Var::make(a.type, a.name); },
            [&](const shared_ptr<Instruction> &i) -> Expr {
                return Var::make(i->type, i->name);
            },
        },
        v->data);
}

// One field of an entry: a piece of the continuation, and who writes it.
struct Field {
    string name;
    Type type;
    // The callee parameter whose value -- or, for a pointer, whose pointee --
    // the callee stores here when it pushes. None for a field only the frame
    // writes.
    optional<size_t> param;
    bool pointee = false;
    // The owner's mutable local this field relocates, if any: the field holds
    // its contents while the entry waits, and the drain gives the entry a
    // local of its own to run with.
    const Instruction *local = nullptr;
    // The owner's definition of what the field holds, when it is one value
    // throughout -- for an argument and a frame value that are the same value
    // to share the field.
    Definition origin;
    // Whether the frame writes it after a saved return, rather than the
    // callee at the push.
    bool frame_writes = false;
};

} // namespace

//===--------------------------------------------------------------------===//
// defer()
//===--------------------------------------------------------------------===//

vector<Type> defer(FuncMap &funcs, const string &func_name,
                   const string &callee_name, const QueueSpec &queue) {
    const string what =
        func_name + ".defer(" + callee_name + ", " + queue.name + ")";

    internal_assert(funcs.contains(func_name))
        << what << ": " << func_name << " is not a function of the program";
    internal_assert(funcs.contains(callee_name))
        << what << ": " << callee_name << " is not a function of the program";
    internal_assert(funcs.contains(queue.owner))
        << what << ": the queue's owner " << queue.owner
        << " is not a function of the program";
    const shared_ptr<Function> F = funcs.at(func_name);
    const shared_ptr<Function> C = funcs.at(callee_name);
    const shared_ptr<Function> O = funcs.at(queue.owner);

    const auto has_attribute = [](const Function &f,
                                  ir::Function::Attribute a) {
        return std::find(f.attributes.begin(), f.attributes.end(), a) !=
               f.attributes.end();
    };
    for (const auto &[name, f] : funcs) {
        internal_assert(!(has_attribute(*f, ir::Function::Attribute::vectorized) &&
                          (name == func_name || name == callee_name ||
                           f->specialized_from == func_name ||
                           f->specialized_from == callee_name)))
            << what << ": " << name << " is a vectorized gang's copy. A push "
            << "inside a gang has to compact the lanes that push, which is not "
            << "built; write the defer before the vectorize, and vectorize the "
            << "drain loop `" << queue.name << "` instead.";
    }

    //===----------------------------------------------------------------===//
    // The deferred calls, and the chain of calls that reaches them
    //===----------------------------------------------------------------===//

    vector<shared_ptr<Block>> sites;
    for (const auto &block : F->blocks) {
        if (const auto *multi =
                std::get_if<Terminator::MultiCall>(&block->terminator.data)) {
            internal_assert(multi->call.name != callee_name)
                << what << ": the calls to " << callee_name << " in "
                << block->name << " are a run of " << multi->varying.size()
                << " -- a branching recursion, a tree traversal. Only a linear "
                << "deferral is supported: a step may make at most one deferred "
                << "call, so that the number of entries in flight never grows "
                << "and the queue can be sized by the producer count. Put a "
                << "branching recursion on a stack with loopify(N) instead.";
            continue;
        }
        const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
        if (call != nullptr && call->call.name == callee_name) {
            sites.push_back(block);
        }
    }
    internal_assert(!sites.empty())
        << what << ": " << func_name << " makes no call to " << callee_name
        << (func_name == callee_name
                ? ". If the schedule also loopifies it, the recursion is a "
                  "loop by the time defer runs: a recursion is run either "
                  "depth first or breadth first, not both."
                : ".");

    const BlockMap fmap = make_block_map(F);
    for (const auto &site : sites) {
        const auto &call = std::get<Terminator::Call>(site->terminator.data);
        internal_assert(is_tail_call(call, fmap))
            << what << ": the call to " << callee_name << " in " << site->name
            << " is not in tail position -- " << func_name << " does more "
            << (call.drop ? "after the call" : "with its result")
            << " than return. The work after the call is state on the call "
            << "stack that the entry would have to carry, and an entry carries "
            << "only the call's arguments. Not supported yet."
            << (sites.size() > 1
                    ? " (" + func_name + " makes " +
                          std::to_string(sites.size()) + " calls to " +
                          callee_name + "; a step that makes one after "
                          "another is a branching recursion, a tree "
                          "traversal, which only a stack -- loopify(N) -- "
                          "runs without recursing. Only a linear deferral is "
                          "supported.)"
                    : "");
    }

    // The chain: every function on a call path from the owner to `func`.
    const set<string> from_owner = reachable_functions(funcs, queue.owner);
    internal_assert(from_owner.contains(func_name))
        << what << ": " << queue.owner << " never calls " << func_name
        << ", directly or through other functions, so nothing it runs would "
        << "push onto " << queue.name;
    set<string> chain;
    for (const string &g : from_owner) {
        if (g == func_name || reachable_functions(funcs, g).contains(func_name)) {
            chain.insert(g);
        }
    }
    internal_assert(!chain.contains(queue.owner))
        << what << ": " << queue.owner << " is on a call cycle with "
        << func_name << ", so the owner of the queue is also a producer of "
        << "its entries. Not supported.";
    const bool callee_in_chain = chain.contains(callee_name);

    // The functions the program can run: reached from an exported function
    // or from `main`. Inlining leaves the function it copied in behind, still
    // calling what it called, and a call from such a function is not a call.
    set<string> live;
    {
        bool any_root = false;
        for (const auto &[name, f] : funcs) {
            if (has_attribute(*f, ir::Function::Attribute::exported) ||
                name == "main") {
                any_root = true;
                live.insert(name);
                const set<string> below = reachable_functions(funcs, name);
                live.insert(below.begin(), below.end());
            }
        }
        if (!any_root) {
            for (const auto &[name, f] : funcs) {
                live.insert(name);
            }
        }
    }

    // A function nothing can reach that calls into the chain would be left
    // calling a function whose shape has changed. It goes: it could never
    // have run.
    for (auto it = funcs.begin(); it != funcs.end();) {
        const bool dead_caller =
            !live.contains(it->first) &&
            std::any_of(chain.begin(), chain.end(), [&](const string &g) {
                return callees_of(*it->second).contains(g);
            });
        it = dead_caller ? funcs.erase(it) : std::next(it);
    }

    // Every call into the chain, by callee: from the owner (the producers),
    // from chain functions (the chain's own links), and the deferred calls.
    map<string, vector<CallSite>> calls_into;
    for (const auto &[name, f] : funcs) {
        if (!live.contains(name)) {
            continue;
        }
        for (const auto &block : f->blocks) {
            const auto *callee = block->terminator.callee();
            if (callee == nullptr || !chain.contains(callee->name)) {
                continue;
            }
            internal_assert(name == queue.owner || chain.contains(name))
                << what << ": " << callee->name << " is on the chain of calls "
                << "from " << queue.owner << " to " << func_name
                << ", and is also called from " << name
                << ", which is not. The chain's functions change shape -- "
                << "they take the queue and return whether they saved state -- "
                << "and a copy for the other callers is not made yet.";
            internal_assert(
                std::holds_alternative<Terminator::Call>(block->terminator.data))
                << what << ": " << name << " calls " << callee->name
                << " in a run of calls (" << block->name
                << "), which a linear deferral cannot pass a queue through.";
            calls_into[callee->name].push_back(CallSite{name, block});
        }
    }

    // The chain is a chain: the only recursion in it is the deferred call,
    // and a function's own tail recursion, which passes the queue and the
    // flag round like any other tail call.
    for (const string &g : chain) {
        for (const string &h : callees_of(*funcs.at(g))) {
            if (!chain.contains(h) || h == g) {
                continue;
            }
            const bool deferred = g == func_name && h == callee_name;
            internal_assert(deferred ||
                            !reachable_functions(funcs, h).contains(g))
                << what << ": " << g << " and " << h
                << " call each other on the way from " << queue.owner << " to "
                << func_name << ", a recursion other than the one being "
                << "deferred. Not supported.";
        }
    }

    // A chain function's call into the chain is in tail position: the value
    // comes straight back up, with nothing of the frame kept. The producer's
    // own call, in the owner, is the one frame allowed to go on.
    for (const auto &[callee, calls] : calls_into) {
        for (const CallSite &cs : calls) {
            if (cs.caller == queue.owner) {
                continue;
            }
            if (cs.caller == func_name && callee == callee_name) {
                continue; // a deferred call, checked above
            }
            const BlockMap cmap = make_block_map(funcs.at(cs.caller));
            internal_assert(is_tail_call(*cs.call(), cmap))
                << what << ": " << cs.caller << " calls " << callee << " in "
                << cs.block->name << " and does more with the result than "
                << "return it. That is state on the call stack between the "
                << "deferred call and the queue's owner, which the entry does "
                << "not carry. Not supported yet.";
        }
    }

    // What flows back up the chain is one type: the callee's return type.
    const Type ret_type = C->ret_type;
    for (const string &g : chain) {
        internal_assert(equals(funcs.at(g)->ret_type, ret_type))
            << what << ": " << g << " returns " << funcs.at(g)->ret_type
            << " where " << callee_name << " returns " << ret_type
            << ", yet its call down the chain is a tail call. This should "
            << "not typecheck.";
    }
    const bool returns_value = !ret_type.is<Void_t>();

    //===----------------------------------------------------------------===//
    // The producer: the owner's call into the chain, and the loop around it
    //===----------------------------------------------------------------===//

    vector<CallSite> producers;
    for (const auto &[callee, calls] : calls_into) {
        for (const CallSite &cs : calls) {
            if (cs.caller == queue.owner) {
                producers.push_back(cs);
            }
        }
    }
    internal_assert(producers.size() == 1)
        << what << ": " << queue.owner << " calls into the chain from "
        << producers.size() << " places; one producer call per queue is what "
        << "is supported. Each would need its own continuation in the drain.";
    const CallSite producer = producers.front();
    Terminator::Call &pcall = *producer.call();

    const Cfg ocfg(*O);
    const DomTree odom = compute_dominator_tree(ocfg);
    const BlockMap omap = make_block_map(O);

    // The owner's region: its whole body for `root`, or one iteration of the
    // named loop.
    const bool root = queue.loop == "root";
    shared_ptr<Block> owner_loop_block; // the block whose ParFor is the loop
    string region_entry = O->blocks.front()->name;
    if (!root) {
        for (const auto &block : O->blocks) {
            const auto *p =
                std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p != nullptr && p->index == queue.loop) {
                owner_loop_block = block;
                region_entry = p->body.name;
            }
        }
        internal_assert(owner_loop_block)
            << what << ": " << queue.owner << " has no parfor named "
            << queue.loop << " to own the queue " << queue.name
            << ". A queue is owned by a parfor of its function, or by `root`.";
    }
    const Cfg region(*O, region_entry);
    internal_assert(region.contains(producer.block->name))
        << what << ": the call to " << pcall.call.name << " in "
        << producer.block->name << " of " << queue.owner << " is not inside "
        << (root ? "the function" : "the loop " + queue.loop)
        << " that owns " << queue.name;

    // A sequential loop between the owner and the producer would have its
    // iterations reordered: an iteration's deferred work runs after the
    // loop, while the next iteration's own work runs before it. Only a
    // parfor permits that. A loop around the owner's own loop is another
    // matter: each of its iterations allocates and drains its own queue.
    {
        const DomTree rdom = compute_dominator_tree(region);
        const LoopForest rloops = compute_loop_forest(region, rdom);
        internal_assert(rloops.innermost(region.id(producer.block->name)) ==
                        nullptr)
            << what << ": the call to " << pcall.call.name << " in "
            << producer.block->name << " is inside a sequential loop of "
            << queue.owner << ". Deferring it would run the loop's iterations "
            << "out of order, which a `for` does not permit; a `parfor` would.";
    }

    // The producer loop: the one parfor of the region whose body holds the
    // call, if any. Nested parfors would multiply the count; not yet.
    shared_ptr<Block> producer_loop_block;
    for (BlockId b : region.rpo) {
        const shared_ptr<Block> &block = region.block(b);
        const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (p == nullptr) {
            continue;
        }
        const Cfg body(*O, p->body.name);
        if (!body.contains(producer.block->name)) {
            continue;
        }
        internal_assert(!producer_loop_block)
            << what << ": the call to " << pcall.call.name << " in "
            << producer.block->name << " is inside two nested parfors of "
            << queue.owner << " (" << producer_loop_block->name << " and "
            << block->name << "). The queue's size would be the product of "
            << "their counts; only one producer loop is supported yet.";
        producer_loop_block = block;
    }

    // Where the drain goes, and what has to dominate it: the block whose
    // parfor produces the entries, or the block that makes the one call.
    const shared_ptr<Block> point =
        producer_loop_block ? producer_loop_block : producer.block;
    const BlockId point_id = ocfg.id(point->name);
    const auto available = [&](const Definition &d) {
        if (!d.value) {
            return false;
        }
        if (d.block.empty()) {
            return true; // a constant
        }
        const BlockId b = ocfg.find(d.block);
        return b != NO_BLOCK && odom.dominates(b, point_id);
    };

    // The continuation of the producer's call: the rest of its iteration.
    Definitions odefs(*O);
    const shared_ptr<Block> k0_entry = omap.at(pcall.cont.name);
    internal_assert(k0_entry->preds.size() == 1)
        << what << ": the continuation " << k0_entry->name << " of the call in "
        << producer.block->name << " has " << k0_entry->preds.size()
        << " predecessors";
    const size_t result_args = pcall.drop ? 0 : 1;
    internal_assert(k0_entry->args.size() == pcall.cont.args.size() + result_args)
        << k0_entry->name << " takes " << k0_entry->args.size()
        << " arguments but its call passes " << pcall.cont.args.size();
    const vector<shared_ptr<Block>> k0 = region_from(*O, k0_entry->name);
    for (const auto &block : k0) {
        const auto *callee = block->terminator.callee();
        internal_assert(callee == nullptr || !chain.contains(callee->name))
            << what << ": after the call to " << pcall.call.name << " in "
            << producer.block->name << ", " << queue.owner << " calls "
            << callee->name << " again, in " << block->name
            << ". Two producer calls per iteration are not supported.";
        internal_assert(
            !std::holds_alternative<Terminator::ParFor>(block->terminator.data))
            << what << ": the continuation of the call to " << pcall.call.name
            << " runs a parfor (" << block->name << "), which the drain would "
            << "run once per entry. Not supported.";
        if (root) {
            const auto *r = std::get_if<Terminator::Return>(&block->terminator.data);
            internal_assert(r == nullptr || r->value == nullptr)
                << what << ": " << queue.owner << " returns a value after the "
                << "deferred work, in " << block->name << ". The value would "
                << "depend on entries not yet run; a root-owned queue needs "
                << "an owner that returns nothing.";
        }
    }

    //===----------------------------------------------------------------===//
    // Which arguments an entry carries
    //===----------------------------------------------------------------===//

    // A parameter of a chain function is invariant when every call of the
    // function passes it one value of the owner's -- an invariant parameter
    // of the caller, or a value computed from such, down to a definition in
    // the owner; the deferred self-call included, so that a recursion's
    // unchanging arguments count. A fixed point, since one parameter's
    // invariance rests on others'. Whether that one value is *available*
    // where the drain runs is a separate question, asked below.
    map<string, vector<bool>> invariant;
    map<string, vector<Definition>> origin; // the owner's value, if invariant
    map<string, Definitions> defs;
    for (const string &g : chain) {
        invariant[g].assign(funcs.at(g)->blocks.front()->args.size(), true);
        origin[g].resize(funcs.at(g)->blocks.front()->args.size());
        defs.emplace(g, Definitions(*funcs.at(g)));
    }
    if (!callee_in_chain) {
        invariant[callee_name].assign(C->blocks.front()->args.size(), true);
        origin[callee_name].resize(C->blocks.front()->args.size());
        // The deferred calls are calls of the callee whatever the chain holds.
        for (const auto &site : sites) {
            calls_into[callee_name].push_back(CallSite{func_name, site});
        }
    }
    defs.emplace(queue.owner, Definitions(*O));

    // Loop-invariant code motion, for the values the drain needs. A value the
    // owner computes inside the producer's iteration from things available
    // before it -- the scene read through a parameter the owner cannot write,
    // a scale from the sample count -- is the same in every iteration, and
    // the drain can have it too: a copy of the computation goes where the
    // queue is made, before the producer loop, and that copy is what the
    // drain uses. The same for a value a chain function computes from its
    // invariant parameters -- the depth limit unwrapped from the integrator
    // -- copied into the owner over the parameters' origins. Only what is
    // recomputable moves: arithmetic on such operands, and a read through a
    // pointer or an array that is a parameter the owner does not write
    // through. The originals stay where they were; the backend's own motion
    // takes care of the duplicates, and what ends up unread is removed.
    const string oentry = O->blocks.front()->name;
    const auto read_only_parameter = [&](const Definition &d) {
        const auto *arg = std::get_if<Argument>(&d.value->data);
        if (arg == nullptr || d.block != oentry) {
            return false;
        }
        for (const Argument &param : O->blocks.front()->args) {
            if (param.name == arg->name) {
                return !param.mutating &&
                       (param.type.is<Ptr_t>() || param.type.is_reference());
            }
        }
        return false;
    };
    const auto copy_into_point = [&](const Instruction &in,
                                     vector<shared_ptr<Value>> operands) {
        auto copy = std::make_shared<Instruction>(
            O->get_unique_name(), in.type, in.op, std::move(operands), point);
        copy->queried_type = in.queried_type;
        copy->intrinsic = in.intrinsic;
        copy->reduce = in.reduce;
        copy->shuffle = in.shuffle;
        point->instrs.push_back(copy);
        auto value = std::make_shared<Value>(copy);
        point->lookups[copy->name] = value;
        return Definition{value, point->name};
    };
    // A definition in function `g` (the owner, or one on the chain), as the
    // owner can have it before the producer loop, or nothing.
    map<std::pair<string, const Instruction *>, optional<Definition>> moved;
    std::function<optional<Definition>(const string &, const Definition &)>
        materialize = [&](const string &g,
                          const Definition &d) -> optional<Definition> {
        if (!d.value) {
            return std::nullopt;
        }
        if (g == queue.owner && available(d)) {
            return d;
        }
        if (d.block.empty()) {
            return d; // a constant
        }
        if (const auto *arg = std::get_if<Argument>(&d.value->data)) {
            if (g == queue.owner) {
                return std::nullopt; // a value of the iteration
            }
            // A chain function's parameter: its origin, if it has one.
            const Block &gentry = *funcs.at(g)->blocks.front();
            if (d.block != gentry.name) {
                return std::nullopt;
            }
            for (size_t k = 0; k < gentry.args.size(); k++) {
                if (gentry.args[k].name == arg->name) {
                    if (!invariant.at(g)[k] || !origin.at(g)[k].value) {
                        return std::nullopt;
                    }
                    return materialize(queue.owner, origin.at(g)[k]);
                }
            }
            return std::nullopt;
        }
        const auto *held = std::get_if<shared_ptr<Instruction>>(&d.value->data);
        internal_assert(held != nullptr);
        const Instruction *in = held->get();
        const auto key = std::make_pair(g, in);
        if (const auto it = moved.find(key); it != moved.end()) {
            return it->second;
        }
        moved[key] = std::nullopt; // against a cycle, and the answer if not
        if (!recomputable(*in)) {
            return std::nullopt;
        }
        Definitions &gdefs = defs.at(g);
        if (in->op == Instruction::Op::Set) {
            // A name for a value, not a computation: the value is what moves.
            return moved[key] = materialize(g, gdefs.of(d.block, in->operands[0]));
        }
        if (in->op == Instruction::Op::Load ||
            in->op == Instruction::Op::ExtractIdx) {
            const auto base = materialize(g, gdefs.of(d.block, in->operands[0]));
            if (!base.has_value() || !read_only_parameter(*base)) {
                return std::nullopt;
            }
        }
        vector<shared_ptr<Value>> operands;
        for (const auto &operand : in->operands) {
            const auto m = materialize(g, gdefs.of(d.block, operand));
            if (!m.has_value()) {
                return std::nullopt;
            }
            operands.push_back(reach(point, m->value));
        }
        return moved[key] = copy_into_point(*in, std::move(operands));
    };
    // An owner-level definition, moved before the producer loop when it can
    // be and left as it is otherwise.
    const auto settle = [&](const Definition &d) {
        const auto m = materialize(queue.owner, d);
        return m.has_value() ? *m : d;
    };

    for (bool changed = true; changed;) {
        changed = false;
        for (auto &[g, inv] : invariant) {
            for (size_t j = 0; j < inv.size(); j++) {
                if (!inv[j]) {
                    continue;
                }
                bool ok = true;
                optional<Definition> agreed;
                const auto consider = [&](const Definition &d) {
                    if (!agreed.has_value()) {
                        agreed = d;
                    } else if (!same_definition(*agreed, d)) {
                        ok = false;
                    }
                };
                for (const CallSite &cs : calls_into[g]) {
                    const Terminator::Call &call = *cs.call();
                    internal_assert(j < call.call.args.size())
                        << cs.caller << " passes " << call.call.args.size()
                        << " arguments to " << g << ", which takes "
                        << inv.size();
                    const Definition d =
                        defs.at(cs.caller).of(cs.block->name, call.call.args[j]);
                    if (cs.caller == queue.owner) {
                        consider(settle(d));
                        continue;
                    }
                    // From a chain function: one of its own invariant
                    // parameters, whose origin is then this one's too, or a
                    // value computed from such, which the owner gets a copy
                    // of.
                    const Block &centry = *funcs.at(cs.caller)->blocks.front();
                    const auto *arg = std::get_if<Argument>(&d.value->data);
                    if (arg != nullptr && d.block == centry.name) {
                        size_t k = centry.args.size();
                        for (size_t i = 0; i < centry.args.size(); i++) {
                            if (centry.args[i].name == arg->name) {
                                k = i;
                            }
                        }
                        internal_assert(k < centry.args.size()) << arg->name;
                        if (!invariant.at(cs.caller)[k]) {
                            ok = false;
                            break;
                        }
                        if (origin.at(cs.caller)[k].value) {
                            consider(settle(origin.at(cs.caller)[k]));
                        }
                        continue;
                    }
                    const auto m = materialize(cs.caller, d);
                    if (!m.has_value()) {
                        ok = false;
                        break;
                    }
                    consider(*m);
                }
                if (!ok) {
                    inv[j] = false;
                    origin[g][j] = Definition{};
                    changed = true;
                } else if (agreed.has_value() &&
                           !same_definition(origin[g][j], *agreed)) {
                    origin[g][j] = *agreed;
                    changed = true;
                }
            }
        }
    }

    // The entry's fields. First the callee's parameters, in order, as the
    // program wrote them (the chain gains one below): left out when the
    // drain has the value in scope; the contents of a mutable local of the
    // producer's iteration, when that is what a pointer argument points at;
    // the value itself otherwise.
    vector<Field> fields;
    const auto field_named = [&](string name) {
        for (bool clash = true; clash;) {
            clash = false;
            for (const Field &f : fields) {
                if (f.name == name) {
                    name += "_";
                    clash = true;
                }
            }
        }
        return name;
    };
    enum class Where { Elided, Stored, Relocated };
    struct ParamPlan {
        Where where = Where::Elided;
        size_t field = 0;
    };
    const Block &centry = *C->blocks.front();
    const size_t nparams = centry.args.size();
    vector<ParamPlan> params(nparams);
    for (size_t j = 0; j < nparams; j++) {
        const Argument &param = centry.args[j];
        const bool inv = invariant.at(callee_name)[j] &&
                         origin.at(callee_name)[j].value != nullptr;
        Definition &org = origin.at(callee_name)[j];
        if (inv) {
            org = settle(org);
        }
        if (inv && available(org)) {
            params[j] = {Where::Elided, 0};
            continue;
        }
        if (const Instruction *local = inv ? as_local(org) : nullptr) {
            Field f;
            f.name = field_named(param.name);
            f.type = local->type.as<Ptr_t>()->etype;
            f.param = j;
            f.pointee = true;
            f.local = local;
            f.origin = org;
            params[j] = {Where::Relocated, fields.size()};
            fields.push_back(std::move(f));
            continue;
        }
        internal_assert(!holds_address(param.type))
            << what << ": the argument " << param.name << " of " << callee_name
            << " is an address (" << param.type << ") that is neither the same "
            << "at every call nor a mutable local of the producer's iteration, "
            << "so it would be stored in the queue -- and might point into a "
            << "frame that has returned by the time the entry runs. Not "
            << "supported.";
        Field f;
        f.name = field_named(param.name);
        f.type = param.type;
        f.param = j;
        if (inv) {
            f.origin = org;
        }
        params[j] = {Where::Stored, fields.size()};
        fields.push_back(std::move(f));
    }

    // Then what the rest of the producer's iteration -- its continuation,
    // which the drain runs for every entry that finishes -- needs of the
    // iteration: each value the continuation takes from before the call is
    // available where the drain runs, or a mutable local of the iteration
    // whose contents the entry carries, or a value of the iteration that
    // the frame writes into the entry once it learns the call was saved.
    // The user's rule: "if the return says we saved state, save this data
    // in the queue location, otherwise act as normal."
    enum class From { Available, Field, Relocated };
    struct CarriedPlan {
        From from = From::Available;
        size_t field = 0;
        Definition def;
    };
    vector<CarriedPlan> carried;
    for (size_t i = 0; i < pcall.cont.args.size(); i++) {
        const Definition d =
            settle(odefs.of(producer.block->name, pcall.cont.args[i]));
        const Argument &arg = k0_entry->args[i + result_args];
        if (available(d)) {
            carried.push_back({From::Available, 0, d});
            continue;
        }
        if (const Instruction *local = as_local(d)) {
            size_t f = fields.size();
            for (size_t k = 0; k < fields.size(); k++) {
                if (fields[k].local == local) {
                    f = k; // the callee is handed it too, and stores it
                }
            }
            if (f == fields.size()) {
                Field field;
                field.name = field_named(arg.name);
                field.type = local->type.as<Ptr_t>()->etype;
                field.pointee = true;
                field.local = local;
                field.origin = d;
                field.frame_writes = true;
                fields.push_back(std::move(field));
            }
            carried.push_back({From::Relocated, f, d});
            continue;
        }
        internal_assert(!holds_address(arg.type))
            << what << ": after the call to " << pcall.call.name << " in "
            << producer.block->name << ", " << queue.owner << " goes on to use "
            << arg.name << ", an address (" << arg.type << ") computed inside "
            << "the producer's iteration that is not one of its mutable "
            << "locals. The drain runs that continuation later, when the "
            << "iteration is gone. Not supported.";
        size_t f = fields.size();
        for (size_t k = 0; k < fields.size(); k++) {
            if (!fields[k].pointee && same_definition(fields[k].origin, d)) {
                f = k; // one value, already stored by the callee
            }
        }
        if (f == fields.size()) {
            Field field;
            field.name = field_named(arg.name);
            field.type = arg.type;
            field.origin = d;
            field.frame_writes = true;
            fields.push_back(std::move(field));
        }
        carried.push_back({From::Field, f, d});
    }
    const bool frame_saves =
        std::any_of(fields.begin(), fields.end(),
                    [](const Field &f) { return f.frame_writes; });

    // BONSAI_EXPLAIN_DEFER=1 prints the entry's plan: which arguments are
    // left out and why, which are stored, which relocated, and what the
    // frame adds -- the question to ask when an entry is bigger than it
    // should be.
    if (std::getenv("BONSAI_EXPLAIN_DEFER") != nullptr) {
        std::cerr << "; " << what << ": the entry holds";
        for (const Field &f : fields) {
            std::cerr << " " << f.name << " : " << f.type << ";";
        }
        std::cerr << "\n";
        for (size_t j = 0; j < nparams; j++) {
            const Argument &param = centry.args[j];
            std::cerr << ";   " << param.name << " : " << param.type << " -- ";
            switch (params[j].where) {
            case Where::Elided:
                std::cerr << "in scope at the drain";
                break;
            case Where::Relocated:
                std::cerr << "a mutable local of the producer's iteration, "
                             "stored as its contents";
                break;
            case Where::Stored:
                std::cerr << "stored: "
                          << (invariant.at(callee_name)[j]
                                  ? "one value down the chain, but computed "
                                    "inside the producer's iteration"
                                  : "differs between calls");
                break;
            }
            std::cerr << "\n";
        }
        for (size_t i = 0; i < carried.size(); i++) {
            const Argument &arg = k0_entry->args[i + result_args];
            std::cerr << ";   continuation uses " << arg.name << " : "
                      << arg.type << " -- ";
            switch (carried[i].from) {
            case From::Available:
                std::cerr << "in scope at the drain";
                break;
            case From::Field:
                std::cerr << "a value of the iteration, written by the frame";
                break;
            case From::Relocated:
                std::cerr << "a mutable local of the iteration, stored as its "
                             "contents";
                break;
            }
            std::cerr << "\n";
        }
    }

    //===----------------------------------------------------------------===//
    // The size of the queue
    //===----------------------------------------------------------------===//

    // One entry per producer iteration: the producer loop's trip count, or one
    // for a lone call. Computed in the block the loop is in, before it runs.
    shared_ptr<Value> count;
    {
        const size_t at = point->instrs.size();
        if (producer_loop_block) {
            const auto &p =
                std::get<Terminator::ParFor>(producer_loop_block->terminator.data);
            const Type itype = p.start->get_type();
            const auto start = constant_of(p.start);
            const auto stride = constant_of(p.stride);
            if (start.has_value() && *start == 0 && stride.has_value() &&
                *stride == 1) {
                count = p.end;
            } else {
                // ceil((end - start) / stride)
                auto diff = insert_instruction(*O, point, at, itype,
                                               Instruction::Op::Sub,
                                               reach_all(point, {p.end, p.start}));
                if (stride.has_value() && *stride == 1) {
                    count = diff;
                } else {
                    auto less = insert_instruction(
                        *O, point, point->instrs.size(), itype,
                        Instruction::Op::Sub,
                        {reach(point, p.stride), index_constant(itype, 1)});
                    auto sum = insert_instruction(
                        *O, point, point->instrs.size(), itype,
                        Instruction::Op::Add, {diff, less});
                    count = insert_instruction(
                        *O, point, point->instrs.size(), itype,
                        Instruction::Op::Div, {sum, reach(point, p.stride)});
                }
            }
        } else {
            count = constant_u32(1);
        }
    }
    shared_ptr<Value> size = count;
    if (queue.capacity.has_value()) {
        const auto bound = constant_of(count);
        internal_assert(bound.has_value())
            << what << ": " << queue.name << " is given a capacity of "
            << *queue.capacity << ", but the number of entries an iteration of "
            << queue.owner << " pushes is only known at run time (the trip "
            << "count of the producer loop). The capacity cannot be checked "
            << "against it, and a queue that overflows is silent. Leave the "
            << "capacity off to size the queue by the count; a capacity below "
            << "it needs a flush policy, which is not built.";
        internal_assert(*queue.capacity >= *bound)
            << what << ": " << queue.name << " is given a capacity of "
            << *queue.capacity << ", but an iteration of " << queue.owner
            << " pushes " << *bound << " entries. Flushing a full queue by "
            << "draining it before the write is not built; the capacity has "
            << "to hold them all.";
        size = constant_u32(*queue.capacity);
    }

    //===----------------------------------------------------------------===//
    // The types
    //===----------------------------------------------------------------===//

    Struct_t::Map entry_fields;
    for (const Field &f : fields) {
        entry_fields.emplace_back(f.name, f.type);
    }
    const Type entry_t = Struct_t::make("Entry_" + queue.name, entry_fields);
    const Type queue_t = Struct_t::make(
        "Queue_" + queue.name,
        {TypedVar("count", u32()), TypedVar("data", Array_t::make(entry_t, Expr()))});
    const Type queue_ptr_t = Ptr_t::make(queue_t);
    vector<Type> made = {entry_t, queue_t};

    // What a chain function returns now: whether it saved its state; which
    // slot it saved it in, when the frame has state of its own to add there;
    // and the value it would have returned.
    Struct_t::Map saved_fields = {TypedVar("saved", Bool_t::make())};
    const size_t slot_index = frame_saves ? saved_fields.size() : 0;
    if (frame_saves) {
        saved_fields.emplace_back("slot", u32());
    }
    const size_t value_index = returns_value ? saved_fields.size() : 0;
    if (returns_value) {
        saved_fields.emplace_back("value", ret_type);
    }
    const bool saved_is_struct = saved_fields.size() > 1;
    const Type saved_t = saved_is_struct
                             ? Struct_t::make("Saved_" + callee_name, saved_fields)
                             : Bool_t::make();
    if (saved_is_struct) {
        made.push_back(saved_t);
    }
    const auto make_saved = [&](const shared_ptr<Block> &block, bool saved,
                                shared_ptr<Value> slot,
                                shared_ptr<Value> value) -> shared_ptr<Value> {
        if (!saved_is_struct) {
            return constant_bool(saved);
        }
        vector<shared_ptr<Value>> parts = {constant_bool(saved)};
        if (frame_saves) {
            parts.push_back(slot ? std::move(slot) : undef_value(u32()));
        }
        if (returns_value) {
            parts.push_back(value ? std::move(value) : undef_value(ret_type));
        }
        return block->make_instruction(saved_t, Instruction::Op::MakeStruct,
                                       std::move(parts));
    };
    const auto saved_of = [&](const shared_ptr<Block> &block,
                              const shared_ptr<Value> &s) -> shared_ptr<Value> {
        if (!saved_is_struct) {
            return s;
        }
        return block->make_instruction(Bool_t::make(), Instruction::Op::LoadField,
                                       {s, constant_u32(0)});
    };
    const auto slot_of = [&](const shared_ptr<Block> &block,
                             const shared_ptr<Value> &s) -> shared_ptr<Value> {
        internal_assert(frame_saves);
        return block->make_instruction(u32(), Instruction::Op::LoadField,
                                       {s, constant_u32(slot_index)});
    };
    const auto value_of = [&](const shared_ptr<Block> &block,
                              const shared_ptr<Value> &s) -> shared_ptr<Value> {
        internal_assert(returns_value);
        return block->make_instruction(ret_type, Instruction::Op::LoadField,
                                       {s, constant_u32(value_index)});
    };

    //===----------------------------------------------------------------===//
    // The chain: take the queue, return the flag, push at the deferred calls
    //===----------------------------------------------------------------===//

    const string qparam = "_queue_" + queue.name;
    map<string, shared_ptr<Value>> queue_of; // each chain function's parameter
    for (const string &g : chain) {
        Function &gf = *funcs.at(g);
        Block &entry = *gf.blocks.front();
        for (const Argument &arg : entry.args) {
            internal_assert(arg.name != qparam)
                << what << ": " << g << " already has a parameter named "
                << qparam;
        }
        queue_of[g] = entry.add_argument(
            Argument{queue_ptr_t, qparam, /*mutating=*/true});
        gf.ret_type = saved_t;
    }

    // Every call from one chain function to another passes the queue on, and
    // its tail return passes the flag on.
    set<const Block *> passthrough_returns;
    for (const auto &[callee, calls] : calls_into) {
        for (const CallSite &cs : calls) {
            if (cs.caller == queue.owner) {
                continue;
            }
            if (cs.caller == func_name && callee == callee_name) {
                continue; // the deferred call: rewritten below
            }
            Terminator::Call &call = *cs.call();
            const shared_ptr<Function> gf = funcs.at(cs.caller);
            const BlockMap gmap = make_block_map(gf);
            const shared_ptr<Block> cont = gmap.at(call.cont.name);
            call.call.args.push_back(reach(cs.block, queue_of.at(cs.caller)));
            if (call.drop) {
                // `g(); return;` becomes `s = g(); return s;`
                call.drop = false;
                const string result = gf->get_unique_name();
                cont->args.insert(cont->args.begin(), Argument{saved_t, result});
                auto value = std::make_shared<Value>(cont->args.front());
                cont->lookups[result] = value;
                cont->terminator.data = Terminator::Return{value};
            } else {
                retype_argument(*cont, cont->args.front().name, saved_t);
            }
            passthrough_returns.insert(cont.get());
        }
    }

    // Every other return of a chain function says it did not save.
    for (const string &g : chain) {
        for (const auto &block : funcs.at(g)->blocks) {
            auto *ret = std::get_if<Terminator::Return>(&block->terminator.data);
            if (ret == nullptr || passthrough_returns.contains(block.get())) {
                continue;
            }
            if (g == func_name &&
                std::any_of(sites.begin(), sites.end(),
                            [&](const shared_ptr<Block> &site) {
                                const auto &call = std::get<Terminator::Call>(
                                    site->terminator.data);
                                return call.cont.name == block->name;
                            })) {
                continue; // a deferred call's continuation: about to go
            }
            internal_assert(returns_value == (ret->value != nullptr))
                << g << " returns " << (ret->value ? "a value" : "nothing")
                << " in " << block->name << " but is declared to return "
                << ret_type;
            shared_ptr<Value> value = returns_value ? reach(block, ret->value)
                                                    : nullptr;
            ret->value = make_saved(block, false, nullptr, value);
        }
    }

    // The deferred calls: push the entry, return saved. The callee writes
    // the fields that are its arguments -- a pointer argument's pointee --
    // and leaves the frame's to the frame.
    for (const auto &site : sites) {
        const auto call = std::get<Terminator::Call>(site->terminator.data);
        vector<shared_ptr<Value>> values;
        for (const Field &f : fields) {
            if (!f.param.has_value()) {
                values.push_back(undef_value(f.type));
                continue;
            }
            shared_ptr<Value> v = reach(site, call.call.args[*f.param]);
            if (f.pointee) {
                v = site->make_instruction(f.type, Instruction::Op::Load, {v});
            }
            values.push_back(std::move(v));
        }
        auto entry = site->make_instruction(entry_t, Instruction::Op::MakeStruct,
                                            std::move(values));
        auto slot = site->make_instruction(u32(), Instruction::Op::Push,
                                           {reach(site, queue_of.at(func_name)),
                                            entry});
        std::get<shared_ptr<Instruction>>(slot->data)->atomic = true;
        site->terminator.data =
            Terminator::Return{make_saved(site, true, slot, nullptr)};
    }
    remove_unreachable_blocks(*F);

    //===----------------------------------------------------------------===//
    // The owner: the queue, the flag at the producer, and the drain
    //===----------------------------------------------------------------===//

    // A queue is a count and a handle to its own storage, an array of entries
    // sized by the producer count; both are `mut` locals of the owner, made
    // where the count is known, before the producer runs, and a queue is
    // handed to the callee the way any mutable local is handed to a callee:
    // as its address.
    //
    // Whether there is one queue or two follows from who pushes onto it. When
    // the deferred callee is on the chain -- a self-recursion -- running an
    // entry pushes its successor onto the queue being drained, so there are
    // two, as pbrt's wavefront integrator has two ray queues: the one being
    // read this round and the one the round's successors go to, indexed by
    // the round's parity (pbrt: `rayQueues[depth & 1]`). When the drain's
    // call cannot reach a push -- the deferred call is to a function off the
    // chain -- one queue and one pass over it are all there is.
    const bool self_feeding = callee_in_chain;
    const int nqueues = self_feeding ? 2 : 1;
    vector<shared_ptr<Value>> stores;
    for (int i = 0; i < nqueues; i++) {
        stores.push_back(make_alloca(
            *O, point, Array_t::make(entry_t, as_expr(size)),
            queue.name + "_entries" + (self_feeding ? "_" + std::to_string(i) : "")));
    }
    const shared_ptr<Value> queues =
        self_feeding
            ? make_alloca(*O, point,
                          Array_t::make(queue_t, UIntImm::make(u32(), 2)),
                          queue.name + "_queue")
            : make_alloca(*O, point, queue_t, queue.name + "_queue");
    for (int i = 0; i < nqueues; i++) {
        auto initial = point->make_instruction(queue_t, Instruction::Op::MakeStruct,
                                               {constant_u32(0), stores[i]});
        auto slot = self_feeding
                        ? point->make_instruction(queue_ptr_t, Instruction::Op::GEP,
                                                  {queues, constant_u32(i)})
                        : queues;
        point->make_side_effect(Instruction::Op::Store, {slot, initial});
    }
    // The address of queue `which`, for passing to the callee: an address is
    // written where it is used (has_no_binding in SSA/CodeGen_Stmt.cpp), so
    // this names the queue in place rather than a copy of it.
    const auto queue_at = [&](const shared_ptr<Block> &block,
                              const shared_ptr<Value> &which) {
        if (!self_feeding) {
            return reach(block, queues);
        }
        return block->make_instruction(queue_ptr_t, Instruction::Op::GEP,
                                       {reach(block, queues), which});
    };
    // The entry storage of queue `which`, as the queue records it.
    const auto entries_at = [&](const shared_ptr<Block> &block,
                                const shared_ptr<Value> &which) {
        if (!self_feeding) {
            return reach(block, stores[0]);
        }
        auto that = block->make_instruction(queue_t, Instruction::Op::ExtractIdx,
                                            {reach(block, queues), which});
        return block->make_instruction(Array_t::make(entry_t, Expr()),
                                       Instruction::Op::LoadField,
                                       {that, constant_u32(1)});
    };
    const auto fresh_block = [&](const string &name) {
        auto block = std::make_shared<Block>();
        block->name = name;
        for (const auto &other : O->blocks) {
            internal_assert(other->name != name)
                << what << ": " << queue.owner << " already has a block named "
                << name;
        }
        block->owner = O;
        O->blocks.push_back(block);
        return block;
    };
    const string prefix = queue.name + "!";

    // The frame's part of an entry, written into the slot the callee saved in
    // the storage `entries`: the iteration's values the continuation needs,
    // and the contents of the iteration's mutable locals the callee was not
    // handed.
    const auto write_frame = [&](const shared_ptr<Block> &block,
                                 const shared_ptr<Value> &entries,
                                 const shared_ptr<Value> &slot,
                                 const std::function<shared_ptr<Value>(const Field &)> &value_of_field) {
        auto entry_ptr = block->make_instruction(Ptr_t::make(entry_t),
                                                 Instruction::Op::GEP,
                                                 {reach(block, entries), slot});
        for (size_t f = 0; f < fields.size(); f++) {
            if (!fields[f].frame_writes) {
                continue;
            }
            auto field_ptr = block->make_instruction(Ptr_t::make(fields[f].type),
                                                     Instruction::Op::FieldPtr,
                                                     {entry_ptr, constant_u32(f)});
            block->make_side_effect(Instruction::Op::Store,
                                    {field_ptr, value_of_field(fields[f])});
        }
    };

    // The producer passes the first queue and acts on the flag: a saved return
    // skips the rest of the iteration, which the drain runs later, after
    // the frame has put its own state into the entry.
    shared_ptr<Block> skip;
    {
        pcall.call.args.push_back(queue_at(producer.block, constant_u32(0)));
        const bool dropped = pcall.drop;
        pcall.drop = false;

        auto dispatch = fresh_block(prefix + "saved");
        skip = fresh_block(prefix + "skip");
        auto flag = dispatch->add_argument(
            Argument{saved_t, O->get_unique_name()});
        // The continuation's other arguments, threaded through.
        vector<shared_ptr<Value>> onwards;
        vector<shared_ptr<Value>> carried_here; // as the dispatch refers to them
        if (!dropped) {
            onwards.push_back(nullptr); // the value, below
        }
        for (size_t i = result_args; i < k0_entry->args.size(); i++) {
            auto v = dispatch->add_argument(k0_entry->args[i]);
            onwards.push_back(v);
            carried_here.push_back(v);
        }
        // Wired in before anything below asks the dispatch for a value it
        // does not have: what it is asked for has to be threaded on from the
        // producer's block, through the call's continuation edge.
        dispatch->preds = {producer.block};
        pcall.cont = Terminator::Jump{dispatch->name, pcall.cont.args};
        k0_entry->preds = {dispatch};
        auto saved = saved_of(dispatch, flag);
        if (!dropped) {
            onwards[0] = value_of(dispatch, flag);
        }
        shared_ptr<Block> on_saved = skip;
        if (frame_saves) {
            on_saved = fresh_block(prefix + "save");
            on_saved->preds = {dispatch};
            dispatch->terminator.data = Terminator::Dispatch{
                saved,
                {Terminator::Jump{k0_entry->name, std::move(onwards)},
                 Terminator::Jump{on_saved->name}}};
            auto slot = slot_of(on_saved, reach(on_saved, flag));
            write_frame(on_saved, stores[0], slot, [&](const Field &f) -> shared_ptr<Value> {
                if (f.local != nullptr) {
                    return on_saved->make_instruction(
                        f.type, Instruction::Op::Load,
                        {reach(on_saved, f.origin.value)});
                }
                for (size_t i = 0; i < carried.size(); i++) {
                    if (carried[i].from == From::Field &&
                        &fields[carried[i].field] == &f) {
                        return reach(on_saved, carried_here[i]);
                    }
                }
                internal_error << "no value for the frame's field " << f.name;
                return nullptr;
            });
            on_saved->terminator.data = Terminator::Jump{skip->name};
            skip->preds = {on_saved};
        } else {
            dispatch->terminator.data = Terminator::Dispatch{
                saved,
                {Terminator::Jump{k0_entry->name, std::move(onwards)},
                 Terminator::Jump{skip->name}}};
            skip->preds = {dispatch};
        }
        // `skip` ends the iteration: a yield of the producer loop's body, or
        // -- for a lone call -- a jump to the drain, set below once it exists.
        skip->terminator.data = Terminator::Yield{};
    }

    // The rest of the producer's iteration, copied for the drain to run per
    // entry. Copied now, before the drain is built: building it threads
    // values through the blocks between the producer and the drain, which
    // for a lone call are these, and the copy is of the continuation as the
    // program has it -- taking the call's value and what it carried, and no
    // more.
    const map<string, shared_ptr<Block>> copies =
        clone_region(*O, k0, "!" + queue.name);
    const shared_ptr<Block> k_entry = copies.at(k0_entry->name);

    // The drain. For a self-feeding queue, a round per bounce, as pbrt's
    // wavefront integrator runs one set of kernels per depth: round r reads
    // the queue r & 1 and its successors go to the other, emptied first, and
    // the drain ends when a round finds its queue empty. Otherwise one pass
    // over the one queue.
    auto drain_entry = fresh_block(prefix + "drain");
    auto body = fresh_block(prefix + "run");
    auto after = fresh_block(prefix + "ran");
    auto exit = fresh_block(prefix + "exit");
    shared_ptr<Block> header, pre, latch;
    if (self_feeding) {
        header = fresh_block(prefix + "round");
        pre = fresh_block(prefix + "batch");
        latch = fresh_block(prefix + "next");
    }

    // Where the drain sits: after the producer loop, in place of its
    // continuation; or, for a lone call, where the iteration ends.
    vector<shared_ptr<Value>> exit_args;
    string exit_target;
    if (producer_loop_block) {
        auto &p = std::get<Terminator::ParFor>(producer_loop_block->terminator.data);
        exit_target = p.cont.name;
        exit_args = p.cont.args;
        p.cont = Terminator::Jump{drain_entry->name};
        drain_entry->preds = {producer_loop_block};
        const shared_ptr<Block> old_cont = omap.at(exit_target);
        std::erase_if(old_cont->preds, [&](const auto &w) {
            return w.lock().get() == producer_loop_block.get();
        });
        old_cont->preds.push_back(exit);
    } else {
        // The rest of the iteration ends by yielding (a loop's) or returning
        // (the function's); both now go through the drain, and so does the
        // skip.
        for (const auto &block : k0) {
            const bool ends =
                std::holds_alternative<Terminator::Yield>(block->terminator.data) ||
                std::holds_alternative<Terminator::Return>(block->terminator.data);
            if (!ends) {
                continue;
            }
            block->terminator.data = Terminator::Jump{drain_entry->name};
            drain_entry->preds.push_back(block);
        }
        skip->terminator.data = Terminator::Jump{drain_entry->name};
        drain_entry->preds.push_back(skip);
    }
    shared_ptr<Value> cur, nxt; // which queue is read, and which pushed to
    if (self_feeding) {
        drain_entry->terminator.data =
            Terminator::Jump{header->name, {constant_u32(0)}};

        // The loop's skeleton first and its contents after: a value from
        // outside the loop is threaded into the header through both its
        // predecessors (Block::get_value), the latch included, and that walk
        // needs every block on the way to have its terminator. The parfor's
        // trip count is filled in once the header has read it.
        auto round = header->add_argument(Argument{u32(), prefix + "round"});
        header->preds = {drain_entry, latch};
        header->terminator.data = Terminator::Dispatch{
            /*cond=*/nullptr,
            {Terminator::Jump{pre->name}, Terminator::Jump{exit->name}}};
        pre->preds = {header};
        pre->terminator.data = Terminator::ParFor{queue.name,
                                                  constant_u32(0),
                                                  /*end=*/nullptr,
                                                  constant_u32(1),
                                                  Terminator::Jump{body->name},
                                                  Terminator::Jump{latch->name}};
        latch->preds = {pre};
        {
            auto next_round = latch->make_instruction(
                u32(), Instruction::Op::Add, {reach(latch, round), constant_u32(1)});
            latch->terminator.data =
                Terminator::Jump{header->name, {next_round}};
        }

        // round: while this round's queue is not empty
        cur = header->make_instruction(u32(), Instruction::Op::BwAnd,
                                       {round, constant_u32(1)});
        auto current = header->make_instruction(
            queue_t, Instruction::Op::ExtractIdx, {reach(header, queues), cur});
        auto pending = header->make_instruction(
            u32(), Instruction::Op::LoadField, {current, constant_u32(0)});
        auto empty = header->make_instruction(
            Bool_t::make(), Instruction::Op::Eq, {pending, constant_u32(0)});
        std::get<Terminator::Dispatch>(header->terminator.data).cond = empty;

        // batch: the other queue is emptied and is where this round's
        // successors go; then every entry of this one is run.
        nxt = pre->make_instruction(u32(), Instruction::Op::Xor,
                                    {reach(pre, cur), constant_u32(1)});
        auto count_ptr = pre->make_instruction(
            Ptr_t::make(u32()), Instruction::Op::FieldPtr,
            {queue_at(pre, nxt), constant_u32(0)});
        pre->make_side_effect(Instruction::Op::Store, {count_ptr, constant_u32(0)});
        std::get<Terminator::ParFor>(pre->terminator.data).end =
            reach(pre, pending);
        body->preds = {pre};
    } else {
        // One pass: every entry there is, and nothing pushes meanwhile.
        auto whole = drain_entry->make_instruction(
            queue_t, Instruction::Op::Load, {reach(drain_entry, queues)});
        auto pending = drain_entry->make_instruction(
            u32(), Instruction::Op::LoadField, {whole, constant_u32(0)});
        drain_entry->terminator.data = Terminator::ParFor{queue.name,
                                                          constant_u32(0),
                                                          pending,
                                                          constant_u32(1),
                                                          Terminator::Jump{body->name},
                                                          Terminator::Jump{exit->name}};
        body->preds = {drain_entry};
    }

    // run(i): the entry, and the call it stands for.
    auto index = body->add_argument(Argument{u32(), queue.name});
    auto entry = body->make_instruction(
        entry_t, Instruction::Op::ExtractIdx,
        {entries_at(body, cur ? reach(body, cur) : nullptr), index});
    // A relocated local: the entry's copy of its contents, given a local of
    // the drain's own to be run with.
    map<size_t, shared_ptr<Value>> locals;
    for (size_t f = 0; f < fields.size(); f++) {
        if (fields[f].local == nullptr) {
            continue;
        }
        auto local = make_alloca(*O, body, fields[f].type);
        auto contents = body->make_instruction(fields[f].type,
                                               Instruction::Op::LoadField,
                                               {entry, constant_u32(f)});
        body->make_side_effect(Instruction::Op::Store, {local, contents});
        locals[f] = local;
    }
    {
        vector<shared_ptr<Value>> args;
        for (size_t j = 0; j < nparams; j++) {
            switch (params[j].where) {
            case Where::Elided:
                args.push_back(reach(body, origin.at(callee_name)[j].value));
                break;
            case Where::Stored:
                args.push_back(body->make_instruction(
                    centry.args[j].type, Instruction::Op::LoadField,
                    {entry, constant_u32(params[j].field)}));
                break;
            case Where::Relocated:
                args.push_back(locals.at(params[j].field));
                break;
            }
        }
        if (callee_in_chain) {
            args.push_back(queue_at(body, reach(body, nxt)));
        }
        body->terminator.data = Terminator::Call{
            Terminator::Jump{callee_name, std::move(args)},
            Terminator::Jump{after->name}, /*drop=*/!callee_in_chain && !returns_value};
    }

    // ran(r): a saved return is an entry for the next round, with the frame's
    // part written; any other is the rest of the producer's iteration,
    // copied here.
    after->preds = {body};
    {
        // The value goes to the copy only where the producer kept it.
        vector<shared_ptr<Value>> onwards;
        shared_ptr<Value> flag;
        shared_ptr<Value> saved;
        if (callee_in_chain) {
            flag = after->add_argument(Argument{saved_t, O->get_unique_name()});
            saved = saved_of(after, flag);
            if (result_args == 1) {
                onwards.push_back(value_of(after, flag));
            }
        } else if (returns_value) {
            auto value =
                after->add_argument(Argument{ret_type, O->get_unique_name()});
            if (result_args == 1) {
                onwards.push_back(value);
            }
        }
        for (const CarriedPlan &c : carried) {
            switch (c.from) {
            case From::Available:
                onwards.push_back(reach(after, c.def.value));
                break;
            case From::Field:
                onwards.push_back(after->make_instruction(
                    fields[c.field].type, Instruction::Op::LoadField,
                    {reach(after, entry), constant_u32(c.field)}));
                break;
            case From::Relocated:
                onwards.push_back(reach(after, locals.at(c.field)));
                break;
            }
        }
        if (saved) {
            auto done = fresh_block(prefix + "done");
            done->preds = {after};
            after->terminator.data = Terminator::Dispatch{
                saved,
                {Terminator::Jump{k_entry->name, std::move(onwards)},
                 Terminator::Jump{done->name}}};
            if (frame_saves) {
                auto slot = slot_of(done, reach(done, flag));
                write_frame(done, entries_at(done, reach(done, nxt)), slot,
                            [&](const Field &f) -> shared_ptr<Value> {
                    for (size_t k = 0; k < fields.size(); k++) {
                        if (&fields[k] != &f) {
                            continue;
                        }
                        if (f.local != nullptr) {
                            return done->make_instruction(
                                f.type, Instruction::Op::Load,
                                {reach(done, locals.at(k))});
                        }
                        // The frame's value is what the entry carried in.
                        return done->make_instruction(
                            f.type, Instruction::Op::LoadField,
                            {reach(done, entry), constant_u32(k)});
                    }
                    internal_error << "no value for the frame's field " << f.name;
                    return nullptr;
                });
            }
            done->terminator.data = Terminator::Yield{};
        } else {
            after->terminator.data =
                Terminator::Jump{k_entry->name, std::move(onwards)};
        }
    }
    // The copy ends the drain's iteration where the original ended the
    // producer's; a return of the function's becomes a yield of the loop's.
    for (const auto &[name, copy] : copies) {
        if (std::holds_alternative<Terminator::Return>(copy->terminator.data)) {
            copy->terminator.data = Terminator::Yield{};
        }
        O->blocks.push_back(copy);
    }

    // exit: on to whatever followed the producer.
    exit->preds = {self_feeding ? header : drain_entry};
    if (producer_loop_block) {
        exit->terminator.data =
            Terminator::Jump{exit_target, reach_all(exit, exit_args)};
    } else if (root) {
        exit->terminator.data = Terminator::Return{};
    } else {
        exit->terminator.data = Terminator::Yield{};
    }

    refresh_preds(*O);
    remove_unreachable_blocks(*O);
    // What was built by rule is looked at once: a field read twice, a
    // comparison of constants (see SSA/Simplify.h).
    simplify(*O);
    for (const string &g : chain) {
        simplify(*funcs.at(g));
    }
    return made;
}

//===--------------------------------------------------------------------===//
// lower_pushes()
//===--------------------------------------------------------------------===//

void lower_pushes(Function &func) {
    for (const auto &block : func.blocks) {
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> push = block->instrs[i];
            if (push->op != Instruction::Op::Push) {
                continue;
            }
            internal_assert(push->operands.size() == 2) << push->operands.size();
            const shared_ptr<Value> q = push->operands[0];
            const shared_ptr<Value> entry = push->operands[1];
            const auto *qptr = q->get_type().as<Ptr_t>();
            internal_assert(qptr) << "push to a non-pointer: " << q->get_type();
            const Type queue_t = qptr->etype;
            const auto *qs = queue_t.as<Struct_t>();
            internal_assert(qs && qs->fields.size() == 2)
                << "push to something that is not a queue: " << queue_t;
            const Type count_t = qs->fields[0].type;
            const Type data_t = qs->fields[1].type;

            size_t at = i;
            // The count's address is a FieldPtr, which is written out where it
            // is used rather than bound to a name (has_no_binding in
            // SSA/CodeGen_Stmt.cpp): it names the count in the queue, where a
            // load of the queue and the address of its count field would name
            // a copy. The slot is claimed by fetch-and-add when the push is
            // atomic, and by a read and a write when it is not.
            auto count_ptr = insert_instruction(func, block, at++,
                                                Ptr_t::make(count_t),
                                                Instruction::Op::FieldPtr,
                                                {q, constant_u32(0)});
            auto whole = insert_instruction(func, block, at++, queue_t,
                                            Instruction::Op::Load, {q});
            const auto one =
                std::make_shared<Value>(Constant{count_t, uint64_t(1)});
            shared_ptr<Value> index;
            if (push->atomic) {
                index = insert_instruction(func, block, at++, count_t,
                                           Instruction::Op::AtomicAdd,
                                           {count_ptr, one});
            } else {
                index = insert_instruction(func, block, at++, count_t,
                                           Instruction::Op::LoadField,
                                           {whole, constant_u32(0)});
                auto next = insert_instruction(func, block, at++, count_t,
                                               Instruction::Op::Add,
                                               {index, one});
                insert_side_effect(block, at++, Instruction::Op::Store,
                                   {count_ptr, next});
            }
            // (*q).data[index] = entry, field by field where the entry is
            // built in place -- a field left undefined is one the frame
            // writes, and is not stored. The array is a handle, so reading it
            // out of a copy of the queue reads the same storage.
            auto data = insert_instruction(func, block, at++, data_t,
                                           Instruction::Op::LoadField,
                                           {whole, constant_u32(1)});
            auto slot = insert_instruction(func, block, at++,
                                           Ptr_t::make(entry->get_type()),
                                           Instruction::Op::GEP, {data, index});
            const auto *built = std::get_if<shared_ptr<Instruction>>(&entry->data);
            const auto *entry_struct = entry->get_type().as<Struct_t>();
            if (built != nullptr && (*built)->op == Instruction::Op::MakeStruct &&
                entry_struct != nullptr) {
                for (size_t k = 0; k < (*built)->operands.size(); k++) {
                    const shared_ptr<Value> &v = (*built)->operands[k];
                    if (is_undef(v)) {
                        continue;
                    }
                    auto field_ptr = insert_instruction(
                        func, block, at++, Ptr_t::make(entry_struct->fields[k].type),
                        Instruction::Op::FieldPtr, {slot, constant_u32(k)});
                    insert_side_effect(block, at++, Instruction::Op::Store,
                                       {field_ptr, v});
                }
            } else {
                insert_side_effect(block, at++, Instruction::Op::Store,
                                   {slot, entry});
            }
            // The push's value was the slot; it is gone with the push.
            replace_uses(func, push.get(), index);
            i = at - 1;
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
