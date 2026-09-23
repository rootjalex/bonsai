#include "SSA/Defer.h"

#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/Definitions.h"
#include "SSA/SSA.h"
#include "SSA/Simplify.h"
#include "SSA/Specialize.h"

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
#include <sstream>
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

Type value_type(const Value &v) {
    return std::visit(
        overloads{[](const Constant &c) { return c.type; },
                  [](const Argument &a) { return a.type; },
                  [](const shared_ptr<Instruction> &i) { return i->type; }},
        v.data);
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
                        bool atomic = false, bool compact = false) {
    auto instr = std::make_shared<Instruction>(op, std::move(operands), block);
    instr->atomic = atomic;
    instr->compact = compact;
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

// Applies `edit` to every copy `block` holds of the argument `name`: its
// entry in the parameter list, its lookup, and every value in the block that
// names the argument. An Argument is copied by value into a Value, so there
// is no one place to change it (see widen_argument in Vectorize.cpp).
void edit_argument(Block &block, const string &name,
                   const std::function<void(Argument &)> &edit) {
    for (Argument &arg : block.args) {
        if (arg.name == name) {
            edit(arg);
        }
    }
    const auto visit = [&](const shared_ptr<Value> &v) {
        if (v && std::holds_alternative<Argument>(v->data)) {
            Argument &a = std::get<Argument>(v->data);
            if (a.name == name) {
                edit(a);
            }
        }
    };
    if (const auto it = block.lookups.find(name); it != block.lookups.end()) {
        visit(it->second);
    }
    for (const auto &instr : block.instrs) {
        for (const auto &operand : instr->operands) {
            visit(operand);
        }
    }
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) {
                       for (auto &a : j.args) {
                           visit(a);
                       }
                   },
                   [&](Terminator::Dispatch &d) {
                       visit(d.cond);
                       for (auto &t : d.targets) {
                           for (auto &a : t.args) {
                               visit(a);
                           }
                       }
                   },
                   [&](Terminator::Return &r) { visit(r.value); },
                   [&](Terminator::ParFor &p) {
                       visit(p.start);
                       visit(p.end);
                       visit(p.stride);
                       for (auto &a : p.body.args) {
                           visit(a);
                       }
                       for (auto &a : p.cont.args) {
                           visit(a);
                       }
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) {
                       for (auto &a : c.call.args) {
                           visit(a);
                       }
                       for (auto &a : c.cont.args) {
                           visit(a);
                       }
                   },
                   [&](Terminator::MultiCall &c) {
                       for (auto &a : c.call.args) {
                           visit(a);
                       }
                       for (auto &a : c.cont.args) {
                           visit(a);
                       }
                       for (auto &vs : c.varying) {
                           for (auto &a : vs) {
                               visit(a);
                           }
                       }
                   },
               },
               block.terminator.data);
}

// Changes the type of the argument `name` of `block`, wherever the block
// holds a copy of it.
void retype_argument(Block &block, const string &name, const Type &type) {
    edit_argument(block, name, [&](Argument &a) { a.type = type; });
}

// Renames the argument `from` of `block` to `to`, wherever the block holds a
// copy of it, its lookup included.
void rename_argument(Block &block, const string &from, const string &to) {
    edit_argument(block, from, [&](Argument &a) { a.name = to; });
    if (const auto it = block.lookups.find(from); it != block.lookups.end()) {
        auto value = it->second;
        block.lookups.erase(it);
        block.lookups[to] = std::move(value);
    }
}

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

} // namespace

// A copy of the blocks `region` of `func`, renamed with `suffix`, for placing
// the rest of a producer's iteration inside the drain -- and, declared in
// SSA/CloneFunction.h, for specialize()'s copy of a loop body per variant.
// Instructions get fresh names, and so does every argument that carried one
// of the old names on, so that the copy and the original can live in one
// function; a name the region refers to but does not define -- a value from
// before the call, threaded in -- keeps its name, since that is what its
// definition is called. Jumps to blocks outside the region are left pointing
// where they were, for the caller to redirect. Predecessors are not set; the
// caller rebuilds them.
map<string, shared_ptr<Block>>
clone_region(Function &func, const vector<shared_ptr<Block>> &region,
             const string &suffix, bool keep_names) {
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
            // copy's scope keeps it apart from the original's. A copy that
            // becomes a function of its own keeps every name.
            const bool keeps_name = instr->name.empty() ||
                                    instr->op == Instruction::Op::Set ||
                                    keep_names;
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

namespace {

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
    // The owner's reducer local this field holds the address of, if any:
    // given a slot of its own beside the queues once the queue's size is
    // known, whose address the field then stores (see the hoist below).
    const Instruction *reducer_local = nullptr;
    // Whether the field holds a reducer's address at all -- a slot's, or
    // one handed down -- so that a drain's read of it counts as one too
    // (Function::reducer_slots) for the deferrals that come after.
    bool reducer_address = false;
    // The owner's definition of what the field holds, when it is one value
    // throughout -- for an argument and a frame value that are the same value
    // to share the field.
    Definition origin;
    // Whether the frame writes it after a saved return, rather than the
    // callee at the push.
    bool frame_writes = false;
};

// One scalar of an entry, which the queue keeps in an array of its own: a
// queue's storage is a struct of arrays. An entry field that is an aggregate
// is taken down to its scalars -- a struct to its fields, a short vector to
// its components -- so that a gang of entries reads each scalar of its
// continuations as one dense vector load, and the compacting push writes
// each with one compress-store; and so that the queue reads as pbrt's
// wavefront queues do, whose `SOA<Ray>` holds an `SOA<Point3f> o` that holds
// `float *x, *y, *z` (pbrt's `soac`). Named after the entry's field and the
// path down to the scalar. A layout for the queue that a schedule asks for
// may replace this; nothing here is the layout language's promise.
struct Leaf {
    string name;
    Type type;
};

// The name of component `k` of a short vector: the geometric letters for a
// vector short enough to have them, the index otherwise.
string component_name(uint32_t k, uint32_t lanes) {
    return lanes <= 4 ? string(1, "xyzw"[k]) : std::to_string(k);
}

// The leaves of a value of `type` named `name`, appended to `out` in the
// order take_apart() reads them and rebuild() puts them back.
void leaves_of(const string &name, const Type &type, vector<Leaf> &out) {
    if (const Struct_t *s = type.as<Struct_t>()) {
        for (const TypedVar &f : s->fields) {
            leaves_of(name + "_" + f.name, f.type, out);
        }
        return;
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        for (uint32_t k = 0; k < v->lanes; k++) {
            leaves_of(name + "_" + component_name(k, v->lanes), v->etype, out);
        }
        return;
    }
    internal_assert(!type.is<Array_t>() && !type.is<Tuple_t>())
        << "[unimplemented] a queue entry's field " << name << " is " << type
        << ", which a struct-of-arrays queue has no array for";
    out.push_back(Leaf{name, type});
}

// Makes an instruction where the caller wants it, and hands back its value.
using Emit = std::function<shared_ptr<Value>(const Type &, Instruction::Op,
                                             vector<shared_ptr<Value>>)>;

// The leaves of `value`, appended to `out` in leaf order: `leaves[l]` is the
// first, and `l` is left after the last. The value is one entry, or a gang
// of `lanes` of them -- each aggregate a struct of gang-wide parts (see
// widen() in IR/Type.cpp) -- and is taken apart until a part is a leaf as
// the queue stores it, or as the gang holds one. An undefined value -- a
// field the frame writes after the push, not the callee -- gives an
// undefined leaf, which nothing stores.
void take_apart(const shared_ptr<Value> &value, const vector<Leaf> &leaves,
                size_t &l, uint32_t lanes, const Emit &emit,
                vector<shared_ptr<Value>> &out) {
    const Type type = value->get_type();
    if (l < leaves.size() &&
        (equals(type, leaves[l].type) ||
         (lanes > 1 && equals(type, widen(leaves[l].type, lanes))))) {
        out.push_back(is_undef(value) ? nullptr : value);
        l++;
        return;
    }
    // Part `k` of the value: the operand a value built in place was built
    // from, where there is one to read off, and a read of the part otherwise.
    const auto part = [&](const Type &t, Instruction::Op op, uint32_t k) {
        if (is_undef(value)) {
            return undef_value(t);
        }
        const auto *built = std::get_if<shared_ptr<Instruction>>(&value->data);
        if (built != nullptr && (*built)->op == Instruction::Op::MakeStruct &&
            k < (*built)->operands.size() &&
            equals((*built)->operands[k]->get_type(), t)) {
            return (*built)->operands[k];
        }
        return emit(t, op, {value, constant_u32(k)});
    };
    if (const Struct_t *s = type.as<Struct_t>()) {
        for (uint32_t k = 0; k < uint32_t(s->fields.size()); k++) {
            take_apart(part(s->fields[k].type, Instruction::Op::LoadField, k),
                       leaves, l, lanes, emit, out);
        }
        return;
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        for (uint32_t k = 0; k < v->lanes; k++) {
            take_apart(part(v->etype, Instruction::Op::ExtractIdx, k), leaves,
                       l, lanes, emit, out);
        }
        return;
    }
    internal_error << "a queue entry holds a " << type << " where the queue "
                   << "stores "
                   << (l < leaves.size() ? leaves[l].name : string("nothing"));
}

// A value of `type` put back together from its leaves, `next_leaf` giving
// each in leaf order: take_apart() inverted, for one entry.
shared_ptr<Value> rebuild(
    const Type &type, const Emit &emit,
    const std::function<shared_ptr<Value>(const Type &)> &next_leaf) {
    vector<shared_ptr<Value>> parts;
    if (const Struct_t *s = type.as<Struct_t>()) {
        for (const TypedVar &f : s->fields) {
            parts.push_back(rebuild(f.type, emit, next_leaf));
        }
    } else if (const Vector_t *v = type.as<Vector_t>()) {
        // A vector built from its components is a MakeStruct of the vector's
        // type, as a vector literal is (see Build in SSA/Convert.cpp).
        for (uint32_t k = 0; k < v->lanes; k++) {
            parts.push_back(rebuild(v->etype, emit, next_leaf));
        }
    } else {
        return next_leaf(type);
    }
    return emit(type, Instruction::Op::MakeStruct, std::move(parts));
}

// The value `pred` passes to `block`'s argument `k` along its edge there, or
// null when `pred` reaches `block` some other way.
shared_ptr<Value> passed_along(const Block &pred, const Block &block, size_t k) {
    shared_ptr<Value> found;
    const auto take = [&](const Terminator::Jump &j) {
        if (j.name == block.name && k < j.args.size()) {
            found = j.args[k];
        }
    };
    std::visit(overloads{
                   [&](const std::monostate &) {},
                   [&](const Terminator::Jump &j) { take(j); },
                   [&](const Terminator::Dispatch &d) {
                       for (const auto &t : d.targets) {
                           take(t);
                       }
                   },
                   [&](const Terminator::Return &) {},
                   [&](const Terminator::ParFor &p) {
                       take(p.body);
                       take(p.cont);
                   },
                   [&](const Terminator::Yield &) {},
                   [&](const Terminator::Call &c) { take(c.cont); },
                   [&](const Terminator::MultiCall &c) { take(c.cont); },
               },
               pred.terminator.data);
    return found;
}

// Points every edge of `block`'s terminator that goes to `from` at `to`
// instead, the arguments as they were.
void retarget(Block &block, const string &from, const string &to) {
    const auto fix = [&](Terminator::Jump &j) {
        if (j.name == from) {
            j.name = to;
        }
    };
    std::visit(overloads{
                   [&](std::monostate &) {},
                   [&](Terminator::Jump &j) { fix(j); },
                   [&](Terminator::Dispatch &d) {
                       for (auto &t : d.targets) {
                           fix(t);
                       }
                   },
                   [&](Terminator::Return &) {},
                   [&](Terminator::ParFor &p) {
                       fix(p.body);
                       fix(p.cont);
                   },
                   [&](Terminator::Yield &) {},
                   [&](Terminator::Call &c) { fix(c.cont); },
                   [&](Terminator::MultiCall &c) { fix(c.cont); },
               },
               block.terminator.data);
}

// The join a spawned deferral leaves implicit. `spawn acc += f(...)` says
// the value goes into the reducer and nothing waits for it; what does wait,
// by the rules of a reducer, is whoever reads the reducer -- the producer's
// continuation, which reads the path's radiance to write the film. Deferred,
// the spawned calls run in a drain that comes after the continuation would
// have: a path that ends in a round has its shadow ray traced at the end of
// that round, after the material drain that ended the path ran the
// continuation. So every place the owner runs a continuation that reads a
// reducer (Function::continuation_entries) becomes a push of the
// continuation's arguments onto a queue of its own, `<queue>_done`, and one
// pass over that queue runs the continuation from `at` on -- the block that
// comes after every round of the drains, or after the spawned drain when
// there are no rounds. pbrt has the same two things by construction: its
// per-sample state (`pixelSampleState`) outlives the bounces, and its film
// write is a pass of its own after the last of them (`UpdateFilm`). What the
// entry carries is what the continuation took: a value of the drain's
// iteration as itself; a mutable local of the iteration as its contents,
// given a local of the pass's own to run with; a reducer's address as the
// address; and nothing for a value in scope where the pass runs.
void join_continuations(const shared_ptr<Function> &O, const string &what,
                        const string &qname, const shared_ptr<Value> &size,
                        const shared_ptr<Block> &alloc_point,
                        const shared_ptr<Block> &at, vector<Type> &made) {
    Function &F = *O;
    const BlockMap omap = make_block_map(O);
    Definitions defs(F);
    const Block &oentry = *F.blocks.front();

    // Whether a definition is a reducer's address: a slot a deferral made,
    // or a reducer the owner was handed.
    const auto reducer_address = [&](const Definition &d) {
        if (!d.value) {
            return false;
        }
        if (const auto *di = std::get_if<shared_ptr<Instruction>>(&d.value->data)) {
            return F.reducer_slots.contains(di->get());
        }
        if (const auto *da = std::get_if<Argument>(&d.value->data);
            da != nullptr && d.block == oentry.name) {
            for (const Argument &p : oentry.args) {
                if (p.name == da->name) {
                    return p.reducer;
                }
            }
        }
        return false;
    };

    // The continuations still there, and whether any reads a reducer.
    vector<shared_ptr<Block>> entries;
    bool reads = false;
    for (const string &name : F.continuation_entries) {
        const auto it = omap.find(name);
        if (it == omap.end()) {
            continue;
        }
        entries.push_back(it->second);
        for (const auto &block : region_from(F, name)) {
            for (const auto &in : block->instrs) {
                if (in->op != Instruction::Op::Load || in->operands.size() != 1) {
                    continue;
                }
                const Definition d = defs.of(block->name, in->operands[0]);
                if (reducer_address(d)) {
                    reads = true;
                }
                // A pointer this form cannot trace to its definition -- a
                // merge -- might be one; taken to be, rather than the read
                // left where it is.
                const auto *da = d.value ? std::get_if<Argument>(&d.value->data) : nullptr;
                if (da != nullptr && d.block != oentry.name && holds_address(da->type)) {
                    reads = true;
                }
            }
        }
    }
    if (!reads || entries.empty()) {
        return;
    }
    for (const auto &K : entries) {
        for (const auto &block : region_from(F, K->name)) {
            internal_assert(
                !std::holds_alternative<Terminator::Return>(block->terminator.data))
                << "[unimplemented] " << what << ": the continuation at "
                << K->name << " reads a reducer the spawned calls add into, "
                << "and returns from " << F.blocks.front()->name << " (in "
                << block->name << "): it has to run after the drain, which is "
                << "built for a continuation that ends a loop iteration.";
        }
    }

    // One signature: the continuations are copies of one another. A copy of
    // the arguments, since the first continuation's own are renamed below
    // and the names are needed as they were.
    const vector<Argument> sig = entries.front()->args;
    for (const auto &K : entries) {
        internal_assert(K->args.size() == sig.size())
            << what << ": the continuations " << entries.front()->name << " and "
            << K->name << " take " << sig.size() << " and " << K->args.size()
            << " arguments";
        for (size_t i = 0; i < sig.size(); i++) {
            internal_assert(equals(K->args[i].type, sig[i].type))
                << what << ": argument " << i << " of " << K->name << " is a "
                << K->args[i].type << ", of " << entries.front()->name << " a "
                << sig[i].type;
        }
    }

    // What each argument is, at every edge into every continuation.
    enum class Kind { Elided, Value, Address, Contents };
    struct Plan {
        Kind kind = Kind::Value;
        Type type;
        Definition def; // for Elided: what the pass reaches instead
    };
    const Cfg cfg(F);
    const DomTree dom = compute_dominator_tree(cfg);
    const BlockId at_id = cfg.id(at->name);
    const auto available = [&](const Definition &d) {
        if (!d.value) {
            return false;
        }
        if (d.block.empty()) {
            return true;
        }
        const BlockId b = cfg.find(d.block);
        return b != NO_BLOCK && dom.dominates(b, at_id);
    };
    vector<Plan> plans(sig.size());
    for (size_t i = 0; i < sig.size(); i++) {
        optional<Plan> plan;
        for (const auto &K : entries) {
            for (const auto &wp : K->preds) {
                const shared_ptr<Block> P = wp.lock();
                if (!P) {
                    continue;
                }
                const shared_ptr<Value> v = passed_along(*P, *K, i);
                internal_assert(v) << what << ": " << P->name << " reaches "
                                   << K->name << " without its argument " << i;
                const Definition d = defs.of(P->name, v);
                Plan here;
                here.type = sig[i].type;
                if (available(d)) {
                    here.kind = Kind::Elided;
                    here.def = d;
                } else if (const Instruction *local = as_local(d)) {
                    here.kind = Kind::Contents;
                    here.type = local->type.as<Ptr_t>()->etype;
                } else if (reducer_address(d)) {
                    here.kind = Kind::Address;
                } else {
                    internal_assert(!holds_address(sig[i].type))
                        << what << ": the continuation at " << K->name
                        << " takes " << sig[i].name << ", an address ("
                        << sig[i].type << ") that is neither in scope after the "
                        << "drains nor a mutable local of the iteration nor a "
                        << "reducer's; it would be stored, and might not "
                        << "outlive the frame.";
                    here.kind = Kind::Value;
                }
                if (!plan.has_value()) {
                    plan = here;
                } else if (plan->kind != here.kind ||
                           (plan->kind == Kind::Elided &&
                            !same_definition(plan->def, here.def))) {
                    // The edges disagree: stored, as a value or an address.
                    const bool addresses =
                        holds_address(sig[i].type) &&
                        plan->kind != Kind::Contents && here.kind != Kind::Contents;
                    internal_assert(!holds_address(sig[i].type) || addresses)
                        << what << ": the continuations take " << sig[i].name
                        << " as a mutable local's address at one place and not "
                        << "at another";
                    plan = Plan{addresses ? Kind::Address : Kind::Value, sig[i].type, {}};
                }
            }
        }
        internal_assert(plan.has_value())
            << what << ": nothing reaches the continuations";
        plans[i] = *plan;
    }

    // The entry and the queue, as any queue's: a struct of arrays behind a
    // count, sized as the queues this joins are, made where they were made.
    Struct_t::Map entry_fields;
    vector<Leaf> leaves;
    vector<std::pair<size_t, size_t>> field_leaves;
    vector<size_t> stored; // the arguments that have a field, in field order
    for (size_t i = 0; i < sig.size(); i++) {
        if (plans[i].kind == Kind::Elided) {
            continue;
        }
        string name = sig[i].name;
        for (bool clash = true; clash;) {
            clash = false;
            for (const TypedVar &f : entry_fields) {
                if (f.name == name) {
                    name += "_";
                    clash = true;
                }
            }
        }
        entry_fields.emplace_back(name, plans[i].type);
        const size_t first = leaves.size();
        leaves_of(name, plans[i].type, leaves);
        field_leaves.emplace_back(first, leaves.size());
        stored.push_back(i);
    }
    const Type entry_t = Struct_t::make("Entry_" + qname, entry_fields);
    Struct_t::Map queue_fields = {TypedVar("count", u32())};
    for (const Leaf &leaf : leaves) {
        queue_fields.emplace_back(leaf.name, Array_t::make(leaf.type, Expr()));
    }
    const Type queue_t = Struct_t::make("Queue_" + qname, queue_fields);
    made.push_back(entry_t);
    made.push_back(queue_t);
    vector<shared_ptr<Value>> stores;
    for (const Leaf &leaf : leaves) {
        stores.push_back(make_alloca(F, alloc_point,
                                     Array_t::make(leaf.type, as_expr(size)),
                                     qname + "_" + leaf.name));
    }
    const shared_ptr<Value> queue = make_alloca(F, alloc_point, queue_t, qname + "_queue");
    {
        vector<shared_ptr<Value>> parts = {constant_u32(0)};
        parts.insert(parts.end(), stores.begin(), stores.end());
        auto initial = alloc_point->make_instruction(queue_t, Instruction::Op::MakeStruct,
                                                     std::move(parts));
        alloc_point->make_side_effect(Instruction::Op::Store, {queue, initial});
    }

    // Each continuation's entry becomes a push of what it was handed, and
    // the end of the iteration that reached it.
    for (const auto &K : entries) {
        auto push = std::make_shared<Block>();
        push->name = K->name + "!push";
        push->owner = O;
        F.blocks.push_back(push);
        vector<shared_ptr<Value>> args;
        for (const Argument &a : K->args) {
            args.push_back(push->add_argument(a));
        }
        for (const auto &wp : K->preds) {
            if (const shared_ptr<Block> P = wp.lock()) {
                retarget(*P, K->name, push->name);
                push->preds.push_back(P);
            }
        }
        K->preds.clear();
        vector<shared_ptr<Value>> values;
        for (size_t i : stored) {
            values.push_back(plans[i].kind == Kind::Contents
                                 ? push->make_instruction(plans[i].type,
                                                          Instruction::Op::Load, {args[i]})
                                 : args[i]);
        }
        auto entry = push->make_instruction(entry_t, Instruction::Op::MakeStruct,
                                            std::move(values));
        auto slot = push->make_instruction(u32(), Instruction::Op::Push,
                                           {reach(push, queue), entry});
        std::get<shared_ptr<Instruction>>(slot->data)->atomic = true;
        push->terminator.data = Terminator::Yield{};
    }

    // The pass: from `at`, whose terminator moves to the pass's exit, over
    // every entry, running the first continuation (the others, now reached
    // by nothing, go); a relocated local is the pass's own, named so that
    // the continuation's loads find it (see the drain, which does the same).
    const shared_ptr<Block> K0 = entries.front();
    const auto fresh = [&](const string &name) {
        auto block = std::make_shared<Block>();
        block->name = name;
        for (const auto &other : F.blocks) {
            internal_assert(other->name != name)
                << what << ": " << F.blocks.front()->name
                << " already has a block named " << name;
        }
        block->owner = O;
        F.blocks.push_back(block);
        return block;
    };
    auto drain = fresh(qname + "!drain");
    auto body = fresh(qname + "!run");
    auto exit = fresh(qname + "!exit");
    exit->terminator.data = std::move(at->terminator.data);
    at->terminator.data = Terminator::Jump{drain->name};
    drain->preds = {at};
    {
        auto whole = drain->make_instruction(queue_t, Instruction::Op::Load,
                                             {reach(drain, queue)});
        auto pending = drain->make_instruction(u32(), Instruction::Op::LoadField,
                                               {whole, constant_u32(0)});
        drain->terminator.data = Terminator::ParFor{qname,
                                                    constant_u32(0),
                                                    pending,
                                                    constant_u32(1),
                                                    Terminator::Jump{body->name},
                                                    Terminator::Jump{exit->name}};
    }
    body->preds = {drain};
    exit->preds = {drain};
    auto index = body->add_argument(Argument{u32(), qname});
    const Emit emit = [&](const Type &t, Instruction::Op op,
                          vector<shared_ptr<Value>> ops) {
        return body->make_instruction(t, op, std::move(ops));
    };
    vector<shared_ptr<Value>> onwards(sig.size());
    for (size_t k = 0; k < stored.size(); k++) {
        const size_t i = stored[k];
        size_t l = field_leaves[k].first;
        auto value = rebuild(plans[i].type, emit, [&](const Type &t) {
            internal_assert(l < field_leaves[k].second)
                << what << ": " << sig[i].name << " has more scalars than "
                << qname << " stores for it";
            return body->make_instruction(t, Instruction::Op::ExtractIdx,
                                          {reach(body, stores[l++]), index});
        });
        if (plans[i].kind == Kind::Contents) {
            auto local = make_alloca(F, body, plans[i].type);
            body->make_side_effect(Instruction::Op::Store, {local, value});
            const string to = std::get<shared_ptr<Instruction>>(local->data)->name;
            for (const auto &block : region_from(F, K0->name)) {
                rename_argument(*block, sig[i].name, to);
            }
            value = local;
        }
        onwards[i] = value;
    }
    for (size_t i = 0; i < sig.size(); i++) {
        if (plans[i].kind == Kind::Elided) {
            onwards[i] = reach(body, plans[i].def.value);
        }
    }
    body->terminator.data = Terminator::Jump{K0->name, std::move(onwards)};
    K0->preds = {body};
}

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

    // A spawned call -- `spawn acc += callee(...)`, Terminator::Call::spawned
    // -- is deferred without waiting: the program has said that nothing
    // after the call depends on its value but the accumulate into the
    // reducer, so the call site pushes and goes on, the chain returns what
    // it always returned, and the drain makes the call and the accumulate.
    // pbrt's shadow ray, traced by its own kernel and added to the pixel's
    // L when it is. The other pushers a schedule may name are for a
    // recursion's first call, which a spawned call is not.
    size_t spawned_sites = 0;
    for (const auto &site : sites) {
        if (std::get<Terminator::Call>(site->terminator.data).spawned) {
            spawned_sites++;
        }
    }
    internal_assert(spawned_sites == 0 || spawned_sites == sites.size())
        << what << ": " << func_name << " calls " << callee_name << " "
        << sites.size() << " times, of which " << spawned_sites
        << " are spawned (`spawn acc += " << callee_name
        << "(...)`); a deferral is of all the calls or of none";
    const bool spawned = spawned_sites > 0;
    internal_assert(!spawned || (queue.also_from.empty() && !queue.initial_push))
        << what << ": the calls to " << callee_name << " are spawned, and "
        << "another function's `.defer(" << callee_name << ", " << queue.name
        << ")` names it as a pusher too. The other pushers of a queue are "
        << "for a recursion's first call; a spawned call's pushers are the "
        << "spawned calls.";
    internal_assert(!spawned || !queue.split.has_value())
        << "[unimplemented] " << what << ": " << queue.name
        << " is split (`" << queue.name << ".specialize(...)`) and its "
        << "entries are spawned calls; a split is built for a staged call's "
        << "queue and a spawned call's queue is drained in one pass, so the "
        << "two would fit -- not built.";

    // The calls to the callee from the other functions the schedule named,
    // pushes too (QueueSpec::also_from): the recursion's first call, made by
    // a chain function between the owner and the callee.
    vector<string> site_function(sites.size(), func_name);
    for (const string &g : queue.also_from) {
        internal_assert(funcs.contains(g))
            << what << ": " << g << " is not a function of the program";
        internal_assert(g != func_name && g != callee_name)
            << what << ": " << g << " is the deferred callee";
        const BlockMap gmap = make_block_map(funcs.at(g));
        size_t found = 0;
        for (const auto &block : funcs.at(g)->blocks) {
            const auto *call = std::get_if<Terminator::Call>(&block->terminator.data);
            if (call == nullptr || call->call.name != callee_name) {
                continue;
            }
            internal_assert(is_tail_call(*call, gmap))
                << what << ": the call to " << callee_name << " in "
                << block->name << " of " << g << " is not in tail position; "
                << "a call pushed onto a queue has to be the last thing its "
                << "function does.";
            sites.push_back(block);
            site_function.push_back(g);
            found++;
        }
        internal_assert(found > 0)
            << g << ".defer(" << callee_name << ", " << queue.name << "): "
            << g << " makes no call to " << callee_name;
    }

    const BlockMap fmap = make_block_map(F);
    for (size_t s = 0; s < sites.size(); s++) {
        if (site_function[s] != func_name) {
            continue;
        }
        const auto &site = sites[s];
        const auto &call = std::get<Terminator::Call>(site->terminator.data);
        internal_assert(spawned || is_tail_call(call, fmap))
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

    // A spawned call's continuation: the accumulate of the call's value into
    // a reducer parameter of the function, first, and then whatever the
    // function goes on to do, which does not read the value (the parser
    // writes the statement so; checked here rather than trusted). The
    // accumulate moves to the drain, so the entry holds the reducer's
    // address beside the call's arguments -- which is safe to store because
    // a reducer parameter is storage of the queue's owner, handed down, and
    // outlives every frame; a reducer local of this function would not.
    struct Spawn {
        Instruction::Op op = Instruction::Op::AccAdd;
        bool atomic = false;
        shared_ptr<Value> target; // the reducer's address, as the site has it
        const Argument *reducer = nullptr; // the parameter it is
    };
    vector<Spawn> spawns(sites.size());
    if (spawned) {
        Definitions fdefs(*F);
        const Cfg fcfg(*F);
        const LoopForest floops =
            compute_loop_forest(fcfg, compute_dominator_tree(fcfg));
        for (size_t s = 0; s < sites.size(); s++) {
            const auto &site = sites[s];
            const auto &call = std::get<Terminator::Call>(site->terminator.data);
            const shared_ptr<Block> cont = fmap.at(call.cont.name);
            internal_assert(!call.drop && cont->preds.size() == 1 &&
                            !cont->args.empty() && !cont->instrs.empty())
                << what << ": the spawned call in " << site->name
                << " has no value, or a continuation that is not its own";
            const Argument &result = cont->args.front();
            const shared_ptr<Instruction> &acc = cont->instrs.front();
            const bool accumulates =
                acc->op == Instruction::Op::AccAdd || acc->op == Instruction::Op::AccMul ||
                acc->op == Instruction::Op::AccSub || acc->op == Instruction::Op::AccMin ||
                acc->op == Instruction::Op::AccMax;
            const auto *added = acc->operands.size() == 2
                                    ? std::get_if<Argument>(&acc->operands[1]->data)
                                    : nullptr;
            internal_assert(accumulates && added != nullptr && added->name == result.name)
                << what << ": the spawned call in " << site->name
                << " is not followed by the accumulate of its value that "
                << "`spawn` writes";
            // Once per run of the function, since the queue holds one entry
            // per run of the entry that reaches it: the call is in no loop
            // of the function, and the function does not call itself after
            // it (checked with the chain, below).
            internal_assert(floops.innermost(fcfg.id(site->name)) == nullptr)
                << what << ": the spawned call in " << site->name
                << " is inside a loop of " << func_name << ", and would push "
                << "once per iteration; the queue holds one entry per run of "
                << "the function, as every entry that runs it pushes at most "
                << "one.";
            const Argument *reducer = fdefs.parameter(cont->name, acc->operands[0]);
            internal_assert(reducer != nullptr && reducer->reducer)
                << what << ": the `spawn` in " << func_name
                << " accumulates into something that is not a reducer "
                << "parameter of " << func_name << ". The drain adds the "
                << "call's value where the reducer is, once this function's "
                << "frame is gone: a reducer parameter is the queue owner's "
                << "storage, handed down, and outlives the frame; a local of "
                << "this function does not.";
            // Nothing else reads the value: no operand or terminator of the
            // function's but the accumulate names it (edit_argument visits
            // those, and a block's own parameter list and lookup, which are
            // taken off the count), and no block below takes it as a
            // parameter, which is how a value is threaded on to a reader.
            size_t readers = 0;
            for (const auto &block : F->blocks) {
                size_t visits = 0;
                edit_argument(*block, result.name, [&](Argument &) { visits++; });
                for (const Argument &a : block->args) {
                    if (a.name == result.name) {
                        internal_assert(block == cont)
                            << what << ": the value of the spawned call in "
                            << site->name << " is threaded on to " << block->name
                            << ", so something below reads it; `spawn` promises "
                            << "the accumulate is its only reader";
                        visits--;
                    }
                }
                if (block->lookups.contains(result.name)) {
                    visits--;
                }
                readers += visits;
            }
            internal_assert(readers == 1)
                << what << ": the value of the spawned call in " << site->name
                << " is read " << readers - 1 << " time(s) beyond its "
                << "accumulate; `spawn` promises the accumulate is its only "
                << "reader";
            spawns[s] = Spawn{acc->op, acc->atomic, acc->operands[0], reducer};
        }
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
    // own call, in the owner, is the one frame allowed to go on. Not for a
    // spawned call: nothing comes back up -- the chain only hands the queue
    // down -- so a chain function may do what it likes after its call. What
    // it may not do is reach the chain twice in one run: the queue holds
    // one entry per run of each entry that reaches it, so a call into the
    // chain is in no loop of its function, and no path from its continuation
    // reaches another such call, the function's call of itself included.
    if (!spawned) {
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
    } else {
        for (const string &g : chain) {
            const shared_ptr<Function> gf = funcs.at(g);
            const BlockMap gmap = make_block_map(gf);
            const Cfg gcfg(*gf);
            const LoopForest gloops =
                compute_loop_forest(gcfg, compute_dominator_tree(gcfg));
            const auto into_chain = [&](const Block &block) {
                const auto *callee = block.terminator.callee();
                return callee != nullptr &&
                       (chain.contains(callee->name) || callee->name == callee_name);
            };
            set<string> reachable;
            for (BlockId b : gcfg.rpo) {
                reachable.insert(gcfg.name(b));
            }
            for (const auto &block : gf->blocks) {
                if (!into_chain(*block) || !reachable.contains(block->name)) {
                    continue;
                }
                internal_assert(gloops.innermost(gcfg.id(block->name)) == nullptr)
                    << what << ": " << g << " calls into the chain in "
                    << block->name << ", inside one of its loops, and so "
                    << "could push more than one entry per run; the queue "
                    << "holds one per run of the entry that reaches it.";
                // Every block a path from the continuation reaches.
                vector<string> pending = successors(*block);
                set<string> seen;
                while (!pending.empty()) {
                    const string next = pending.back();
                    pending.pop_back();
                    if (!seen.insert(next).second) {
                        continue;
                    }
                    const Block &there = *gmap.at(next);
                    internal_assert(!into_chain(there))
                        << what << ": " << g << " calls "
                        << block->terminator.callee()->name << " in "
                        << block->name << " and " << there.terminator.callee()->name
                        << " in " << there.name << " on a path after it, both "
                        << "on the way to " << callee_name << ", and so could "
                        << "push two entries per run; the queue holds one per "
                        << "run of the entry that reaches it.";
                    for (const string &succ : successors(there)) {
                        pending.push_back(succ);
                    }
                }
            }
        }
    }

    // What flows back up the chain is one type: the callee's return type.
    // For a spawned call nothing flows up; the value is accumulated by the
    // drain, so there has to be one.
    const Type ret_type = C->ret_type;
    if (!spawned) {
        for (const string &g : chain) {
            internal_assert(equals(funcs.at(g)->ret_type, ret_type))
                << what << ": " << g << " returns " << funcs.at(g)->ret_type
                << " where " << callee_name << " returns " << ret_type
                << ", yet its call down the chain is a tail call. This should "
                << "not typecheck.";
        }
    }
    const bool returns_value = !ret_type.is<Void_t>();
    internal_assert(!spawned || returns_value)
        << what << ": the spawned " << callee_name << " returns nothing to "
        << "accumulate";

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
    internal_assert(!producers.empty())
        << what << ": " << queue.owner << " never calls into the chain";
    // Several producers only for a spawned call: the drain runs no
    // continuation of theirs, so each is only a call that hands the queue
    // down, and the drain goes after the last of them.
    internal_assert(spawned || producers.size() == 1)
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
        string variants;
        for (const auto &block : O->blocks) {
            const auto *p =
                std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p != nullptr && p->index.rfind(queue.loop + "!", 0) == 0) {
                variants += (variants.empty() ? "" : ", ") + p->index;
            }
        }
        internal_assert(owner_loop_block)
            << what << ": " << queue.owner << " has no parfor named "
            << queue.loop << " to own the queue " << queue.name
            << ". A queue is owned by a parfor of its function, or by `root`."
            << (variants.empty()
                    ? ""
                    : " " + queue.owner + " is specialized, and its loops are "
                          "the variants' (" + variants + "): a queue is one "
                          "variant's, `" + queue.owner + "[<Variant>].queue(" +
                          queue.loop + ")`.");
    }
    const Cfg region(*O, region_entry);
    for (const CallSite &pc : producers) {
        internal_assert(region.contains(pc.block->name))
            << what << ": the call to " << pc.call()->call.name << " in "
            << pc.block->name << " of " << queue.owner << " is not inside "
            << (root ? "the function" : "the loop " + queue.loop)
            << " that owns " << queue.name;
    }

    // A sequential loop between the owner and the producer would have its
    // iterations reordered: an iteration's deferred work runs after the
    // loop, while the next iteration's own work runs before it. Only a
    // parfor permits that. A loop around the owner's own loop is another
    // matter: each of its iterations allocates and drains its own queue.
    {
        const DomTree rdom = compute_dominator_tree(region);
        const LoopForest rloops = compute_loop_forest(region, rdom);
        for (const CallSite &pc : producers) {
            internal_assert(rloops.innermost(region.id(pc.block->name)) == nullptr)
                << what << ": the call to " << pc.call()->call.name << " in "
                << pc.block->name << " is inside a sequential loop of "
                << queue.owner << ". Deferring it would run the loop's "
                << "iterations out of order, which a `for` does not permit; a "
                << "`parfor` would.";
        }
    }

    // The producer loop: the one parfor of the region whose body holds the
    // call, if any. Nested parfors would multiply the count; not yet. One
    // per producer.
    vector<shared_ptr<Block>> producer_loops;
    for (const CallSite &pc : producers) {
        shared_ptr<Block> loop;
        for (BlockId b : region.rpo) {
            const shared_ptr<Block> &block = region.block(b);
            const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p == nullptr) {
                continue;
            }
            const Cfg body(*O, p->body.name);
            if (!body.contains(pc.block->name)) {
                continue;
            }
            internal_assert(!loop)
                << what << ": the call to " << pc.call()->call.name << " in "
                << pc.block->name << " is inside two nested parfors of "
                << queue.owner << " (" << loop->name << " and " << block->name
                << "). The queue's size would be the product of their counts; "
                << "only one producer loop is supported yet.";
            loop = block;
        }
        producer_loops.push_back(loop);
    }
    shared_ptr<Block> producer_loop_block = producer_loops.front();

    // Where the drain goes, and what has to dominate it: the block whose
    // parfor produces the entries, or the block that makes the one call.
    // With several producers, the last of their loops: the one every other
    // dominates, so that every push has been made by the time the drain
    // runs -- pbrt's shadow rays, pushed by the material kernels in turn
    // and traced after the last of them.
    shared_ptr<Block> point =
        producer_loop_block ? producer_loop_block : producer.block;
    if (producers.size() > 1) {
        string named;
        for (size_t i = 0; i < producers.size(); i++) {
            internal_assert(producer_loops[i])
                << what << ": the call to " << producers[i].call()->call.name
                << " in " << producers[i].block->name << " is not inside a "
                << "parfor of " << queue.owner << ", while another call into "
                << "the chain is; with several producers the drain goes after "
                << "the last of their loops, so each has to be in one.";
            named += (named.empty() ? "" : ", ") + producer_loops[i]->name;
        }
        for (const auto &loop : producer_loops) {
            if (odom.dominates(ocfg.id(point->name), ocfg.id(loop->name))) {
                point = loop;
            }
        }
        for (const auto &loop : producer_loops) {
            internal_assert(odom.dominates(ocfg.id(loop->name), ocfg.id(point->name)))
                << what << ": the loops of " << queue.owner << " that call into "
                << "the chain (" << named << ") do not run one after another: "
                << "none comes after all the others, so there is no one place "
                << "for the drain to run once every push has been made.";
        }
        producer_loop_block = point;
    }
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
    // A spawned call's drain runs none of it, so it is looked at only for
    // a lone producer, whose iteration's end is where the drain goes.
    Definitions odefs(*O);
    shared_ptr<Block> k0_entry;
    vector<shared_ptr<Block>> k0;
    const size_t result_args = pcall.drop ? 0 : 1;
    if (!spawned || !producer_loop_block) {
        k0_entry = omap.at(pcall.cont.name);
        internal_assert(k0_entry->preds.size() == 1)
            << what << ": the continuation " << k0_entry->name << " of the call in "
            << producer.block->name << " has " << k0_entry->preds.size()
            << " predecessors";
        internal_assert(k0_entry->args.size() == pcall.cont.args.size() + result_args)
            << k0_entry->name << " takes " << k0_entry->args.size()
            << " arguments but its call passes " << pcall.cont.args.size();
        k0 = region_from(*O, k0_entry->name);
    }
    for (const auto &block : k0) {
        const auto *callee = block->terminator.callee();
        internal_assert(spawned || callee == nullptr || !chain.contains(callee->name))
            << what << ": after the call to " << pcall.call.name << " in "
            << producer.block->name << ", " << queue.owner << " calls "
            << callee->name << " again, in " << block->name
            << ". Two producer calls per iteration are not supported.";
        internal_assert(
            spawned ||
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
    // Why a parameter is not invariant, when it is not: the two values that
    // disagreed, for the message of a refusal that rests on it.
    map<string, map<size_t, string>> disagreement;
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
                const CallSite *considering = nullptr;
                const auto consider = [&](const Definition &d) {
                    if (!agreed.has_value()) {
                        agreed = d;
                    } else if (!same_definition(*agreed, d)) {
                        ok = false;
                        // For the message, when this ends in a refusal.
                        const auto shown = [](const Definition &x) {
                            std::ostringstream os;
                            if (x.value) {
                                std::visit(overloads{
                                               [&](const Constant &c) { c.dump(os); },
                                               [&](const Argument &a) { os << a.name; },
                                               [&](const shared_ptr<Instruction> &i) {
                                                   os << i->name;
                                               },
                                           },
                                           x.value->data);
                            }
                            if (!x.block.empty()) {
                                os << " of " << x.block;
                            }
                            return os.str();
                        };
                        disagreement[g][j] =
                            shown(*agreed) + " at one call and " + shown(d) +
                            " at the call in " + considering->block->name +
                            " of " + considering->caller;
                    }
                };
                for (const CallSite &cs : calls_into[g]) {
                    considering = &cs;
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
                            disagreement[g][j] =
                                "handed " + arg->name + ", a parameter of " +
                                cs.caller + " that is itself not one value at "
                                "every call" +
                                (disagreement[cs.caller].contains(k)
                                     ? " (" + disagreement[cs.caller][k] + ")"
                                     : "");
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
                        {
                            std::ostringstream os;
                            if (d.value) {
                                std::visit(overloads{
                                               [&](const Constant &c) { c.dump(os); },
                                               [&](const Argument &a) { os << a.name; },
                                               [&](const shared_ptr<Instruction> &i) {
                                                   os << i->name;
                                               },
                                           },
                                           d.value->data);
                            }
                            disagreement[g][j] =
                                "handed " + os.str() + " by the call in " +
                                cs.block->name + " of " + cs.caller +
                                ", a value of that function's own (defined in " +
                                d.block + ") that cannot be computed before the "
                                "producer loop";
                        }
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
        if (param.reducer) {
            // A reducer is storage every continuation of the iteration adds
            // into, in this queue and any other: its address is stored,
            // never its contents. A local of the producer's iteration is
            // given a slot in an array made beside the queues (below, once
            // the queue's size is known) and the field holds that slot's
            // address; a reducer the producer was itself handed is the
            // address it was given. Either outlives every frame, which is
            // what makes the address safe to store.
            internal_assert(inv)
                << what << ": the reducer " << param.name << " of "
                << callee_name << " is given different storage at different "
                << "calls; a reducer is one place for the whole iteration";
            Field f;
            f.name = field_named(param.name);
            f.type = param.type;
            f.param = j;
            f.origin = org;
            f.reducer_local = as_local(org);
            f.reducer_address = true;
            params[j] = {Where::Stored, fields.size()};
            fields.push_back(std::move(f));
            continue;
        }
        // A spawned call's continuation goes on at once, so nothing the
        // callee could write is the caller's any more: a spawned call takes
        // values, and reducers.
        internal_assert(!spawned || !param.mutating)
            << what << ": the spawned " << callee_name << " takes " << param.name
            << " as `mut`; the call runs later, and what it wrote would reach "
            << "no one.";
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
            << "supported."
            << (disagreement[callee_name].contains(j)
                    ? " It is " + disagreement[callee_name][j] + "."
                    : invariant.at(callee_name)[j]
                          ? " It is one value down the chain, but computed inside "
                            "the producer's iteration."
                          : "");
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

    // A spawned call's accumulate: the reducer's address goes in the entry
    // beside the arguments, for the drain to add the value into. A reducer
    // is one place for the whole iteration, so the address is one of: a
    // local of the owner's iteration, given a slot beside the queues like
    // any reducer local (the hoist below); a slot an earlier deferral made,
    // read out of its entry (Function::reducer_slots); or a reducer the
    // owner was itself handed. The first is found when every call down the
    // chain hands the same local on; when the calls disagree -- the spawned
    // function is reached from two drains, each handing on what its entry
    // held -- each route is followed up to the owner to see that it starts
    // from storage that outlives the frames.
    optional<size_t> acc_field;
    if (spawned) {
        const Argument *reducer = spawns.front().reducer;
        for (const Spawn &sp : spawns) {
            internal_assert(sp.reducer->name == reducer->name)
                << "[unimplemented] " << what << ": the spawned calls to "
                << callee_name << " accumulate into different reducers ("
                << reducer->name << ", " << sp.reducer->name << ")";
        }
        const Block &fentry = *F->blocks.front();
        size_t k = fentry.args.size();
        for (size_t i = 0; i < fentry.args.size(); i++) {
            if (fentry.args[i].name == reducer->name) {
                k = i;
            }
        }
        internal_assert(k < fentry.args.size()) << what << ": " << reducer->name;
        Field f;
        f.name = field_named(reducer->name);
        f.type = reducer->type;
        f.reducer_address = true;
        const bool inv = invariant.at(func_name)[k] &&
                         origin.at(func_name)[k].value != nullptr;
        if (inv) {
            f.origin = settle(origin.at(func_name)[k]);
            f.reducer_local = as_local(f.origin);
        } else {
            // Every route from the owner: what it hands the parameter.
            set<std::pair<string, size_t>> seen;
            const std::function<void(const string &, size_t)> check =
                [&](const string &g, size_t j) {
                    if (!seen.insert({g, j}).second) {
                        return;
                    }
                    for (const CallSite &cs : calls_into[g]) {
                        const shared_ptr<Value> &arg = cs.call()->call.args[j];
                        if (cs.caller == queue.owner) {
                            const Definition d = odefs.of(cs.block->name, arg);
                            const auto *di = std::get_if<shared_ptr<Instruction>>(&d.value->data);
                            const auto *da = std::get_if<Argument>(&d.value->data);
                            const bool slot = di != nullptr && O->reducer_slots.contains(di->get());
                            const bool handed = da != nullptr && d.block == oentry;
                            internal_assert(slot || handed)
                                << what << ": the reducer " << reducer->name
                                << " of " << func_name << " is given different "
                                << "storage by different calls, and the call in "
                                << cs.block->name << " of " << queue.owner
                                << " hands the chain storage that is not a "
                                << "reducer slot's or the owner's own parameter; "
                                << "the address would be stored in the queue and "
                                << "might not outlive the frame.";
                            continue;
                        }
                        const Argument *up = defs.at(cs.caller).parameter(cs.block->name, arg);
                        internal_assert(up != nullptr)
                            << what << ": " << cs.caller << " hands " << g
                            << " a reducer that is not one of its parameters, in "
                            << cs.block->name;
                        const Block &uentry = *funcs.at(cs.caller)->blocks.front();
                        size_t uk = uentry.args.size();
                        for (size_t i = 0; i < uentry.args.size(); i++) {
                            if (uentry.args[i].name == up->name) {
                                uk = i;
                            }
                        }
                        internal_assert(uk < uentry.args.size()) << up->name;
                        check(cs.caller, uk);
                    }
                };
            check(func_name, k);
        }
        acc_field = fields.size();
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
    for (size_t i = 0; !spawned && i < pcall.cont.args.size(); i++) {
        const Definition d =
            settle(odefs.of(producer.block->name, pcall.cont.args[i]));
        const Argument &arg = k0_entry->args[i + result_args];
        if (available(d)) {
            carried.push_back({From::Available, 0, d});
            continue;
        }
        // A reducer's address is stored as the address it is: the slot the
        // callee's parameter field holds, when the callee is handed the same
        // local; or a slot an earlier deferral made, which outlives every
        // frame (Function::reducer_slots).
        bool reducer_address = false;
        if (const Instruction *local = as_local(d)) {
            size_t f = fields.size();
            for (size_t k = 0; k < fields.size(); k++) {
                if (fields[k].reducer_local == local) {
                    f = k;
                }
            }
            if (f != fields.size()) {
                carried.push_back({From::Field, f, d});
                continue;
            }
        } else if (const auto *di = std::get_if<shared_ptr<Instruction>>(&d.value->data);
                   di != nullptr && O->reducer_slots.contains(di->get())) {
            reducer_address = true;
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
        internal_assert(!holds_address(arg.type) || reducer_address)
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
            field.reducer_address = reducer_address;
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
    // Where the queue's storage is made: before the producer loop, or --
    // when that loop is another queue's drain, inside its rounds -- where
    // the drained queue's storage was made, before the rounds.
    shared_ptr<Block> alloc_point = point;
    // Whether the producer loop is another queue's drain, and if so the
    // block after that queue's rounds (Function::OwnedQueue::after).
    bool producer_is_drain = false;
    string after_rounds;
    {
        const size_t at = point->instrs.size();
        const auto *producer_loop =
            producer_loop_block
                ? &std::get<Terminator::ParFor>(producer_loop_block->terminator.data)
                : nullptr;
        const auto drained =
            producer_loop ? O->queue_sizes.find(producer_loop->index)
                          : O->queue_sizes.end();
        if (drained != O->queue_sizes.end()) {
            // The producer loop is another queue's drain (a drain's parfor
            // is named after its queue): a round's count is known only at
            // run time, but each entry it runs pushes at most one here, so
            // that queue's capacity bounds this one -- pbrt's material queue
            // sized as its ray queue is.
            count = drained->second.size;
            alloc_point = omap.at(drained->second.point);
            producer_is_drain = true;
            after_rounds = drained->second.after;
        } else if (producer_loop_block) {
            const auto &p = *producer_loop;
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
    O->queue_sizes[queue.name] =
        Function::OwnedQueue{size, alloc_point->name, after_rounds};

    // A reducer local of the producer's iteration: a slot per iteration in
    // an array made where the queue's storage is, and the slot's address in
    // place of the local, so that every continuation of the iteration -- in
    // this queue and in any other -- adds into the one place (pbrt's
    // pixelSampleState.L). The local's initializing store becomes the
    // slot's; the field stores the address.
    for (Field &f : fields) {
        if (f.reducer_local == nullptr) {
            continue;
        }
        const Instruction *local = f.reducer_local;
        const shared_ptr<Block> home = local->owner.lock();
        internal_assert(home) << what << ": the reducer " << f.name
                              << " outlived its block";
        const Type etype = local->type.as<Ptr_t>()->etype;
        auto slots = make_alloca(*O, alloc_point,
                                 Array_t::make(etype, as_expr(size)),
                                 queue.name + "_" + f.name + "_slots");
        shared_ptr<Value> index = constant_u32(0);
        if (producer_loop_block) {
            // The iteration's slot is its index in the producer loop, the
            // loop body's first argument.
            const auto &p = std::get<Terminator::ParFor>(producer_loop_block->terminator.data);
            const shared_ptr<Block> pbody = omap.at(p.body.name);
            internal_assert(!pbody->args.empty()) << what << ": " << pbody->name
                                                  << " has no index";
            index = reach(home, std::make_shared<Value>(pbody->args.front()));
        }
        size_t at = home->instrs.size();
        for (size_t i = 0; i < home->instrs.size(); i++) {
            if (home->instrs[i].get() == local) {
                at = i + 1;
            }
        }
        internal_assert(at <= home->instrs.size())
            << what << ": " << f.name << " is not in " << home->name;
        auto slot = std::make_shared<Instruction>(
            O->get_unique_name(), local->type, Instruction::Op::GEP,
            vector<shared_ptr<Value>>{reach(home, slots), index}, home);
        home->instrs.insert(home->instrs.begin() + at, slot);
        auto slot_value = std::make_shared<Value>(slot);
        home->lookups[slot->name] = slot_value;
        replace_uses(*O, local, slot_value);
        O->reducer_slots.insert(slot.get());
        f.origin = Definition{slot_value, home->name};
        f.reducer_local = nullptr;
    }

    //===----------------------------------------------------------------===//
    // The types
    //===----------------------------------------------------------------===//

    // The entry is the continuation as a value, which is what the callee
    // pushes; the queue stores it as a struct of arrays, one per scalar of it
    // (see Leaf), behind the count.
    Struct_t::Map entry_fields;
    for (const Field &f : fields) {
        entry_fields.emplace_back(f.name, f.type);
    }
    const Type entry_t = Struct_t::make("Entry_" + queue.name, entry_fields);
    vector<Leaf> leaves;
    vector<std::pair<size_t, size_t>> field_leaves; // each field's [first, last)
    for (const Field &f : fields) {
        const size_t first = leaves.size();
        leaves_of(f.name, f.type, leaves);
        field_leaves.emplace_back(first, leaves.size());
    }
    Struct_t::Map queue_fields = {TypedVar("count", u32())};
    for (const Leaf &leaf : leaves) {
        for (const TypedVar &other : queue_fields) {
            internal_assert(other.name != leaf.name)
                << what << ": two scalars of " << queue.name << "'s entry are "
                << "both named " << leaf.name << " once their fields' names "
                << "and their paths are joined";
        }
        queue_fields.emplace_back(leaf.name, Array_t::make(leaf.type, Expr()));
    }
    const Type queue_t = Struct_t::make("Queue_" + queue.name, queue_fields);

    // The queues a split makes of this one (QueueSpec::split): a leaf of
    // the split's tree per path of variants, each with a copy of the callee
    // in which the keys on its path have their variants' tags (SSA/
    // Specialize.h), named by the path -- `hits!Some!Diffuse`, its drain
    // loop of the same name. One entry, the queue itself, when there is no
    // split. The callee the schedule named stays for what else calls it.
    struct SubQueue {
        string path;
        string callee;
    };
    vector<SubQueue> subqueues;
    const bool split = queue.split.has_value();
    if (split) {
        internal_assert(!callee_in_chain)
            << "[unimplemented] " << what << ": " << queue.name
            << " is split (`" << queue.name << ".specialize(...)`) and is a "
            << "self-feeding queue; a split is built for a queue drained in "
            << "one pass -- a stage's";
        internal_assert(queue.adt_storages != nullptr)
            << what << ": no variant storages for the split's keys";
        const std::function<void(const QueueSpec::Split *, const string &,
                                 const string &)>
            grow = [&](const QueueSpec::Split *node, const string &path,
                       const string &callee) {
                if (node == nullptr || node->key.empty()) {
                    subqueues.push_back(SubQueue{path, callee});
                    return;
                }
                const shared_ptr<Function> cf = funcs.at(callee);
                const shared_ptr<Value> kv = key_value(*cf, node->key);
                internal_assert(kv) << what << ": " << callee
                                    << " has no value named " << node->key
                                    << " to split " << path << " on";
                const vector<KeyVariant> variants =
                    key_variants(value_type(*kv), *queue.adt_storages);
                internal_assert(!variants.empty())
                    << what << ": " << node->key << " in " << callee << " is a "
                    << value_type(*kv) << ", not a variant type or an optional";
                for (const auto &[label, _] : node->under) {
                    internal_assert(std::any_of(variants.begin(), variants.end(),
                                                [&](const KeyVariant &v) {
                                                    return v.label == label;
                                                }))
                        << what << ": " << path << "[" << label << "]: "
                        << node->key << " has no variant " << label;
                }
                for (const KeyVariant &v : variants) {
                    const string sub_callee = callee + "!" + v.label;
                    funcs[sub_callee] = specialize_function(
                        *cf, sub_callee, node->key, v, *queue.adt_storages);
                    // And into the callees the copy hands the value to --
                    // `material_bxdf(material, ...)`, where the material's
                    // match is -- so that the copy is one variant's program
                    // all the way down, as a loop's specialization is (SSA/
                    // Specialize.h): pbrt's material kernel evaluates one
                    // material.
                    specialize_callees(funcs, funcs.at(sub_callee)->blocks,
                                       node->key + "!" + v.label, v,
                                       *queue.adt_storages);
                    const auto child = node->under.find(v.label);
                    grow(child == node->under.end() ? nullptr : &child->second,
                         path + "!" + v.label, sub_callee);
                }
            };
        grow(&*queue.split, queue.name, callee_name);
    } else {
        subqueues.push_back(SubQueue{queue.name, callee_name});
    }
    // Which of the split's queues a path names.
    const auto subqueue_index = [&](const string &path) {
        for (size_t q = 0; q < subqueues.size(); q++) {
            if (subqueues[q].path == path) {
                return q;
            }
        }
        internal_error << what << ": no queue at " << path;
        return size_t(0);
    };
    // Each queue of the split is sized and placed as the queue is, and a
    // later deferral whose producer is one of their drains finds it by the
    // drain's name (Function::queue_sizes).
    for (const SubQueue &sq : subqueues) {
        O->queue_sizes[sq.path] = O->queue_sizes.at(queue.name);
    }

    // What the callee is handed to push onto: the queue's address, or the
    // split's whole family, which the push indexes by the variants it finds.
    const Type queue_ptr_t =
        split ? Array_t::make(queue_t, UIntImm::make(u32(), subqueues.size()))
              : Ptr_t::make(queue_t);
    // The address of one queue, wherever one is picked out of several.
    const Type queue_addr_t = Ptr_t::make(queue_t);
    vector<Type> made = {entry_t, queue_t};

    // What a chain function returns now: whether it saved its state; which
    // slot it saved it in, when the frame has state of its own to add there;
    // and the value it would have returned.
    Struct_t::Map saved_fields = {TypedVar("saved", Bool_t::make())};
    const size_t slot_index = frame_saves ? saved_fields.size() : 0;
    if (frame_saves) {
        saved_fields.emplace_back("slot", u32());
    }
    // Which queue of a split the slot is in, for the frame to write its
    // part there.
    const bool saved_names_queue = frame_saves && split;
    const size_t queue_index = saved_names_queue ? saved_fields.size() : 0;
    if (saved_names_queue) {
        saved_fields.emplace_back("queue", u32());
    }
    const size_t value_index = returns_value ? saved_fields.size() : 0;
    if (returns_value) {
        saved_fields.emplace_back("value", ret_type);
    }
    const bool saved_is_struct = saved_fields.size() > 1;
    const Type saved_t = saved_is_struct
                             ? Struct_t::make("Saved_" + callee_name, saved_fields)
                             : Bool_t::make();
    if (saved_is_struct && !spawned) {
        made.push_back(saved_t);
    }
    const auto make_saved = [&](const shared_ptr<Block> &block, bool saved,
                                shared_ptr<Value> slot, shared_ptr<Value> value,
                                shared_ptr<Value> which = nullptr) -> shared_ptr<Value> {
        if (!saved_is_struct) {
            return constant_bool(saved);
        }
        vector<shared_ptr<Value>> parts = {constant_bool(saved)};
        if (frame_saves) {
            parts.push_back(slot ? std::move(slot) : undef_value(u32()));
        }
        if (saved_names_queue) {
            parts.push_back(which ? std::move(which) : undef_value(u32()));
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
    // The queue the slot is in: the split's, or the one there is.
    const auto queue_of_saved = [&](const shared_ptr<Block> &block,
                                    const shared_ptr<Value> &s) -> shared_ptr<Value> {
        if (!saved_names_queue) {
            return constant_u32(0);
        }
        return block->make_instruction(u32(), Instruction::Op::LoadField,
                                       {s, constant_u32(queue_index)});
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
        if (!spawned) {
            gf.ret_type = saved_t;
        }
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
            if (spawned) {
                continue; // the queue goes down; nothing comes up
            }
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
        if (spawned) {
            break; // the chain returns what it returned
        }
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

    // The push of `entry`, made in `block` of `fn`, onto what `base` names:
    // the queue, or the split's family, of which the entry's keys select
    // one -- each split's key computed here from `params`, the values the
    // callee's parameters have at this push (SSA/Specialize.h, key_tag_at),
    // and dispatched on, a block per variant, down to the leaf that pushes.
    // A key below a variant is computed only under it, since it may be
    // defined only there (the material of a hit, under `Some`). `finish`
    // ends every block that pushed, given its slot and which queue that is
    // in.
    const auto push_entry =
        [&](const shared_ptr<Function> &fn, const shared_ptr<Block> &block,
            const shared_ptr<Value> &base, const shared_ptr<Value> &entry,
            const map<string, shared_ptr<Value>> &params,
            const std::function<void(const shared_ptr<Block> &,
                                     const shared_ptr<Value> &,
                                     const shared_ptr<Value> &)> &finish) {
            const auto push_onto = [&](const shared_ptr<Block> &at,
                                       const shared_ptr<Value> &q, size_t which) {
                auto slot = at->make_instruction(u32(), Instruction::Op::Push,
                                                 {q, reach(at, entry)});
                std::get<shared_ptr<Instruction>>(slot->data)->atomic = true;
                finish(at, slot, constant_u32(which));
            };
            if (!split) {
                push_onto(block, reach(block, base), 0);
                return;
            }
            std::function<void(const shared_ptr<Block> &, const QueueSpec::Split *,
                               const string &, const string &)>
                descend = [&](const shared_ptr<Block> &at,
                              const QueueSpec::Split *node, const string &path,
                              const string &callee) {
                    if (node == nullptr || node->key.empty()) {
                        const size_t which = subqueue_index(path);
                        auto q = at->make_instruction(
                            queue_addr_t, Instruction::Op::GEP,
                            {reach(at, base), constant_u32(which)});
                        push_onto(at, q, which);
                        return;
                    }
                    const Function &cf = *funcs.at(callee);
                    map<string, shared_ptr<Value>> here;
                    for (const auto &[pname, pvalue] : params) {
                        here[pname] = reach(at, pvalue);
                    }
                    auto tag = key_tag_at(*at, cf, node->key, here,
                                          *queue.adt_storages);
                    const vector<KeyVariant> variants = key_variants(
                        value_type(*key_value(cf, node->key)), *queue.adt_storages);
                    // The targets first and the dispatch on them, so that
                    // what each target then asks for can be threaded in
                    // through its edge.
                    Terminator::Dispatch dispatch;
                    dispatch.cond = tag;
                    vector<shared_ptr<Block>> targets;
                    for (const KeyVariant &v : variants) {
                        auto target = std::make_shared<Block>();
                        target->name = at->name + "!" + v.label;
                        target->owner = fn;
                        target->preds = {at};
                        fn->blocks.push_back(target);
                        dispatch.targets.push_back(Terminator::Jump{target->name, {}});
                        targets.push_back(target);
                    }
                    at->terminator.data = std::move(dispatch);
                    for (size_t k = 0; k < variants.size(); k++) {
                        const auto child = node->under.find(variants[k].label);
                        descend(targets[k],
                                child == node->under.end() ? nullptr : &child->second,
                                path + "!" + variants[k].label,
                                callee + "!" + variants[k].label);
                    }
                };
            descend(block, &*queue.split, queue.name, callee_name);
        };
    // The values the callee's parameters have at a call of it, by name.
    const auto params_at = [&](const shared_ptr<Block> &block,
                               const vector<shared_ptr<Value>> &args) {
        map<string, shared_ptr<Value>> params;
        const vector<Argument> &cparams = C->blocks.front()->args;
        for (size_t j = 0; j < cparams.size() && j < args.size(); j++) {
            params[cparams[j].name] = reach(block, args[j]);
        }
        return params;
    };

    // The deferred calls: push the entry, return saved. The callee writes
    // the fields that are its arguments -- a pointer argument's pointee --
    // and leaves the frame's to the frame.
    for (size_t s = 0; s < sites.size(); s++) {
        const auto &site = sites[s];
        const auto call = std::get<Terminator::Call>(site->terminator.data);
        vector<shared_ptr<Value>> values;
        for (size_t f = 0; f < fields.size(); f++) {
            const Field &field = fields[f];
            if (acc_field.has_value() && f == *acc_field) {
                values.push_back(reach(site, spawns[s].target));
                continue;
            }
            if (!field.param.has_value()) {
                values.push_back(undef_value(field.type));
                continue;
            }
            shared_ptr<Value> v = reach(site, call.call.args[*field.param]);
            if (field.pointee) {
                v = site->make_instruction(field.type, Instruction::Op::Load, {v});
            }
            values.push_back(std::move(v));
        }
        auto entry = site->make_instruction(entry_t, Instruction::Op::MakeStruct,
                                            std::move(values));
        // A spawned call's site goes on: to its continuation, less the
        // accumulate the drain now makes and the value it took.
        shared_ptr<Block> spawn_cont;
        if (spawned) {
            spawn_cont = fmap.at(call.cont.name);
            spawn_cont->instrs.erase(spawn_cont->instrs.begin());
            const string result = spawn_cont->args.front().name;
            spawn_cont->args.erase(spawn_cont->args.begin());
            spawn_cont->lookups.erase(result);
        }
        push_entry(funcs.at(site_function[s]), site,
                   queue_of.at(site_function[s]), entry,
                   params_at(site, call.call.args),
                   [&](const shared_ptr<Block> &at, const shared_ptr<Value> &slot,
                       const shared_ptr<Value> &which) {
                       if (spawned) {
                           at->terminator.data = Terminator::Jump{
                               spawn_cont->name, reach_all(at, call.cont.args)};
                           spawn_cont->preds = {at};
                           return;
                       }
                       at->terminator.data = Terminator::Return{
                           make_saved(at, true, slot, nullptr, which)};
                   });
    }
    remove_unreachable_blocks(*F);
    for (const string &g : queue.also_from) {
        remove_unreachable_blocks(*funcs.at(g));
    }

    //===----------------------------------------------------------------===//
    // The owner: the queue, the flag at the producer, and the drain
    //===----------------------------------------------------------------===//

    // A queue is a count and the handles to its own storage, an array per
    // scalar of the entry sized by the producer count (see Leaf); all are
    // `mut` locals of the owner, made where the count is known, before the
    // producer runs, and a queue is handed to the callee the way any mutable
    // local is handed to a callee: as its address.
    //
    // Whether the drain goes round follows from the chain: when the deferred
    // callee is on it -- a self-recursion -- running the entries pushes their
    // successors, and the drain runs a round per bounce until a round finds
    // nothing. When the drain's call cannot reach a push -- the deferred call
    // is to a function off the chain -- one pass over the entries is all
    // there is. Whether there is one queue or two follows from who pushes
    // onto it. When running an entry pushes its successor onto the queue
    // being drained, there are two, as pbrt's wavefront integrator has two
    // ray queues: the one being read this round and the one the round's
    // successors go to, indexed by the round's parity (pbrt: `rayQueues[depth
    // & 1]`). When the successors are pushed by another queue's drain, after
    // this one's pass is over -- the recursion's call is in the staged rest
    // of the callee, so the cycle is rays to hits to rays -- one queue serves
    // every round: its count is read for the pass and reset before it, and
    // the other drain fills it again (QueueSpec::drain_pushes_self).
    const bool rounds = callee_in_chain;
    const bool double_buffered = rounds && queue.drain_pushes_self;
    // Two for a double buffer, one per queue of a split, one otherwise; the
    // several are one array, indexed by the round's parity or the split's
    // leaf.
    const size_t nqueues = double_buffered ? 2 : subqueues.size();
    const bool many = nqueues > 1;
    // stores[i][l]: queue i's array for leaf l, named after both.
    vector<vector<shared_ptr<Value>>> stores(nqueues);
    for (size_t i = 0; i < nqueues; i++) {
        const string qname = double_buffered ? queue.name : subqueues[i].path;
        for (const Leaf &leaf : leaves) {
            stores[i].push_back(make_alloca(
                *O, alloc_point, Array_t::make(leaf.type, as_expr(size)),
                qname + "_" + leaf.name +
                    (double_buffered ? "_" + std::to_string(i) : "")));
        }
    }
    const shared_ptr<Value> queues =
        many ? make_alloca(*O, alloc_point,
                           Array_t::make(queue_t, UIntImm::make(u32(), nqueues)),
                           queue.name + "_queue")
             : make_alloca(*O, alloc_point, queue_t, queue.name + "_queue");
    for (size_t i = 0; i < nqueues; i++) {
        vector<shared_ptr<Value>> parts = {constant_u32(0)};
        parts.insert(parts.end(), stores[i].begin(), stores[i].end());
        auto initial = point->make_instruction(queue_t, Instruction::Op::MakeStruct,
                                               std::move(parts));
        auto slot = many ? point->make_instruction(queue_addr_t, Instruction::Op::GEP,
                                                   {queues, constant_u32(i)})
                         : queues;
        point->make_side_effect(Instruction::Op::Store, {slot, initial});
    }
    // The address of queue `which`, for passing to the callee: an address is
    // written where it is used (has_no_binding in SSA/CodeGen_Stmt.cpp), so
    // this names the queue in place rather than a copy of it.
    const auto queue_at = [&](const shared_ptr<Block> &block,
                              const shared_ptr<Value> &which) {
        if (!many) {
            return reach(block, queues);
        }
        return block->make_instruction(queue_addr_t, Instruction::Op::GEP,
                                       {reach(block, queues), which});
    };
    // The storage of queue `which`, as the queue records it: a function from
    // a leaf to its array, so that a reader of several leaves finds the queue
    // once.
    const auto storage_of = [&](const shared_ptr<Block> &block,
                                const shared_ptr<Value> &which)
        -> std::function<shared_ptr<Value>(size_t)> {
        if (!many) {
            return [&, block](size_t l) { return reach(block, stores[0][l]); };
        }
        // Each handle read through the queue's address, so that an entry of
        // many scalars reads the handles it needs and not the whole queue.
        auto that = queue_at(block, which);
        return [&, block, that](size_t l) {
            const Type array_t = Array_t::make(leaves[l].type, Expr());
            auto handle = block->make_instruction(
                Ptr_t::make(array_t), Instruction::Op::FieldPtr,
                {that, constant_u32(1 + l)});
            return block->make_instruction(array_t, Instruction::Op::Load,
                                           {handle});
        };
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
    // queue `which`: the iteration's values the continuation needs, and the
    // contents of the iteration's mutable locals the callee was not handed.
    // A scalar at a time, into each one's array.
    const auto write_frame = [&](const shared_ptr<Block> &block,
                                 const shared_ptr<Value> &which,
                                 const shared_ptr<Value> &slot,
                                 const std::function<shared_ptr<Value>(const Field &)> &value_of_field) {
        const auto storage = storage_of(block, which);
        const Emit emit = [&](const Type &t, Instruction::Op op,
                              vector<shared_ptr<Value>> ops) {
            return block->make_instruction(t, op, std::move(ops));
        };
        for (size_t f = 0; f < fields.size(); f++) {
            if (!fields[f].frame_writes) {
                continue;
            }
            vector<shared_ptr<Value>> parts;
            size_t l = field_leaves[f].first;
            take_apart(value_of_field(fields[f]), leaves, l, 1, emit, parts);
            internal_assert(l == field_leaves[f].second &&
                            parts.size() == l - field_leaves[f].first)
                << what << ": the frame's " << fields[f].name << " came apart "
                << "into " << parts.size() << " scalars, not "
                << field_leaves[f].second - field_leaves[f].first;
            for (size_t k = 0; k < parts.size(); k++) {
                const size_t leaf = field_leaves[f].first + k;
                auto place = block->make_instruction(
                    Ptr_t::make(leaves[leaf].type), Instruction::Op::GEP,
                    {storage(leaf), slot});
                block->make_side_effect(Instruction::Op::Store,
                                        {place, parts[k]});
            }
        }
    };

    // The producer passes the first queue and acts on the flag: a saved return
    // skips the rest of the iteration, which the drain runs later, after
    // the frame has put its own state into the entry.
    shared_ptr<Block> skip;
    if (spawned) {
        // Each producer hands the queue down and goes on; nothing comes back
        // for it to act on.
        for (const CallSite &pc : producers) {
            pc.call()->call.args.push_back(queue_at(pc.block, constant_u32(0)));
        }
    } else {
        pcall.call.args.push_back(split ? reach(producer.block, queues)
                                        : queue_at(producer.block, constant_u32(0)));
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
            write_frame(on_saved, queue_of_saved(on_saved, reach(on_saved, flag)),
                        slot, [&](const Field &f) -> shared_ptr<Value> {
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

        // The initial push: the owner's call itself pushes the first entry
        // and says saved, so that every step runs from the drain -- pbrt's
        // camera rays go through the ray queue and are traced by nothing
        // else. The call's arguments become the entry as a deferred call's
        // do; the frame's part is written by the save path above, which the
        // constant flag now always takes.
        if (queue.initial_push) {
            const Terminator::Call was = pcall;
            vector<shared_ptr<Value>> values;
            for (const Field &f : fields) {
                if (!f.param.has_value()) {
                    values.push_back(undef_value(f.type));
                    continue;
                }
                shared_ptr<Value> v = reach(producer.block, was.call.args[*f.param]);
                if (f.pointee) {
                    v = producer.block->make_instruction(f.type, Instruction::Op::Load,
                                                         {v});
                }
                values.push_back(std::move(v));
            }
            auto entry = producer.block->make_instruction(
                entry_t, Instruction::Op::MakeStruct, std::move(values));
            push_entry(O, producer.block,
                       split ? queues : queue_at(producer.block, constant_u32(0)),
                       entry, params_at(producer.block, was.call.args),
                       [&](const shared_ptr<Block> &at, const shared_ptr<Value> &slot,
                           const shared_ptr<Value> &which) {
                           auto flag = make_saved(at, true, slot, nullptr, which);
                           vector<shared_ptr<Value>> onwards{flag};
                           for (const auto &a : was.cont.args) {
                               onwards.push_back(reach(at, a));
                           }
                           if (at != producer.block) {
                               dispatch->preds.push_back(at);
                           }
                           at->terminator.data =
                               Terminator::Jump{dispatch->name, std::move(onwards)};
                       });
        }
    }

    // The rest of the producer's iteration, copied for the drain to run per
    // entry -- a copy per queue of a split, since each drain runs it with
    // its own storage. Copied now, before the drain is built: building it
    // threads values through the blocks between the producer and the drain,
    // which for a lone call are these, and the copy is of the continuation
    // as the program has it -- taking the call's value and what it carried,
    // and no more.
    vector<map<string, shared_ptr<Block>>> copies_of;
    for (const SubQueue &sq : subqueues) {
        if (spawned) {
            break; // the drain runs no continuation
        }
        copies_of.push_back(clone_region(*O, k0, "!" + sq.path));
    }
    // Where the owner now runs the rest of a producer's iteration, for a
    // spawned deferral's join to find (Function::continuation_entries): the
    // program's own continuation and each drain's copy of it, when the
    // producer is the program's call; when the producer is another drain,
    // its continuation is that drain's -- a flag dispatch and then the
    // program's, already recorded -- so what is recorded is the copies of
    // what was recorded inside it.
    if (!spawned) {
        const set<string> recorded = O->continuation_entries;
        if (!producer_is_drain) {
            O->continuation_entries.insert(k0_entry->name);
        }
        for (const auto &copies : copies_of) {
            if (!producer_is_drain) {
                O->continuation_entries.insert(copies.at(k0_entry->name)->name);
            }
            for (const string &r : recorded) {
                if (const auto it = copies.find(r); it != copies.end()) {
                    O->continuation_entries.insert(it->second->name);
                }
            }
        }
    }

    // The drain. For a self-feeding queue, a round per bounce, as pbrt's
    // wavefront integrator runs one set of kernels per depth: round r reads
    // the queue r & 1 and its successors go to the other, emptied first, and
    // the drain ends when a round finds its queue empty. Otherwise one pass
    // over each queue there is -- the one, or the split's in order.
    auto drain_entry = fresh_block(prefix + "drain");
    vector<shared_ptr<Block>> bodies, afters;
    for (const SubQueue &sq : subqueues) {
        bodies.push_back(fresh_block(sq.path + "!run"));
        afters.push_back(fresh_block(sq.path + "!ran"));
    }
    auto exit = fresh_block(prefix + "exit");
    shared_ptr<Block> header, pre, latch;
    if (rounds) {
        header = fresh_block(prefix + "round");
        pre = fresh_block(prefix + "batch");
        latch = fresh_block(prefix + "next");
        // What has to wait for every round runs from the exit
        // (Function::OwnedQueue::after).
        O->queue_sizes[queue.name].after = exit->name;
        for (const SubQueue &sq : subqueues) {
            O->queue_sizes[sq.path].after = exit->name;
        }
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
        if (skip) {
            skip->terminator.data = Terminator::Jump{drain_entry->name};
            drain_entry->preds.push_back(skip);
        }
    }
    shared_ptr<Value> cur, nxt; // which queue is read, and which pushed to
    shared_ptr<Value> next_queue; // the address of the one pushed to
    if (rounds) {
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
                                                  Terminator::Jump{bodies[0]->name},
                                                  Terminator::Jump{latch->name}};
        latch->preds = {pre};
        {
            auto next_round = latch->make_instruction(
                u32(), Instruction::Op::Add, {reach(latch, round), constant_u32(1)});
            latch->terminator.data =
                Terminator::Jump{header->name, {next_round}};
        }

        // round: while this round's queue is not empty
        shared_ptr<Value> current;
        if (double_buffered) {
            cur = header->make_instruction(u32(), Instruction::Op::BwAnd,
                                           {round, constant_u32(1)});
            current = header->make_instruction(
                queue_t, Instruction::Op::ExtractIdx, {reach(header, queues), cur});
        } else {
            cur = constant_u32(0);
            current = header->make_instruction(queue_t, Instruction::Op::Load,
                                               {reach(header, queues)});
        }
        auto pending = header->make_instruction(
            u32(), Instruction::Op::LoadField, {current, constant_u32(0)});
        auto empty = header->make_instruction(
            Bool_t::make(), Instruction::Op::Eq, {pending, constant_u32(0)});
        std::get<Terminator::Dispatch>(header->terminator.data).cond = empty;

        // batch: the queue this round's successors go to is emptied -- the
        // other one, or this one now that its count has been read for the
        // pass and nothing the pass runs pushes onto it -- and then every
        // entry of this round's is run. Its address is made here, once per
        // round and before the parfor, so that a later deferral whose
        // producer is this drain finds the value the drain passes its callee
        // defined where its own drain can reach it.
        nxt = double_buffered
                  ? pre->make_instruction(u32(), Instruction::Op::Xor,
                                          {reach(pre, cur), constant_u32(1)})
                  : cur;
        next_queue = queue_at(pre, nxt);
        auto count_ptr = pre->make_instruction(
            Ptr_t::make(u32()), Instruction::Op::FieldPtr,
            {next_queue, constant_u32(0)});
        pre->make_side_effect(Instruction::Op::Store, {count_ptr, constant_u32(0)});
        std::get<Terminator::ParFor>(pre->terminator.data).end =
            reach(pre, pending);
        bodies[0]->preds = {pre};
    }

    // A pass of the drain over queue `q`: `body` runs an entry -- the call
    // it stands for, to the callee that queue has -- and `after` acts on
    // what came back.
    const auto build_pass = [&](size_t q, const shared_ptr<Block> &body,
                                const shared_ptr<Block> &after) {
        const string &qpath = subqueues[q].path;
        // The producer's continuation, copied for this drain: none for a
        // spawned call, whose drain only accumulates.
        const map<string, shared_ptr<Block>> *copies =
            spawned ? nullptr : &copies_of[q];
        const shared_ptr<Block> k_entry =
            spawned ? nullptr : copies->at(k0_entry->name);

        // run(i): the entry, and the call it stands for. The entry's fields
        // are read as they are asked for, each put back together from its
        // scalars read out of their arrays at the entry's index, in `run`
        // whichever block of the iteration asks: every block of it follows
        // `run`, and a value is threaded to where it is used.
        auto index = body->add_argument(Argument{u32(), qpath});
        const auto run_storage = storage_of(
            body, double_buffered ? reach(body, cur) : constant_u32(q));
        map<size_t, shared_ptr<Value>> field_values;
        const auto entry_field = [&](const shared_ptr<Block> &block,
                                     size_t f) -> shared_ptr<Value> {
            auto it = field_values.find(f);
            if (it == field_values.end()) {
                size_t l = field_leaves[f].first;
                const Emit emit = [&](const Type &t, Instruction::Op op,
                                      vector<shared_ptr<Value>> ops) {
                    return body->make_instruction(t, op, std::move(ops));
                };
                auto value = rebuild(fields[f].type, emit, [&](const Type &t) {
                    internal_assert(l < field_leaves[f].second)
                        << what << ": " << fields[f].name << " has more scalars "
                        << "than the queue stores for it";
                    return body->make_instruction(t, Instruction::Op::ExtractIdx,
                                                  {run_storage(l++), index});
                });
                internal_assert(l == field_leaves[f].second)
                    << what << ": " << fields[f].name << " has fewer scalars "
                    << "than the queue stores for it";
                if (fields[f].reducer_address) {
                    // Read back out of the entry, the address is still a
                    // reducer's: a later deferral may store it too.
                    if (const auto *vi = std::get_if<shared_ptr<Instruction>>(&value->data)) {
                        O->reducer_slots.insert(vi->get());
                    }
                }
                it = field_values.emplace(f, std::move(value)).first;
            }
            return reach(block, it->second);
        };
        // A relocated local: the entry's copy of its contents, given a local
        // of the drain's own to be run with.
        map<size_t, shared_ptr<Value>> locals;
        for (size_t f = 0; f < fields.size(); f++) {
            if (fields[f].local == nullptr) {
                continue;
            }
            auto local = make_alloca(*O, body, fields[f].type);
            body->make_side_effect(Instruction::Op::Store,
                                   {local, entry_field(body, f)});
            locals[f] = local;
        }
        // The copy of the continuation takes a relocated local as a
        // parameter named after the producer's storage for it, and the drain
        // hands it its own storage instead. A value threaded through blocks
        // keeps its name in this form (Block::get_value), and a pass that
        // unwinds the threading -- promote_allocas, which turns a local
        // written once and read after into a value -- finds the pointer's
        // loads by that name; so the copy's parameter is renamed after the
        // drain's storage, throughout the copy.
        if (!spawned) {
            internal_assert(k_entry->args.size() == result_args + carried.size())
                << what << ": the continuation " << k_entry->name << " takes "
                << k_entry->args.size() << " arguments, not " << result_args
                << " + " << carried.size();
        }
        for (size_t i = 0; i < carried.size(); i++) {
            if (carried[i].from != From::Relocated) {
                continue;
            }
            const string from = k_entry->args[result_args + i].name;
            const string to = std::get<shared_ptr<Instruction>>(
                                  locals.at(carried[i].field)->data)
                                  ->name;
            for (const auto &[name, copy] : *copies) {
                rename_argument(*copy, from, to);
            }
        }
        {
            vector<shared_ptr<Value>> args;
            for (size_t j = 0; j < nparams; j++) {
                switch (params[j].where) {
                case Where::Elided:
                    args.push_back(reach(body, origin.at(callee_name)[j].value));
                    break;
                case Where::Stored:
                    args.push_back(entry_field(body, params[j].field));
                    break;
                case Where::Relocated:
                    args.push_back(locals.at(params[j].field));
                    break;
                }
            }
            if (callee_in_chain) {
                args.push_back(reach(body, next_queue));
            }
            body->terminator.data = Terminator::Call{
                Terminator::Jump{subqueues[q].callee, std::move(args)},
                Terminator::Jump{after->name},
                /*drop=*/!callee_in_chain && !returns_value};
        }

        // ran(r): a saved return is an entry for the next round, with the
        // frame's part written; any other is the rest of the producer's
        // iteration, copied here.
        after->preds = {body};
        if (spawned) {
            // The accumulate the call site gave up: the value, into the
            // reducer whose address the entry carries. Plain, as the
            // program's was: one entry per run of the producer's entry, and
            // each of those has a reducer of its own.
            auto value = after->add_argument(Argument{ret_type, O->get_unique_name()});
            after->make_side_effect(spawns.front().op,
                                    {entry_field(after, *acc_field), value},
                                    spawns.front().atomic);
            after->terminator.data = Terminator::Yield{};
        } else {
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
                    onwards.push_back(entry_field(after, c.field));
                    break;
                case From::Relocated:
                    onwards.push_back(reach(after, locals.at(c.field)));
                    break;
                }
            }
            if (saved) {
                auto done = fresh_block(qpath + "!done");
                done->preds = {after};
                after->terminator.data = Terminator::Dispatch{
                    saved,
                    {Terminator::Jump{k_entry->name, std::move(onwards)},
                     Terminator::Jump{done->name}}};
                if (frame_saves) {
                    auto slot = slot_of(done, reach(done, flag));
                    write_frame(done, reach(done, nxt), slot,
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
                            return entry_field(done, k);
                        }
                        internal_error << "no value for the frame's field "
                                       << f.name;
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
        // producer's; a return of the function's becomes a yield of the
        // loop's.
        if (copies != nullptr) {
            for (const auto &[name, copy] : *copies) {
                if (std::holds_alternative<Terminator::Return>(copy->terminator.data)) {
                    copy->terminator.data = Terminator::Yield{};
                }
                O->blocks.push_back(copy);
            }
        }
    };

    shared_ptr<Block> exit_from = drain_entry; // the block before `exit`
    if (rounds) {
        build_pass(0, bodies[0], afters[0]);
    } else {
        // One pass over each queue, the split's in order: every entry there
        // is, and nothing pushes meanwhile.
        shared_ptr<Block> from = drain_entry;
        for (size_t q = 0; q < subqueues.size(); q++) {
            const bool last = q + 1 == subqueues.size();
            const shared_ptr<Block> to =
                last ? exit : fresh_block(subqueues[q + 1].path + "!drain");
            auto whole = from->make_instruction(
                queue_t, Instruction::Op::Load, {queue_at(from, constant_u32(q))});
            auto pending = from->make_instruction(
                u32(), Instruction::Op::LoadField, {whole, constant_u32(0)});
            from->terminator.data =
                Terminator::ParFor{subqueues[q].path,
                                   constant_u32(0),
                                   pending,
                                   constant_u32(1),
                                   Terminator::Jump{bodies[q]->name},
                                   Terminator::Jump{to->name}};
            bodies[q]->preds = {from};
            if (!last) {
                to->preds = {from};
            }
            exit_from = from;
            build_pass(q, bodies[q], afters[q]);
            from = to;
        }
    }

    // exit: on to whatever followed the producer.
    exit->preds = {rounds ? header : exit_from};
    if (!rounds) {
        // Drained, so empty: said, for a queue whose producer runs again --
        // a stage's queue inside another queue's rounds is filled and
        // drained once per round.
        for (size_t q = 0; q < subqueues.size(); q++) {
            auto count_ptr = exit->make_instruction(
                Ptr_t::make(u32()), Instruction::Op::FieldPtr,
                {queue_at(exit, constant_u32(q)), constant_u32(0)});
            exit->make_side_effect(Instruction::Op::Store,
                                   {count_ptr, constant_u32(0)});
        }
    }
    if (producer_loop_block) {
        exit->terminator.data =
            Terminator::Jump{exit_target, reach_all(exit, exit_args)};
    } else if (root) {
        exit->terminator.data = Terminator::Return{};
    } else {
        exit->terminator.data = Terminator::Yield{};
    }

    // The join a spawned call leaves implicit: a continuation that reads a
    // reducer runs once every add is in -- after every round when the
    // drain is inside another queue's rounds, after the drain otherwise.
    if (spawned) {
        if (!producer_is_drain) {
            // The producer is the program's own call, so its continuation
            // has not been recorded by an earlier deferral's drain.
            O->continuation_entries.insert(pcall.cont.name);
        }
        const shared_ptr<Block> after_all =
            after_rounds.empty() ? exit : omap.at(after_rounds);
        join_continuations(O, what, queue.name + "_done", size, alloc_point,
                           after_all, made);
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

namespace {

// Each lane's rank among the lanes of `mask` that are on -- how many lanes
// before it are on -- as one index of `wide` (an unsigned vector as wide as
// the mask) per lane, so that the lanes that push are numbered 0, 1, 2, .. in
// lane order and take consecutive slots; a lane that is off holds a number
// nothing reads. CUDA's coalesced_group::thread_rank(), ispc's
// exclusive_scan_add over the mask. Built as an exclusive prefix sum of the
// mask read as ones and zeros, by a Hillis-Steele scan (Hillis & Steele,
// "Data Parallel Algorithms", CACM 1986): log2(lanes) rounds, each adding to
// every lane the sum 2^k lanes before it, a shuffle with a zero fill and an
// add apiece, inserted into `block` at `at`.
shared_ptr<Value> lane_rank(Function &func, const shared_ptr<Block> &block,
                            size_t &at, const shared_ptr<Value> &mask,
                            const Type &wide) {
    const uint32_t lanes = wide.lanes();
    const Type count = wide.element_of();
    auto ones = insert_instruction(func, block, at++, wide,
                                   Instruction::Op::Cast, {mask});
    auto zero = insert_instruction(
        func, block, at++, wide, Instruction::Op::Bc,
        {std::make_shared<Value>(Constant{count, uint64_t(0)}),
         constant_u32(lanes)});
    // `v` shifted `by` lanes along, the first `by` lanes filled with zeros:
    // in each lane, the value `by` lanes before it, and nothing before the
    // first.
    const auto before = [&](const shared_ptr<Value> &v, uint32_t by) {
        auto shifted = insert_instruction(func, block, at++, wide,
                                          Instruction::Op::Shuffle, {zero, v});
        auto &indices =
            std::get<shared_ptr<Instruction>>(shifted->data)->shuffle;
        indices.resize(lanes);
        for (uint32_t i = 0; i < lanes; i++) {
            indices[i] = i < by ? int(i) : int(lanes + i - by);
        }
        return shifted;
    };
    // Exclusive: each lane starts from whether the lane before it is on, and
    // the rounds sum everything before that.
    shared_ptr<Value> sum = before(ones, 1);
    for (uint32_t by = 1; by < lanes; by *= 2) {
        sum = insert_instruction(func, block, at++, wide, Instruction::Op::Add,
                                 {sum, before(sum, by)});
    }
    return sum;
}

} // namespace

void lower_pushes(Function &func) {
    for (const auto &block : func.blocks) {
        for (size_t i = 0; i < block->instrs.size(); i++) {
            const shared_ptr<Instruction> push = block->instrs[i];
            if (push->op != Instruction::Op::Push) {
                continue;
            }
            internal_assert(push->operands.size() == 2 ||
                            push->operands.size() == 3)
                << push->operands.size();
            const shared_ptr<Value> q = push->operands[0];
            const shared_ptr<Value> entry = push->operands[1];
            const shared_ptr<Value> mask =
                push->operands.size() == 3 ? push->operands[2] : nullptr;
            const auto *qptr = q->get_type().as<Ptr_t>();
            internal_assert(qptr)
                << "push to a non-pointer: " << q->get_type()
                << (q->get_type().is_vector()
                        ? " -- a gang pushes onto one queue, not a queue per lane"
                        : "");
            const Type queue_t = qptr->etype;
            const auto *qs = queue_t.as<Struct_t>();
            internal_assert(qs && qs->fields.size() >= 2 &&
                            qs->fields[0].name == "count")
                << "push to something that is not a queue: " << queue_t;
            const Type count_t = qs->fields[0].type;
            // The queue's storage: an array per scalar of the entry (see
            // Leaf), in the entry's order, behind the count.
            vector<Leaf> leaves;
            for (size_t k = 1; k < qs->fields.size(); k++) {
                internal_assert(qs->fields[k].type.is<Array_t>())
                    << "push to something that is not a queue: " << queue_t
                    << " holds a " << qs->fields[k].type;
                leaves.push_back(
                    Leaf{qs->fields[k].name, qs->fields[k].type.element_of()});
            }

            // A gang's push, told by its value: one slot per lane. The lanes
            // that push -- those the mask has on, or all of them -- compact
            // into consecutive slots: the count advances once by how many
            // there are, and each takes the slot at its rank among them, in
            // lane order (ispc's packed_store_active). With the entries laid
            // out a scalar at a time, each scalar of the gang's entries is
            // one compress-store (`vpcompressd`) from the first slot claimed,
            // and a plain vector store when every lane pushes. A scalar push
            // is a gang of one with no mask.
            const Vector_t *gang = push->type.as<Vector_t>();
            const uint32_t lanes = gang != nullptr ? gang->lanes : 1;
            internal_assert(gang != nullptr || mask == nullptr)
                << "a scalar push under a mask in " << block->name;
            internal_assert(gang == nullptr || equals(count_t, u32()))
                << "a gang's push counts in " << count_t << ", not u32";

            size_t at = i;
            // The count's address is a FieldPtr, which is written out where it
            // is used rather than bound to a name (has_no_binding in
            // SSA/CodeGen_Stmt.cpp): it names the count in the queue, where a
            // load of the queue and the address of its count field would name
            // a copy. The slots are claimed by fetch-and-add when the push is
            // atomic, and by a read and a write when it is not.
            auto count_ptr = insert_instruction(func, block, at++,
                                                Ptr_t::make(count_t),
                                                Instruction::Op::FieldPtr,
                                                {q, constant_u32(0)});
            // How many entries this push adds.
            shared_ptr<Value> added;
            if (gang == nullptr) {
                added = std::make_shared<Value>(Constant{count_t, uint64_t(1)});
            } else if (mask != nullptr) {
                added = insert_instruction(func, block, at++, count_t,
                                           Instruction::Op::Popcount, {mask});
            } else {
                added = std::make_shared<Value>(
                    Constant{count_t, uint64_t(lanes)});
            }
            // The first slot they take: the count before.
            shared_ptr<Value> first;
            if (push->atomic) {
                first = insert_instruction(func, block, at++, count_t,
                                           Instruction::Op::AtomicAdd,
                                           {count_ptr, added});
            } else {
                first = insert_instruction(func, block, at++, count_t,
                                           Instruction::Op::Load, {count_ptr});
                auto next = insert_instruction(func, block, at++, count_t,
                                               Instruction::Op::Add,
                                               {first, added});
                insert_side_effect(block, at++, Instruction::Op::Store,
                                   {count_ptr, next});
            }
            // Where the entries go: the slots from `first`. A gang that pushes
            // from every lane writes `lanes` consecutive slots, one vector
            // store per scalar at ramp(first, 1); one that pushes under a mask
            // writes each scalar with a compress-store from the address of
            // slot `first`, which puts the lanes that are on into consecutive
            // slots itself.
            shared_ptr<Value> dense; // the gang's slots, when every lane pushes
            if (gang != nullptr && mask == nullptr) {
                dense = insert_instruction(
                    func, block, at++, push->type, Instruction::Op::Ramp,
                    {first,
                     std::make_shared<Value>(Constant{count_t, uint64_t(1)})});
            }
            // The entry's scalars, as the push holds them: for a gang, each a
            // gang-wide vector. One the frame writes after the push is
            // undefined here, and is not stored.
            vector<shared_ptr<Value>> parts;
            {
                size_t l = 0;
                const Emit emit = [&](const Type &t, Instruction::Op op,
                                      vector<shared_ptr<Value>> ops) {
                    return insert_instruction(func, block, at++, t, op,
                                              std::move(ops));
                };
                take_apart(entry, leaves, l, lanes, emit, parts);
                internal_assert(l == leaves.size() && parts.size() == l)
                    << "an entry of " << parts.size() << " scalars pushed onto "
                    << queue_t << ", which stores " << leaves.size();
            }
            // Each array's handle is read through the queue's address -- the
            // handles the entry needs, not the whole queue -- and names the
            // storage itself.
            for (size_t l = 0; l < leaves.size(); l++) {
                if (parts[l] == nullptr) {
                    continue;
                }
                auto handle = insert_instruction(
                    func, block, at++, Ptr_t::make(qs->fields[1 + l].type),
                    Instruction::Op::FieldPtr, {q, constant_u32(1 + l)});
                auto array = insert_instruction(func, block, at++,
                                                qs->fields[1 + l].type,
                                                Instruction::Op::Load, {handle});
                auto place = insert_instruction(
                    func, block, at++,
                    dense ? Vector_t::make(Ptr_t::make(leaves[l].type), lanes)
                          : Ptr_t::make(leaves[l].type),
                    Instruction::Op::GEP, {array, dense ? dense : first});
                vector<shared_ptr<Value>> operands = {place, parts[l]};
                if (mask != nullptr) {
                    operands.push_back(mask);
                }
                insert_side_effect(block, at++, Instruction::Op::Store,
                                   std::move(operands), /*atomic=*/false,
                                   /*compact=*/mask != nullptr);
            }
            // The push's value was each lane's slot: the first, plus its rank
            // among the lanes that push -- its lane index when they all do.
            // Wanted by a frame that writes its part into the slot after the
            // push, and built only then: the rank is a scan over the mask.
            shared_ptr<Value> index = first;
            if (dense != nullptr) {
                index = dense;
            } else if (gang != nullptr && has_uses(func, push.get())) {
                auto rank = lane_rank(func, block, at, mask, push->type);
                auto base = insert_instruction(
                    func, block, at++, push->type, Instruction::Op::Bc,
                    {first, constant_u32(lanes)});
                index = insert_instruction(func, block, at++, push->type,
                                           Instruction::Op::Add, {base, rank});
            }
            replace_uses(func, push.get(), index);
            i = at - 1;
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
