#include "SSA/RecordHoist.h"

#include "Error.h"
#include "IR/Expr.h"
#include "IR/Printer.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/Simplify.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
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

constexpr const char *kRecordName = "_RTRecord";
constexpr const char *kFillerName = "bonsai_rt_record_fill";

bool explaining() { return std::getenv("BONSAI_EXPLAIN_RECORD_HOIST") != nullptr; }

const Instruction *as_instr(const shared_ptr<Value> &v) {
    const auto *p = std::get_if<shared_ptr<Instruction>>(&v->data);
    return p != nullptr ? p->get() : nullptr;
}

const Argument *as_arg(const shared_ptr<Value> &v) {
    return std::get_if<Argument>(&v->data);
}

bool is_record_type(const Type &t) {
    const Struct_t *s = t.as<Struct_t>();
    return s != nullptr && s->name == kRecordName;
}

// Whether `f` reads the hit-group record.
bool reads_record(const Function &f) {
    for (const auto &block : f.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op == Instruction::Op::Intrinsic &&
                instr->intrinsic == ir::Intrinsic::rt_sbt_record) {
                return true;
            }
        }
    }
    return false;
}

// The field a record read takes: `load_field(<record>, k)` with k constant.
std::optional<size_t> record_field_read(const Instruction &instr) {
    if (instr.op != Instruction::Op::LoadField || instr.operands.size() != 2 ||
        !is_record_type(instr.operands[0]->get_type())) {
        return std::nullopt;
    }
    const auto *c = std::get_if<Constant>(&instr.operands[1]->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return size_t(*u);
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return size_t(*i);
    }
    return std::nullopt;
}

// The arm a record field belongs to, from its name: `_<variant>_<field>`
// (ir::rt_record_field) for an arm's field; none for base, holder and the
// tag.
std::optional<size_t> arm_of_field(const string &name) {
    if (name.size() < 3 || name[0] != '_' || !std::isdigit(name[1])) {
        return std::nullopt;
    }
    size_t end = 1;
    while (end < name.size() && std::isdigit(name[end])) {
        end++;
    }
    if (end >= name.size() || name[end] != '_') {
        return std::nullopt;
    }
    return size_t(std::stoull(name.substr(1, end - 1)));
}

//===--------------------------------------------------------------------===//
// Binding times
//===--------------------------------------------------------------------===//

// When a value is known: at the launch (a parameter, a constant, what is
// computed from them alone), from the record (the record's fields and what
// is computed from them and from read-only storage at such indices), or
// only at the hit (the primitive index, the ray, memory anything writes,
// and whatever depends on them).
enum class Time { Launch = 0, Record = 1, Hit = 2 };

Time later(Time a, Time b) { return a > b ? a : b; }

// Whether `op` computes its value from its operands alone: no memory, no
// effect, nothing of the hit's state.
bool pure_op(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::Inf:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Popcount:
    case Instruction::Op::Ramp:
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
    default:
        return false;
    }
}

// Whether `instr` reads memory: an element of an array handle, or through a
// pointer. (An extract of a vector's lane is arithmetic; see pure_op.)
bool memory_read(const Instruction &instr) {
    if (instr.op == Instruction::Op::ExtractIdx) {
        return instr.operands.size() == 2 && instr.operands[0]->get_type().is<Array_t>();
    }
    return instr.op == Instruction::Op::Load;
}

struct Times {
    map<const Instruction *, Time> instrs;
    map<string, Time> args; // block arguments, by name

    Time of(const shared_ptr<Value> &v) const {
        if (std::holds_alternative<Constant>(v->data)) {
            return Time::Launch;
        }
        if (const Argument *a = as_arg(v)) {
            const auto it = args.find(a->name);
            return it == args.end() ? Time::Launch : it->second;
        }
        const auto it = instrs.find(as_instr(v));
        return it == instrs.end() ? Time::Hit : it->second;
    }
};

// The binding times of `f`'s values, to a fixed point: a block argument is
// as late as the latest value any edge passes it, and a parameter is the
// launch's.
Times analyze(const Function &f) {
    Times t;
    map<string, const Block *> by_name;
    for (const auto &block : f.blocks) {
        by_name[block->name] = block.get();
    }
    const auto raise_arg = [&](const string &name, Time time) -> bool {
        auto [it, added] = t.args.emplace(name, time);
        if (!added && it->second >= time) {
            return false;
        }
        it->second = time;
        return true;
    };
    const auto raise_jump = [&](const Terminator::Jump &j, size_t first_arg_offset,
                                Time first) -> bool {
        const auto target = by_name.find(j.name);
        if (target == by_name.end()) {
            return false;
        }
        bool changed = false;
        const vector<Argument> &params = target->second->args;
        for (size_t i = 0; i < first_arg_offset && i < params.size(); i++) {
            changed = raise_arg(params[i].name, first) || changed;
        }
        for (size_t i = 0; i < j.args.size(); i++) {
            const size_t k = first_arg_offset + i;
            if (k < params.size()) {
                changed = raise_arg(params[k].name, t.of(j.args[i])) || changed;
            }
        }
        return changed;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto &block : f.blocks) {
            for (const auto &instr : block->instrs) {
                Time time = Time::Hit;
                if (instr->op == Instruction::Op::Intrinsic) {
                    time = instr->intrinsic == ir::Intrinsic::rt_sbt_record ? Time::Record
                                                                             : Time::Hit;
                } else if (memory_read(*instr)) {
                    // Read-only storage at an index known when the base is.
                    if (instr->operands[0]->get_type().is_readonly()) {
                        time = Time::Launch;
                        for (const auto &operand : instr->operands) {
                            time = later(time, t.of(operand));
                        }
                    }
                } else if (pure_op(instr->op)) {
                    time = Time::Launch;
                    for (const auto &operand : instr->operands) {
                        time = later(time, t.of(operand));
                    }
                }
                auto [it, added] = t.instrs.emplace(instr.get(), time);
                if (!added && it->second != time) {
                    it->second = time;
                    changed = true;
                } else if (added) {
                    changed = true;
                }
            }
            std::visit(
                overloads{
                    [&](const std::monostate &) {},
                    [&](const Terminator::Jump &j) { changed = raise_jump(j, 0, Time::Hit) || changed; },
                    [&](const Terminator::Dispatch &d) {
                        for (const auto &j : d.targets) {
                            changed = raise_jump(j, 0, Time::Hit) || changed;
                        }
                    },
                    [&](const Terminator::Return &) {},
                    [&](const Terminator::ParFor &p) {
                        // The body's first argument is the index.
                        changed = raise_jump(p.body, 1, Time::Hit) || changed;
                        changed = raise_jump(p.cont, 0, Time::Hit) || changed;
                    },
                    [&](const Terminator::Yield &) {},
                    [&](const Terminator::Call &c) {
                        // The continuation's first argument is the call's
                        // value, unknown here.
                        changed = raise_jump(c.cont, c.drop ? 0 : 1, Time::Hit) || changed;
                    },
                    [&](const Terminator::MultiCall &c) {
                        changed = raise_jump(c.cont, c.drop ? 0 : 1, Time::Hit) || changed;
                    },
                    [&](const Terminator::Spawn &s) {
                        changed = raise_jump(s.cont, 0, Time::Hit) || changed;
                    },
                    [&](const Terminator::Join &j) { changed = raise_jump(j.next, 0, Time::Hit) || changed; },
                    [&](const Terminator::Defer &) {},
                },
                block->terminator.data);
        }
    }
    return t;
}

// Whether `v` carries a part known from the record: a record-time value, or
// a value computed from one and others -- a triangle built from the
// record's mesh and the primitive index, a shape's payload packing both
// into its words -- through which the callee's reads at the record's values
// are reached once the call is copied in and the packing folds.
bool has_record_part(const shared_ptr<Value> &v, const Times &t) {
    if (t.of(v) == Time::Record) {
        return true;
    }
    const Instruction *instr = as_instr(v);
    if (instr == nullptr || !pure_op(instr->op)) {
        return false;
    }
    for (const auto &operand : instr->operands) {
        if (has_record_part(operand, t)) {
            return true;
        }
    }
    return false;
}

// Whether `from`'s calls reach `to`.
bool calls_reach(const RecordFuncMap &funcs, const string &from, const string &to) {
    set<string> seen;
    vector<string> work{from};
    while (!work.empty()) {
        const string name = work.back();
        work.pop_back();
        if (!seen.insert(name).second) {
            continue;
        }
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue;
        }
        for (const auto &block : it->second->blocks) {
            if (const auto *call = block->terminator.callee()) {
                if (call->name == to) {
                    return true;
                }
                work.push_back(call->name);
            }
        }
    }
    return false;
}

//===--------------------------------------------------------------------===//
// The hoist
//===--------------------------------------------------------------------===//

// The instructions `root` is computed from, in dependence order, stopping
// at record field reads; the record fields and the parameters they read.
// Empty if the slice has an input this hoist does not carry (a value of the
// launch that is not an array parameter, a block argument).
// An operand of a slice's instruction, as the slice stands on its own: a
// constant, an array parameter of the program (by its name), a field of the
// record, or an earlier instruction of the slice. The slice is copied from
// this form -- not from the program's instructions, which the hoist then
// replaces and the simplification after it removes.
struct SymOperand {
    enum class Kind { Constant, Param, Field, Ref } kind = Kind::Ref;
    shared_ptr<Value> constant; // Kind::Constant
    string param;               // Kind::Param
    size_t index = 0;           // Kind::Field: the record's field; Kind::Ref: the slice's
};

struct SymInstr {
    Instruction::Op op;
    Type type;
    Type queried_type;
    ir::Intrinsic::OpType intrinsic;
    ir::VectorReduce::OpType reduce;
    vector<int> shuffle;
    vector<SymOperand> operands;
};

struct Slice {
    vector<SymInstr> instrs; // dependence order, the root last
    set<size_t> fields;
    vector<const Instruction *> record_reads; // the `load_field(<record>, k)`s
    map<string, Type> params;                 // by the program's parameter names
    bool has_load = false;
    bool ok = true;
};

// A record-time load hoisted into the record: the field it becomes, the arm
// whose fields its address reads, and the instructions that compute it
// (its slice, in an order each may be evaluated in), whose inputs are record
// field reads, constants and array parameters.
struct Hoisted {
    string name;
    Type type;
    size_t arm = 0;
    vector<SymInstr> slice;   // dependence order, the root last
    map<string, Type> params; // the array parameters read, by name
};

// The parameter of `f` that the block argument `name` carries: the argument
// followed up the single predecessors, each passing what it was passed,
// to the entry block. None where the chain forks or passes a computed
// value. (An inlined callee's parameters are block arguments of its entry
// copy, renamed; the array the call passed is the program's parameter.)
std::optional<std::pair<string, Type>> parameter_of(const Function &f, const string &name) {
    for (const auto &block : f.blocks) {
        for (size_t k = 0; k < block->args.size(); k++) {
            if (block->args[k].name != name) {
                continue;
            }
            if (block == f.blocks.front()) {
                return std::make_pair(name, block->args[k].type);
            }
            if (block->preds.size() != 1) {
                return std::nullopt;
            }
            const shared_ptr<Block> pred = block->preds[0].lock();
            if (pred == nullptr) {
                return std::nullopt;
            }
            const shared_ptr<Value> passed = passed_to(*pred, *block, k);
            if (passed == nullptr) {
                return std::nullopt;
            }
            if (const Argument *a = as_arg(passed)) {
                return parameter_of(f, a->name);
            }
            return std::nullopt;
        }
    }
    return std::nullopt;
}

Slice slice_of(const Function &f, const shared_ptr<Instruction> &root, const Times &t) {
    Slice s;
    map<const Instruction *, size_t> position; // of the slice's instructions
    // The operand `v` in the slice's own terms, visiting what it depends on
    // first; none when the slice cannot carry it.
    std::function<std::optional<SymOperand>(const shared_ptr<Value> &)> visit =
        [&](const shared_ptr<Value> &v) -> std::optional<SymOperand> {
        if (!s.ok) {
            return std::nullopt;
        }
        if (std::holds_alternative<Constant>(v->data)) {
            SymOperand o;
            o.kind = SymOperand::Kind::Constant;
            o.constant = std::make_shared<Value>(*v);
            return o;
        }
        if (const Argument *a = as_arg(v)) {
            if (t.of(v) == Time::Launch && a->type.is<Array_t>()) {
                const auto param = parameter_of(f, a->name);
                if (!param) {
                    s.ok = false; // an array no parameter of the program names
                    return std::nullopt;
                }
                s.params.emplace(param->first, param->second);
                SymOperand o;
                o.kind = SymOperand::Kind::Param;
                o.param = param->first;
                return o;
            }
            s.ok = false; // a block argument carrying a record-time value
            return std::nullopt;
        }
        const Instruction *instr = as_instr(v);
        if (instr == nullptr) {
            s.ok = false;
            return std::nullopt;
        }
        if (const auto k = record_field_read(*instr)) {
            s.fields.insert(*k);
            s.record_reads.push_back(instr);
            SymOperand o;
            o.kind = SymOperand::Kind::Field;
            o.index = *k;
            return o;
        }
        if (const auto at = position.find(instr); at != position.end()) {
            SymOperand o;
            o.kind = SymOperand::Kind::Ref;
            o.index = at->second;
            return o;
        }
        if (instr->op == Instruction::Op::Intrinsic ||
            !(pure_op(instr->op) || memory_read(*instr))) {
            s.ok = false; // the record itself read whole, or an effect
            return std::nullopt;
        }
        SymInstr sym;
        sym.op = instr->op;
        sym.type = instr->type;
        sym.queried_type = instr->queried_type;
        sym.intrinsic = instr->intrinsic;
        sym.reduce = instr->reduce;
        sym.shuffle = instr->shuffle;
        for (const auto &operand : instr->operands) {
            const auto o = visit(operand);
            if (!o) {
                s.ok = false;
                return std::nullopt;
            }
            sym.operands.push_back(*o);
        }
        if (memory_read(*instr)) {
            s.has_load = true;
        }
        position[instr] = s.instrs.size();
        s.instrs.push_back(std::move(sym));
        SymOperand o;
        o.kind = SymOperand::Kind::Ref;
        o.index = position[instr];
        return o;
    };
    visit(std::make_shared<Value>(root));
    return s;
}

// What a slice computes, as a string: the same computation in two programs
// (the any-hit's and the closest-hit's read of the mesh's first index) is
// one field of the record, not two.
string slice_key(const Slice &s) {
    std::ostringstream key;
    for (const SymInstr &instr : s.instrs) {
        key << int(instr.op) << ":" << instr.type << "(";
        for (const SymOperand &o : instr.operands) {
            switch (o.kind) {
            case SymOperand::Kind::Constant:
                key << "c";
                std::get<Constant>(o.constant->data).dump(key);
                break;
            case SymOperand::Kind::Param:
                key << "p" << o.param;
                break;
            case SymOperand::Kind::Field:
                key << "r" << o.index;
                break;
            case SymOperand::Kind::Ref:
                key << "#" << o.index;
                break;
            }
            key << ",";
        }
        key << ");";
    }
    return key.str();
}

// `value`, in scope in block `from`, made available in block `to`: threaded
// through the blocks between as an argument of each, along a chain of
// single predecessors -- the shape an inlined callee's straight-line body
// has below the block that read the record. Null where the chain forks,
// which this hoist does not thread through. (This form passes every value
// a block uses that it does not define as the block's argument; see
// SSA/CodeGen_Stmt.cpp on what an entry binding has to name.)
shared_ptr<Value> thread_to(Function &F, const shared_ptr<Value> &value,
                            const shared_ptr<Block> &from, const shared_ptr<Block> &to) {
    if (to == from) {
        return value;
    }
    if (to->preds.size() != 1) {
        return nullptr;
    }
    const shared_ptr<Block> pred = to->preds[0].lock();
    if (pred == nullptr) {
        return nullptr;
    }
    const shared_ptr<Value> in_pred = thread_to(F, value, from, pred);
    if (in_pred == nullptr) {
        return nullptr;
    }
    Argument a;
    a.type = value->get_type();
    a.name = F.get_unique_name();
    to->args.push_back(a);
    for (Terminator::Jump *j : jumps_of(*pred)) {
        if (j->name == to->name) {
            j->args.push_back(in_pred);
        }
    }
    return std::make_shared<Value>(a);
}

// Rewrites every struct named `_RTRecord` inside `type` to `record`.
Type retyped(const Type &type, const Type &record) {
    if (is_record_type(type)) {
        return record;
    }
    if (const Array_t *a = type.as<Array_t>()) {
        Type etype = retyped(a->etype, record);
        return etype.same_as(a->etype) ? type : Array_t::make(etype, a->size, a->readonly);
    }
    if (const Ptr_t *p = type.as<Ptr_t>()) {
        Type etype = retyped(p->etype, record);
        return etype.same_as(p->etype) ? type : Ptr_t::make(etype, p->readonly);
    }
    return type;
}

void retype(Function &f, const Type &record) {
    for (const auto &block : f.blocks) {
        for (Argument &a : block->args) {
            a.type = retyped(a.type, record);
        }
        for (const auto &instr : block->instrs) {
            instr->type = retyped(instr->type, record);
            if (instr->queried_type.defined()) {
                instr->queried_type = retyped(instr->queried_type, record);
            }
        }
    }
    for_each_value(f, [&](shared_ptr<Value> &v) {
        if (Argument *a = std::get_if<Argument>(&v->data)) {
            a->type = retyped(a->type, record);
        } else if (Constant *c = std::get_if<Constant>(&v->data)) {
            c->type = retyped(c->type, record);
        }
    });
}

shared_ptr<Value> constant_u32(uint64_t k) {
    return std::make_shared<Value>(Constant{UInt_t::make(32), k});
}

} // namespace

void inline_record_callees(RecordFuncMap &funcs) {
    for (auto &[name, f] : funcs) {
        if (!reads_record(*f)) {
            continue;
        }
        bool again = true;
        for (size_t rounds = 0; again && rounds < 64; rounds++) {
            again = false;
            const Times t = analyze(*f);
            const vector<shared_ptr<Block>> blocks = f->blocks;
            for (const shared_ptr<Block> &block : blocks) {
                const auto *c = std::get_if<Terminator::Call>(&block->terminator.data);
                if (c == nullptr) {
                    continue;
                }
                const auto callee = funcs.find(c->call.name);
                if (callee == funcs.end() || callee->second->blocks.empty() ||
                    c->call.name == name || calls_reach(funcs, c->call.name, name)) {
                    continue;
                }
                bool record_part = false;
                for (const auto &arg : c->call.args) {
                    record_part = record_part || has_record_part(arg, t);
                }
                if (!record_part) {
                    continue;
                }
                if (explaining()) {
                    std::cerr << "record hoist: " << name << " inlines " << c->call.name
                              << " at " << block->name << "\n";
                }
                inline_call(funcs, f, block);
                // A struct the caller built and the copy takes apart is its
                // fields again, so that the next round sees the record's
                // values where the callee read them.
                simplify(*f);
                again = true;
                break;
            }
        }
    }
}

void hoist_record_loads(RecordFuncMap &funcs, ir::Program &program) {
    const auto record_it = program.types.find(kRecordName);
    if (record_it == program.types.end()) {
        return;
    }
    const auto filler_it = funcs.find(kFillerName);
    if (filler_it == funcs.end() || filler_it->second->blocks.empty()) {
        return; // no held arm: nothing is filled, nothing to hoist into
    }
    const Struct_t *record = record_it->second.as<Struct_t>();
    internal_assert(record != nullptr) << kRecordName << " is " << record_it->second;
    const size_t base_fields = record->fields.size();

    vector<Hoisted> added;
    map<string, size_t> known; // a slice's key -> its index in `added`
    for (auto &[name, f] : funcs) {
        if (name == kFillerName || !reads_record(*f)) {
            continue;
        }
        const Times t = analyze(*f);
        refresh_preds(*f); // for the threading below
        map<string, shared_ptr<Value>> record_in_block;
        // Which instructions each value is read by, for the frontier.
        map<const Instruction *, vector<const Instruction *>> users;
        set<const Instruction *> read_by_terminator;
        for (const auto &block : f->blocks) {
            for (const auto &instr : block->instrs) {
                for (const auto &operand : instr->operands) {
                    if (const Instruction *o = as_instr(operand)) {
                        users[o].push_back(instr.get());
                    }
                }
            }
            for_each_value(block->terminator, [&](shared_ptr<Value> &v) {
                if (const Instruction *o = as_instr(v)) {
                    read_by_terminator.insert(o);
                }
            });
        }
        // Which record-time values the hit needs: read by a terminator or
        // by an instruction of a later time, or by a record-time
        // instruction the hit needs -- the arithmetic between a load and
        // its use stays in the program, over the hoisted field.
        set<const Instruction *> reaches_hit;
        for (bool changed = true; changed;) {
            changed = false;
            for (const auto &[instr, time] : t.instrs) {
                if (time != Time::Record || reaches_hit.contains(instr)) {
                    continue;
                }
                bool needed = read_by_terminator.contains(instr);
                for (const Instruction *u : users[instr]) {
                    const auto ut = t.instrs.find(u);
                    needed = needed || ut == t.instrs.end() || ut->second != Time::Record ||
                             reaches_hit.contains(u);
                }
                if (needed) {
                    reaches_hit.insert(instr);
                    changed = true;
                }
            }
        }
        // Whether a loaded struct is read field by field alone: every use a
        // `load_field` at a constant, none by a terminator.
        const auto narrowed_to_fields = [&](const Instruction *row) {
            if (read_by_terminator.contains(row)) {
                return false;
            }
            const auto found = users.find(row);
            if (found == users.end() || found->second.empty()) {
                return false;
            }
            for (const Instruction *u : found->second) {
                if (u->op != Instruction::Op::LoadField || u->operands.size() != 2 ||
                    !std::holds_alternative<Constant>(u->operands[1]->data)) {
                    return false;
                }
            }
            return true;
        };
        // The frontier: record-time loads the hit needs, whose address reads
        // an arm's fields, and with the record value in scope in their
        // block (threaded there when it is not).
        struct Candidate {
            shared_ptr<Block> block;
            size_t at;
            Slice slice;
            size_t arm;
            shared_ptr<Value> record_value;
        };
        vector<Candidate> candidates;
        for (const auto &block : f->blocks) {
            for (size_t i = 0; i < block->instrs.size(); i++) {
                const shared_ptr<Instruction> &instr = block->instrs[i];
                const auto ti = t.instrs.find(instr.get());
                if (ti == t.instrs.end() || ti->second != Time::Record ||
                    !reaches_hit.contains(instr.get()) || instr->type.is<Ptr_t>() ||
                    instr->type.is<Array_t>() || instr->type.is<Void_t>()) {
                    continue;
                }
                // A load, or a field of a loaded struct that is read field by
                // field: the record carries the fields read of a row, not
                // the row (pbrt's record holds the mesh's attributes, not
                // every word of the mesh).
                if (memory_read(*instr)) {
                    if (instr->type.is<Struct_t>() && narrowed_to_fields(instr.get())) {
                        continue;
                    }
                } else if (instr->op == Instruction::Op::LoadField &&
                           instr->operands.size() == 2 &&
                           std::holds_alternative<Constant>(instr->operands[1]->data)) {
                    const Instruction *row = as_instr(instr->operands[0]);
                    const auto rt = row != nullptr ? t.instrs.find(row) : t.instrs.end();
                    if (row == nullptr || !memory_read(*row) || rt == t.instrs.end() ||
                        rt->second != Time::Record || !narrowed_to_fields(row)) {
                        continue;
                    }
                } else {
                    continue;
                }
                Slice slice = slice_of(*f, instr, t);
                if (!slice.ok) {
                    continue;
                }
                std::optional<size_t> arm;
                bool one_arm = true;
                for (const size_t k : slice.fields) {
                    if (k >= base_fields) {
                        one_arm = false; // a field this pass added: not again
                        break;
                    }
                    const auto a = arm_of_field(record->fields[k].name);
                    if (!a) {
                        continue;
                    }
                    if (arm && *arm != *a) {
                        one_arm = false;
                    }
                    arm = a;
                }
                if (!one_arm || !arm) {
                    if (explaining()) {
                        std::cerr << "record hoist: " << name << " keeps " << instr->name
                                  << " (reads no one arm's fields)\n";
                    }
                    continue;
                }
                // The record value, in this block: the one the slice's
                // record read took, threaded here from that read's block
                // (once per block).
                shared_ptr<Value> record_value;
                if (const auto cached = record_in_block.find(block->name);
                    cached != record_in_block.end()) {
                    record_value = cached->second;
                } else {
                    const Instruction *read = slice.record_reads.front();
                    const shared_ptr<Block> read_block = read->owner.lock();
                    internal_assert(read_block != nullptr) << read->name << " has no block";
                    record_value = thread_to(*f, read->operands[0], read_block, block);
                    if (record_value) {
                        record_in_block[block->name] = record_value;
                    }
                }
                if (!record_value) {
                    if (explaining()) {
                        std::cerr << "record hoist: " << name << " keeps " << instr->name
                                  << " (the record is not in scope in its block, and "
                                  << "the way there forks)\n";
                    }
                    continue;
                }
                candidates.push_back(
                    Candidate{block, i, std::move(slice), *arm, record_value});
            }
        }
        // The residual program: each candidate a read of its new field.
        // Later candidates first, so that the positions recorded stay.
        for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
            Candidate &c = *it;
            const shared_ptr<Instruction> root = c.block->instrs[c.at];
            // One field per computation, however many programs make it.
            const string key = slice_key(c.slice);
            size_t index;
            if (const auto found = known.find(key); found != known.end()) {
                index = found->second;
            } else {
                index = added.size();
                known[key] = index;
                Hoisted h;
                h.arm = c.arm;
                h.name = "_" + std::to_string(c.arm) + "_h" + std::to_string(index);
                h.type = root->type;
                h.slice = std::move(c.slice.instrs);
                h.params = std::move(c.slice.params);
                added.push_back(std::move(h));
            }
            const size_t field = base_fields + index;
            auto read = std::make_shared<Instruction>(
                f->get_unique_name(), root->type, Instruction::Op::LoadField,
                vector<shared_ptr<Value>>{c.record_value, constant_u32(field)}, c.block);
            c.block->instrs.insert(c.block->instrs.begin() + long(c.at), read);
            if (explaining()) {
                std::cerr << "record hoist: " << name << " reads " << added[index].name
                          << " for " << root->name << " (" << root->type << ")\n";
            }
            replace_uses(*f, root.get(), std::make_shared<Value>(read));
        }
        if (!candidates.empty()) {
            simplify(*f); // the slices left behind
        }
    }
    if (added.empty()) {
        return;
    }

    // The record's type, with the hoisted fields after the ones it had.
    Struct_t::Map fields = record->fields;
    for (const Hoisted &h : added) {
        fields.push_back(TypedVar{h.name, h.type});
    }
    const Type new_record = Struct_t::make(kRecordName, std::move(fields), record->attributes);
    program.types[kRecordName] = new_record;
    for (auto &[name, f] : funcs) {
        retype(*f, new_record);
    }

    // The filler: in the arm whose fields a hoisted load's address reads,
    // the slice computed from the arm's record values and the arrays, and
    // the value built into the record; a zero in the other arms. The
    // records are built by the match on the holder's variant, which the
    // lowering may have left as a function of its own (`_match<n>`) that the
    // filler calls: it is copied into the filler first, so that the arms
    // are the filler's blocks, reached from its entry.
    Function &F = *filler_it->second;
    const auto builds_record = [&](const Function &g) {
        for (const auto &block : g.blocks) {
            for (const auto &instr : block->instrs) {
                if (instr->op == Instruction::Op::MakeStruct && is_record_type(instr->type)) {
                    return true;
                }
            }
        }
        return false;
    };
    for (bool again = true; again;) {
        again = false;
        const vector<shared_ptr<Block>> blocks = F.blocks;
        for (const shared_ptr<Block> &block : blocks) {
            const auto *c = std::get_if<Terminator::Call>(&block->terminator.data);
            if (c == nullptr) {
                continue;
            }
            const auto callee = funcs.find(c->call.name);
            if (callee == funcs.end() || callee->second->blocks.empty() ||
                callee->second == filler_it->second || !builds_record(*callee->second)) {
                continue;
            }
            inline_call(funcs, filler_it->second, block);
            again = true;
            break;
        }
    }
    refresh_preds(F);
    const shared_ptr<Block> entry = F.blocks.front();
    for (auto &[gname, gf] : funcs) {
        Function &G = *gf;
        const bool in_filler = gf == filler_it->second;
        bool changed = false;
        map<string, shared_ptr<Value>> threaded; // "<block>/<param>" -> its value there
    for (const shared_ptr<Block> &block : G.blocks) {
        for (size_t at = 0; at < block->instrs.size(); at++) {
            const shared_ptr<Instruction> M = block->instrs[at];
            if (M->op != Instruction::Op::MakeStruct || !is_record_type(M->type) ||
                M->operands.size() != base_fields) {
                continue;
            }
            changed = true;
            if (!in_filler) {
                // A build outside the filler -- the match's own function,
                // copied into the filler above and left for the dead-code
                // sweep -- takes the record's shape and nothing more.
                for (const Hoisted &h : added) {
                    M->operands.push_back(undef_value(h.type));
                }
                continue;
            }
            const auto *tag = std::get_if<Constant>(&M->operands[2]->data);
            internal_assert(tag != nullptr) << kFillerName << " builds a record with a tag "
                                            << "that is not a constant";
            const uint64_t arm = std::holds_alternative<uint64_t>(tag->data)
                                     ? std::get<uint64_t>(tag->data)
                                     : uint64_t(std::get<int64_t>(tag->data));
            // The slice's values are made here, before the record is built.
            vector<shared_ptr<Instruction>> made;
            for (const Hoisted &h : added) {
                if (h.arm != arm) {
                    // Nothing reads the field in this arm's records.
                    const size_t before = block->instrs.size();
                    shared_ptr<Value> zero = zero_value(h.type, G, block);
                    // zero_value appends; the record build must follow.
                    vector<shared_ptr<Instruction>> appended(
                        block->instrs.begin() + long(before), block->instrs.end());
                    block->instrs.erase(block->instrs.begin() + long(before), block->instrs.end());
                    made.insert(made.end(), appended.begin(), appended.end());
                    M->operands.push_back(zero);
                    continue;
                }
                // The arrays the slice reads: parameters of the filler by
                // the program's names, added when it lacks them, and
                // threaded from the entry to this block.
                map<string, shared_ptr<Value>> values;
                for (const auto &[pname, ptype] : h.params) {
                    const string where = block->name + "/" + pname;
                    if (const auto seen = threaded.find(where); seen != threaded.end()) {
                        values[pname] = seen->second;
                        continue;
                    }
                    shared_ptr<Value> in_entry;
                    for (const Argument &a : entry->args) {
                        if (a.name == pname) {
                            in_entry = std::make_shared<Value>(a);
                        }
                    }
                    if (!in_entry) {
                        Argument a;
                        a.type = ptype;
                        a.name = pname;
                        entry->args.push_back(a);
                        in_entry = std::make_shared<Value>(a);
                    }
                    shared_ptr<Value> in_block = thread_to(G, in_entry, entry, block);
                    internal_assert(in_block != nullptr)
                        << kFillerName << ": the record built in " << block->name
                        << " is reached from the entry by more than one way; the "
                        << "hoist cannot thread `" << pname << "` there";
                    threaded[where] = in_block;
                    values[pname] = in_block;
                }
                // The slice, each instruction over the values made so far.
                vector<shared_ptr<Value>> copies;
                const auto mapped = [&](const SymOperand &o) -> shared_ptr<Value> {
                    switch (o.kind) {
                    case SymOperand::Kind::Constant:
                        return std::make_shared<Value>(*o.constant);
                    case SymOperand::Kind::Param: {
                        const auto found = values.find(o.param);
                        internal_assert(found != values.end())
                            << h.name << "'s slice reads `" << o.param
                            << "`, which is not one of its array parameters";
                        return found->second;
                    }
                    case SymOperand::Kind::Field:
                        internal_assert(o.index < base_fields)
                            << h.name << " reads field " << o.index;
                        return M->operands[o.index];
                    case SymOperand::Kind::Ref:
                        internal_assert(o.index < copies.size())
                            << h.name << "'s slice reads its instruction " << o.index
                            << " before computing it";
                        return copies[o.index];
                    }
                    return nullptr;
                };
                for (const SymInstr &instr : h.slice) {
                    vector<shared_ptr<Value>> operands;
                    for (const SymOperand &o : instr.operands) {
                        operands.push_back(mapped(o));
                    }
                    auto copy = std::make_shared<Instruction>(
                        G.get_unique_name(), instr.type, instr.op, std::move(operands),
                        block);
                    copy->queried_type = instr.queried_type;
                    copy->intrinsic = instr.intrinsic;
                    copy->reduce = instr.reduce;
                    copy->shuffle = instr.shuffle;
                    made.push_back(copy);
                    copies.push_back(std::make_shared<Value>(copy));
                }
                internal_assert(!copies.empty()) << h.name << "'s slice has no root";
                M->operands.push_back(copies.back());
            }
            block->instrs.insert(block->instrs.begin() + long(at), made.begin(), made.end());
            at += made.size();
        }
    }
        if (changed) {
            refresh_preds(G);
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
