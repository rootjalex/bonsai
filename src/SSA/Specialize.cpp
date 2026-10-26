#include "SSA/Specialize.h"

#include "Error.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/Definitions.h"
#include "SSA/Rewrite.h"
#include "SSA/Simplify.h"
#include "Utils.h"

#include <functional>
#include <map>
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

// Every place a block holds a value: its instructions' operands and its
// terminator's, whatever kind it is.
template <typename F>
void for_each_value(Block &block, const F &fn) {
    const auto all = [&](vector<shared_ptr<Value>> &vs) {
        for (auto &v : vs) {
            fn(v);
        }
    };
    for (auto &instr : block.instrs) {
        all(instr->operands);
    }
    std::visit(overloads{
                   [](std::monostate &) {},
                   [&](Terminator::Jump &j) { all(j.args); },
                   [&](Terminator::Dispatch &d) {
                       fn(d.cond);
                       for (auto &t : d.targets) {
                           all(t.args);
                       }
                   },
                   [&](Terminator::Return &r) { fn(r.value); },
                   [&](Terminator::ParFor &p) {
                       fn(p.start);
                       fn(p.end);
                       fn(p.stride);
                       all(p.body.args);
                       all(p.cont.args);
                   },
                   [](Terminator::Yield &) {},
                   [&](Terminator::Spawn &s) {
                       all(s.call.args);
                       all(s.cont.args);
                   },
                   [&](Terminator::Join &j) { all(j.next.args); },
                   [&](Terminator::Defer &d) {
                       fn(d.queue);
                       fn(d.entry);
                   },
                   [&](Terminator::Call &c) {
                       all(c.call.args);
                       all(c.cont.args);
                   },
                   [&](Terminator::MultiCall &c) {
                       all(c.call.args);
                       all(c.cont.args);
                       for (auto &vs : c.varying) {
                           all(vs);
                       }
                       all(c.keys);
                       all(c.conds);
                   },
               },
               block.terminator.data);
}

// Whether the blocks `region` mention the name `param` -- as an operand, a
// condition, a value a terminator passes, or a block's argument. A body that
// never reads the parameter has nothing to specialize.
bool mentions(const vector<shared_ptr<Block>> &region, const string &param) {
    bool found = false;
    for (const auto &block : region) {
        for (const Argument &arg : block->args) {
            found = found || arg.name == param;
        }
        for_each_value(*block, [&](shared_ptr<Value> &v) {
            const auto *a = v ? std::get_if<Argument>(&v->data) : nullptr;
            found = found || (a != nullptr && a->name == param);
        });
    }
    return found;
}

// Every use of the name `from` in `blocks` renamed `to`: the operands and
// terminator values that name it, and the arguments of every block but
// `keep`, which carry it on under its name -- `keep` is the body's head,
// whose argument of that name is the value as given, and stays.
void rename_uses(const vector<shared_ptr<Block>> &blocks, const Block &keep,
                 const string &from, const string &to) {
    for (const auto &block : blocks) {
        for_each_value(*block, [&](shared_ptr<Value> &v) {
            const auto *a = v ? std::get_if<Argument>(&v->data) : nullptr;
            if (a != nullptr && a->name == from) {
                Argument renamed = *a;
                renamed.name = to;
                v = std::make_shared<Value>(renamed);
            }
        });
        if (block.get() == &keep) {
            continue;
        }
        for (Argument &a : block->args) {
            if (a.name == from) {
                a.name = to;
            }
        }
        if (const auto it = block->lookups.find(from); it != block->lookups.end()) {
            shared_ptr<Value> v = it->second;
            block->lookups.erase(it);
            if (auto *a = std::get_if<Argument>(&v->data)) {
                a->name = to;
            }
            block->lookups[to] = v;
        }
    }
}

// The values a jump passes, as the block `from` can name them: an argument
// threaded in through `from`'s predecessors, anything else as it is.
vector<shared_ptr<Value>> threaded(Block &from,
                                   const vector<shared_ptr<Value>> &args) {
    vector<shared_ptr<Value>> out;
    for (const auto &v : args) {
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            out.push_back(from.get_value(a->name, a->type));
        } else {
            out.push_back(v);
        }
    }
    return out;
}

shared_ptr<Value> constant_u32(uint64_t v) {
    return std::make_shared<Value>(Constant{UInt_t::make(32), v});
}

// An instruction emitted where the caller is building: its type, operation
// and operands; the caller names and places it.
using Emit = std::function<shared_ptr<Value>(const Type &, Instruction::Op,
                                             vector<shared_ptr<Value>>)>;

// The bool field `f` read out of `payload`, the arm's words, as the match
// lowering reads it (Lower/WordStorage.cpp, read_words): the byte of its
// word, `((payload[word] >> shift) & 0xff) != 0`.
shared_ptr<Value> read_fixed_field(const FixedField &f,
                                   const shared_ptr<Value> &payload,
                                   const Emit &emit) {
    const Type u32 = UInt_t::make(32);
    shared_ptr<Value> word =
        emit(u32, Instruction::Op::ExtractIdx, {payload, constant_u32(f.word)});
    if (f.shift != 0) {
        word = emit(u32, Instruction::Op::Shr, {word, constant_u32(f.shift)});
    }
    shared_ptr<Value> byte =
        emit(u32, Instruction::Op::BwAnd, {word, constant_u32(0xff)});
    return emit(Bool_t::make(), Instruction::Op::Ne, {byte, constant_u32(0)});
}

// `payload` with the field's byte set to `value`: its word with the byte
// cleared and the value put in, `(payload[word] & ~(0xff << shift)) |
// (value << shift)`, and the words rebuilt lane by lane around it (a vector
// is a MakeStruct of its lanes, SSA/SSA.cpp). Every read of the field out
// of this payload then reads the value, which the simplifier's known-bits
// rule folds to the constant (SSA/Simplify.cpp).
shared_ptr<Value> force_fixed_field(const FixedField &f, uint64_t value,
                                    const shared_ptr<Value> &payload,
                                    const Emit &emit) {
    const Type payload_t = payload->get_type();
    const Vector_t *words = payload_t.as<Vector_t>();
    internal_assert(words != nullptr && words->etype.is_uint() &&
                    words->etype.bits() == 32 && f.word < words->lanes)
        << "a variant's payload that is not 32-bit words: " << payload_t;
    const Type u32 = UInt_t::make(32);
    vector<shared_ptr<Value>> lanes;
    for (uint32_t k = 0; k < words->lanes; k++) {
        shared_ptr<Value> word =
            emit(u32, Instruction::Op::ExtractIdx, {payload, constant_u32(k)});
        if (k == f.word) {
            const uint64_t mask = (uint64_t(0xff) << f.shift) & 0xffffffffu;
            word = emit(u32, Instruction::Op::BwAnd,
                        {word, constant_u32(~mask & 0xffffffffu)});
            if (value != 0) {
                word = emit(u32, Instruction::Op::BwOr,
                            {word, constant_u32((value & 0xff) << f.shift)});
            }
        }
        lanes.push_back(word);
    }
    return emit(payload_t, Instruction::Op::MakeStruct, std::move(lanes));
}

// The reads of the payload words a copy rewrote (force_fixed_field) at a
// constant word read the word built there: `extract_idx(words, k)` is the
// k-th operand of the make_struct, the forced word where k is the field's.
// This is what the arm's binding of the field has become once the copy's
// struct has folded into the reads of it (the LoadField rule, SSA/Simplify.
// h); the byte the binding then takes out of the forced word folds to the
// constant by the known-bits rule there. Done here rather than as a rule of
// the simplifier because a run of calls whose arguments are the lanes of one
// vector is the shape the sort lowering recognizes a gang's run by (SSA/
// SortRecursion.cpp, lane_of), and a general fold of a lane out of a built
// vector would take that shape apart. A read names the words by the
// instruction or by the argument a block was handed them as.
void fold_forced_words(Function &f,
                       const vector<shared_ptr<Instruction>> &words) {
    map<string, shared_ptr<Instruction>> by_name;
    for (const auto &w : words) {
        by_name[w->name] = w;
    }
    vector<std::pair<shared_ptr<Instruction>, shared_ptr<Value>>> replace;
    for (const auto &block : f.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::ExtractIdx ||
                instr->operands.size() != 2) {
                continue;
            }
            string name;
            if (const auto *a = std::get_if<Argument>(&instr->operands[0]->data)) {
                name = a->name;
            } else if (const auto *i = std::get_if<shared_ptr<Instruction>>(
                           &instr->operands[0]->data)) {
                name = (*i)->name;
            }
            const auto it = by_name.find(name);
            const Constant *c = std::get_if<Constant>(&instr->operands[1]->data);
            if (it == by_name.end() || c == nullptr) {
                continue;
            }
            const std::optional<uint64_t> lane = std::visit(
                overloads{
                    [](uint64_t u) -> std::optional<uint64_t> { return u; },
                    [](int64_t i) -> std::optional<uint64_t> {
                        return i < 0 ? std::nullopt
                                     : std::optional<uint64_t>(uint64_t(i));
                    },
                    [](const auto &) -> std::optional<uint64_t> {
                        return std::nullopt;
                    },
                },
                c->data);
            if (lane.has_value() && *lane < it->second->operands.size()) {
                replace.emplace_back(instr, it->second->operands[*lane]);
            }
        }
    }
    for (const auto &[read, word] : replace) {
        replace_uses(f, read.get(), word);
    }
}

// The instruction a value is, for the words force_fixed_field built.
shared_ptr<Instruction> instruction_of(const shared_ptr<Value> &v) {
    const auto *i = std::get_if<shared_ptr<Instruction>>(&v->data);
    internal_assert(i != nullptr) << "a forced payload that is not an instruction";
    return *i;
}

// Whether `name` carries `!label` as one of its segments: a loop's block or
// index named by the copies that made it (`!par_body_0!VolPath!True`).
bool carries(const string &name, const string &label) {
    const string segment = "!" + label;
    return name.ends_with(segment) || name.find(segment + "!") != string::npos;
}

} // namespace

void specialize_loops(FuncMap &fmap, const string &fname, const string &param_name,
                      const string &variant, const string &loop_name,
                      const string &arm, const string &field_name,
                      const map<string, ir::Program::AdtStorage> &storages) {
    const bool field_form = !field_name.empty();
    const string key = field_form ? arm + "::" + field_name : param_name;
    const auto fit = fmap.find(fname);
    internal_assert(fit != fmap.end())
        << fname << ".specialize(" << key << "): no function " << fname;
    Function &f = *fit->second;
    string spelled = fname;
    for (size_t at = 0; at < variant.size();) {
        const size_t next = variant.find('!', at + 1);
        spelled += "[" + variant.substr(at + 1, next - at - 1) + "]";
        at = next == string::npos ? variant.size() : next;
    }
    const string where = spelled + ".specialize(" +
                         (loop_name.empty() ? "" : loop_name + ", ") + key + ")";

    // The field form names no parameter: it is the one whose variant type
    // has an arm of that name, and there has to be one such.
    string param = param_name;
    if (field_form) {
        for (const Argument &a : f.blocks.front()->args) {
            const Ptr_t *p = a.type.as<Ptr_t>();
            const Struct_t *s = (p != nullptr ? p->etype : a.type).as<Struct_t>();
            const auto it = s != nullptr ? storages.find(s->name) : storages.end();
            if (it == storages.end()) {
                continue;
            }
            for (const auto &[vname, tag] : it->second.variants) {
                if (vname != arm) {
                    continue;
                }
                internal_assert(param.empty())
                    << where << ": both " << param << " and " << a.name
                    << " are of a variant type with an arm named " << arm
                    << "; the field form takes one parameter's arm";
                param = a.name;
            }
        }
        internal_assert(!param.empty())
            << where << ": no parameter of " << fname
            << " is of a variant type with an arm named " << arm
            << "; the field form names an arm of a parameter's variant type "
            << "and a bool field of it, `<Arm>::<field>`";
    }

    // The parameter, and what its variant type is stored as. A variant
    // parameter arrives as its storage struct, or as a pointer to one
    // (Lower/Mutability.cpp hands aggregates over by pointer).
    const Argument *parameter = nullptr;
    for (const Argument &a : f.blocks.front()->args) {
        if (a.name == param) {
            parameter = &a;
        }
    }
    internal_assert(parameter != nullptr)
        << where << ": " << fname << " has no parameter " << param;
    // Or a bool -- pbrt's `haveMedia`, fixed for the whole render as the
    // integrator is -- copied for `False` and `True`: Halide's
    // `specialize(cond)` exactly, the value its own tag (KeyTag).
    const bool boolean = parameter->type.is<Bool_t>();
    const Ptr_t *pointer = parameter->type.as<Ptr_t>();
    const Type struct_type = pointer != nullptr ? pointer->etype : parameter->type;
    const Struct_t *storage_t = struct_type.as<Struct_t>();
    const auto storage_it = storage_t != nullptr ? storages.find(storage_t->name)
                                                 : storages.end();
    if (!boolean && storage_it == storages.end()) {
        string known;
        for (const auto &[name, _] : storages) {
            known += (known.empty() ? "" : ", ") + name;
        }
        internal_error << where << ": " << param << " is a " << parameter->type
                       << ", not a variant type or a bool (the variant types "
                       << "stored: " << (known.empty() ? "none" : known) << ")";
    }
    const ir::Program::AdtStorage *storage = boolean ? nullptr : &storage_it->second;
    // The copies, in the order the dispatch below takes them: its k-th
    // target on k, so a variant's tag has to be its position, and a bool's
    // false comes first -- as does a field's, whose copies are a bool's.
    vector<std::pair<string, uint64_t>> variants;
    size_t tag_at = 0;
    size_t payload_at = 0;
    if (boolean) {
        variants = {{"False", 0}, {"True", 1}};
    } else {
        internal_assert(storage->inline_storage)
            << "[unimplemented] " << where << ": " << storage_t->name
            << " is stored as a tagged index, whose tag is the top bits of one "
            << "word; specialize() handles a tag beside the payload";
        tag_at = find_struct_index(storage->tag_field, storage_t->fields);
        payload_at = find_struct_index(storage->payload_field, storage_t->fields);
        for (size_t k = 0; k < storage->variants.size(); k++) {
            internal_assert(storage->variants[k].second == k)
                << "[unimplemented] " << where << ": variant "
                << storage->variants[k].first << " has tag "
                << storage->variants[k].second << ", not its position " << k;
            if (!field_form) {
                variants.push_back(storage->variants[k]);
            }
        }
    }
    // The field form: the arm's field, a bool, found in the arm's fields as
    // stored (ir::Program::AdtStorage::fields_of) -- its byte of the
    // payload's words -- and the arm's tag, which every copy keeps.
    std::optional<FixedField> fixed;
    if (field_form) {
        variants = {{"False", 0}, {"True", 1}};
        const auto fields_it = storage->fields_of.find(arm);
        internal_assert(fields_it != storage->fields_of.end())
            << "[unimplemented] " << where << ": the arm " << arm
            << " is stored by index, its fields in a pool rather than in the "
            << "value; the field form reads a field out of the payload";
        const ir::Program::AdtStorage::Field *found = nullptr;
        string known;
        for (const auto &fld : fields_it->second) {
            if (fld.name == field_name) {
                found = &fld;
            }
            known += (known.empty() ? "" : ", ") + fld.name;
        }
        internal_assert(found != nullptr)
            << where << ": " << arm << " has no field " << field_name
            << " (its fields: " << (known.empty() ? "none" : known) << ")";
        internal_assert(found->type.is<Bool_t>())
            << "[unimplemented] " << where << ": " << arm << "::" << field_name
            << " is a " << found->type << "; the field form of specialize "
            << "takes a bool field -- Halide's specialize(cond) -- and copies "
            << "the loop for false and true";
        FixedField ff;
        ff.arm = arm;
        ff.name = field_name;
        for (const auto &[vname, tag] : storage->variants) {
            if (vname == arm) {
                ff.arm_tag = tag;
            }
        }
        ff.word = found->offset / 4;
        ff.shift = unsigned(found->offset % 4) * 8;
        fixed = ff;
    }

    // The loops to specialize: the function's parallel loops that are not
    // inside another -- a thread loop inside a block loop is copied with the
    // block loop's body. Bound or not: under a bind each copy is a kernel or
    // a thread body of its own, and a sequential loop is copied the same
    // way, which is what lets the copies be checked without hardware.
    vector<shared_ptr<Block>> loop_blocks;
    for (const auto &block : f.blocks) {
        if (std::holds_alternative<Terminator::ParFor>(block->terminator.data)) {
            loop_blocks.push_back(block);
        }
    }
    vector<shared_ptr<Block>> outermost;
    for (const auto &block : loop_blocks) {
        // The loop form: the loop named, wherever it sits, with every copy
        // an earlier specialize made of it (`shadow!VolPath`), as the
        // directives that name a loop find them (resolve_loops).
        if (!loop_name.empty()) {
            const string &index =
                std::get<Terminator::ParFor>(block->terminator.data).index;
            if ((index == loop_name || index.rfind(loop_name + "!", 0) == 0) &&
                (!field_form || carries(index, arm))) {
                outermost.push_back(block);
            }
            continue;
        }
        bool inside = false;
        for (const auto &other : loop_blocks) {
            if (other == block) {
                continue;
            }
            const auto &loop = std::get<Terminator::ParFor>(other->terminator.data);
            if (Cfg(f, loop.body.name).contains(*block)) {
                inside = true;
                break;
            }
        }
        // One variant's loops only, when the directive named one: an earlier
        // specialize left them as the selectors it made, `p!VolPath`, whose
        // names end in the variant's suffix (or carry it, once specialized
        // again: `p!VolPath!Halton`).
        // The field form reads the arm's payload, so its loops are the
        // arm's copies alone, whatever the directive was written on.
        const bool of_variant =
            (variant.empty() || block->name.ends_with(variant) ||
             block->name.find(variant + "!") != string::npos) &&
            (!field_form || carries(block->name, arm));
        if (!inside && of_variant) {
            outermost.push_back(block);
        }
    }
    internal_assert(!outermost.empty())
        << where << ": " << fname
        << (field_form
                ? " has no loop of the " + arm + " arm" +
                      (loop_name.empty() ? "" : " named " + loop_name) +
                      ": `" + fname + ".specialize(" + param +
                      ")` makes one per arm first, and the field form is "
                      "written on the arm's, `" + fname + "[" + arm +
                      "].specialize(" + key + ")`"
            : !loop_name.empty()
                ? " has no loop named " + loop_name
                : variant.empty()
                      ? string(" has no parallel loop; specialize() copies a "
                               "parfor's body per variant")
                      : " has no loop of that variant; `" + spelled +
                            "` names a copy an earlier specialize made");

    // The payload words the copies rewrite (force_fixed_field), whose reads
    // are folded once the copies' structs have folded into them.
    vector<shared_ptr<Instruction>> forced_words;
    for (const shared_ptr<Block> &at : outermost) {
        const string body_name =
            std::get<Terminator::ParFor>(at->terminator.data).body.name;
        const Cfg region_cfg(f, body_name);
        const vector<shared_ptr<Block>> region = region_cfg.blocks();
        // The parameter as the region refers to it: by its name, or -- for
        // one that arrives by pointer -- by the name of a whole read of it
        // that a deferral hoisted before the loop (`@2335 = load(sampler)`,
        // SSA/Defer.cpp), which is how a drain's body knows the sampler. A
        // body that refers to it no way has nothing to specialize.
        vector<string> aliases;
        if (mentions(region, param)) {
            aliases.push_back(param);
        }
        if (pointer != nullptr) {
            Definitions defs(f, /*lenient=*/true);
            set<string> seen;
            for (const auto &block : region) {
                for_each_value(*block, [&](shared_ptr<Value> &v) {
                    string name;
                    if (const auto *a = v ? std::get_if<Argument>(&v->data) : nullptr) {
                        name = a->name;
                    } else if (const auto *i = v ? std::get_if<shared_ptr<Instruction>>(&v->data)
                                                 : nullptr) {
                        name = (*i)->name;
                    }
                    if (name.empty() || name == param || !seen.insert(name).second) {
                        return;
                    }
                    const Definition d = defs.of(block->name, v);
                    const auto *i = d.value ? std::get_if<shared_ptr<Instruction>>(&d.value->data)
                                            : nullptr;
                    // A read made before the loop: one inside the body is a
                    // read of the pointer the copy's own slot answers.
                    if (i == nullptr || d.block.empty() || region_cfg.contains(d.block) ||
                        (*i)->op != Instruction::Op::Load ||
                        (*i)->operands.size() != 1) {
                        return;
                    }
                    const Argument *p = defs.parameter(d.block, (*i)->operands[0]);
                    if (p != nullptr && p->name == param) {
                        aliases.push_back(name);
                    }
                });
            }
        }
        if (aliases.empty()) {
            continue;
        }

        // The tag's source, read where the loop was, before the loop is
        // copied: threading the parameter into the loop's block may give
        // the block an argument and pass it on along the loop's own
        // continuation -- a round loop carries it back around through the
        // drain's block -- and the copies below have to carry it too.
        shared_ptr<Value> given_here = at->get_value(param, parameter->type);
        const Terminator::ParFor loop =
            std::get<Terminator::ParFor>(at->terminator.data);

        // One copy of the body per variant, each entered by a selector block
        // that is the loop as it was, over the copy; the loop's block then
        // dispatches on the tag to the selectors. Wired first, so that what
        // the copies need can be threaded in through the selectors.
        struct Copy {
            string variant;
            uint64_t tag;
            map<string, shared_ptr<Block>> blocks;
            shared_ptr<Block> selector;
        };
        vector<Copy> made;
        Terminator::Dispatch dispatch;
        for (const auto &[vname, tag] : variants) {
            Copy copy;
            copy.variant = vname;
            copy.tag = tag;
            copy.blocks = clone_region(f, region, "!" + vname);
            for (const auto &block : region) {
                f.blocks.push_back(copy.blocks.at(block->name));
            }
            copy.selector = std::make_shared<Block>();
            copy.selector->name = at->name + "!" + vname;
            copy.selector->owner = at->owner;
            Terminator::ParFor entered = loop;
            entered.body.name = copy.blocks.at(loop.body.name)->name;
            copy.selector->terminator.data = entered;
            f.blocks.push_back(copy.selector);
            dispatch.targets.push_back(Terminator::Jump{copy.selector->name, {}});
            made.push_back(std::move(copy));
        }

        // The tag, read from the value above where the loop was, and the
        // dispatch on it.
        const auto at_instr = [&](const string &name, const Type &type,
                                  Instruction::Op op,
                                  vector<shared_ptr<Value>> operands) {
            auto instr = std::make_shared<Instruction>(name, type, op,
                                                       std::move(operands), at);
            at->instrs.push_back(instr);
            auto value = std::make_shared<Value>(instr);
            at->lookups[name] = value;
            return value;
        };
        // Or, for the field form, the field's byte of the payload's words.
        shared_ptr<Value> tag_value;
        if (boolean) {
            tag_value = given_here; // its own tag
        } else {
            const size_t read_at = fixed.has_value() ? payload_at : tag_at;
            const Type &read_t = storage_t->fields[read_at].type;
            shared_ptr<Value> read;
            if (pointer != nullptr) {
                shared_ptr<Value> ptr =
                    at_instr(f.get_unique_name(), Ptr_t::make(read_t),
                             Instruction::Op::FieldPtr,
                             {given_here, constant_u32(read_at)});
                read = at_instr(f.get_unique_name(), read_t,
                                Instruction::Op::Load, {ptr});
            } else {
                read = at_instr(f.get_unique_name(), read_t,
                                Instruction::Op::LoadField,
                                {given_here, constant_u32(read_at)});
            }
            if (fixed.has_value()) {
                const Emit emit = [&](const Type &t, Instruction::Op op,
                                      vector<shared_ptr<Value>> operands) {
                    return at_instr(f.get_unique_name(), t, op,
                                    std::move(operands));
                };
                tag_value = read_fixed_field(*fixed, read, emit);
            } else {
                tag_value = read;
            }
        }
        dispatch.cond = tag_value;
        at->terminator.data = std::move(dispatch);
        refresh_preds(f);

        for (Copy &copy : made) {
            Block &head = *copy.blocks.at(loop.body.name);
            // What the selector's loop passes its body and its continuation,
            // and its bounds, as the selector can name them: the values the
            // original loop had, threaded in from the loop's block.
            auto &entered = std::get<Terminator::ParFor>(copy.selector->terminator.data);
            entered.body.args = threaded(*copy.selector, entered.body.args);
            entered.cont.args = threaded(*copy.selector, entered.cont.args);
            for (shared_ptr<Value> *bound : {&entered.start, &entered.end, &entered.stride}) {
                if (const auto *a = std::get_if<Argument>(&(*bound)->data)) {
                    *bound = copy.selector->get_value(a->name, a->type);
                }
            }

            // Inside the copy, `param` is the specialized value: every use
            // of each of its names renamed to one of this copy's own,
            // defined in the head as the given value with the tag set to
            // this variant's. The given value keeps its name, threaded into
            // the head as before. An alias that is a whole read of the
            // pointer is the struct itself, and is specialized by value.
            vector<shared_ptr<Block>> copied;
            for (const auto &block : region) {
                copied.push_back(copy.blocks.at(block->name));
            }
            struct Specialized {
                string fresh;
                shared_ptr<Value> spec;
                bool by_pointer = false;
            };
            vector<Specialized> specialized;
            vector<shared_ptr<Instruction>> prologue;
            const auto make = [&](const string &name, const Type &type,
                                  Instruction::Op op,
                                  vector<shared_ptr<Value>> operands) {
                auto instr = std::make_shared<Instruction>(
                    name, type, op, std::move(operands), head.shared_from_this());
                prologue.push_back(instr);
                auto value = std::make_shared<Value>(instr);
                head.lookups[name] = value;
                return value;
            };
            for (const string &alias : aliases) {
                const bool by_pointer = alias == param && pointer != nullptr;
                const Type alias_type = alias == param ? parameter->type : struct_type;
                const string fresh = alias + "!" + copy.variant;
                rename_uses(copied, head, alias, fresh);
                shared_ptr<Value> given = head.get_value(alias, alias_type);
                // The given value as a whole -- read through the pointer,
                // when that is how it arrived -- and its fields, the tag
                // replaced. For a bool, the constant.
                shared_ptr<Value> whole =
                    by_pointer ? make(f.get_unique_name(), struct_type,
                                      Instruction::Op::Load, {given})
                               : given;
                // For the field form the tag is the arm's, and the payload
                // is the given one with the field's byte set to the copy's
                // value.
                const uint64_t tag = fixed.has_value() ? fixed->arm_tag : copy.tag;
                vector<shared_ptr<Value>> fields;
                for (size_t i = 0; !boolean && i < storage_t->fields.size(); i++) {
                    if (i == tag_at) {
                        fields.push_back(std::make_shared<Value>(
                            storage->tag_type.is_uint()
                                ? Constant{storage->tag_type, uint64_t(tag)}
                                : Constant{storage->tag_type, int64_t(tag)}));
                        continue;
                    }
                    shared_ptr<Value> read = make(f.get_unique_name(),
                                                  storage_t->fields[i].type,
                                                  Instruction::Op::LoadField,
                                                  {whole, constant_u32(i)});
                    if (fixed.has_value() && i == payload_at) {
                        const Emit emit = [&](const Type &t, Instruction::Op op,
                                              vector<shared_ptr<Value>> operands) {
                            return make(f.get_unique_name(), t, op,
                                        std::move(operands));
                        };
                        read = force_fixed_field(*fixed, copy.tag, read, emit);
                        forced_words.push_back(instruction_of(read));
                    }
                    fields.push_back(read);
                }
                shared_ptr<Value> spec;
                if (boolean) {
                    spec = std::make_shared<Value>(
                        Constant{parameter->type, copy.tag != 0});
                } else if (by_pointer) {
                    // Stored where the body can point at it: a slot under
                    // the copy's name, holding the specialized value.
                    spec = make(f.get_unique_name(), struct_type,
                                Instruction::Op::MakeStruct, std::move(fields));
                    shared_ptr<Value> slot =
                        make(fresh, parameter->type, Instruction::Op::Alloca, {});
                    prologue.push_back(std::make_shared<Instruction>(
                        Instruction::Op::Store, vector<shared_ptr<Value>>{slot, spec},
                        head.shared_from_this()));
                } else {
                    spec = make(fresh, alias_type, Instruction::Op::MakeStruct,
                                std::move(fields));
                }
                specialized.push_back({fresh, spec, by_pointer});
            }
            head.instrs.insert(head.instrs.begin(), prologue.begin(),
                               prologue.end());

            // The copy's loops are the variant's: `p` becomes `p!VolPath`,
            // so that a directive written after this one can name the loop
            // of one variant (`render[VolPath].bind(p, GPUBlock)`) or every
            // variant's (`render.bind(p, GPUBlock)`, which finds them all;
            // see resolve_loops in SSA/Convert.cpp). The index is the body's
            // first argument and is threaded through the body under its
            // name, so the name changes wherever the copy carries it.
            const auto rename_loop = [&](Terminator::ParFor &p) {
                const string was = p.index;
                p.index = was + "!" + copy.variant;
                rename_uses(copied, *copy.selector, was, p.index);
            };
            rename_loop(entered);
            for (const auto &block : copied) {
                if (auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data)) {
                    rename_loop(*p);
                }
            }

            // The value is followed into the callees it is passed to: a
            // callee given the specialized value is copied with that
            // parameter's tag fixed, and the copy called instead, so that
            // its match folds too -- `integrator_li!VolPath` has the volpath
            // arm alone. Recursively, as far as the value is passed on.
            KeyVariant kv{copy.variant, fixed.has_value() ? fixed->arm_tag : copy.tag, {}};
            if (fixed.has_value()) {
                kv.field = *fixed;
                kv.field->value = copy.tag;
            }
            for (const Specialized &s : specialized) {
                specialize_callees(fmap, copied, s.fresh, kv, storages);
            }

            // The uses then read the struct itself. By value, in place of
            // its name, which the simplifier's field rule cannot see through
            // (SSA/Simplify.h). By pointer, in place of each whole read of
            // the slot: the slot is written above and nowhere else -- the
            // body never assigns a parameter that arrived by pointer only
            // because variants travel that way (Lower/Mutability.cpp) -- and
            // would otherwise be promoted at the end (SSA/PromoteAllocas.h),
            // after the last simplify. Either way the match's read of the
            // tag sees the struct, and folds. After the callees, which found
            // the value by its name.
            for (const Specialized &s : specialized) {
                const auto is_fresh = [&](const shared_ptr<Value> &u) {
                    const auto *a = u ? std::get_if<Argument>(&u->data) : nullptr;
                    return a != nullptr && a->name == s.fresh;
                };
                if (s.by_pointer) {
                    vector<shared_ptr<Instruction>> reads;
                    for (const auto &block : copied) {
                        for (const auto &instr : block->instrs) {
                            if (instr->op == Instruction::Op::Load &&
                                instr->operands.size() == 1 &&
                                is_fresh(instr->operands[0])) {
                                reads.push_back(instr);
                            }
                        }
                    }
                    for (const auto &read : reads) {
                        replace_uses(f, read.get(), s.spec);
                    }
                } else {
                    for (const auto &block : copied) {
                        for_each_value(*block, [&](shared_ptr<Value> &u) {
                            if (is_fresh(u)) {
                                u = s.spec;
                            }
                        });
                    }
                }
            }
        }
        refresh_preds(f);
    }

    // The bodies the loops no longer enter.
    remove_unreachable_blocks(f);
    // The tag is a constant in every copy: the matches on it fold to their
    // arms now, so that what the directives after this one schedule is the
    // variant's code alone (SSA/Simplify.h). The field form's reads of its
    // forced words fold once the structs have folded into them, and the
    // field's byte is then a constant too.
    simplify(f);
    if (!forced_words.empty()) {
        fold_forced_words(f, forced_words);
        simplify(f);
    }
}

void specialize_callees(FuncMap &fmap, const vector<shared_ptr<Block>> &blocks,
                        const string &value, const KeyVariant &v,
                        const map<string, ir::Program::AdtStorage> &storages) {
    // Whether `arg` is the value: named so, or a load through the pointer
    // named so -- a variant parameter arrives by pointer (Lower/Mutability
    // .cpp), and a callee taking it by value is handed what the pointer
    // holds.
    const std::function<bool(const shared_ptr<Value> &)> names =
        [&](const shared_ptr<Value> &arg) -> bool {
        if (!arg) {
            return false;
        }
        if (const auto *a = std::get_if<Argument>(&arg->data)) {
            return a->name == value;
        }
        if (const auto *i = std::get_if<shared_ptr<Instruction>>(&arg->data)) {
            if ((*i)->name == value) {
                return true;
            }
            // Through a load of the pointer, for a callee taking the value;
            // through the address of the value, for a callee taking a
            // pointer to it, which is how a variant travels.
            return ((*i)->op == Instruction::Op::Load ||
                    (*i)->op == Instruction::Op::AddressOf) &&
                   !(*i)->operands.empty() && names((*i)->operands[0]);
        }
        return false;
    };
    for (const auto &block : blocks) {
        // A call, or a spawned one (Terminator::Spawn): the same edge to a
        // callee handed the value.
        Terminator::Jump *call = nullptr;
        if (auto *c = std::get_if<Terminator::Call>(&block->terminator.data)) {
            call = &c->call;
        } else if (auto *s = std::get_if<Terminator::Spawn>(&block->terminator.data)) {
            call = &s->call;
        }
        if (call == nullptr) {
            continue;
        }
        const auto callee = fmap.find(call->name);
        if (callee == fmap.end()) {
            continue;
        }
        const vector<Argument> &params = callee->second->blocks.front()->args;
        for (size_t j = 0; j < call->args.size() && j < params.size(); j++) {
            if (!names(call->args[j]) ||
                key_variants(params[j].type, storages).empty()) {
                continue;
            }
            const string clone_name = call->name + "!" + v.label;
            const bool fresh_clone = !fmap.contains(clone_name);
            if (fresh_clone) {
                fmap[clone_name] = specialize_function(
                    *callee->second, clone_name, params[j].name, v, storages);
            }
            call->name = clone_name;
            if (fresh_clone) {
                // Inside the copy the parameter is the specialized value,
                // under the name specialize_function gave it.
                specialize_callees(fmap, fmap.at(clone_name)->blocks,
                                   params[j].name + "!" + v.label, v, storages);
            }
            break; // one specialized value per call
        }
    }
}

//===----------------------------------------------------------------------===//
// A queue's specialize: the drained function per variant of a value
//===----------------------------------------------------------------------===//

namespace {

Type type_of(const Value &v) {
    return std::visit(
        overloads{[](const Constant &c) { return c.type; },
                  [](const Argument &a) { return a.type; },
                  [](const shared_ptr<Instruction> &i) { return i->type; }},
        v.data);
}

// Lower/Options.cpp stores an optional as a struct named `_option<n>` of
// its `value` and a `set` flag.
bool is_optional_storage(const Struct_t &s) {
    return s.fields.size() == 2 && s.name.rfind("_option", 0) == 0 &&
           s.fields[0].name == "value" && s.fields[1].name == "set";
}

// Where a value's tag is and what it can be. A bool is its own tag -- no
// field to read, the value itself -- with the two variants `False` and
// `True`, in that order, since a dispatch on a bool takes its second target
// when true. Capitalized as a variant's name is, because `true` and `false`
// are keywords and a schedule names a copy by its label (`render[True]`).
struct KeyTag {
    std::optional<size_t> field;
    Type type;
    vector<KeyVariant> variants;
};

std::optional<KeyTag>
key_tag_of(const Type &type,
           const map<string, ir::Program::AdtStorage> &storages,
           const string &what) {
    if (type.is<Bool_t>()) {
        return KeyTag{std::nullopt, Bool_t::make(), {{"False", 0}, {"True", 1}}};
    }
    const Struct_t *s = type.as<Struct_t>();
    if (s == nullptr) {
        return std::nullopt;
    }
    if (is_optional_storage(*s)) {
        return KeyTag{1, Bool_t::make(), {{"None", 0}, {"Some", 1}}};
    }
    const auto it = storages.find(s->name);
    if (it == storages.end()) {
        return std::nullopt;
    }
    const ir::Program::AdtStorage &storage = it->second;
    internal_assert(storage.inline_storage)
        << "[unimplemented] " << what << ": " << s->name
        << " is stored as a tagged index, whose tag is the top bits of one "
        << "word; a queue's specialize handles a tag beside the payload";
    KeyTag tag{find_struct_index(storage.tag_field, s->fields),
               storage.tag_type,
               {}};
    for (size_t k = 0; k < storage.variants.size(); k++) {
        internal_assert(storage.variants[k].second == k)
            << "[unimplemented] " << what << ": variant "
            << storage.variants[k].first << " has tag "
            << storage.variants[k].second << ", not its position " << k;
        tag.variants.push_back(
            KeyVariant{storage.variants[k].first, storage.variants[k].second});
    }
    return tag;
}

shared_ptr<Value> tag_constant(const KeyTag &tag, uint64_t value) {
    if (tag.type.is<Bool_t>()) {
        return std::make_shared<Value>(Constant{tag.type, value != 0});
    }
    return std::make_shared<Value>(tag.type.is_uint()
                                       ? Constant{tag.type, uint64_t(value)}
                                       : Constant{tag.type, int64_t(value)});
}

// What a key may be computed by, for the push to compute it too: nothing
// that traps, allocates or has an effect. A load is allowed and its source
// checked by the caller.
bool computes_a_key(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Cast:
    case Instruction::Op::Eq:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::GEP:
    case Instruction::Op::LAnd:
    case Instruction::Op::LOr:
    case Instruction::Op::Leq:
    case Instruction::Op::Load:
    case Instruction::Op::LoadField:
    case Instruction::Op::Lt:
    case Instruction::Op::MakeStruct:
    case Instruction::Op::Max:
    case Instruction::Op::Min:
    case Instruction::Op::Mul:
    case Instruction::Op::Ne:
    case Instruction::Op::Not:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
    case Instruction::Op::Set:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::Sub:
    case Instruction::Op::Xor:
        return true;
    default:
        return false;
    }
}

} // namespace

vector<KeyVariant>
key_variants(const Type &type,
             const map<string, ir::Program::AdtStorage> &storages) {
    // Through a pointer: a variant parameter arrives by one.
    const Type held = type.is<Ptr_t>() ? type.as<Ptr_t>()->etype : type;
    const auto tag = key_tag_of(held, storages, "key_variants");
    return tag.has_value() ? tag->variants : vector<KeyVariant>{};
}

shared_ptr<Value> key_value(const Function &func, const string &key) {
    for (const auto &block : func.blocks) {
        for (const auto &in : block->instrs) {
            if (in->name == key) {
                return std::make_shared<Value>(in);
            }
        }
    }
    for (const Argument &a : func.blocks.front()->args) {
        if (a.name == key) {
            return std::make_shared<Value>(a);
        }
    }
    return nullptr;
}

shared_ptr<Function>
specialize_function(const Function &func, const string &name,
                    const string &key, const KeyVariant &v,
                    const map<string, ir::Program::AdtStorage> &storages) {
    const string what =
        func.blocks.front()->name + " with " + key + " = " + v.label;
    shared_ptr<Function> copy = clone_function(func);
    // A function's entry block carries its name; whatever in the copy names
    // the original -- a jump to the entry, a call of itself -- follows.
    const string was = copy->blocks.front()->name;
    copy->blocks.front()->name = name;
    for (const auto &block : copy->blocks) {
        for (Terminator::Jump *jump : jumps_of(*block)) {
            if (jump->name == was) {
                jump->name = name;
            }
        }
        // A call of itself, by whichever terminator makes it: Call,
        // MultiCall or Spawn.
        if (Terminator::Jump *callee = block->terminator.callee();
            callee != nullptr && callee->name == was) {
            callee->name = name;
        }
    }

    const shared_ptr<Value> given = key_value(*copy, key);
    internal_assert(given) << what << ": " << func.blocks.front()->name
                           << " has no value named " << key;
    const Type type = type_of(*given);
    // A variant parameter arrives by pointer (Lower/Mutability.cpp): the
    // value is what it points at, and the copy gives the parameter's name to
    // a slot of its own holding the specialized value, so that the body's
    // reads through the pointer read the struct with the tag fixed.
    const bool by_pointer = type.is<Ptr_t>();
    const Type held = by_pointer ? type.as<Ptr_t>()->etype : type;
    const auto tag = key_tag_of(held, storages, what);
    internal_assert(tag.has_value())
        << what << ": " << key << " is a " << held
        << ", not a variant type, an optional or a bool";
    const Struct_t *s = held.as<Struct_t>();
    internal_assert(!tag->field.has_value() || s != nullptr) << what;

    // The specialized value is defined right after the key: after its
    // instruction, or first thing in the entry for a parameter. It is the
    // key's fields with the tag replaced -- or, for a bool, the constant.
    const string fresh = key + "!" + v.label;
    copy->reserve_name(fresh);
    shared_ptr<Block> at;
    size_t position = 0;
    const auto *instr = std::get_if<shared_ptr<Instruction>>(&given->data);
    if (instr != nullptr) {
        at = (*instr)->owner.lock();
        internal_assert(at) << what << ": " << key << " outlived its block";
        for (size_t i = 0; i < at->instrs.size(); i++) {
            if (at->instrs[i].get() == instr->get()) {
                position = i + 1;
            }
        }
    } else {
        at = copy->blocks.front();
    }
    internal_assert(!by_pointer || instr == nullptr)
        << "[unimplemented] " << what << ": " << key
        << " is a pointer computed inside the function";
    vector<shared_ptr<Instruction>> prologue;
    const auto make = [&](const string &iname, const Type &itype,
                          Instruction::Op op,
                          vector<shared_ptr<Value>> operands) {
        auto in = std::make_shared<Instruction>(iname, itype, op,
                                                std::move(operands), at);
        prologue.push_back(in);
        auto value = std::make_shared<Value>(in);
        at->lookups[iname] = value;
        return value;
    };
    const shared_ptr<Value> whole =
        by_pointer ? make(copy->get_unique_name(), held, Instruction::Op::Load, {given})
                   : given;
    // A field fixed with the tag (the field form of a loop's specialize,
    // followed here from the loop): the payload's words with its byte set.
    std::optional<size_t> payload_at;
    if (v.field.has_value()) {
        const auto st = s != nullptr ? storages.find(s->name) : storages.end();
        internal_assert(st != storages.end())
            << what << ": " << v.field->arm << "::" << v.field->name
            << " fixed on a value that is not a variant type's";
        payload_at = find_struct_index(st->second.payload_field, s->fields);
    }
    vector<shared_ptr<Value>> fields;
    vector<shared_ptr<Instruction>> forced_words;
    for (size_t i = 0; tag->field.has_value() && i < s->fields.size(); i++) {
        if (i == *tag->field) {
            fields.push_back(tag_constant(*tag, v.tag));
            continue;
        }
        shared_ptr<Value> read = make(copy->get_unique_name(), s->fields[i].type,
                                      Instruction::Op::LoadField,
                                      {whole, constant_u32(i)});
        if (payload_at.has_value() && i == *payload_at) {
            const Emit emit = [&](const Type &t, Instruction::Op op,
                                  vector<shared_ptr<Value>> operands) {
                return make(copy->get_unique_name(), t, op, std::move(operands));
            };
            read = force_fixed_field(*v.field, v.field->value, read, emit);
            forced_words.push_back(instruction_of(read));
        }
        fields.push_back(read);
    }
    shared_ptr<Value> spec;
    if (!tag->field.has_value()) {
        internal_assert(!by_pointer) << what << ": a bool arrives by value";
        spec = tag_constant(*tag, v.tag);
    } else if (by_pointer) {
        spec = make(copy->get_unique_name(), held, Instruction::Op::MakeStruct,
                    std::move(fields));
        const shared_ptr<Value> slot =
            make(fresh, type, Instruction::Op::Alloca, {});
        prologue.push_back(std::make_shared<Instruction>(
            Instruction::Op::Store, vector<shared_ptr<Value>>{slot, spec}, at));
    } else {
        spec = make(fresh, type, Instruction::Op::MakeStruct, std::move(fields));
    }

    // Every use of the key reads the specialized value: an instruction's
    // uses by pointer -- the reads of its fields above are not in any block
    // yet, and keep the key -- and a parameter's by name.
    if (instr != nullptr) {
        for (const auto &block : copy->blocks) {
            for_each_value(*block, [&](shared_ptr<Value> &u) {
                const auto *ui = u ? std::get_if<shared_ptr<Instruction>>(&u->data)
                                   : nullptr;
                if (ui != nullptr && ui->get() == instr->get()) {
                    u = spec;
                }
            });
        }
        at->lookups[key] = spec;
        // The blocks below were handed the value under its name; under the
        // new name now, so that what they pass it on to is seen to be
        // passed the specialized value (specialize_callees), and the struct
        // is found through the argument (SSA/Simplify.cpp, defined_by).
        rename_uses(copy->blocks, *at, key, fresh);
    } else {
        rename_uses(copy->blocks, *at, key, fresh);
        // The uses then read the struct itself rather than its name: a name
        // is what a block below receives, which the simplifier's field rule
        // cannot see through (SSA/Simplify.h), and the fold -- of the match
        // on the tag, and of the arm's reads of the other fields -- is the
        // point. The entry defines the struct, so every block reaches it.
        // By pointer, the whole reads of the slot are the struct: the slot
        // is written once, in the prologue, and the body never assigns a
        // parameter that arrived by pointer only because variants travel
        // that way; its other uses -- a field's address, the pointer handed
        // on -- keep the slot.
        if (by_pointer) {
            vector<shared_ptr<Instruction>> reads;
            for (const auto &block : copy->blocks) {
                for (const auto &in : block->instrs) {
                    const auto *a = in->op == Instruction::Op::Load && in->operands.size() == 1
                                        ? std::get_if<Argument>(&in->operands[0]->data)
                                        : nullptr;
                    if (a != nullptr && a->name == fresh) {
                        reads.push_back(in);
                    }
                }
            }
            for (const auto &read : reads) {
                replace_uses(*copy, read.get(), spec);
            }
        } else {
            for (const auto &block : copy->blocks) {
                for_each_value(*block, [&](shared_ptr<Value> &u) {
                    const auto *a = u ? std::get_if<Argument>(&u->data) : nullptr;
                    if (a != nullptr && a->name == fresh) {
                        u = spec;
                    }
                });
            }
        }
    }
    at->instrs.insert(at->instrs.begin() + position, prologue.begin(),
                      prologue.end());
    // The tag is a constant now: every match on the key folds to its arm;
    // and a fixed field's reads fold once the struct has (fold_forced_words).
    simplify(*copy);
    if (!forced_words.empty()) {
        fold_forced_words(*copy, forced_words);
        simplify(*copy);
    }
    return copy;
}

shared_ptr<Value>
key_tag_at(Block &block, const Function &func, const string &key,
           const map<string, shared_ptr<Value>> &params,
           const map<string, ir::Program::AdtStorage> &storages) {
    const string what = func.blocks.front()->name + "'s " + key + " at a push";
    const shared_ptr<Value> given = key_value(func, key);
    internal_assert(given) << what << ": " << func.blocks.front()->name
                           << " has no value named " << key;
    const Type type = type_of(*given);
    internal_assert(!type.is<Ptr_t>())
        << "[unimplemented] " << what << ": " << key << " is handed by pointer";
    const auto tag = key_tag_of(type, storages, what);
    internal_assert(tag.has_value())
        << what << ": " << key << " is a " << type
        << ", not a variant type or an optional";

    Definitions defs(func);
    const string entry = func.blocks.front()->name;
    // The parameter a chain of addresses bottoms out in, if it does: what a
    // load in the key's computation reads through.
    const std::function<const Argument *(const shared_ptr<Value> &, const string &)>
        param_behind = [&](const shared_ptr<Value> &v,
                           const string &in) -> const Argument * {
        const Definition d = defs.of(in, v);
        if (const auto *a = std::get_if<Argument>(&d.value->data)) {
            if (d.block != entry) {
                return nullptr;
            }
            for (const Argument &p : func.blocks.front()->args) {
                if (p.name == a->name) {
                    return &p;
                }
            }
            return nullptr;
        }
        const auto *i = std::get_if<shared_ptr<Instruction>>(&d.value->data);
        if (i == nullptr || (*i)->operands.empty()) {
            return nullptr;
        }
        switch ((*i)->op) {
        case Instruction::Op::GEP:
        case Instruction::Op::FieldPtr:
        case Instruction::Op::ExtractIdx:
        case Instruction::Op::LoadField:
        case Instruction::Op::Set:
            return param_behind((*i)->operands[0], (*i)->owner.lock()->name);
        default:
            return nullptr;
        }
    };

    map<const Instruction *, shared_ptr<Value>> made;
    std::function<shared_ptr<Value>(const shared_ptr<Value> &, const string &)>
        copy_of;
    copy_of = [&](const shared_ptr<Value> &v,
                  const string &in) -> shared_ptr<Value> {
        return std::visit(
            overloads{
                [&](const Constant &) { return v; },
                [&](const Argument &a) -> shared_ptr<Value> {
                    const Definition d = defs.of(in, v);
                    if (const auto *da = std::get_if<Argument>(&d.value->data)) {
                        internal_assert(d.block == entry)
                            << what << ": " << key << " depends on " << a.name
                            << ", to which " << d.block << " gives different "
                            << "values on different paths; a key is one "
                            << "computation of the entry";
                        const auto it = params.find(da->name);
                        internal_assert(it != params.end())
                            << what << ": the push has no value for the "
                            << "parameter " << da->name;
                        return it->second;
                    }
                    return copy_of(d.value, d.block);
                },
                [&](const shared_ptr<Instruction> &i) -> shared_ptr<Value> {
                    if (const auto it = made.find(i.get()); it != made.end()) {
                        return it->second;
                    }
                    if (i->op == Instruction::Op::Set) {
                        // A program's name for a value: the value.
                        return copy_of(i->operands[0], i->owner.lock()->name);
                    }
                    internal_assert(computes_a_key(i->op))
                        << what << ": " << key << " depends on " << i->name
                        << ", which is not a computation the push can repeat "
                        << "from the entry";
                    const string here = i->owner.lock()->name;
                    if (i->op == Instruction::Op::Load ||
                        (i->op == Instruction::Op::ExtractIdx &&
                         (type_of(*i->operands[0]).is<Ptr_t>() ||
                          type_of(*i->operands[0]).is_reference()))) {
                        const Argument *through = param_behind(i->operands[0], here);
                        internal_assert(through != nullptr && !through->mutating)
                            << what << ": " << key << " depends on " << i->name
                            << ", a read of memory that is not through a "
                            << "parameter nothing writes; the push might read "
                            << "something else than the drain";
                    }
                    vector<shared_ptr<Value>> operands;
                    for (const auto &o : i->operands) {
                        operands.push_back(copy_of(o, here));
                    }
                    auto out = block.make_instruction(i->type, i->op,
                                                      std::move(operands));
                    made[i.get()] = out;
                    return out;
                },
            },
            v->data);
    };
    const auto *instr = std::get_if<shared_ptr<Instruction>>(&given->data);
    const shared_ptr<Value> here =
        copy_of(given, instr != nullptr ? (*instr)->owner.lock()->name : entry);
    if (!tag->field.has_value()) {
        return here; // a bool is its own tag
    }
    return block.make_instruction(tag->type, Instruction::Op::LoadField,
                                  {here, constant_u32(*tag->field)});
}

//===--------------------------------------------------------------------===//
// A specialize on a gang's state
//===--------------------------------------------------------------------===//

namespace {

// The mask parameter of a gang's masked copy (`!mask`, the last parameter,
// SSA/Vectorize.cpp), or null for a function that takes none.
const Argument *mask_parameter(const Function &func) {
    for (const Argument &a : func.blocks.front()->args) {
        if (a.name == "!mask") {
            return &a;
        }
    }
    return nullptr;
}

// Every parfor of `func` renamed with `suffix`, and every use of its index
// with it: a copy made for an arm carries the arm's label on its loops, as
// the copies specialize_loops makes do (`p!VolPath`), so that a directive
// written on the arm resolves to them (resolve_loops, SSA/Convert.cpp).
void suffix_loops(Function &func, const string &suffix) {
    vector<std::pair<string, string>> renamed;
    for (const auto &block : func.blocks) {
        if (auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data)) {
            const string was = p->index;
            p->index = was + suffix;
            renamed.emplace_back(was, p->index);
        }
    }
    // The index is declared by the loop's body and threaded on under its
    // name; the entry declares no loop's index, so nothing is kept from
    // the renaming.
    for (const auto &[was, now] : renamed) {
        rename_uses(func.blocks, *func.blocks.front(), was, now);
    }
}

// Every call `func` makes of `was` made of `now` instead: an arm's
// recursion is the arm's own.
void retarget_calls(Function &func, const string &was, const string &now) {
    for (const auto &block : func.blocks) {
        if (Terminator::Jump *callee = block->terminator.callee();
            callee != nullptr && callee->name == was) {
            callee->name = now;
        }
    }
}

// Whether a gang copy's loop (Function::gang_loop) is the loop a directive
// named: the name itself, lowering's `_` spelling of it, or a variant of
// either (`i_lane!VolPath`), as resolve_loops matches a loop's name.
bool is_gang_of(const string &gang_loop, const string &wanted) {
    for (const string &candidate : {wanted, "_" + wanted}) {
        if (gang_loop == candidate ||
            (gang_loop.size() > candidate.size() &&
             gang_loop.compare(0, candidate.size(), candidate) == 0 &&
             gang_loop[candidate.size()] == '!')) {
            return true;
        }
    }
    return false;
}

// An instruction appended to `block`, named and looked up by its name.
shared_ptr<Value> emit(Function &f, Block &block, const Type &type,
                       Instruction::Op op, vector<shared_ptr<Value>> operands) {
    auto instr = std::make_shared<Instruction>(f.get_unique_name(), type, op,
                                               std::move(operands),
                                               block.shared_from_this());
    block.instrs.push_back(instr);
    auto value = std::make_shared<Value>(instr);
    block.lookups[instr->name] = value;
    return value;
}

// `count <compare> threshold`, in `block`.
shared_ptr<Value> compare_count(Function &f, Block &block,
                                const shared_ptr<Value> &count,
                                ir::Specialize::Compare compare,
                                uint64_t threshold) {
    const shared_ptr<Value> k = constant_u32(threshold);
    const auto make = [&](Instruction::Op op, shared_ptr<Value> a,
                          shared_ptr<Value> b) {
        return emit(f, block, Bool_t::make(), op, {std::move(a), std::move(b)});
    };
    switch (compare) {
    case ir::Specialize::Compare::LT:
        return make(Instruction::Op::Lt, count, k);
    case ir::Specialize::Compare::LE:
        return make(Instruction::Op::Leq, count, k);
    case ir::Specialize::Compare::GT:
        return make(Instruction::Op::Lt, k, count);
    case ir::Specialize::Compare::GE:
        return make(Instruction::Op::Leq, k, count);
    case ir::Specialize::Compare::EQ:
        return make(Instruction::Op::Eq, count, k);
    case ir::Specialize::Compare::NE:
        return make(Instruction::Op::Ne, count, k);
    }
    internal_error << "compare_count: unknown comparison";
    return nullptr;
}

// A block of `f` named `name` that takes `params`, calls `callee` with them
// and returns what the call returns: where a dispatch on the count hands a
// call to an arm.
shared_ptr<Block> call_and_return(const shared_ptr<Function> &f,
                                  const string &name,
                                  const vector<Argument> &params,
                                  const string &callee) {
    const bool returns_value = !f->ret_type.is<Void_t>();
    auto call = std::make_shared<Block>();
    call->name = name;
    call->owner = f;
    vector<shared_ptr<Value>> args;
    for (const Argument &p : params) {
        args.push_back(call->add_argument(p));
    }
    auto ret = std::make_shared<Block>();
    ret->name = name + "!ret";
    ret->owner = f;
    shared_ptr<Value> value;
    if (returns_value) {
        value = ret->add_argument(Argument{f->ret_type, "!ret"});
    }
    ret->terminator.data = Terminator::Return{value};
    call->terminator.data =
        Terminator::Call{Terminator::Jump{callee, std::move(args)},
                         Terminator::Jump{ret->name, {}},
                         /*drop=*/!returns_value};
    f->blocks.push_back(call);
    f->blocks.push_back(ret);
    return call;
}

// The count of the lanes of `mask` that are on, compared as the directive
// wrote it, as instructions in `block`.
shared_ptr<Value> count_test(Function &f, Block &block,
                             const shared_ptr<Value> &mask,
                             ir::Specialize::Compare compare,
                             uint64_t threshold) {
    const shared_ptr<Value> count =
        emit(f, block, UInt_t::make(32), Instruction::Op::Popcount, {mask});
    return compare_count(f, block, count, compare, threshold);
}

// One arm of `of`: a copy of the gang copy `x` as it stands, its loops and
// its recursion the arm's own, entered through a test of the count that
// hands the call to `other` when the condition does not pick this arm.
void make_arm(FuncMap &fmap, const string &of, const Function &x,
              const string &label, const string &other, bool runs_when_true,
              ir::Specialize::Compare compare, uint64_t threshold) {
    const string name = of + "!" + label;
    shared_ptr<Function> arm = clone_function(x);
    arm->specialized_from = of;
    arm->gang_loop = x.gang_loop;
    suffix_loops(*arm, "!" + label);
    retarget_calls(*arm, of, name);
    fmap[name] = arm;

    // A function is named by its entry block: the switch is the arm's
    // entry under the arm's name, and the copied entry follows it.
    const shared_ptr<Block> body = arm->blocks.front();
    body->name = name + "!entry";
    auto guard = std::make_shared<Block>();
    guard->name = name;
    guard->owner = arm;
    vector<shared_ptr<Value>> args;
    for (const Argument &p : body->args) {
        args.push_back(guard->add_argument(p));
    }
    const shared_ptr<Value> mask = guard->lookups.at("!mask");
    const shared_ptr<Value> cond = count_test(*arm, *guard, mask, compare, threshold);
    arm->blocks.insert(arm->blocks.begin(), guard);
    const shared_ptr<Block> divert =
        call_and_return(arm, "!switch!" + (runs_when_true ? string("False")
                                                            : string("True")),
                        body->args, other);
    const Terminator::Jump stay{body->name, args};
    const Terminator::Jump go{divert->name, args};
    guard->terminator.data = Terminator::Dispatch{
        cond, runs_when_true ? vector<Terminator::Jump>{go, stay}
                             : vector<Terminator::Jump>{stay, go}};
    refresh_preds(*arm);
}

// `x`'s body replaced by the test and a call of the arm it picks.
void make_dispatcher(const shared_ptr<Function> &x, const string &on_true,
                     const string &on_false, ir::Specialize::Compare compare,
                     uint64_t threshold) {
    const vector<Argument> params = x->blocks.front()->args;
    const string entry_name = x->blocks.front()->name;
    x->blocks.clear();
    auto entry = std::make_shared<Block>();
    entry->name = entry_name;
    entry->owner = x;
    vector<shared_ptr<Value>> args;
    for (const Argument &p : params) {
        args.push_back(entry->add_argument(p));
    }
    x->blocks.push_back(entry);
    const shared_ptr<Value> mask = entry->lookups.at("!mask");
    const shared_ptr<Value> cond = count_test(*x, *entry, mask, compare, threshold);
    const shared_ptr<Block> to_true = call_and_return(x, "!True", params, on_true);
    const shared_ptr<Block> to_false = call_and_return(x, "!False", params, on_false);
    entry->terminator.data = Terminator::Dispatch{
        cond, {Terminator::Jump{to_false->name, args},
               Terminator::Jump{to_true->name, args}}};
    refresh_preds(*x);
}

// The gang copies a directive over the state of the gang `loop` means by
// `fname` -- the copies of `fname` itself and of the recursions it stands
// for (recursive_functions_from, SSA/Rewrite.h) that the vectorize of
// `loop` made (Function::gang_loop) and that take a mask -- in name order.
// The helpers a function calls are not among them: a condition on
// `occluded`'s lanes is tested where occluded is called and at each call
// of its recursion, not at every call of its box or triangle test.
vector<string> gang_copies_of(const FuncMap &fmap, const string &fname,
                              const string &loop) {
    set<string> candidates = recursive_functions_from(fmap, fname);
    for (const auto &[name, f] : fmap) {
        if (f->specialized_from == fname) {
            candidates.insert(name);
        }
    }
    vector<string> copies;
    for (const string &name : candidates) {
        const Function &f = *fmap.at(name);
        if (is_gang_of(f.gang_loop, loop) && mask_parameter(f) != nullptr) {
            copies.push_back(name);
        }
    }
    return copies;
}

} // namespace

void specialize_gang(FuncMap &fmap, const string &fname, const string &gang,
                     ir::Specialize::Compare compare, uint64_t threshold) {
    const string where = fname + ".specialize(count(" + gang + ") ...)";
    internal_assert(fmap.contains(fname)) << where << ": no function " << fname;
    const vector<string> targets = gang_copies_of(fmap, fname, gang);
    if (targets.empty()) {
        string gangs;
        for (const string &name : functions_standing_for(fmap, fname)) {
            const Function &f = *fmap.at(name);
            if (!f.gang_loop.empty()) {
                gangs += (gangs.empty() ? "" : ", ") + name + " (" + f.gang_loop + ")";
            }
        }
        internal_error
            << where << ": no copy of " << fname
            << " or of its recursion was made by vectorizing " << gang
            << ". The condition counts the lanes of a gang, so `vectorize(" << gang
            << ")` comes first, on the loop that calls " << fname << ". "
            << (gangs.empty() ? "No gang copy of anything it stands for exists."
                              : "The gang copies it stands for, with the loop "
                                "that made each: " + gangs + ".");
    }
    for (const string &name : targets) {
        const shared_ptr<Function> x = fmap.at(name);
        const string on_true = name + "!True";
        const string on_false = name + "!False";
        make_arm(fmap, name, *x, "True", on_false, /*runs_when_true=*/true,
                 compare, threshold);
        make_arm(fmap, name, *x, "False", on_true, /*runs_when_true=*/false,
                 compare, threshold);
        make_dispatcher(x, on_true, on_false, compare, threshold);
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
