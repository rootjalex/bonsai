#include "SSA/Specialize.h"

#include "Error.h"
#include "IR/Type.h"
#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "Utils.h"

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

} // namespace

void specialize_loops(FuncMap &fmap, const string &fname, const string &param,
                      const map<string, ir::Program::AdtStorage> &storages) {
    const auto fit = fmap.find(fname);
    internal_assert(fit != fmap.end())
        << fname << ".specialize(" << param << "): no function " << fname;
    Function &f = *fit->second;
    const string where = fname + ".specialize(" + param + ")";

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
    const Ptr_t *pointer = parameter->type.as<Ptr_t>();
    const Type struct_type = pointer != nullptr ? pointer->etype : parameter->type;
    const Struct_t *storage_t = struct_type.as<Struct_t>();
    const auto storage_it = storage_t != nullptr ? storages.find(storage_t->name)
                                                 : storages.end();
    if (storage_it == storages.end()) {
        string known;
        for (const auto &[name, _] : storages) {
            known += (known.empty() ? "" : ", ") + name;
        }
        internal_error << where << ": " << param << " is a " << parameter->type
                       << ", not a variant type (the variant types stored: "
                       << (known.empty() ? "none" : known) << ")";
    }
    const ir::Program::AdtStorage &storage = storage_it->second;
    internal_assert(storage.inline_storage)
        << "[unimplemented] " << where << ": " << storage_t->name
        << " is stored as a tagged index, whose tag is the top bits of one "
        << "word; specialize() handles a tag beside the payload";
    const size_t tag_at = find_struct_index(storage.tag_field, storage_t->fields);
    // The dispatch below takes its k-th target on k, so the tags have to be
    // the variants' positions.
    for (size_t k = 0; k < storage.variants.size(); k++) {
        internal_assert(storage.variants[k].second == k)
            << "[unimplemented] " << where << ": variant "
            << storage.variants[k].first << " has tag "
            << storage.variants[k].second << ", not its position " << k;
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
        if (!inside) {
            outermost.push_back(block);
        }
    }
    internal_assert(!outermost.empty())
        << where << ": " << fname
        << " has no parallel loop; specialize() copies a parfor's body per "
        << "variant";

    for (const shared_ptr<Block> &at : outermost) {
        const Terminator::ParFor loop =
            std::get<Terminator::ParFor>(at->terminator.data);
        const Cfg region_cfg(f, loop.body.name);
        const vector<shared_ptr<Block>> region = region_cfg.blocks();
        if (!mentions(region, param)) {
            continue;
        }

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
        for (const auto &[vname, tag] : storage.variants) {
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

        // The tag, read where the loop was, and the dispatch on it.
        shared_ptr<Value> given_here = at->get_value(param, parameter->type);
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
        shared_ptr<Value> tag_value;
        if (pointer != nullptr) {
            shared_ptr<Value> tag_ptr =
                at_instr(f.get_unique_name(), Ptr_t::make(storage.tag_type),
                         Instruction::Op::FieldPtr, {given_here, constant_u32(tag_at)});
            tag_value = at_instr(f.get_unique_name(), storage.tag_type,
                                 Instruction::Op::Load, {tag_ptr});
        } else {
            tag_value = at_instr(f.get_unique_name(), storage.tag_type,
                                 Instruction::Op::LoadField,
                                 {given_here, constant_u32(tag_at)});
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
            // of the name renamed to one of this copy's own, defined in the
            // head as the given value with the tag set to this variant's.
            // The given value keeps the parameter's name, threaded into the
            // head as before.
            vector<shared_ptr<Block>> copied;
            for (const auto &block : region) {
                copied.push_back(copy.blocks.at(block->name));
            }
            const string fresh = param + "!" + copy.variant;
            rename_uses(copied, head, param, fresh);
            shared_ptr<Value> given = head.get_value(param, parameter->type);

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
            // The given value as a whole -- read through the pointer, when
            // that is how it arrived -- and its fields, the tag replaced.
            shared_ptr<Value> whole =
                pointer != nullptr
                    ? make(f.get_unique_name(), struct_type, Instruction::Op::Load,
                           {given})
                    : given;
            vector<shared_ptr<Value>> fields;
            for (size_t i = 0; i < storage_t->fields.size(); i++) {
                if (i == tag_at) {
                    fields.push_back(std::make_shared<Value>(
                        storage.tag_type.is_uint()
                            ? Constant{storage.tag_type, uint64_t(copy.tag)}
                            : Constant{storage.tag_type, int64_t(copy.tag)}));
                    continue;
                }
                fields.push_back(make(f.get_unique_name(),
                                      storage_t->fields[i].type,
                                      Instruction::Op::LoadField,
                                      {whole, constant_u32(i)}));
            }
            if (pointer != nullptr) {
                // Stored where the body can point at it: a slot under the
                // copy's name, holding the specialized value.
                shared_ptr<Value> spec =
                    make(f.get_unique_name(), struct_type,
                         Instruction::Op::MakeStruct, std::move(fields));
                shared_ptr<Value> slot =
                    make(fresh, parameter->type, Instruction::Op::Alloca, {});
                prologue.push_back(std::make_shared<Instruction>(
                    Instruction::Op::Store, vector<shared_ptr<Value>>{slot, spec},
                    head.shared_from_this()));
            } else {
                make(fresh, parameter->type, Instruction::Op::MakeStruct,
                     std::move(fields));
            }
            head.instrs.insert(head.instrs.begin(), prologue.begin(),
                               prologue.end());
        }
        refresh_preds(f);
    }

    // The bodies the loops no longer enter.
    remove_unreachable_blocks(f);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
