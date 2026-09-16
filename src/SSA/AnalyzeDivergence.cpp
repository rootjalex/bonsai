#include "SSA/AnalyzeDivergence.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <algorithm>
#include <map>
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

// Resolves an argument reference against a scope map: the argument is in the
// set if its own block lists it, or if a dominating block that it is visible
// from does.
bool in_set(const set<std::pair<string, string>> &args,
            const map<string, set<string>> &scope, const string &block,
            const Argument &a) {
    if (args.count({block, a.name})) {
        return true;
    }
    const auto it = scope.find(block);
    return it != scope.end() && it->second.count(a.name) > 0;
}

} // namespace

bool Divergence::is_varying(const string &block, const Value &v) const {
    return std::visit(overloads{
                          [&](const shared_ptr<Instruction> &i) {
                              return instrs.count(i.get()) > 0;
                          },
                          [&](const Constant &) { return false; },
                          [&](const Argument &a) {
                              return in_set(args, in_scope, block, a);
                          },
                      },
                      v.data);
}

bool Divergence::points_to_varying(const string &block, const Value &v) const {
    return std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &i) {
                return pointee_instrs.count(i.get()) > 0;
            },
            [&](const Constant &) { return false; },
            [&](const Argument &a) {
                return in_set(pointee_args, pointee_in_scope, block, a);
            },
        },
        v.data);
}

namespace {

// Values are identified by name: the SSA builder threads a definition through
// block arguments under its own name, so two references to the same name --
// in whatever blocks -- are two references to the same definition.
string value_key(const Value &v) {
    return std::visit(
        overloads{
            [](const shared_ptr<Instruction> &i) { return "%" + i->name; },
            [](const Constant &c) {
                std::ostringstream os;
                c.dump(os);
                return os.str();
            },
            [](const Argument &a) { return "%" + a.name; },
        },
        v.data);
}

// One edge's worth of argument passing: the values `from`'s terminator hands
// to `target`, and the index of the target argument the first of them binds
// to.
struct ArgumentFlow {
    string from;
    size_t first_arg = 0;
    const vector<shared_ptr<Value>> *values = nullptr;
};

// The argument flows out of `block` that stay inside the region, keyed by
// target block. The callee of a Call is not one of them: it belongs to
// another function's CFG, and its arguments are bound by that function's own
// analysis.
void collect_argument_flows(const Block &block,
                            map<string, vector<ArgumentFlow>> &into) {
    auto add = [&](const Terminator::Jump &j, size_t first_arg) {
        into[j.name].push_back({block.name, first_arg, &j.args});
    };

    std::visit(overloads{
                   [&](const std::monostate &) {
                       internal_error << "Block " << block.name
                                      << " has no terminator.";
                   },
                   [&](const Terminator::Jump &j) { add(j, 0); },
                   [&](const Terminator::Dispatch &d) {
                       for (const auto &target : d.targets) {
                           add(target, 0);
                       }
                   },
                   [&](const Terminator::Return &) {},
                   [&](const Terminator::ParFor &) {
                       internal_error << "TODO: nested ParFor in " << block.name
                                      << " during divergence analysis.";
                   },
                   [&](const Terminator::Yield &) {},
                   [&](const Terminator::Call &c) {
                       // A returned value is prepended to the continuation's
                       // arguments, so the explicitly passed ones start at 1.
                       add(c.cont, c.drop ? 0 : 1);
                   },
                   [&](const Terminator::MultiCall &c) {
                       add(c.cont, c.drop ? 0 : 1);
                   },
               },
               block.terminator.data);
}

template <typename T>
const T &lookup_or(const map<string, T> &m, const string &key,
                   const T &fallback) {
    const auto it = m.find(key);
    return it == m.end() ? fallback : it->second;
}

// What each block can name of the arguments in `args`: its own, and those of
// every block that dominates it. Reverse postorder visits a block after every
// block that dominates it, so one pass suffices.
//
// A block does not only refer to its own arguments. Linearization drops the
// arguments it turns into blends, after which the blocks that used to receive
// them refer to the dominating definition directly -- and a loop header's
// arguments, which survive, are referred to that way from everywhere in the
// loop.
map<string, set<string>>
compute_scope(const vector<string> &rpo, const DomTree &dom,
              const BlockMap &all_blocks,
              const set<std::pair<string, string>> &args) {
    map<string, set<string>> scope_of;
    for (const string &name : rpo) {
        set<string> &scope = scope_of[name];
        const auto idom = dom.idom.find(name);
        if (idom != dom.idom.end() && idom->second != name) {
            const auto outer = scope_of.find(idom->second);
            if (outer != scope_of.end()) {
                scope = outer->second;
            }
        }
        for (const Argument &arg : all_blocks.at(name)->args) {
            if (args.count({name, arg.name})) {
                scope.insert(arg.name);
            } else {
                // Shadows whatever a dominator called it.
                scope.erase(arg.name);
            }
        }
    }
    return scope_of;
}

// Does this value address memory -- a pointer, or an array handle, which is
// the address of its elements (see Type::is_reference)?
bool addresses_memory(const Type &type) {
    return type.defined() && (type.is<Ptr_t>() || type.is_reference());
}

// The pointers of the region, in equivalence classes of "derived from the
// same allocation": a field's address, an element's address, a copy, and the
// block arguments that thread any of them are all the one allocation seen
// from different places, and the memory they address is laid out one way for
// all of them. Whether that layout is per lane is then a fact about the
// class -- a pointer is keyed by its instruction name, or by the block that
// declares the argument it is, so that the same argument referred to from
// several blocks is one key.
struct PointerClasses {
    // Union-find over keys.
    map<string, string> parent;

    // The block declaring the argument `name` as seen from `block`: the block
    // itself, or the nearest dominator that declares it (see compute_scope for
    // why a reference can be that far from its declaration).
    const DomTree &dom;
    const BlockMap &blocks;

    PointerClasses(const DomTree &dom, const BlockMap &blocks)
        : dom(dom), blocks(blocks) {}

    string owner_of(const string &block, const string &name) const {
        string at = block;
        while (true) {
            const auto it = blocks.find(at);
            if (it != blocks.end()) {
                for (const Argument &arg : it->second->args) {
                    if (arg.name == name) {
                        return at;
                    }
                }
            }
            const auto idom = dom.idom.find(at);
            if (idom == dom.idom.end() || idom->second == at) {
                return block;
            }
            at = idom->second;
        }
    }

    // The key of `v` as referenced from `block`, or empty if it is not a
    // pointer (a constant, or a value of some other type).
    string key(const string &block, const Value &v) const {
        if (!addresses_memory(v.get_type())) {
            return "";
        }
        return std::visit(
            overloads{
                [](const shared_ptr<Instruction> &i) { return "%" + i->name; },
                [](const Constant &) { return string(); },
                [&](const Argument &a) {
                    return owner_of(block, a.name) + "/" + a.name;
                },
            },
            v.data);
    }

    string find(const string &k) {
        auto it = parent.find(k);
        if (it == parent.end()) {
            parent[k] = k;
            return k;
        }
        if (it->second == k) {
            return k;
        }
        const string root = find(it->second);
        parent[k] = root;
        return root;
    }

    void unite(const string &a, const string &b) {
        if (a.empty() || b.empty()) {
            return;
        }
        const string ra = find(a), rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    }
};

} // namespace

Divergence
analyze_divergence(const Function &func, const string &entry,
                   const set<string> &varying_seeds,
                   const set<const Instruction *> &varying_instrs,
                   const set<std::pair<string, string>> &varying_args,
                   const set<string> &pointee_seeds,
                   const shared_ptr<Value> &entry_mask,
                   const set<string> &masked_seeds) {
    // Is this the mask the whole region runs under? Compared by name, the
    // way this form identifies a definition wherever it is referenced.
    const auto is_entry_mask = [&](const Value &v) {
        if (entry_mask == nullptr) {
            return false;
        }
        return value_key(v) == value_key(*entry_mask);
    };
    // A store or accumulate made by some lanes only: it carries a mask that is
    // not the region's own.
    const auto narrowed = [&](const Instruction &instr) {
        return instr.operands.size() > 2 && !is_entry_mask(*instr.operands[2]);
    };

    const BlockMap all_blocks = make_block_map(func);
    const AdjacencyMap all_succs = compute_successors(func);

    // Restrict the CFG to the region: a ParFor body ends at its Yield, and a
    // Call's callee is outside it, so this is exactly the set of blocks whose
    // lanes execute together.
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
    const AdjacencyMap preds = compute_predecessors(succs);
    const vector<string> rpo = reverse_postorder(entry, succs);
    const DomTree dom = compute_dominator_tree(entry, succs, preds, rpo);
    const DomTree pdom = compute_post_dominator_tree(entry, succs, preds);
    const ControlDependence cdep = compute_control_dependence(succs, pdom);

    map<string, vector<ArgumentFlow>> flows;
    for (const string &name : region) {
        collect_argument_flows(*all_blocks.at(name), flows);
    }

    const set<Edge> no_edges;
    const vector<string> no_names;
    const vector<ArgumentFlow> no_flows;

    //===------------------------------------------------------------------===//
    // Pointer classes
    //===------------------------------------------------------------------===//

    // How pointers derive from one another does not depend on what varies, so
    // the classes are settled before the solve. A pointer-producing
    // instruction is derived from its pointer operands; a pointer-typed block
    // argument from every value passed to it. Allocations, addresses taken,
    // loads and call results derive from nothing.
    PointerClasses classes(dom, all_blocks);
    for (const string &name : region) {
        const Block &block = *all_blocks.at(name);
        for (const shared_ptr<Instruction> &instr : block.instrs) {
            if (!addresses_memory(instr->type) ||
                instr->op == Instruction::Op::Alloca ||
                instr->op == Instruction::Op::Alloc ||
                instr->op == Instruction::Op::AddressOf ||
                instr->op == Instruction::Op::Load) {
                continue;
            }
            const string self = classes.key(name, Value(instr));
            for (const shared_ptr<Value> &operand : instr->operands) {
                classes.unite(self, classes.key(name, *operand));
            }
        }
        const vector<ArgumentFlow> &incoming = lookup_or(flows, name, no_flows);
        for (size_t j = 0; j < block.args.size(); j++) {
            if (!addresses_memory(block.args[j].type)) {
                continue;
            }
            const string self = name + "/" + block.args[j].name;
            for (const ArgumentFlow &flow : incoming) {
                if (j < flow.first_arg ||
                    j - flow.first_arg >= flow.values->size()) {
                    continue;
                }
                const Value &v = *(*flow.values)[j - flow.first_arg];
                classes.unite(self, classes.key(flow.from, v));
            }
        }
    }

    // The classes whose memory is the gang's own, and so may be laid out per
    // lane: the region's allocations, the temporaries an address-of makes,
    // and the parameters the caller says are its per-lane locals. Anything
    // else -- an array parameter, a pointer into the scene -- is memory the
    // lanes share with the world, and its layout is not this region's to
    // change.
    set<string> owned;
    for (const string &name : region) {
        for (const shared_ptr<Instruction> &instr :
             all_blocks.at(name)->instrs) {
            if (instr->op == Instruction::Op::Alloca ||
                instr->op == Instruction::Op::Alloc ||
                instr->op == Instruction::Op::AddressOf) {
                owned.insert(classes.find(classes.key(name, Value(instr))));
            }
        }
    }
    for (const string &seed : pointee_seeds) {
        owned.insert(classes.find(entry + "/" + seed));
    }

    // Classes found to be per lane. Grows monotonically with everything else.
    set<string> per_lane;
    for (const string &seed : pointee_seeds) {
        per_lane.insert(classes.find(entry + "/" + seed));
    }

    Divergence result;
    for (const string &seed : varying_seeds) {
        result.args.insert({entry, seed});
    }
    result.instrs.insert(varying_instrs.begin(), varying_instrs.end());
    result.args.insert(varying_args.begin(), varying_args.end());
    for (const string &seed : masked_seeds) {
        if (region.count(seed)) {
            result.masked.insert(seed);
        }
    }

    // Is `v`, referenced from `block`, a shared pointer to per-lane memory?
    auto points_per_lane = [&](const string &block, const Value &v) {
        const string k = classes.key(block, v);
        return !k.empty() && per_lane.count(classes.find(k)) > 0;
    };

    //===------------------------------------------------------------------===//
    // Solve
    //===------------------------------------------------------------------===//

    // A store of a per-lane value through a pointer that reads as shared,
    // noted during a pass of the solve and judged only once the solve has
    // converged (see below).
    struct PendingWrite {
        string block;
        shared_ptr<Value> place;
        const Instruction *by;
    };
    vector<PendingWrite> pending;

    // Monotone: every rule only ever adds to the sets, so this terminates.
    //
    // Two fixpoints, one inside the other. The inner one settles what varies
    // given what memory is laid out per lane. Only then are the stores of a
    // per-lane value through a shared pointer judged: whether such a store
    // makes an allocation per lane, or is a race, depends on the place
    // *staying* uniform -- and in the middle of a pass the value may already
    // read as varying while the index the place is computed from has not
    // yet been reached, as a tracker seeded varying is stored at an exit the
    // gang's own loop index only later proves to vary. A judgement made
    // there would be wrong. Laying an allocation out per lane changes what
    // loads through it vary, so the outer loop then runs the inner one again.
    bool again = true;
    while (again) {
        again = false;
        pending.clear();

    bool changed = true;
    while (changed) {
        changed = false;
        auto mark = [&](auto &s, const auto &key) {
            changed |= s.insert(key).second;
        };

        // A pointer written per lane makes the whole allocation per lane --
        // if it is an allocation this region may lay out. Writing a per-lane
        // value through a pointer into shared memory is a race between the
        // lanes in the original program and has no meaning here. Noted here,
        // judged after the solve.
        auto mark_written = [&](const string &block,
                                const shared_ptr<Value> &place,
                                const Instruction &by) {
            pending.push_back({block, place, &by});
        };

        result.in_scope = compute_scope(rpo, dom, all_blocks, result.args);

        for (const string &name : rpo) {
            const Block &block = *all_blocks.at(name);

            // A block executes with a partial mask if it is control dependent
            // on a divergent branch, or on a branch in a block that is itself
            // only partially executed: a block's mask is always a subset of
            // the masks of the blocks it is control dependent on.
            for (const auto &[from, to] : lookup_or(cdep, name, no_edges)) {
                if (result.branches.count(from) || result.masked.count(from)) {
                    mark(result.masked, name);
                }
            }
            const bool under_mask = result.masked.count(name) > 0;

            // At a join whose predecessors are only partially executed, lanes
            // arrive along different edges, so an argument that is not passed
            // the same definition along every edge disagrees between lanes.
            const vector<string> &block_preds =
                lookup_or(preds, name, no_names);
            const bool divergent_join =
                block_preds.size() > 1 &&
                std::any_of(block_preds.begin(), block_preds.end(),
                            [&](const string &p) {
                                return result.masked.count(p) > 0;
                            });

            const vector<ArgumentFlow> &incoming =
                lookup_or(flows, name, no_flows);
            for (size_t j = 0; j < block.args.size(); j++) {
                std::optional<string> common_key;
                bool varying = false;
                for (const ArgumentFlow &flow : incoming) {
                    if (j < flow.first_arg ||
                        j - flow.first_arg >= flow.values->size()) {
                        continue;
                    }
                    const Value &v = *(*flow.values)[j - flow.first_arg];
                    varying |= result.is_varying(flow.from, v);
                    const string key = value_key(v);
                    varying |= divergent_join && common_key.has_value() &&
                               *common_key != key;
                    common_key = key;
                }
                if (varying) {
                    mark(result.args, std::pair{name, block.args[j].name});
                }
            }

            for (const shared_ptr<Instruction> &instr : block.instrs) {
                switch (instr->op) {
                // A cross-lane reduction asks a question about the gang as a
                // whole, so its answer is the same for every lane however
                // much its operand varies. This is what makes the latch of a
                // uniformized divergent loop a branch the gang can take
                // together (Moll & Hack section 5).
                case Instruction::Op::Any:
                    continue;

                // The address of a value the lanes disagree about is one
                // address, of a temporary with a slot per lane.
                case Instruction::Op::AddressOf:
                    if (result.is_varying(name, *instr->operands[0])) {
                        mark(per_lane, classes.find(classes.key(
                                           name, Value(instr))));
                    }
                    continue;

                // Reads per-lane memory: one value per lane. An element read
                // out of a per-lane array is one too, whichever element -- the
                // array holds a lane's own copy of every element.
                case Instruction::Op::Load:
                case Instruction::Op::ExtractIdx:
                    if (result.is_varying(name, *instr->operands[0]) ||
                        points_per_lane(name, *instr->operands[0]) ||
                        (instr->operands.size() > 1 &&
                         result.is_varying(name, *instr->operands[1]))) {
                        mark(result.instrs, instr.get());
                    }
                    continue;

                // A store through one shared address of a value the lanes
                // disagree about makes the memory per lane. So does a mask
                // alone, when the memory is the gang's: the lanes it leaves
                // out keep what was there, and afterwards the slots differ.
                // (A mask on a store into shared memory means the store
                // happens if any lane is on, and changes nothing about the
                // memory's layout.)
                //
                // A store through per-lane *addresses* is each lane writing
                // its own place -- a scatter. Into memory the gang shares with
                // the world, that says nothing about what any of those places
                // holds for the others. Into an allocation of the region's
                // own it says everything: in the program as written every
                // lane had an allocation of its own -- a traversal's stack,
                // pushed at each lane's own depth -- so two lanes at the same
                // index were never writing one place, and in one shared copy
                // they would be.
                case Instruction::Op::Store:
                    if (!result.is_varying(name, *instr->operands[0])) {
                        if (result.is_varying(name, *instr->operands[1])) {
                            mark_written(name, instr->operands[0], *instr);
                        } else if (narrowed(*instr) || under_mask) {
                            const string k =
                                classes.key(name, *instr->operands[0]);
                            if (!k.empty() && owned.count(classes.find(k))) {
                                mark(per_lane, classes.find(k));
                            }
                        }
                    } else {
                        const string k = classes.key(name, *instr->operands[0]);
                        if (!k.empty() && owned.count(classes.find(k))) {
                            mark(per_lane, classes.find(k));
                        }
                    }
                    break;

                // An accumulate into shared memory of a per-lane value is a
                // cross-lane reduction, and leaves the memory shared; only an
                // accumulate into the gang's own memory makes that per lane
                // -- at a per-lane place always, for the reason given at
                // Store.
                case Instruction::Op::AccAdd:
                case Instruction::Op::AccMul:
                case Instruction::Op::AccSub:
                case Instruction::Op::AccMin:
                case Instruction::Op::AccMax:
                case Instruction::Op::AccArgmin:
                case Instruction::Op::AccArgmax: {
                    const string k = classes.key(name, *instr->operands[0]);
                    const bool writes_per_lane =
                        under_mask || narrowed(*instr) ||
                        result.is_varying(name, *instr->operands[0]) ||
                        result.is_varying(name, *instr->operands[1]);
                    if (writes_per_lane && !k.empty() &&
                        owned.count(classes.find(k))) {
                        mark(per_lane, classes.find(k));
                    }
                    break;
                }

                default:
                    break;
                }

                // Everything else -- a store and an accumulate included, which
                // are per lane whenever what they write is -- is varying when
                // any operand is.

                const bool varying =
                    std::any_of(instr->operands.begin(), instr->operands.end(),
                                [&](const shared_ptr<Value> &v) {
                                    return result.is_varying(name, *v);
                                });
                if (varying) {
                    mark(result.instrs, instr.get());
                }
            }

            if (const auto *d =
                    std::get_if<Terminator::Dispatch>(&block.terminator.data)) {
                if (result.is_varying(name, *d->cond)) {
                    mark(result.branches, name);
                }
            }

            // A call on uniform arguments cannot produce a varying result, but
            // a call on varying ones -- or one that reads per-lane memory --
            // may produce anything, and may write anything into any per-lane
            // memory it is handed. Until divergence is solved across
            // functions, that is what is assumed of it. A call made under a
            // mask writes only some lanes' slots, which is the same as a
            // masked store.
            auto call_effects = [&](const vector<shared_ptr<Value>> &args,
                                    const Terminator::Jump &cont, bool drop) {
                // The region's own mask, handed on to a callee specialized to
                // run under it, is not what makes a call varying.
                const bool varying_call =
                    std::any_of(args.begin(), args.end(),
                                [&](const shared_ptr<Value> &v) {
                                    return !is_entry_mask(*v) &&
                                           (result.is_varying(name, *v) ||
                                            points_per_lane(name, *v));
                                });
                if (!drop && varying_call) {
                    const Block &cont_block = *all_blocks.at(cont.name);
                    internal_assert(!cont_block.args.empty())
                        << "Call continuation " << cont_block.name
                        << " takes no result argument.";
                    mark(result.args,
                         std::pair{cont_block.name, cont_block.args[0].name});
                }
                if (varying_call || under_mask) {
                    for (const shared_ptr<Value> &v : args) {
                        const string k = classes.key(name, *v);
                        if (!k.empty() && owned.count(classes.find(k))) {
                            mark(per_lane, classes.find(k));
                        }
                    }
                }
            };
            if (const auto *c =
                    std::get_if<Terminator::Call>(&block.terminator.data)) {
                call_effects(c->call.args, c->cont, c->drop);
            }
            if (const auto *c =
                    std::get_if<Terminator::MultiCall>(&block.terminator.data)) {
                for (size_t i = 0; i < c->varying.size(); i++) {
                    call_effects(c->call_args(i), c->cont, c->drop);
                }
            }
        }
    }

        // The stores noted above, on the converged state. One whose place
        // has since turned out to vary is a scatter, each lane writing its
        // own place, and says nothing about the memory's layout.
        for (const PendingWrite &w : pending) {
            if (result.is_varying(w.block, *w.place)) {
                continue;
            }
            const string k = classes.key(w.block, *w.place);
            if (k.empty()) {
                continue;
            }
            const string root = classes.find(k);
            if (!owned.count(root)) {
                std::cerr << "--- the function as analyzed:\n";
                func.dump(std::cerr);
                w.by->dump(std::cerr);
                // How the place came to be read as shared, for whoever has
                // to work out whether the program or the analysis is wrong.
                if (const auto *i =
                        std::get_if<shared_ptr<Instruction>>(&w.place->data)) {
                    std::cerr << "\nThe place, ";
                    (*i)->dump(std::cerr);
                    std::cerr << ", is uniform; its operands are:";
                    for (const auto &operand : (*i)->operands) {
                        std::cerr << "\n  ";
                        operand->dump(std::cerr);
                        std::cerr << (result.is_varying(w.block, *operand)
                                          ? " (varying)"
                                          : " (uniform)");
                    }
                }
                internal_error
                    << "\nThe write above, in " << w.block << ", stores a "
                    << "value that differs between the lanes of a gang "
                    << "through a pointer the lanes share, into memory the "
                    << "gang does not own. Every lane of a parfor writing its "
                    << "own value to one place is a race; index the place by "
                    << "the loop variable, or accumulate into it.";
            }
            again |= per_lane.insert(root).second;
        }
    }

    //===------------------------------------------------------------------===//
    // Report the pointer classes as values
    //===------------------------------------------------------------------===//

    for (const string &name : region) {
        const Block &block = *all_blocks.at(name);
        for (const shared_ptr<Instruction> &instr : block.instrs) {
            if (addresses_memory(instr->type) &&
                points_per_lane(name, Value(instr))) {
                result.pointee_instrs.insert(instr.get());
            }
        }
        for (const Argument &arg : block.args) {
            if (addresses_memory(arg.type) &&
                per_lane.count(classes.find(name + "/" + arg.name))) {
                result.pointee_args.insert({name, arg.name});
            }
        }
    }
    result.pointee_in_scope =
        compute_scope(rpo, dom, all_blocks, result.pointee_args);

    return result;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
