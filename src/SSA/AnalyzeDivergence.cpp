#include "SSA/AnalyzeDivergence.h"

#include "SSA/Analysis.h"

#include "Utils.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
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
    BlockId from = NO_BLOCK;
    size_t first_arg = 0;
    const vector<shared_ptr<Value>> *values = nullptr;
};

// The argument flows out of the block `from` that stay inside the region, by
// target block. The callee of a Call is not one of them: it belongs to
// another function's CFG, and its arguments are bound by that function's own
// analysis.
void collect_argument_flows(const Cfg &cfg, BlockId from,
                            vector<vector<ArgumentFlow>> &into) {
    const Block &block = cfg[from];
    auto add = [&](const Terminator::Jump &j, size_t first_arg) {
        into[cfg.id(j.name)].push_back({from, first_arg, &j.args});
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
compute_scope(const Cfg &cfg, const DomTree &dom,
              const set<std::pair<string, string>> &args) {
    vector<set<string>> scope_of(cfg.size());
    for (BlockId b : cfg.rpo) {
        const string &name = cfg.name(b);
        set<string> &scope = scope_of[b];
        if (dom.contains(b) && dom.idom[b] != b) {
            scope = scope_of[dom.idom[b]];
        }
        for (const Argument &arg : cfg[b].args) {
            if (args.count({name, arg.name})) {
                scope.insert(arg.name);
            } else {
                // Shadows whatever a dominator called it.
                scope.erase(arg.name);
            }
        }
    }
    map<string, set<string>> by_name;
    for (BlockId b : cfg.rpo) {
        by_name.emplace(cfg.name(b), std::move(scope_of[b]));
    }
    return by_name;
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
    const Cfg &cfg;
    const DomTree &dom;

    PointerClasses(const Cfg &cfg, const DomTree &dom) : cfg(cfg), dom(dom) {}

    BlockId owner_of(BlockId block, const string &name) const {
        BlockId at = block;
        while (true) {
            for (const Argument &arg : cfg[at].args) {
                if (arg.name == name) {
                    return at;
                }
            }
            if (!dom.contains(at) || dom.idom[at] == at) {
                return block;
            }
            at = dom.idom[at];
        }
    }

    // The key of `v` as referenced from `block`, or empty if it is not a
    // pointer (a constant, or a value of some other type).
    string key(BlockId block, const Value &v) const {
        if (!addresses_memory(v.get_type())) {
            return "";
        }
        return std::visit(
            overloads{
                [](const shared_ptr<Instruction> &i) { return "%" + i->name; },
                [](const Constant &) { return string(); },
                [&](const Argument &a) {
                    return cfg.name(owner_of(block, a.name)) + "/" + a.name;
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

    // The region: a ParFor body ends at its Yield, and a Call's callee is
    // outside it, so this is exactly the set of blocks whose lanes execute
    // together.
    const Cfg cfg(func, entry);
    const DomTree dom = compute_dominator_tree(cfg);
    const LoopForest loops = compute_loop_forest(cfg, dom);
    const DomTree pdom = compute_post_dominator_tree(cfg);
    const ControlDependence cdep = compute_control_dependence(cfg, pdom);

    vector<vector<ArgumentFlow>> flows(cfg.size());
    for (BlockId b = 0; b < cfg.size(); b++) {
        collect_argument_flows(cfg, b, flows);
    }

    //===------------------------------------------------------------------===//
    // Pointer classes
    //===------------------------------------------------------------------===//

    // How pointers derive from one another does not depend on what varies, so
    // the classes are settled before the solve. A pointer-producing
    // instruction is derived from its pointer operands; a pointer-typed block
    // argument from every value passed to it. Allocations, addresses taken,
    // loads and call results derive from nothing.
    PointerClasses classes(cfg, dom);
    for (BlockId b = 0; b < cfg.size(); b++) {
        const Block &block = cfg[b];
        const string &name = block.name;
        for (const shared_ptr<Instruction> &instr : block.instrs) {
            if (!addresses_memory(instr->type) ||
                instr->op == Instruction::Op::Alloca ||
                instr->op == Instruction::Op::Alloc ||
                instr->op == Instruction::Op::AddressOf ||
                instr->op == Instruction::Op::Load) {
                continue;
            }
            const string self = classes.key(b, Value(instr));
            for (const shared_ptr<Value> &operand : instr->operands) {
                classes.unite(self, classes.key(b, *operand));
            }
        }
        const vector<ArgumentFlow> &incoming = flows[b];
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
    for (BlockId b = 0; b < cfg.size(); b++) {
        for (const shared_ptr<Instruction> &instr : cfg[b].instrs) {
            if (instr->op == Instruction::Op::Alloca ||
                instr->op == Instruction::Op::Alloc ||
                instr->op == Instruction::Op::AddressOf) {
                owned.insert(classes.find(classes.key(b, Value(instr))));
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
        if (cfg.contains(seed)) {
            result.masked.insert(seed);
        }
    }

    // Is `v`, referenced from `block`, a shared pointer to per-lane memory?
    auto points_per_lane = [&](BlockId block, const Value &v) {
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
        BlockId block;
        shared_ptr<Value> place;
        const Instruction *by;
    };
    vector<PendingWrite> pending;

    // The joins where the paths out of a divergent branch meet again: the
    // blocks two disjoint paths from two of its successors reach. Only at
    // these does an argument passed differently along the incoming edges
    // disagree between lanes -- Karrenberg & Hack's sync dependence ("Whole-
    // Function Vectorization", CGO 2011). Found the way LLVM's
    // SyncDependenceAnalysis finds them: each successor's paths carry its
    // label forward, a block handed two labels is a join and carries its own
    // from there on, so what is reached only through a join is reached one
    // way. Along forward edges only: what a back edge carries is the same
    // for every lane that reached the latch, and a loop some lanes leave
    // early is the temporal case, decided at the header below.
    //
    // Not the iterated dominance frontier, though that is where Cytron et al.
    // would place a phi for a value the two sides define differently: the
    // frontier iterates through the latch to the header of a loop the branch
    // sits in, and the header is reached the same way by every lane -- its
    // two edges are chosen by the loop's own uniform test. Read as a join it
    // made a leaf's loop over its primitives divergent, and with it the
    // primitives, and the instance one of them was, whenever the leaf was
    // visited by only the rays that hit its node.
    //
    // Grows monotonically with `result.branches`; `joins_of` remembers each
    // branch's joins.
    vector<size_t> rpo_index(cfg.size(), 0);
    for (size_t i = 0; i < cfg.rpo.size(); i++) {
        rpo_index[cfg.rpo[i]] = i;
    }
    const auto joins_of_branch = [&](BlockId branch) {
        BlockSet joins(cfg.size());
        vector<BlockId> label(cfg.size(), NO_BLOCK);
        for (BlockId s : cfg.succs[branch]) {
            if (label[s] == NO_BLOCK) {
                label[s] = s; // two edges to one block are one path
            }
        }
        for (BlockId x : cfg.rpo) {
            if (label[x] == NO_BLOCK) {
                continue;
            }
            for (BlockId y : cfg.succs[x]) {
                if (rpo_index[y] <= rpo_index[x]) {
                    continue; // a back edge
                }
                if (label[y] == NO_BLOCK) {
                    label[y] = label[x];
                } else if (label[y] != label[x]) {
                    joins.insert(y);
                    label[y] = y;
                }
            }
        }
        return joins;
    };
    BlockSet joins_found(cfg.size());
    BlockSet sync_joins(cfg.size());

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
        auto mark_written = [&](BlockId block, const shared_ptr<Value> &place,
                                const Instruction &by) {
            pending.push_back({block, place, &by});
        };

        result.in_scope = compute_scope(cfg, dom, result.args);

        for (const string &branch_name : result.branches) {
            const BlockId branch = cfg.id(branch_name);
            if (!joins_found.insert(branch)) {
                continue;
            }
            for (BlockId join : joins_of_branch(branch)) {
                sync_joins.insert(join);
            }
        }

        for (BlockId b : cfg.rpo) {
            const Block &block = cfg[b];
            const string &name = block.name;

            // A block executes with a partial mask if it is control dependent
            // on a divergent branch, or on a branch in a block that is itself
            // only partially executed: a block's mask is always a subset of
            // the masks of the blocks it is control dependent on.
            for (const auto &[from, to] : cdep[b]) {
                const string &from_name = cfg.name(from);
                if (result.branches.count(from_name) ||
                    result.masked.count(from_name)) {
                    mark(result.masked, name);
                }
            }
            const bool under_mask = result.masked.count(name) > 0;

            // At a join where the paths of a branch the lanes disagree about
            // meet (`sync_joins`), lanes arrive along different edges, so an
            // argument that is not passed the same definition along every
            // edge disagrees between lanes. Running under a mask is not by
            // itself that: a join of two edges a uniform branch chose between
            // -- the guard around an arm no lane is on (the BOSCC gadget in
            // SSA/Linearize.h), an `if` on a uniform value inside a
            // loop, or the header of a loop that runs inside a divergent
            // branch's arm -- is reached the same way by every live lane, and
            // its argument varies only if a value passed does.
            // The exception is a loop header whose lanes are at different
            // iterations: a lane that left the loop on an earlier iteration
            // keeps the value it had while the others go on, so the header
            // is divergent in time whatever the branches inside. That is so
            // when the loop has an exit some lanes take and others do not --
            // a latch control dependent, through blocks of the loop, on a
            // branch the lanes disagree about, which is the question
            // SSA/UniformizeLoops.h asks before folding a loop -- or when the
            // loop has already been folded and its blocks seeded as masked.
            // A branch outside the loop does not count: a loop that runs
            // under an outer condition is entered and left by the same lanes
            // together, and its latch's mask is the loop's own. Reading such
            // a latch as divergent would fold a loop whose trips every lane
            // agrees on, and everything indexed by it would vary -- a leaf's
            // primitives, and the instance one of them is, when the leaf is
            // only visited by the rays that hit its node.
            const vector<BlockId> &block_preds = cfg.preds[b];
            const Loop *loop = loops.find(b);
            const auto leaves_divergently = [&](BlockId latch) {
                if (loop == nullptr) {
                    return false;
                }
                const string &latch_name = cfg.name(latch);
                if (masked_seeds.count(latch_name)) {
                    result.divergent_headers.emplace(
                        name,
                        "latch " + latch_name + " is in a loop already folded");
                    return true;
                }
                BlockSet seen(cfg.size());
                vector<BlockId> work{latch};
                while (!work.empty()) {
                    const BlockId at = work.back();
                    work.pop_back();
                    if (!seen.insert(at)) {
                        continue;
                    }
                    for (const auto &[from, to] : cdep[at]) {
                        if (!loop->blocks.contains(from)) {
                            continue;
                        }
                        if (result.branches.count(cfg.name(from))) {
                            result.divergent_headers.emplace(
                                name, "latch " + latch_name +
                                          " is control dependent on the "
                                          "branch in " +
                                          cfg.name(from));
                            return true;
                        }
                        work.push_back(from);
                    }
                }
                return false;
            };
            const bool divergent_join =
                block_preds.size() > 1 &&
                (sync_joins.contains(b) ||
                 std::any_of(block_preds.begin(), block_preds.end(),
                             [&](BlockId p) {
                                 return loop != nullptr &&
                                        std::binary_search(
                                            loop->latches.begin(),
                                            loop->latches.end(), p) &&
                                        leaves_divergently(p);
                             }));

            const vector<ArgumentFlow> &incoming = flows[b];
            for (size_t j = 0; j < block.args.size(); j++) {
                std::optional<string> common_key;
                bool varying = false;
                // BONSAI_EXPLAIN_ARG=<name> prints the first time an argument
                // of that name is marked varying in any block, and why: the
                // edge that passed a varying value, or the join whose edges
                // passed different ones.
                static const char *const explain_arg =
                    std::getenv("BONSAI_EXPLAIN_ARG");
                const bool explain = explain_arg != nullptr &&
                                     block.args[j].name == explain_arg &&
                                     !result.args.count({name, block.args[j].name});
                for (const ArgumentFlow &flow : incoming) {
                    if (j < flow.first_arg ||
                        j - flow.first_arg >= flow.values->size()) {
                        continue;
                    }
                    const Value &v = *(*flow.values)[j - flow.first_arg];
                    const string &from_name = cfg.name(flow.from);
                    const bool passed_varying = result.is_varying(from_name, v);
                    const string key = value_key(v);
                    const bool differs = divergent_join &&
                                         common_key.has_value() &&
                                         *common_key != key;
                    if (explain && (passed_varying || differs)) {
                        std::cerr << "--- " << block.args[j].name << " of "
                                  << name << " is varying: from " << from_name
                                  << " it is passed ";
                        v.dump(std::cerr);
                        std::cerr << (passed_varying ? ", which varies"
                                                     : "")
                                  << (differs ? ", unlike another edge into "
                                                "this join of a divergent "
                                                "branch's paths"
                                              : "")
                                  << "\n";
                        if (passed_varying) {
                            explain_varying(func, result, from_name, v, 1);
                        }
                    }
                    varying |= passed_varying || differs;
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
                // together (Moll & Hack section 5). A vote is the gang's one
                // decision on a bool the lanes may disagree about, and is
                // uniform for the same reason; it is what keeps the children
                // of a sorted run uniform when each lane's key would have
                // ordered them differently.
                case Instruction::Op::Any:
                case Instruction::Op::Popcount:
                case Instruction::Op::Vote:
                    continue;

                // A push's value is the slot its entry took, and the lanes
                // take slots of their own whatever they push -- a gang that
                // pushes one value sixteen times over makes sixteen entries
                // -- so the slot is per lane however uniform the operands.
                case Instruction::Op::Push:
                    mark(result.instrs, instr.get());
                    continue;

                // The address of a value the lanes disagree about is one
                // address, of a temporary with a slot per lane.
                case Instruction::Op::AddressOf:
                    if (result.is_varying(name, *instr->operands[0])) {
                        mark(per_lane,
                             classes.find(classes.key(b, Value(instr))));
                    }
                    continue;

                // Reads per-lane memory: one value per lane. An element read
                // out of a per-lane array is one too, whichever element -- the
                // array holds a lane's own copy of every element.
                case Instruction::Op::Load:
                case Instruction::Op::ExtractIdx:
                    if (result.is_varying(name, *instr->operands[0]) ||
                        points_per_lane(b, *instr->operands[0]) ||
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
                            mark_written(b, instr->operands[0], *instr);
                        } else if (narrowed(*instr) || under_mask) {
                            const string k =
                                classes.key(b, *instr->operands[0]);
                            if (!k.empty() && owned.count(classes.find(k))) {
                                mark(per_lane, classes.find(k));
                            }
                        }
                    } else {
                        const string k = classes.key(b, *instr->operands[0]);
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
                    const string k = classes.key(b, *instr->operands[0]);
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
                                            points_per_lane(b, *v));
                                });
                if (!drop && varying_call) {
                    const Block &cont_block = cfg[cfg.id(cont.name)];
                    internal_assert(!cont_block.args.empty())
                        << "Call continuation " << cont_block.name
                        << " takes no result argument.";
                    mark(result.args,
                         std::pair{cont_block.name, cont_block.args[0].name});
                }
                if (varying_call || under_mask) {
                    for (const shared_ptr<Value> &v : args) {
                        const string k = classes.key(b, *v);
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
            const string &written_in = cfg.name(w.block);
            if (result.is_varying(written_in, *w.place)) {
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
                        std::cerr << (result.is_varying(written_in, *operand)
                                          ? " (varying)"
                                          : " (uniform)");
                    }
                }
                internal_error
                    << "\nThe write above, in " << written_in << ", stores a "
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

    for (BlockId b = 0; b < cfg.size(); b++) {
        const Block &block = cfg[b];
        const string &name = block.name;
        for (const shared_ptr<Instruction> &instr : block.instrs) {
            if (addresses_memory(instr->type) &&
                points_per_lane(b, Value(instr))) {
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
    result.pointee_in_scope = compute_scope(cfg, dom, result.pointee_args);

    return result;
}

void explain_varying(const Function &func, const Divergence &div,
                     const string &block, const Value &v, int depth) {
    if (depth > 12) {
        std::cerr << string(2 * depth, ' ') << "...\n";
        return;
    }
    const string pad(2 * depth, ' ');
    std::visit(
        overloads{
            [&](const shared_ptr<Instruction> &i) {
                std::cerr << pad;
                i->dump(std::cerr);
                std::cerr << "\n";
                for (const auto &operand : i->operands) {
                    if (div.is_varying(block, *operand)) {
                        explain_varying(func, div, block, *operand, depth + 1);
                    }
                }
            },
            [&](const Constant &c) {
                std::cerr << pad;
                c.dump(std::cerr);
                std::cerr << "\n";
            },
            [&](const Argument &a) {
                std::cerr << pad << "argument " << a.name << " : " << a.type
                          << ", declared varying by:\n";
                for (const auto &owner : func.blocks) {
                    if (!div.args.count({owner->name, a.name})) {
                        continue;
                    }
                    std::cerr << pad << "  " << owner->name
                              << (div.masked.count(owner->name) ? " (masked)"
                                                                : "");
                    if (const auto why = div.divergent_headers.find(owner->name);
                        why != div.divergent_headers.end()) {
                        std::cerr << " (lanes at different iterations: "
                                  << why->second << ")";
                    }
                    std::cerr << ", passed:\n";
                    size_t index = 0;
                    for (; index < owner->args.size(); index++) {
                        if (owner->args[index].name == a.name) {
                            break;
                        }
                    }
                    for (const auto &pred : func.blocks) {
                        for (Terminator::Jump *jump : jumps_of(*pred)) {
                            if (jump->name != owner->name) {
                                continue;
                            }
                            const size_t offset =
                                owner->args.size() - jump->args.size();
                            if (index < offset) {
                                continue;
                            }
                            const Value &passed = *jump->args[index - offset];
                            std::cerr << pad << "    from " << pred->name
                                      << (div.masked.count(pred->name)
                                              ? " (masked)"
                                              : "")
                                      << ": ";
                            passed.dump(std::cerr);
                            std::cerr << (div.is_varying(pred->name, passed)
                                              ? " (varying)"
                                              : " (uniform)")
                                      << "\n";
                        }
                    }
                }
            },
        },
        v.data);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
