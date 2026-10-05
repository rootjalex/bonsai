#include "SSA/GateDrains.h"

#include "Error.h"
#include "SSA/Analysis.h"
#include "SSA/Specialize.h"
#include "Utils.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::map;
using std::optional;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

// A fact under which a queue can be fed: bit `bit` of the owner's mask
// parameter `name` (a split variant occurs in the scene), or the owner's
// boolean `name` (pbrt's haveMedia, threaded to wherever the gate is built
// by Block::get_value). A gate is a disjunction of conjunctions of these --
// small by construction: one alternative per way the queue is pushed.
struct Conjunct {
    enum class Kind { MaskBit, OwnerBool } kind;
    string name;
    uint64_t bit = 0;
    // For an OwnerBool that is an instruction: the block defining it, so a
    // gate tests it only where that block dominates (a parameter's is
    // empty, the entry dominating everything).
    string def_block;
    bool operator<(const Conjunct &o) const {
        return std::tie(kind, name, bit, def_block) <
               std::tie(o.kind, o.name, o.bit, o.def_block);
    }
    bool operator==(const Conjunct &o) const {
        return kind == o.kind && name == o.name && bit == o.bit &&
               def_block == o.def_block;
    }
};
using Alt = set<Conjunct>;

struct Gate {
    // No alternatives: the queue cannot be fed. An empty alternative: it
    // always can (true). Alternatives are kept unique and few; past a
    // handful the gate collapses to true, which only costs a launch the
    // queue's emptiness makes a no-op.
    vector<Alt> alts;

    static Gate truth() { return Gate{{Alt{}}}; }
    bool is_true() const {
        return std::any_of(alts.begin(), alts.end(),
                           [](const Alt &a) { return a.empty(); });
    }
    bool is_false() const { return alts.empty(); }

    void add_alt(const Alt &a) {
        if (is_true()) {
            return;
        }
        if (a.empty()) {
            *this = truth();
            return;
        }
        // Absorption: an alternative implied by a weaker one adds nothing
        // -- (bit AND hs) under an existing (hs) -- and one weaker than
        // existing alternatives replaces them.
        for (const Alt &have : alts) {
            if (std::includes(a.begin(), a.end(), have.begin(), have.end())) {
                return;
            }
        }
        alts.erase(std::remove_if(alts.begin(), alts.end(),
                                  [&](const Alt &have) {
                                      return std::includes(have.begin(),
                                                           have.end(),
                                                           a.begin(), a.end());
                                  }),
                   alts.end());
        alts.push_back(a);
        if (alts.size() > 8) {
            // Too many ways in to spell out: keep what every way agrees on
            // -- the conjuncts all alternatives share -- as one alternative.
            // A superset of the gate, so only ever more launches; it is what
            // turns eleven (arm-bit AND have_subsurface) alternatives into
            // pbrt's one haveSubsurface.
            Alt common = alts.front();
            for (const Alt &other : alts) {
                Alt kept;
                std::set_intersection(common.begin(), common.end(),
                                      other.begin(), other.end(),
                                      std::inserter(kept, kept.begin()));
                common = std::move(kept);
            }
            *this = Gate{{std::move(common)}};
        }
    }
    void or_gate(const Gate &g) {
        for (const Alt &a : g.alts) {
            add_alt(a);
        }
    }
    // this AND the extra conjuncts, distributed over the alternatives.
    Gate and_with(const Alt &extra) const {
        Gate out;
        for (const Alt &a : alts) {
            Alt widened = a;
            widened.insert(extra.begin(), extra.end());
            out.add_alt(widened);
        }
        return out;
    }
    bool operator==(const Gate &o) const {
        if (alts.size() != o.alts.size()) {
            return false;
        }
        vector<Alt> a = alts, b = o.alts;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        return a == b;
    }
};

// One drain: the owner's block whose terminator is the parfor over the
// queue (`<family>!drain` holds the first leaf's, `<family>!<Label>!drain`
// the rest), the queue's full name (`hits!Matte`, `probes`), and the body
// region, for knowing which drain a call or push runs inside.
struct Drain {
    Function *owner = nullptr;
    shared_ptr<Block> block;
    string queue;
    string family;
    string label; // empty when the queue is not split
    set<string> region;
};

// One push instruction: where it is, which queue it feeds (the split leaf
// by the constant the queue pointer was indexed with, as lower_pushes reads
// it), and the dispatch conditions that dominate it, each resolvable to an
// owner boolean or dropped.
struct PushSite {
    Function *func = nullptr;
    shared_ptr<Block> block;
    string family;
    optional<uint64_t> leaf;
    vector<shared_ptr<Value>> conds; // conditions on the true edges above
};

const Argument *as_argument(const shared_ptr<Value> &v) {
    return v == nullptr ? nullptr : std::get_if<Argument>(&v->data);
}

const shared_ptr<Instruction> *as_instruction(const shared_ptr<Value> &v) {
    return v == nullptr ? nullptr
                        : std::get_if<shared_ptr<Instruction>>(&v->data);
}

optional<uint64_t> constant_of(const shared_ptr<Value> &v) {
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return std::nullopt;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u;
    }
    if (const auto *i = std::get_if<int64_t>(&c->data)) {
        return *i >= 0 ? optional<uint64_t>(uint64_t(*i)) : std::nullopt;
    }
    return std::nullopt;
}

// The queue a value stands for: the name of the parameter or local it
// traces to -- `_queue_<family>` as a chain function is handed it,
// `<family>_queue` as the owner makes it -- and the leaf a constant GEP
// picked, exactly the reading lower_pushes takes (SSA/Defer.cpp).
optional<std::pair<string, optional<uint64_t>>>
queue_of(const shared_ptr<Value> &q) {
    shared_ptr<Value> base = q;
    optional<uint64_t> leaf;
    if (const auto *qi = as_instruction(base);
        qi != nullptr && (*qi)->op == Instruction::Op::GEP &&
        (*qi)->operands.size() == 2) {
        if (const auto k = constant_of((*qi)->operands[1])) {
            leaf = *k;
            base = (*qi)->operands[0];
        }
    }
    string name;
    if (const Argument *a = as_argument(base)) {
        name = a->name;
    } else if (const auto *bi = as_instruction(base)) {
        name = (*bi)->name;
    } else {
        return std::nullopt;
    }
    if (name.rfind("_queue_", 0) == 0) {
        return {{name.substr(7), leaf}};
    }
    const string suffix = "_queue";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
            0) {
        return {{name.substr(0, name.size() - suffix.size()), leaf}};
    }
    return std::nullopt;
}

// The function's CFG and dominator tree, built once and shared by every
// block queried in it. A snapshot: taken before any gate rewrites the
// owner, which is fine, since the conditions are read before the rewrites.
struct Doms {
    std::unique_ptr<Cfg> cfg;
    DomTree tree;
    LoopForest loops;
};

const Doms &doms_of(const Function &f,
                    map<const Function *, Doms> &cache) {
    auto it = cache.find(&f);
    if (it == cache.end()) {
        Doms d;
        d.cfg = std::make_unique<Cfg>(f);
        d.tree = compute_dominator_tree(*d.cfg);
        d.loops = compute_loop_forest(*d.cfg, d.tree);
        it = cache.emplace(&f, std::move(d)).first;
    }
    return it->second;
}

// The dispatch conditions that hold wherever `block` runs: each dominator
// of the block that ends in a two-way dispatch on a boolean whose true
// target also dominates the block -- every path in takes the true edge --
// contributes its condition, however many joins lie between (the inlined
// match between pbrt's `haveSubsurface &&` test and the BSSRDF push). A
// false edge's negation and a join's disjunction are dropped rather than
// represented, which only weakens a gate toward launching.
vector<shared_ptr<Value>> dominating_conditions(const Function &func,
                                                const Block &push_block,
                                                map<const Function *, Doms>
                                                    &cache) {
    vector<shared_ptr<Value>> conds;
    const Doms &d = doms_of(func, cache);
    const BlockId b = d.cfg->find(push_block);
    if (b == NO_BLOCK || !d.tree.contains(b)) {
        return conds;
    }
    for (BlockId at = d.tree.idom[b]; at != NO_BLOCK;
         at = d.tree.idom[at] == at ? NO_BLOCK : d.tree.idom[at]) {
        const Block &D = (*d.cfg)[at];
        if (const auto *disp =
                std::get_if<Terminator::Dispatch>(&D.terminator.data)) {
            if (disp->targets.size() == 2 && disp->cond != nullptr &&
                disp->cond->get_type().is<Bool_t>()) {
                const BlockId t1 = d.cfg->find(disp->targets[1].name);
                const BlockId t0 = d.cfg->find(disp->targets[0].name);
                if (t1 != NO_BLOCK && t1 != t0 && t1 != at &&
                    d.tree.dominates(t1, b)) {
                    conds.push_back(disp->cond);
                }
            }
        }
    }
    return conds;
}

// A condition taken apart through its conjunctions: an `a && b` the
// frontend made one boolean value of contributes both sides, each resolved
// or dropped on its own -- `have_subsurface && is_transmissive(..)` gates
// on the first and forgets the second.
void conjunct_values(const shared_ptr<Value> &v,
                     vector<shared_ptr<Value>> &out) {
    if (const auto *i = as_instruction(v);
        i != nullptr &&
        ((*i)->op == Instruction::Op::LAnd ||
         ((*i)->op == Instruction::Op::BwAnd &&
          (*i)->type.is<Bool_t>()))) {
        for (const auto &operand : (*i)->operands) {
            conjunct_values(operand, out);
        }
        return;
    }
    out.push_back(v);
}

// An owner boolean as owner_bool spells it -- `name`, or
// `name\ndef_block\nowner` for an instruction -- as the conjunct it stands
// for; def_block carries the owner's name behind a second newline.
Conjunct owner_conjunct(const string &spelled) {
    Conjunct c{Conjunct::Kind::OwnerBool, spelled, 0, {}};
    if (const auto p = spelled.find('\n'); p != string::npos) {
        c.name = spelled.substr(0, p);
        c.def_block = spelled.substr(p + 1);
    }
    return c;
}

// Every call of each function, by callee name: the caller and the block
// whose terminator makes the call.
struct CallSites {
    map<string, vector<std::pair<Function *, Block *>>> of;
};

CallSites index_calls(const FuncMap &fmap) {
    CallSites out;
    for (const auto &[name, f] : fmap) {
        for (const auto &block : f->blocks) {
            if (const auto *c =
                    std::get_if<Terminator::Call>(&block->terminator.data)) {
                out.of[c->call.name].push_back({f.get(), block.get()});
            }
        }
    }
    return out;
}

// The argument a call passes for the callee's parameter named `param`.
shared_ptr<Value> call_argument(const Block &site, const Function &callee,
                                const string &param) {
    const auto *c = std::get_if<Terminator::Call>(&site.terminator.data);
    internal_assert(c != nullptr) << site.name;
    const auto &params = callee.blocks.front()->args;
    for (size_t i = 0; i < params.size() && i < c->call.args.size(); i++) {
        if (params[i].name == param) {
            return c->call.args[i];
        }
    }
    return nullptr;
}

} // namespace

vector<ir::Program::ArmMask>
gate_drains(FuncMap &fmap,
            const std::vector<ir::QueueSpecialize> &queue_splits,
            const map<string, ir::Program::AdtStorage> &storages) {
    // --- The drains, found by what they drain: every drain's parfor still
    // carries its queue here (Terminator::ParFor::queue_base, set by
    // Defer.cpp and moved to the outer loop by a split; the host drains'
    // copies are cleared only after this pass, SSA/Convert.cpp), so a loop
    // is a drain of family `f` exactly when its queue_base names
    // `f_queue` / `_queue_f` -- whatever a schedule's split or specialize
    // renamed the loop itself to. Which leaf of a split it drains is the
    // variant's label in the loop's name, matched below once the split's
    // variants are known; a copy a bool specialize made (`shadow!False`)
    // matches no variant and shares the family's gate, as it shares the
    // family's queue.
    vector<Drain> drains;
    for (const auto &[fname, f] : fmap) {
        if (f->queue_sizes.empty()) {
            continue;
        }
        refresh_preds(*f);
        for (const auto &block : f->blocks) {
            const auto *loop =
                std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (loop == nullptr || loop->queue_base == nullptr) {
                continue;
            }
            const auto q = queue_of(loop->queue_base);
            if (!q || !f->queue_sizes.contains(q->first)) {
                continue;
            }
            Drain d;
            d.owner = f.get();
            d.block = block;
            d.family = q->first;
            d.queue = q->first; // the leaf's label joins below, if any
            drains.push_back(d);
        }
    }
    // A split's outer loop carries the queue onward; where both halves
    // still name it, keep the outermost -- the one whose region holds the
    // other.
    for (Drain &d : drains) {
        const auto *loop =
            std::get_if<Terminator::ParFor>(&d.block->terminator.data);
        const Cfg region(*d.owner, loop->body.name);
        for (const shared_ptr<Block> &b : region.blocks()) {
            d.region.insert(b->name);
        }
    }
    for (auto it = drains.begin(); it != drains.end();) {
        const bool inner = std::any_of(
            drains.begin(), drains.end(), [&](const Drain &outer) {
                return &outer != &*it && outer.owner == it->owner &&
                       outer.family == it->family &&
                       outer.region.contains(it->block->name);
            });
        it = inner ? drains.erase(it) : it + 1;
    }
    if (drains.empty()) {
        return {};
    }

    // --- The split masks: one u64 parameter on the owner per ADT a queue
    // is split on, bit = variant tag; the leaf's drain wears its bit. The
    // drained function is read off a leaf body's call (`hits!Matte!run`
    // calls `shade!Matte`), the key's type off the function the schedule
    // named it in, and an `option` key gets no mask. Only an unnested split
    // is gated; a deeper one launches as before.
    vector<ir::Program::ArmMask> masks;
    map<string, Alt> mask_alt; // leaf queue name -> its MaskBit conjunct
    map<string, vector<KeyVariant>> family_variants;
    for (const ir::QueueSpecialize &qs : queue_splits) {
        if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
            std::fprintf(stderr, "gate_drains: split %s key %s under %zu\n",
                         qs.queue.c_str(), qs.key.c_str(), qs.under.size());
        }
        if (!qs.under.empty()) {
            continue;
        }
        // The function the key lives in: the callee a drain of this family
        // runs -- a copy where the split replaced the original, and a copy
        // keeps the key's name and type.
        string drained;
        Function *owner = nullptr;
        for (const Drain &d : drains) {
            if (d.family != qs.queue) {
                continue;
            }
            for (const auto &block : d.owner->blocks) {
                if (!d.region.contains(block->name)) {
                    continue;
                }
                const auto *c =
                    std::get_if<Terminator::Call>(&block->terminator.data);
                if (c != nullptr && fmap.contains(c->call.name) &&
                    key_value(*fmap.at(c->call.name), qs.key) != nullptr) {
                    drained = c->call.name;
                    owner = d.owner;
                    break;
                }
            }
            if (!drained.empty()) {
                break;
            }
        }
        if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
            std::fprintf(stderr,
                         "gate_drains: split %s drained '%s' (in fmap: %d)\n",
                         qs.queue.c_str(), drained.c_str(),
                         drained.empty() ? 0 : int(fmap.contains(drained)));
        }
        if (drained.empty() || !fmap.contains(drained)) {
            continue;
        }
        const shared_ptr<Value> key = key_value(*fmap.at(drained), qs.key);
        if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
            std::fprintf(stderr, "gate_drains: split %s key value %s\n",
                         qs.queue.c_str(), key == nullptr ? "NULL" : "found");
        }
        if (key == nullptr) {
            continue;
        }
        const Type key_type = key->get_type();
        const Struct_t *key_struct = key_type.as<Struct_t>();
        if (key_struct == nullptr || !storages.contains(key_struct->name)) {
            continue; // a bool or an option: two queues, no mask to index
        }
        const vector<KeyVariant> variants = key_variants(key_type, storages);
        if (variants.empty()) {
            continue;
        }
        const string adt = key_struct->name;
        string param;
        for (const char ch : adt) {
            param += char(std::tolower(static_cast<unsigned char>(ch)));
        }
        param += "_arms";

        ir::Program::ArmMask m;
        for (const auto &[fname, f] : fmap) {
            if (f.get() == owner) {
                m.func = fname;
                break;
            }
        }
        m.param = param;
        m.adt = adt;
        bool fresh = true;
        for (const Argument &a : owner->blocks.front()->args) {
            if (a.name == param) {
                fresh = false; // two splits on one ADT share the mask
            }
        }
        for (const KeyVariant &v : variants) {
            m.arm_bits.push_back({v.label, v.tag});
            m.all_on |= uint64_t(1) << v.tag;
            mask_alt[qs.queue + "!" + v.label] = {
                Conjunct{Conjunct::Kind::MaskBit, param, v.tag, {}}};
        }
        family_variants[qs.queue] = variants;
        if (fresh) {
            Argument arg;
            arg.name = param;
            arg.type = UInt_t::make(64);
            owner->blocks.front()->add_argument(arg);
            // Every call of the owner inside the program passes every arm
            // present -- the sound default a driver that computes nothing
            // passes too; the driver that knows the scene is the exported
            // caller, through the generated header's constants.
            for (const auto &[gname, g] : fmap) {
                for (const auto &block : g->blocks) {
                    if (auto *c = std::get_if<Terminator::Call>(
                            &block->terminator.data)) {
                        if (c->call.name == m.func) {
                            c->call.args.push_back(std::make_shared<Value>(
                                Constant{UInt_t::make(64), m.all_on}));
                        }
                    }
                }
            }
            masks.push_back(m);
        }
    }

    // Which leaf each drain is: the variant's label in the loop's name,
    // delimited -- `hits_blk!Diffuse` is the Diffuse leaf's launch,
    // `hits!DiffuseTransmission` is not Diffuse's -- longest label first. A
    // drain matching no variant keeps the family's name: an unsplit queue,
    // or a bool specialize's copy (`shadow!False`), which drains the same
    // family queue as its sibling and shares the family's gate.
    for (Drain &d : drains) {
        const auto vs = family_variants.find(d.family);
        if (vs == family_variants.end()) {
            continue;
        }
        const auto *loop =
            std::get_if<Terminator::ParFor>(&d.block->terminator.data);
        vector<const KeyVariant *> ordered;
        for (const KeyVariant &v : vs->second) {
            ordered.push_back(&v);
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const KeyVariant *a, const KeyVariant *b) {
                      return a->label.size() > b->label.size();
                  });
        for (const KeyVariant *v : ordered) {
            const string needle = "!" + v->label;
            const size_t at = loop->index.find(needle);
            if (at == string::npos) {
                continue;
            }
            const size_t after = at + needle.size();
            if (after < loop->index.size() && loop->index[after] != '!' &&
                loop->index[after] != '_') {
                continue;
            }
            d.label = v->label;
            d.queue = d.family + "!" + v->label;
            break;
        }
    }
    map<Function *, map<string, string>> region_queue; // block -> queue
    for (const Drain &d : drains) {
        for (const string &bname : d.region) {
            region_queue[d.owner][bname] = d.queue;
        }
    }

    if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
        std::fprintf(stderr, "gate_drains: %zu masks, %zu mask_alts\n",
                     masks.size(), mask_alt.size());
        for (const auto &m : masks) {
            std::fprintf(stderr, "  mask %s on %s (adt %s, all 0x%llx)\n",
                         m.param.c_str(), m.func.c_str(), m.adt.c_str(),
                         (unsigned long long)m.all_on);
        }
        for (const Drain &d : drains) {
            std::fprintf(stderr, "  drain %s (family %s) in %s\n",
                         d.queue.c_str(), d.family.c_str(),
                         d.block->name.c_str());
        }
    }

    // --- The pushes, with their dominating conditions.
    map<const Function *, Doms> doms_cache;
    vector<PushSite> pushes;
    for (const auto &[fname, f] : fmap) {
        refresh_preds(*f);
        for (const auto &block : f->blocks) {
            for (const auto &instr : block->instrs) {
                if (instr->op != Instruction::Op::Push) {
                    continue;
                }
                const auto q = queue_of(instr->operands[0]);
                if (!q) {
                    continue;
                }
                PushSite p;
                p.func = f.get();
                p.block = block;
                p.family = q->first;
                p.leaf = q->second;
                p.conds = dominating_conditions(*f, *block, doms_cache);
                pushes.push_back(std::move(p));
            }
        }
    }

    const CallSites calls = index_calls(fmap);
    if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
        std::fprintf(stderr, "gate_drains: %zu pushes\n", pushes.size());
        for (const PushSite &p : pushes) {
            std::fprintf(stderr, "  push to %s[%lld] in %s, %zu conds\n",
                         p.family.c_str(),
                         p.leaf ? (long long)*p.leaf : -1,
                         p.block->name.c_str(), p.conds.size());
        }
    }
    const set<Function *> owners = [&] {
        set<Function *> o;
        for (const Drain &d : drains) {
            o.insert(d.owner);
        }
        return o;
    }();

    // A condition as the owner's one boolean, if it is one: an argument
    // threaded from the owner itself, or a parameter every caller hands the
    // same owner boolean -- pbrt's haveMedia, passed down the chain by the
    // invariant-parameter hoist (SSA/Defer.cpp). Anything else -- an
    // entry's own state, callers that disagree -- is no gate.
    std::function<optional<string>(Function *, const shared_ptr<Value> &,
                                   set<std::pair<Function *, string>> &)>
        owner_bool = [&](Function *F, const shared_ptr<Value> &v,
                         set<std::pair<Function *, string>> &visiting)
        -> optional<string> {
        if (v == nullptr || !v->get_type().is<Bool_t>()) {
            return std::nullopt;
        }
        const Argument *a = as_argument(v);
        if (owners.contains(F)) {
            // The owner's value by the name the gate will fetch it under:
            // a parameter, or an instruction the owner computes once --
            // pbrt's haveSubsurface unwrapped from the integrator -- whose
            // name Block::get_value threads to the drain. An instruction's
            // defining block rides along, for the emission to test only
            // where it dominates (a per-band value does not gate a drain
            // outside its band).
            if (a != nullptr) {
                return a->name;
            }
            if (const auto *i = as_instruction(v);
                i != nullptr && !(*i)->name.empty()) {
                const auto def = (*i)->owner.lock();
                if (def == nullptr) {
                    return std::nullopt;
                }
                string fname;
                for (const auto &[n, fn] : fmap) {
                    if (fn.get() == F) {
                        fname = n;
                    }
                }
                return (*i)->name + "\n" + def->name + "\n" + fname;
            }
            return std::nullopt;
        }
        if (a == nullptr) {
            return std::nullopt;
        }
        bool is_param = false;
        for (const Argument &pa : F->blocks.front()->args) {
            if (pa.name == a->name) {
                is_param = true;
            }
        }
        if (!is_param) {
            return std::nullopt;
        }
        string fname;
        for (const auto &[n, f] : fmap) {
            if (f.get() == F) {
                fname = n;
            }
        }
        if (!visiting.insert({F, a->name}).second) {
            return std::nullopt;
        }
        const auto sites = calls.of.find(fname);
        if (sites == calls.of.end() || sites->second.empty()) {
            return std::nullopt;
        }
        optional<string> agreed;
        for (const auto &[G, B] : sites->second) {
            const shared_ptr<Value> arg = call_argument(*B, *F, a->name);
            const optional<string> up = owner_bool(G, arg, visiting);
            if (!up || (agreed && *agreed != *up)) {
                return std::nullopt;
            }
            agreed = up;
        }
        return agreed;
    };

    // --- The gates, to a fixpoint from "nothing can be fed": a queue's
    // gate is its mask bit AND the union over its pushes of the gate of
    // wherever the push runs AND what dominates the push; a push outside
    // every drain -- the producer's own iteration -- runs unconditionally.
    map<string, Gate> gates;
    for (const Drain &d : drains) {
        gates[d.queue] = Gate{};
    }
    const auto leaf_name = [&](const PushSite &p) -> optional<string> {
        if (!p.leaf) {
            return gates.contains(p.family) ? optional<string>(p.family)
                                            : std::nullopt;
        }
        // The k-th leaf, in the split's variant order -- the order the
        // leaves' storage was built in (SSA/Defer.cpp).
        const auto vs = family_variants.find(p.family);
        if (vs == family_variants.end() || *p.leaf >= vs->second.size()) {
            return std::nullopt;
        }
        const string q = p.family + "!" + vs->second[*p.leaf].label;
        return gates.contains(q) ? optional<string>(q) : std::nullopt;
    };

    // Whether a conjunct holds wherever any drain tests it: a parameter
    // always; an owner instruction only when it is defined outside every
    // loop -- a loop-carried value read at a drain is not the value the
    // push was made under, a double-buffered queue's entries draining a
    // round after they are pushed -- and in a block dominating every drain
    // of its owner, so the producer drain that feeds a queue launches under
    // at least what the consumer's gate tests. One rule for every use, so
    // the fixpoint's contexts and the emitted dispatches agree.
    const auto admit = [&](const Conjunct &c) -> bool {
        if (c.kind != Conjunct::Kind::OwnerBool || c.def_block.empty()) {
            return true;
        }
        const auto split_at = c.def_block.find('\n');
        if (split_at == string::npos) {
            return false;
        }
        const string def = c.def_block.substr(0, split_at);
        const string owner_name = c.def_block.substr(split_at + 1);
        const auto fit = fmap.find(owner_name);
        if (fit == fmap.end()) {
            return false;
        }
        const Doms &od = doms_of(*fit->second, doms_cache);
        const BlockId db = od.cfg->find(def);
        if (db == NO_BLOCK || od.loops.depth(db) != 0) {
            return false;
        }
        for (const Drain &d : drains) {
            if (d.owner != fit->second.get()) {
                continue;
            }
            const BlockId at = od.cfg->find(*d.block);
            if (at == NO_BLOCK || !od.tree.dominates(db, at)) {
                return false;
            }
        }
        return true;
    };

    // The gate of the place a function's code runs, from the previous
    // round's queue gates; cycles read as true, which only launches.
    for (int round = 0; round < 16; round++) {
        map<Function *, Gate> fctx;
        std::function<Gate(Function *, set<Function *> &)> context =
            [&](Function *F, set<Function *> &visiting) -> Gate {
            if (owners.contains(F)) {
                return Gate::truth();
            }
            if (const auto it = fctx.find(F); it != fctx.end()) {
                return it->second;
            }
            if (!visiting.insert(F).second) {
                return Gate::truth();
            }
            string fname;
            for (const auto &[n, f] : fmap) {
                if (f.get() == F) {
                    fname = n;
                }
            }
            Gate g;
            const auto sites = calls.of.find(fname);
            if (sites != calls.of.end()) {
                for (const auto &[G, B] : sites->second) {
                    Gate at;
                    if (owners.contains(G)) {
                        const auto &regions = region_queue[G];
                        const auto r = regions.find(B->name);
                        at = r == regions.end() ? Gate::truth()
                                                : gates.at(r->second);
                    } else {
                        at = context(G, visiting);
                    }
                    // What dominates the call itself -- the subsurface
                    // chain's `have_subsurface && ..` sits at the call of
                    // vol_subsurface, not inside it.
                    Alt conjuncts;
                    const auto doms = dominating_conditions(*G, *B, doms_cache);
                    vector<shared_ptr<Value>> parts;
                    for (const shared_ptr<Value> &cond : doms) {
                        conjunct_values(cond, parts);
                    }
                    for (const shared_ptr<Value> &cond : parts) {
                        set<std::pair<Function *, string>> vb;
                        if (const optional<string> nm =
                                owner_bool(G, cond, vb)) {
                            const Conjunct c = owner_conjunct(*nm);
                            if (admit(c)) {
                                conjuncts.insert(c);
                            }
                        }
                    }
                    g.or_gate(at.and_with(conjuncts));
                }
            }
            visiting.erase(F);
            fctx[F] = g;
            return g;
        };

        map<string, Gate> next;
        for (const Drain &d : drains) {
            next[d.queue] = Gate{};
        }
        for (const PushSite &p : pushes) {
            const optional<string> q = leaf_name(p);
            if (!q) {
                continue;
            }
            Gate at;
            if (owners.contains(p.func)) {
                const auto &regions = region_queue[p.func];
                const auto r = regions.find(p.block->name);
                at = r == regions.end() ? Gate::truth() : gates.at(r->second);
            } else {
                set<Function *> visiting;
                at = context(p.func, visiting);
            }
            Alt conjuncts;
            vector<shared_ptr<Value>> parts;
            for (const shared_ptr<Value> &cond : p.conds) {
                conjunct_values(cond, parts);
            }
            for (const shared_ptr<Value> &cond : parts) {
                set<std::pair<Function *, string>> visiting;
                if (const optional<string> name =
                        owner_bool(p.func, cond, visiting)) {
                    const Conjunct c = owner_conjunct(*name);
                    if (admit(c)) {
                        conjuncts.insert(c);
                    }
                }
            }
            next[*q].or_gate(at.and_with(conjuncts));
        }
        for (auto &[qname, g] : next) {
            const auto m = mask_alt.find(qname);
            if (m != mask_alt.end()) {
                g = g.and_with(m->second);
            }
        }
        if (next == gates) {
            break;
        }
        gates = std::move(next);
    }

    if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
        for (const auto &[q, g] : gates) {
            std::string s = g.is_true() ? "TRUE" : g.is_false() ? "FALSE" : "";
            for (const Alt &a : g.alts) {
                s += " |";
                for (const Conjunct &c : a) {
                    s += " " + c.name +
                         (c.kind == Conjunct::Kind::MaskBit
                              ? "[" + std::to_string(c.bit) + "]"
                              : "");
                }
            }
            std::fprintf(stderr, "  gate %s =%s\n", q.c_str(), s.c_str());
        }
    }

    // --- The dispatches: each gated drain's block keeps everything it did
    // -- the counts it reads, the storage of queues after it -- and only
    // the loop moves behind the test, into a selector entered on true; the
    // false edge jumps straight to the loop's continuation, whose arguments
    // the block already names. Zero iterations store nothing, so the skip
    // is the launch's cost and nothing else.
    for (Function *O : owners) {
        doms_of(*O, doms_cache); // snapshots, before any rewrite below
    }
    for (const Drain &d : drains) {
        Function &O = *d.owner;
        Block &B = *d.block;
        // Every conjunct in a gate was admitted as testable at every drain
        // of its owner (see `admit`), so the gate is read as it stands.
        const Gate &g = gates.at(d.queue);
        if (g.is_true() || g.is_false()) {
            // True: nothing testable gates it. False: unfeedable with no
            // fact to test would mean dead schedule machinery; launch as
            // before rather than judge it here.
            continue;
        }

        const auto boolean = Bool_t::make();
        const auto u64 = UInt_t::make(64);
        const auto c64 = [&](uint64_t v) {
            return std::make_shared<Value>(Constant{u64, v});
        };
        shared_ptr<Value> cond;
        for (const Alt &alt : g.alts) {
            shared_ptr<Value> all;
            for (const Conjunct &c : alt) {
                shared_ptr<Value> one;
                if (c.kind == Conjunct::Kind::OwnerBool) {
                    one = B.get_value(c.name, boolean);
                } else {
                    auto m = B.get_value(c.name, u64);
                    auto s = B.make_instruction(u64, Instruction::Op::Shr,
                                                {m, c64(c.bit)});
                    auto b = B.make_instruction(u64, Instruction::Op::BwAnd,
                                                {s, c64(1)});
                    one = B.make_instruction(boolean, Instruction::Op::Ne,
                                             {b, c64(0)});
                }
                all = all == nullptr
                          ? one
                          : B.make_instruction(boolean,
                                               Instruction::Op::LAnd,
                                               {all, one});
            }
            cond = cond == nullptr
                       ? all
                       : B.make_instruction(boolean, Instruction::Op::LOr,
                                            {cond, all});
        }
        internal_assert(cond != nullptr) << d.queue;

        // The loop read only now: building the condition can grow this very
        // parfor's argument lists -- threading a mask up from a later drain
        // climbs the round loop's back edge and re-enters the blocks below,
        // this one's continuation among them -- and a copy taken before
        // that would put the stale lists back.
        const Terminator::ParFor loop =
            std::get<Terminator::ParFor>(B.terminator.data);

        auto sel = std::make_shared<Block>();
        sel->name = B.name + "!launch";
        sel->owner = B.owner;
        sel->terminator.data = loop;
        O.blocks.push_back(sel);

        Terminator::Dispatch dispatch;
        dispatch.cond = cond;
        dispatch.targets.push_back(
            Terminator::Jump{loop.cont.name, loop.cont.args});
        dispatch.targets.push_back(Terminator::Jump{sel->name, {}});
        B.terminator.data = std::move(dispatch);
        refresh_preds(O);

        auto &entered = std::get<Terminator::ParFor>(sel->terminator.data);
        const auto thread_value = [&](shared_ptr<Value> &v) {
            if (const Argument *a = as_argument(v)) {
                v = sel->get_value(a->name, a->type);
            }
        };
        entered.body.args = [&] {
            vector<shared_ptr<Value>> out;
            for (auto v : entered.body.args) {
                thread_value(v);
                out.push_back(v);
            }
            return out;
        }();
        entered.cont.args = [&] {
            vector<shared_ptr<Value>> out;
            for (auto v : entered.cont.args) {
                thread_value(v);
                out.push_back(v);
            }
            return out;
        }();
        thread_value(entered.start);
        thread_value(entered.end);
        thread_value(entered.stride);
        if (entered.capacity != nullptr) {
            thread_value(entered.capacity);
        }
        if (entered.queue_base != nullptr) {
            thread_value(entered.queue_base);
        }
        if (entered.count_address != nullptr) {
            thread_value(entered.count_address);
        }
        refresh_preds(O);
    }

    if (std::getenv("BONSAI_DEBUG_GATES") != nullptr) {
        // Every edge's argument count against its target's parameter list,
        // so a mismatch is caught here rather than at the relooper.
        for (Function *O : owners) {
            map<string, size_t> takes;
            for (const auto &block : O->blocks) {
                takes[block->name] = block->args.size();
            }
            const auto check = [&](const string &from,
                                   const Terminator::Jump &j, size_t offset) {
                if (takes.contains(j.name) &&
                    takes[j.name] != j.args.size() + offset) {
                    std::fprintf(stderr,
                                 "gate_drains: BAD EDGE %s -> %s: %zu args "
                                 "for %zu params\n",
                                 from.c_str(), j.name.c_str(),
                                 j.args.size() + offset, takes[j.name]);
                }
            };
            for (const auto &block : O->blocks) {
                std::visit(
                    overloads{
                        [&](const Terminator::Jump &j) {
                            check(block->name, j, 0);
                        },
                        [&](const Terminator::Dispatch &d) {
                            for (const auto &t : d.targets) {
                                check(block->name, t, 0);
                            }
                        },
                        [&](const Terminator::ParFor &p) {
                            check(block->name, p.body, 1);
                            check(block->name, p.cont, 0);
                        },
                        [&](const auto &) {}},
                    block->terminator.data);
            }
        }
    }

    return masks;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
