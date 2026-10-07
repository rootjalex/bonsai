#include "Opt/NarrowLoads.h"

#include "Opt/ValueClass.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"
#include "IR/WriteLoc.h"

#include "Error.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace bonsai {
namespace opt {

using namespace ir;

namespace {

// Rewrites every `x.f` (and, for a vector-typed load, every `x[k]` at a
// constant k) to its narrowed name and counts anything else done with `x`:
// a bare use that is neither means the whole value is wanted and the let
// must stay as it is. The keys are field names for a struct and the
// constant indices' digits for a vector.
struct NarrowUses : Mutator {
    std::string base;
    const std::map<std::string, std::string> *field_names = nullptr;
    std::set<std::string> fields_seen;
    bool whole_use = false;
    // The names bound to the value whole -- the base and every transitive
    // `let x = <one of them>`, inlining's parameter binds. Their field
    // reads are the value's, and the binds themselves dissolve in the
    // rewrite (visit(LetStmt), visit(Sequence)).
    std::set<std::string> aliases;

    bool matches(const Expr &e) {
        const Var *var = e.as<Var>();
        return var != nullptr &&
               (var->name == base || aliases.count(var->name) > 0);
    }

    Stmt visit(const LetStmt *node) override {
        if (node->loc.accesses.empty() && matches(node->value)) {
            // An inliner's parameter bind: not a whole use. Its reads are
            // the value's, counted and rewritten through the alias; the
            // bind itself goes dead with the base and DCE collects both
            // (the narrowing is additive -- nothing is erased here).
            aliases.insert(node->loc.base);
            return node;
        }
        return Mutator::visit(node);
    }

    Expr visit(const Access *node) override {
        if (matches(node->value)) {
            fields_seen.insert(node->field);
            const auto named = field_names->find(node->field);
            if (named == field_names->end()) {
                // Counting pass: leave it, the caller only wants
                // `fields_seen` and `whole_use`.
                return node;
            }
            return Var::make(node->type, named->second);
        }
        return Mutator::visit(node);
    }

    Expr visit(const Extract *node) override {
        if (matches(node->vec)) {
            const ir::UIntImm *u = node->idx.as<ir::UIntImm>();
            const ir::IntImm *i = node->idx.as<ir::IntImm>();
            if ((u == nullptr && i == nullptr) || node->mask.defined()) {
                whole_use = true;
                return node;
            }
            const std::string key =
                std::to_string(u != nullptr ? u->value : uint64_t(i->value));
            fields_seen.insert(key);
            const auto named = field_names->find(key);
            if (named == field_names->end()) {
                return node;
            }
            return Var::make(node->type, named->second);
        }
        return Mutator::visit(node);
    }

    Expr visit(const Var *node) override {
        if (node->name == base || aliases.count(node->name) > 0) {
            whole_use = true;
        }
        return node;
    }
};

// The merge-temp form of the same dead-field problem: the inliner gives a
// value-returning match a result temp -- `r : mut T` allocated, every arm of
// the switch ending in a whole-struct `r = e_k`, every later use a field
// read `r.f` (Opt/Inline.cpp's Nester) -- and the fields nothing reads stay
// alive through every arm because the store is whole. Scalar replacement of
// aggregates (Muchnick 12.2) on exactly that shape: one temp per field READ,
// each arm's store split into that field's stores (`r$f = (e_k).f`, the
// unread fields' chains left for DCE to reap arm by arm), each read renamed.
// Same point, same values; the store's value is duplicated per field, so it
// must be duplication-safe (pure by ValueClass, memory reads included --
// re-read at the same statement).
struct TempUses : Mutator {
    std::string base;
    Type temp_type;
    // Field name -> the per-field temp; empty map = the counting pass.
    const std::map<std::string, std::string> *field_names = nullptr;
    std::set<std::string> fields_seen;
    bool whole_use = false;   // a read of r outside `r.f`
    bool bad_store = false;   // a partial store, a masked one, or an unsafe value
    int64_t *counter = nullptr;

    Expr visit(const Access *node) override {
        if (const Var *var = node->value.as<Var>();
            var != nullptr && var->name == base) {
            fields_seen.insert(node->field);
            const auto named = field_names->find(node->field);
            if (named == field_names->end()) {
                return node;
            }
            return Var::make(node->type, named->second);
        }
        return Mutator::visit(node);
    }

    Expr visit(const Var *node) override {
        if (node->name == base) {
            whole_use = true;
        }
        return node;
    }

    Stmt visit(const Store *node) override {
        if (node->loc.base != base) {
            return Mutator::visit(node);
        }
        if (!node->loc.accesses.empty() || node->mask.defined() ||
            !classify(node->value).sinkable) {
            bad_store = true;
            return node;
        }
        if (field_names->empty()) {
            // Counting: the stored value's own reads still count.
            mutate(node->value);
            return node;
        }
        const Struct_t *s = temp_type.as<Struct_t>();
        std::vector<Stmt> split;
        for (const TypedVar &f : s->fields) {
            const auto named = field_names->find(f.name);
            if (named == field_names->end()) {
                continue;
            }
            split.push_back(Store::make(
                WriteLoc(named->second, f.type),
                Access::make(f.name, mutate(node->value), f.type)));
        }
        return Sequence::make(std::move(split));
    }
};

struct NarrowImpl : Mutator {
    int64_t counter = 0;

    // Copy propagation, scoped to one sequence: `let a = b` (a bare
    // variable) has every later use of `a` read `b` instead, and the alias
    // dies if nothing is left reading it. Inlining leaves such a let where
    // it bound a parameter, and one bare-variable use is all it takes to
    // make a whole aggregate look wanted whole. The walk stops rewriting at
    // anything that redefines or writes either name (Crossing), and the
    // alias is kept whenever a use survives past that.
    void propagate_copies(std::vector<Stmt> &stmts) {
        for (size_t i = 0; i < stmts.size(); i++) {
            const LetStmt *let = stmts[i].as<LetStmt>();
            if (let == nullptr || !let->loc.accesses.empty()) {
                continue;
            }
            const Var *source = let->value.as<Var>();
            if (source == nullptr) {
                continue;
            }
            RenameVar rename;
            rename.from = let->loc.base;
            rename.to = source->name;
            size_t j = i + 1;
            for (; j < stmts.size(); j++) {
                const Crossing c = crossing_of(stmts[j]);
                if (c.barrier || c.defines.count(rename.from) > 0 ||
                    c.defines.count(rename.to) > 0 ||
                    c.writes.count(rename.from) > 0 ||
                    c.writes.count(rename.to) > 0) {
                    break;
                }
                stmts[j] = rename.mutate(stmts[j]);
            }
            bool survives = false;
            for (size_t k = j; k < stmts.size() && !survives; k++) {
                for (const TypedVar &fv : gather_free_vars(stmts[k])) {
                    survives = survives || fv.name == rename.from;
                }
            }
            if (!survives) {
                stmts.erase(stmts.begin() + ptrdiff_t(i));
                i--;
            }
        }
    }

    // The merge-temp split, run over one Allocate of a struct temp at
    // stmts[i]; true when it rewrote.
    bool split_temp(std::vector<Stmt> &stmts, size_t i) {
        const Allocate *alloc = stmts[i].as<Allocate>();
        // The two private kinds only: a device or shared allocation is
        // plumbing someone else may know by name.
        if (alloc == nullptr || !alloc->loc.accesses.empty() ||
            (alloc->memory != Allocate::Stack &&
             alloc->memory != Allocate::Heap)) {
            return false;
        }
        const Struct_t *s = alloc->loc.base_type.as<Struct_t>();
        if (s == nullptr) {
            return false;
        }
        if (alloc->value.defined() && !classify(alloc->value).sinkable) {
            return false;
        }
        const std::string &name = alloc->loc.base;
        TempUses count;
        count.base = name;
        count.temp_type = alloc->loc.base_type;
        static const std::map<std::string, std::string> no_names;
        count.field_names = &no_names;
        for (size_t j = i + 1; j < stmts.size(); j++) {
            count.mutate(stmts[j]);
        }
        if (count.whole_use || count.bad_store || count.fields_seen.empty() ||
            count.fields_seen.size() == s->fields.size()) {
            return false;
        }
        std::map<std::string, std::string> names;
        std::vector<Stmt> split;
        for (const TypedVar &f : s->fields) {
            if (count.fields_seen.count(f.name) == 0) {
                continue;
            }
            const std::string fresh =
                name + "$" + f.name + std::to_string(counter++);
            names.emplace(f.name, fresh);
            split.push_back(Allocate::make(
                WriteLoc(fresh, f.type),
                alloc->value.defined()
                    ? Access::make(f.name, alloc->value, f.type)
                    : Expr(),
                alloc->memory, alloc->unaliased));
        }
        TempUses rewrite;
        rewrite.base = name;
        rewrite.temp_type = alloc->loc.base_type;
        rewrite.field_names = &names;
        std::vector<Stmt> rewritten;
        rewritten.reserve(stmts.size() - i - 1);
        for (size_t j = i + 1; j < stmts.size(); j++) {
            rewritten.push_back(rewrite.mutate(stmts[j]));
        }
        if (rewrite.whole_use || rewrite.bad_store) {
            return false;
        }
        for (const Stmt &st : rewritten) {
            for (const TypedVar &v : gather_free_vars(st)) {
                if (v.name == name) {
                    return false;
                }
            }
        }
        std::copy(rewritten.begin(), rewritten.end(),
                  stmts.begin() + ptrdiff_t(i) + 1);
        stmts.erase(stmts.begin() + ptrdiff_t(i));
        stmts.insert(stmts.begin() + ptrdiff_t(i), split.begin(), split.end());
        return true;
    }

    Stmt visit(const Sequence *node) override {
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());
        for (const Stmt &s : node->stmts) {
            stmts.push_back(mutate(s));
        }
        // Copies first: an inliner's parameter alias is one bare-variable
        // use, which would make every aggregate behind it look wanted
        // whole. Then the merge temps; then the loads.
        propagate_copies(stmts);
        for (size_t i = 0; i < stmts.size(); i++) {
            split_temp(stmts, i);
        }
        // The load pattern, to a fixpoint: narrowing a struct load makes
        // lets of its fields, and a field that is itself an aggregate --
        // the element's payload words -- narrows on the next round.
        for (int round = 0; round < 8; round++) {
            const size_t before = size_t(counter);
        for (size_t i = 0; i < stmts.size(); i++) {
            const LetStmt *let = stmts[i].as<LetStmt>();
            if (let == nullptr || !let->loc.accesses.empty()) {
                continue;
            }
            // A load: a dereference, or an access chain rooted at one or at
            // a reference -- an element read through a tree's storage is
            // `ref.field`, the same chain the code generator's PtrTo walks
            // as an address (CodeGen_LLVM's addressable_chain). The chain
            // is duplicated once per read field, so it must be pure.
            const bool memory_chain = [&] {
                Expr e = let->value;
                while (true) {
                    if (const Deref *d = e.as<Deref>()) {
                        return !d->mask.defined();
                    }
                    // A reference anywhere down the chain makes the whole
                    // chain a load: an element read is `layout.elems[i]`,
                    // and the reference is the array-typed FIELD, not the
                    // layout struct the walk bottoms out at.
                    if (e.type().defined() && e.type().is_reference()) {
                        return true;
                    }
                    if (const Access *a = e.as<Access>()) {
                        e = a->value;
                        continue;
                    }
                    if (const Extract *x = e.as<Extract>()) {
                        e = x->vec;
                        continue;
                    }
                    return false;
                }
            }();
            if (!memory_chain || !classify(let->value).sinkable) {
                continue;
            }
            // What the load can narrow to: a struct's named fields, or --
            // for the word vectors the layouts store aggregates as -- the
            // lanes read at constant indices. A short unpacked numeric
            // vector (two to four lanes, a point or a spectrum) is the
            // machine's own kind and stays one load, the boundary
            // SSA/Defer.cpp's whole_vector draws: splitting those costs a
            // gang its transposed reads and a warp its one coalesced load.
            struct Piece {
                std::string key;
                Type type;
                int lane; // -1 for a struct field
            };
            std::vector<Piece> pieces;
            const Struct_t *s = let->value.type().as<Struct_t>();
            const Vector_t *v = let->value.type().as<Vector_t>();
            if (s != nullptr) {
                for (const TypedVar &f : s->fields) {
                    pieces.push_back(Piece{f.name, f.type, -1});
                }
            } else if (v != nullptr &&
                       !(v->lanes >= 2 && v->lanes <= 4 && !v->packed &&
                         !v->etype.is_bool())) {
                for (uint32_t k = 0; k < v->lanes; k++) {
                    pieces.push_back(
                        Piece{std::to_string(k), v->etype, int(k)});
                }
            } else {
                continue;
            }
            const std::string &name = let->loc.base;
            // Which pieces the rest of the sequence reads, and whether it
            // ever wants the whole value.
            NarrowUses count;
            count.base = name;
            static const std::map<std::string, std::string> no_names;
            count.field_names = &no_names;
            for (size_t j = i + 1; j < stmts.size(); j++) {
                count.mutate(stmts[j]);
            }
            // Whole-value use, or dead (DCE's business): no split. A struct
            // every field of which is read is left whole too; a PACKED
            // vector splits even then -- packed is the layouts' storage
            // kind, never the machine's own load, and the split lane loads
            // then sink one by one into the branch arms that alone read
            // them (Opt/Sink runs next), which one whole load never could.
            if (count.whole_use || count.fields_seen.empty() ||
                (count.fields_seen.size() == pieces.size() &&
                 !(v != nullptr && v->packed))) {
                continue;
            }
            // One let per read piece, in declaration order, at the very
            // place the aggregate load stood -- `x$f = *&(*p).f`, the
            // address chain the code generator steps without loading the
            // aggregate (CodeGen_LLVM's PtrTo visitor).
            std::map<std::string, std::string> names;
            std::vector<Stmt> narrowed;
            for (const Piece &p : pieces) {
                if (count.fields_seen.count(p.key) == 0) {
                    continue;
                }
                const std::string fresh =
                    name + "$" + (p.lane < 0 ? p.key : "w" + p.key) +
                    std::to_string(counter++);
                names.emplace(p.key, fresh);
                narrowed.push_back(LetStmt::make(
                    WriteLoc(fresh, p.type),
                    Deref::make(PtrTo::make(
                        p.lane < 0
                            ? Access::make(p.key, let->value, p.type)
                            : Extract::make(let->value, p.lane)))));
            }
            // Additive: the narrowed lets go in right after the load, the
            // reads are rewritten onto them, and the load -- now dead,
            // along with any parameter binds that aliased it -- is DCE's
            // (the pass after next). Nothing is erased, so a use some
            // shape kept from the rewrite still reads the original and is
            // simply correct.
            NarrowUses rewrite;
            rewrite.base = name;
            rewrite.field_names = &names;
            for (size_t j = i + 1; j < stmts.size(); j++) {
                stmts[j] = rewrite.mutate(stmts[j]);
            }
            stmts.insert(stmts.begin() + ptrdiff_t(i) + 1,
                         narrowed.begin(), narrowed.end());
            i += narrowed.size();
        }
        if (size_t(counter) == before) {
            break;
        }
        }
        return Sequence::make(std::move(stmts));
    }
};

} // namespace

ir::FuncMap NarrowLoads::run(ir::FuncMap funcs,
                             const CompilerOptions &options) const {
    for (auto &[name, f] : funcs) {
        NarrowImpl impl;
        f->body = impl.mutate(f->body);
    }
    return funcs;
}

} // namespace opt
} // namespace bonsai
