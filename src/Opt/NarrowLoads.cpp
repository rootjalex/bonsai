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

// Rewrites every `x.f` to its narrowed name and counts anything else done
// with `x`: a bare use that is not a field read means the whole value is
// wanted and the let must stay as it is.
struct NarrowUses : Mutator {
    std::string base;
    const std::map<std::string, std::string> *field_names = nullptr;
    std::set<std::string> fields_seen;
    bool whole_use = false;

    Expr visit(const Access *node) override {
        if (const Var *var = node->value.as<Var>(); var != nullptr &&
                                                    var->name == base) {
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

    Expr visit(const Var *node) override {
        if (node->name == base) {
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
        // Merge temps first: splitting one exposes nothing to the load
        // pattern below, but a narrowed load's struct could in principle
        // feed a temp, so the order narrows more.
        for (size_t i = 0; i < stmts.size(); i++) {
            split_temp(stmts, i);
        }
        for (size_t i = 0; i < stmts.size(); i++) {
            const LetStmt *let = stmts[i].as<LetStmt>();
            if (let == nullptr || !let->loc.accesses.empty()) {
                continue;
            }
            const Deref *load = let->value.as<Deref>();
            if (load == nullptr || load->mask.defined()) {
                continue;
            }
            const Struct_t *s = let->value.type().as<Struct_t>();
            if (s == nullptr) {
                continue;
            }
            const std::string &name = let->loc.base;
            // Which fields the rest of the sequence reads, and whether it
            // ever wants the whole value.
            NarrowUses count;
            count.base = name;
            static const std::map<std::string, std::string> no_names;
            count.field_names = &no_names;
            for (size_t j = i + 1; j < stmts.size(); j++) {
                count.mutate(stmts[j]);
            }
            if (count.whole_use || count.fields_seen.empty() ||
                count.fields_seen.size() == s->fields.size()) {
                // Whole-value use; dead (DCE's business); or every field
                // read, nothing to narrow.
                continue;
            }
            // One let per read field, in declaration order, at the very
            // place the aggregate load stood -- `x$f = *&(*p).f`, the
            // address chain the code generator steps without loading the
            // aggregate (CodeGen_LLVM's PtrTo visitor).
            std::map<std::string, std::string> names;
            std::vector<Stmt> narrowed;
            for (const TypedVar &f : s->fields) {
                if (count.fields_seen.count(f.name) == 0) {
                    continue;
                }
                const std::string fresh =
                    name + "$" + f.name + std::to_string(counter++);
                names.emplace(f.name, fresh);
                narrowed.push_back(LetStmt::make(
                    WriteLoc(fresh, f.type),
                    Deref::make(PtrTo::make(
                        Access::make(f.name, let->value, f.type)))));
            }
            NarrowUses rewrite;
            rewrite.base = name;
            rewrite.field_names = &names;
            std::vector<Stmt> rewritten;
            rewritten.reserve(stmts.size() - i - 1);
            for (size_t j = i + 1; j < stmts.size(); j++) {
                rewritten.push_back(rewrite.mutate(stmts[j]));
            }
            // The verification that nothing was missed: a use the pattern
            // walk did not reach (a shape the mutator does not step into)
            // leaves the name free in what follows. The narrowing is then
            // dropped whole rather than shipped wrong.
            bool still_free = false;
            for (const Stmt &s : rewritten) {
                for (const TypedVar &v : gather_free_vars(s)) {
                    still_free = still_free || v.name == name;
                }
            }
            if (still_free) {
                continue;
            }
            std::copy(rewritten.begin(), rewritten.end(),
                      stmts.begin() + ptrdiff_t(i) + 1);
            stmts.erase(stmts.begin() + ptrdiff_t(i));
            stmts.insert(stmts.begin() + ptrdiff_t(i),
                         narrowed.begin(), narrowed.end());
            i += narrowed.size() - 1;
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
