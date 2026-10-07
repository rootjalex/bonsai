#include "Opt/NarrowLoads.h"

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

struct NarrowImpl : Mutator {
    int64_t counter = 0;

    Stmt visit(const Sequence *node) override {
        std::vector<Stmt> stmts;
        stmts.reserve(node->stmts.size());
        for (const Stmt &s : node->stmts) {
            stmts.push_back(mutate(s));
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
