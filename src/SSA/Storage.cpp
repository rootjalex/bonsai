#include "SSA/Storage.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
#include "SSA/Analysis.h"

#include "Error.h"
#include "Utils.h"

#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::shared_ptr;
using std::string;
using std::vector;

shared_ptr<Value> make_alloca(Function &func, const shared_ptr<Block> &block,
                              const Type &type, const string &name) {
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

Expr as_expr(const shared_ptr<Value> &v) {
    return std::visit(
        overloads{
            [&](const Constant &c) -> Expr {
                return std::visit(
                    overloads{
                        [](const bool &b) -> Expr { return BoolImm::make(b); },
                        [&](const int64_t &i) -> Expr {
                            return IntImm::make(c.type, i);
                        },
                        [&](const uint64_t &u) -> Expr {
                            return UIntImm::make(c.type, u);
                        },
                        [&](const double &d) -> Expr {
                            return FloatImm::make(c.type, d);
                        },
                        [&](const std::string &s) -> Expr {
                            return StringImm::make(s);
                        },
                        [&](const Undefined &) -> Expr { return Undef::make(c.type); },
                    },
                    c.data);
            },
            [&](const Argument &a) -> Expr { return Var::make(a.type, a.name); },
            [&](const shared_ptr<Instruction> &i) -> Expr {
                return Var::make(i->type, i->name);
            },
        },
        v->data);
}

namespace {

struct RenameInTypes : ir::Mutator {
    const std::map<string, ir::Expr> &renames;
    explicit RenameInTypes(const std::map<string, ir::Expr> &renames)
        : renames(renames) {}

    ir::Expr visit(const ir::Var *node) override {
        const auto it = renames.find(node->name);
        return it == renames.end() ? ir::Expr(node) : it->second;
    }
    Type visit(const ir::Array_t *node) override {
        Type etype = mutate(node->etype);
        ir::Expr size = node->size.defined() ? mutate(node->size) : node->size;
        if (etype.same_as(node->etype) && size.same_as(node->size)) {
            return node;
        }
        return ir::Array_t::make(std::move(etype), std::move(size));
    }
    Type visit(const ir::DynArray_t *node) override {
        Type etype = mutate(node->etype);
        ir::Expr capacity =
            node->capacity.defined() ? mutate(node->capacity) : node->capacity;
        if (etype.same_as(node->etype) && capacity.same_as(node->capacity)) {
            return node;
        }
        return ir::DynArray_t::make(std::move(etype), std::move(capacity));
    }
};

// The sizes an array type carries, outermost first.
void sizes_of(const Type &type, vector<Expr> &out) {
    if (const auto *array = type.as<Array_t>()) {
        if (array->size.defined()) {
            out.push_back(array->size);
        }
        sizes_of(array->etype, out);
    } else if (const auto *dyn = type.as<DynArray_t>()) {
        if (dyn->capacity.defined()) {
            out.push_back(dyn->capacity);
        }
        sizes_of(dyn->etype, out);
    } else if (const auto *ptr = type.as<Ptr_t>()) {
        sizes_of(ptr->etype, out);
    }
}

} // namespace

Type rename_in_type(const Type &type, const std::map<string, Expr> &renames) {
    RenameInTypes renamer(renames);
    return renamer.mutate(type);
}

void rename_in_types(Function &func, const std::map<string, Expr> &renames) {
    RenameInTypes renamer(renames);
    for (const auto &block : func.blocks) {
        for (Argument &arg : block->args) {
            arg.type = renamer.mutate(arg.type);
        }
        for (const auto &instr : block->instrs) {
            instr->type = renamer.mutate(instr->type);
            if (instr->queried_type.defined()) {
                instr->queried_type = renamer.mutate(instr->queried_type);
            }
        }
    }
    for_each_value(func, [&](shared_ptr<Value> &v) {
        if (auto *a = std::get_if<Argument>(&v->data)) {
            a->type = renamer.mutate(a->type);
        }
    });
}

vector<string> names_in_type(const Type &type) {
    vector<Expr> sizes;
    sizes_of(type, sizes);
    vector<string> names;
    for (const Expr &size : sizes) {
        for (const TypedVar &var : gather_free_vars(size)) {
            names.push_back(var.name);
        }
    }
    return names;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
