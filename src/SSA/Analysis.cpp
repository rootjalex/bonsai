#include "SSA/Analysis.h"

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <iostream>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::optional;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::tuple;
using std::vector;

namespace {

struct ValueOrigin {
    enum class Kind {
        Known = 0,
        Merged = 1,
        Unknown = 2,
    } kind = Kind::Unknown;
    // only filled for Known:
    // constant = to_string, instruction = name, argument = name
    std::string value;

    static ValueOrigin MakeArgument(const ssa::Argument &a) {
        return ValueOrigin(Kind::Known, a.name);
    }
    static ValueOrigin MakeConstant(const ssa::Constant &c) {
        std::string s = std::visit(
            overloads{
                [](const std::string &v) -> std::string { return v; },
                [](bool v) -> std::string { return v ? "true" : "false"; },
                [](auto v) -> std::string { return std::to_string(v); }},
            c.data);
        return ValueOrigin(Kind::Known, std::move(s));
    }
    static ValueOrigin MakeInstruction(const ssa::Instruction &i) {
        return ValueOrigin(Kind::Known, i.name);
    }
    static ValueOrigin MakeUnknown() { return ValueOrigin(Kind::Unknown, ""); }

    static ValueOrigin MakeMerge(const ValueOrigin &a, const ValueOrigin &b) {
        if (a.kind == Kind::Unknown) {
            return b;
        } else if (b.kind == Kind::Unknown) {
            return a;
        } else if (a != b) {
            // TODO: make unique merge here to rm constraint below.
            return ValueOrigin(Kind::Merged, "");
        }
        // Equal.
        return a;
    }
    // For call results and parfor loop indexes
    static ValueOrigin MakeMerge() {
        // TODO: make unique merge here to rm constraint below.
        return ValueOrigin(Kind::Merged, "");
    }

    bool operator==(const ValueOrigin &o) const {
        // TODO: relax Merged != restriction
        return (kind != Kind::Merged) && (kind == o.kind && value == o.value);
    }
    bool operator!=(const ValueOrigin &o) const { return !(*this == o); }

  private:
    ValueOrigin(Kind kind, std::string value)
        : kind(kind), value(std::move(value)) {}
};

// block -> [origin for arg in block.args]
using OriginMap = map<string, vector<ValueOrigin>>;

OriginMap make_origin_map(const ssa::Function &func) {
    OriginMap omap;

    const auto bmap = make_block_map(func);

    // Insert empties
    bool entry_block = true;
    for (const auto &block : func.blocks) {
        omap[block->name] =
            vector<ValueOrigin>(block->args.size(), ValueOrigin::MakeUnknown());
        if (entry_block) {
            for (size_t i = 0; i < block->args.size(); i++) {
                // These are arguments in the entry.
                omap[block->name][i] =
                    ValueOrigin::MakeArgument(block->args[i]);
            }
        }
        entry_block = false;
    }

    // Now iterate over all terminators and insert.
    // TODO: how to track uniqueness through cycles?
    // Should this be DFS, linear order, backwards order?

    for (const auto &block : func.blocks) {
        auto handle_jump = [&](const Terminator::Jump &j,
                               const bool non_drop_call) {
            // Don't care about individual jumps (except to entry!)
            internal_assert(bmap.contains(j.name)) << j.name;
            if (j.name != func.blocks[0]->name &&
                bmap.at(j.name)->preds.size() < 2) {
                std::cout << "skipping " << j.name << " in terminator of "
                          << block->name << "\n";
                return;
            }

            const size_t offset = non_drop_call ? 1 : 0;

            internal_assert(omap.contains(j.name) &&
                            omap[j.name].size() == offset + j.args.size())
                << "Bad argument count in jump to " << j.name
                << " in terminator of " << block->name;

            auto &om = omap[j.name];

            if (non_drop_call) {
                // *always* a phi node.
                om[0] = ValueOrigin::MakeMerge();
            }

            for (size_t i = 0; i < j.args.size(); i++) {
                auto v =
                    std::visit(overloads{
                                   [&](const std::shared_ptr<Instruction> &i) {
                                       return ValueOrigin::MakeInstruction(*i);
                                   },
                                   [&](const Constant &c) {
                                       return ValueOrigin::MakeConstant(c);
                                   },
                                   [&](const Argument &a) {
                                       return ValueOrigin::MakeArgument(a);
                                   },
                               },
                               j.args[i]->data);
                om[offset + i] = ValueOrigin::MakeMerge(om[offset + i], v);
            }
        };

        std::visit(
            overloads{
                [&](const std::monostate &m) {
                    internal_error
                        << "Monostate terminator found in make_origin_map";
                },
                [&](const Terminator::Jump &j) { handle_jump(j, false); },
                [&](const Terminator::Dispatch &d) {
                    for (const auto &t : d.targets) {
                        handle_jump(t, false);
                    }
                },
                [&](const Terminator::Return &r) {},
                [&](const Terminator::ParFor &p) {
                    handle_jump(p.body, true);
                    handle_jump(p.cont, false);
                },
                [&](const Terminator::Yield &y) {},
                [&](const Terminator::Call &call) {
                    handle_jump(call.cont, !call.drop);
                },
            },
            block->terminator.data);
    }

    // TODO: iterate to a fixed point or no??

    return omap;
}

} // namespace

BlockMap make_block_map(const shared_ptr<Function> &func) {
    BlockMap bmap;
    for (const auto &block : func->blocks) {
        bmap[block->name] = block;
    }
    return bmap;
}

BlockMap make_block_map(const Function &func) {
    BlockMap bmap;
    for (const auto &block : func.blocks) {
        bmap[block->name] = block;
    }
    return bmap;
}

ArgMutabilityMap get_mutability_map(const ssa::Function &func) {
    OriginMap omap = make_origin_map(func);
    /*
    for (const auto &[bname, vec] : omap) {

        std::cout << bname << " = {";
        bool first = true;
        for (const auto &v : vec) {
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << static_cast<int>(v.kind);
        }
        std::cout << "}\n";
    }
    */

    ArgMutabilityMap result;

    for (const auto &block : func.blocks) {
        const size_t num_args = block->args.size();
        if (num_args == 0) {
            result[block->name] = {};
            continue;
        }

        internal_assert(omap.contains(block->name)) << block->name;
        const auto om = omap.at(block->name);
        internal_assert(om.size() == num_args)
            << block->name << " has " << num_args
            << " but origin map stores: " << om.size() << " entries";

        std::vector<bool> is_mutable(num_args, false);

        for (size_t i = 0; i < num_args; i++) {
            is_mutable[i] = om[i].kind == ValueOrigin::Kind::Merged;
        }
        result[block->name] = std::move(is_mutable);
    }
    return result;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
