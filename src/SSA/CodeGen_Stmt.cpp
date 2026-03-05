#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

Expr codegen_value(const std::shared_ptr<Value> &v) {
    return std::visit(
        overloads{
            [](const std::shared_ptr<Instruction> &i) {
                return Var::make(i->type, i->name);
            },
            [](const Constant &c) {
                return std::visit(
                    overloads{
                        [](const bool &b) { return BoolImm::make(b); },
                        [&](const int64_t &i) {
                            return IntImm::make(c.type, i);
                        },
                        [&](const uint64_t &u) {
                            return UIntImm::make(c.type, u);
                        },
                        [&](const double &d) {
                            return FloatImm::make(c.type, d);
                        },
                        [&](const std::string &s) {
                            return StringImm::make(s);
                        },
                    },
                    c.data);
            },
            [](const Argument &a) { return Var::make(a.type, a.name); },
        },
        v->data);
}

bool is_side_effecty(Instruction::Op op) {
    switch (op) {
    case Instruction::Op::AccAdd:
    case Instruction::Op::AccMul:
    case Instruction::Op::AccSub:
    case Instruction::Op::AccArgmin:
    case Instruction::Op::AccArgmax:
    case Instruction::Op::AccMin:
    case Instruction::Op::AccMax:
    case Instruction::Op::Append:
    case Instruction::Op::Store:
    case Instruction::Op::Alloc:
    case Instruction::Op::Alloca:
        return true;
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::Bc:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Eq:
    case Instruction::Op::ExtractIdx:
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
    case Instruction::Op::Mod:
    case Instruction::Op::Mul:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Set:
    case Instruction::Op::Sub:
        return false;
    }
}

WriteLoc codegen_gep(const std::shared_ptr<Value> &v) {
    // Check if this is an instruction
    if (auto instr = std::get_if<std::shared_ptr<Instruction>>(&v->data)) {
        const auto &i = *instr;

        if (i->op == Instruction::Op::GEP) {
            internal_assert(i->operands.size() == 2)
                << "Malformed GEP: expected 2 operands";

            // Recursively unwrap base
            WriteLoc loc = codegen_gep(i->operands[0]);

            // Add index access
            Expr idx = codegen_value(i->operands[1]);
            loc.add_index_access(idx);

            return loc;
        }

        // Base case: non-GEP instruction -> variable
        internal_assert(!i->name.empty())
            << "Cannot form WriteLoc from unnamed instruction";

        return WriteLoc(i->name, i->type);
    }

    // Argument base case
    if (auto arg = std::get_if<Argument>(&v->data)) {
        return WriteLoc(arg->name, arg->type);
    }

    v->dump(std::cerr);
    internal_error << "GEP base must be instruction or argument";
}

std::vector<Expr>
codegen_values(const std::vector<std::shared_ptr<Value>> &operands) {
    std::vector<Expr> args;
    args.reserve(operands.size());

    for (const auto &operand : operands) {
        args.push_back(codegen_value(operand));
    }
    return args;
}

Accumulate::OpType codegen_acc_op(const Instruction::Op &op) {
    switch (op) {
    case Instruction::Op::AccAdd:
        return Accumulate::OpType::Add;
    case Instruction::Op::AccMul:
        return Accumulate::OpType::Mul;
    case Instruction::Op::AccSub:
        return Accumulate::OpType::Sub;
    case Instruction::Op::AccArgmin:
        return Accumulate::OpType::Argmin;
    case Instruction::Op::AccArgmax:
        return Accumulate::OpType::Argmax;
    case Instruction::Op::AccMin:
        return Accumulate::OpType::Min;
    case Instruction::Op::AccMax:
        return Accumulate::OpType::Max;
    default: {
        internal_error << static_cast<int>(op);
    }
    }
}

uint64_t get_const_u64(const Expr &e) {
    internal_assert(e.is<UIntImm>()) << e;
    return e.as<UIntImm>()->value;
}

Stmt codegen_instruction(const Instruction &instr) {
    if (is_side_effecty(instr.op)) {
        switch (instr.op) {
        case Instruction::Op::AccAdd:
        case Instruction::Op::AccMul:
        case Instruction::Op::AccSub:
        case Instruction::Op::AccArgmin:
        case Instruction::Op::AccArgmax:
        case Instruction::Op::AccMin:
        case Instruction::Op::AccMax: {
            internal_assert(instr.operands.size() == 2)
                << instr.operands.size();
            WriteLoc loc = codegen_gep(instr.operands[0]);
            auto op = codegen_acc_op(instr.op);
            auto val = codegen_value(instr.operands[1]);
            return Accumulate::make(std::move(loc), op, std::move(val));
        }
        case Instruction::Op::Append:
            internal_error << "TODO: Append codegen!\n";
        case Instruction::Op::Store: {
            internal_assert(instr.operands.size() == 2)
                << instr.operands.size();
            WriteLoc loc = codegen_gep(instr.operands[0]);
            Expr val = codegen_value(instr.operands[1]);
            return Store::make(std::move(loc), std::move(val));
        }
        case Instruction::Op::Alloc:
            internal_error << "TODO: Alloc codegen!\n";
        case Instruction::Op::Alloca:
            internal_error << "TODO: Alloca codegen!\n";
        default:
            instr.dump(std::cerr);
            internal_error << "TODO: side_effecty codegen for ^";
        }
    }

    // Codegen LetStmt

    std::vector<Expr> args = codegen_values(instr.operands);

    Expr value;

    switch (instr.op) {
    case Instruction::Op::Abs: {
        value = Intrinsic::make(Intrinsic::OpType::abs, std::move(args));
        break;
    }
    case Instruction::Op::Add: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Add, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Bc: {
        internal_assert(args.size() == 2) << args.size();
        const uint64_t lanes = get_const_u64(args[1]);
        value = Broadcast::make(lanes, std::move(args[0]));
        break;
    }
    case Instruction::Op::Cast: {
        internal_assert(args.size() == 1) << args.size();
        value = Cast::make(instr.type, std::move(args[0]));
        break;
    }
    case Instruction::Op::Div: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Div, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Eps: {
        internal_assert(args.size() == 0) << args.size();
        value = Extrema::make(instr.type, Extrema::eps);
        break;
    }
    case Instruction::Op::Eq: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Eq, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::ExtractIdx: {
        internal_assert(args.size() == 2) << args.size();
        value = Extract::make(std::move(args[0]), std::move(args[1]));
        break;
    }
    case Instruction::Op::GEP: {
        // TODO: is this right?
        internal_assert(args.size() == 2) << args.size();
        Expr temp = Extract::make(std::move(args[0]), std::move(args[1]));
        value = PtrTo::make(temp);
        break;
    }
    case Instruction::Op::LAnd: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::LAnd, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::LOr: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::LOr, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Leq: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Le, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    // case Instruction::Op::Load:
    case Instruction::Op::LoadField: {
        internal_assert(args.size() == 2) << args.size();
        const Struct_t *struct_t = args[0].type().as<Struct_t>();
        internal_assert(struct_t) << args[0].type();
        const uint64_t idx = get_const_u64(args[1]);
        internal_assert(idx < struct_t->fields.size())
            << idx << " versus " << struct_t->fields.size() << " in "
            << args[0].type();
        value = Access::make(struct_t->fields[idx].name, std::move(args[0]));
        break;
    }
    case Instruction::Op::Lt: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Lt, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::MakeStruct: {
        value = Build::make(instr.type, std::move(args));
        break;
    }
    case Instruction::Op::Max: {
        internal_assert(args.size() == 2) << args.size();
        value = Intrinsic::make(Intrinsic::OpType::max, std::move(args));
        break;
    }
    case Instruction::Op::Min: {
        internal_assert(args.size() == 2) << args.size();
        value = Intrinsic::make(Intrinsic::OpType::min, std::move(args));
        break;
    }
    case Instruction::Op::Mod: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Mod, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Mul: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Mul, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Reinterpret: {
        internal_assert(args.size() == 1) << args.size();
        value =
            Cast::make(instr.type, std::move(args[0]), Cast::Mode::Reinterpret);
        break;
    }
    case Instruction::Op::Set: {
        internal_assert(args.size() == 1) << args.size();
        internal_assert(!instr.name.starts_with("@")) << instr.name;
        // TODO: this might be a store?
        return Allocate::make(WriteLoc(instr.name, instr.type), args[0]);
    }
    case Instruction::Op::Sub: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Sub, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    default: {
        instr.dump(std::cerr);
        internal_error << "TODO";
    }
    }

    // Eventually, need to sanitize names. Maybe not here though.
    return LetStmt::make(WriteLoc(instr.name, instr.type), std::move(value));
}

// Recursively inline a pure SSA value as an expression,
// following def-use chains instead of emitting let-bindings.
Expr inline_expr(const std::shared_ptr<Value> &v) {
    auto *ip = std::get_if<std::shared_ptr<Instruction>>(&v->data);
    if (!ip)
        return codegen_value(v); // Constant or Argument — use as-is

    const Instruction &instr = **ip;
    if (is_side_effecty(instr.op))
        return codegen_value(v); // don't inline side effects

    // Recursively inline operands
    std::vector<Expr> args;
    for (auto &op : instr.operands)
        args.push_back(inline_expr(op));

    switch (instr.op) {
    case Instruction::Op::Lt:
        return BinOp::make(BinOp::OpType::Lt, args[0], args[1]);
    case Instruction::Op::Leq:
        return BinOp::make(BinOp::OpType::Le, args[0], args[1]);
    case Instruction::Op::Eq:
        return BinOp::make(BinOp::OpType::Eq, args[0], args[1]);
    case Instruction::Op::LAnd:
        return BinOp::make(BinOp::OpType::LAnd, args[0], args[1]);
    case Instruction::Op::LOr:
        return BinOp::make(BinOp::OpType::LOr, args[0], args[1]);
    case Instruction::Op::Add:
        return BinOp::make(BinOp::OpType::Add, args[0], args[1]);
    case Instruction::Op::Sub:
        return BinOp::make(BinOp::OpType::Sub, args[0], args[1]);
    case Instruction::Op::Mul:
        return BinOp::make(BinOp::OpType::Mul, args[0], args[1]);
    case Instruction::Op::Div:
        return BinOp::make(BinOp::OpType::Div, args[0], args[1]);
    case Instruction::Op::Mod:
        return BinOp::make(BinOp::OpType::Mod, args[0], args[1]);
    case Instruction::Op::Cast:
        return Cast::make(instr.type, args[0]);
    case Instruction::Op::Set:
        return args[0];
    default:
        // Not inlineable — fall back to variable reference
        return codegen_value(v);
    }
}

std::set<std::string> reachable(const std::string &name,
                                const BlockMap &block_map) {
    auto get_successors =
        [&](const std::string &name) -> std::vector<std::string> {
        auto &block = block_map.at(name);
        std::vector<std::string> succs;
        std::visit(
            overloads{
                [&](const Terminator::Jump &j) { succs.push_back(j.name); },
                [&](const Terminator::Dispatch &d) {
                    for (auto &t : d.targets)
                        succs.push_back(t.name);
                },
                [&](const Terminator::ParFor &p) {
                    // This is always enclosed, not considered "reachable".
                    // succs.push_back(p.body.name);
                    succs.push_back(p.cont.name);
                },
                [&](const Terminator::Return &) {},
                [&](const Terminator::Yield &) {},
                [&](const Terminator::Call &c) {
                    succs.push_back(c.cont.name);
                },
                [&](const std::monostate &) {},
            },
            block->terminator.data);
        return succs;
    };

    // BFS from each branch, find first common successor
    auto reachable = [&](const std::string &start) {
        std::set<std::string> seen;
        std::queue<std::string> q;
        q.push(start);
        while (!q.empty()) {
            auto name = q.front();
            q.pop();
            if (!seen.insert(name).second)
                continue;
            auto succs = get_successors(name);
            for (auto &s : succs)
                q.push(std::move(s));
        }
        return seen;
    };

    return reachable(name);
}

// Helper: find the merge/join block for a dispatch
// Returns the name of the first block reachable from BOTH branches
// that hasn't been visited yet (i.e., the post-dominator)
std::string
find_merge_block(const std::string &true_branch,
                 const std::string &false_branch, const BlockMap &block_map,
                 const std::set<std::string> &already_visited) { // <-- add this

    auto get_successors =
        [&](const std::string &name) -> std::vector<std::string> {
        auto &block = block_map.at(name);
        std::vector<std::string> succs;
        std::visit(
            overloads{
                [&](const Terminator::Jump &j) { succs.push_back(j.name); },
                [&](const Terminator::Dispatch &d) {
                    for (auto &t : d.targets)
                        succs.push_back(t.name);
                },
                [&](const Terminator::ParFor &p) {
                    succs.push_back(p.body.name);
                    succs.push_back(p.cont.name);
                },
                [&](const Terminator::Return &) {},
                [&](const Terminator::Yield &) {},
                [&](const Terminator::Call &c) {
                    succs.push_back(c.cont.name);
                },
                [&](const std::monostate &) {},
            },
            block->terminator.data);
        return succs;
    };

    auto reachable = [&](const std::string &start) {
        std::unordered_set<std::string> seen;
        std::queue<std::string> q;
        q.push(start);
        while (!q.empty()) {
            auto name = q.front();
            q.pop();
            if (!seen.insert(name).second)
                continue;
            if (already_visited.count(name) && name != start)
                continue; // don't follow back-edges
            auto succs = get_successors(name);
            for (auto &s : succs)
                q.push(std::move(s));
        }
        return seen;
    };

    auto from_true = reachable(true_branch);
    auto from_false = reachable(false_branch);

    std::queue<std::string> q;
    std::unordered_set<std::string> local_visited;
    q.push(true_branch);
    while (!q.empty()) {
        auto name = q.front();
        q.pop();
        if (!local_visited.insert(name).second)
            continue;
        if (already_visited.count(name) && name != true_branch)
            continue; // don't follow back-edges
        if (from_false.count(name) && name != true_branch)
            return name;
        auto succs = get_successors(name);
        for (auto &s : succs)
            q.push(std::move(s));
    }
    return "";
}

using DominatorMap = std::map<std::string, std::set<std::string>>;

DominatorMap compute_dominators(const ssa::Function &func,
                                const BlockMap &block_map) {
    DominatorMap dom;

    // Initialize
    for (auto &b : func.blocks) {
        for (auto &bb : func.blocks) {
            dom[b->name].insert(bb->name);
        }
    }

    const std::string entry = func.blocks[0]->name;
    dom[entry] = {entry};

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto &b : func.blocks) {
            if (b->name == entry)
                continue;

            std::set<std::string> new_dom;
            bool first = true;

            for (auto &wp : b->preds) {
                auto p = wp.lock();
                internal_assert(p);
                if (first) {
                    new_dom = dom[p->name];
                    first = false;
                } else {
                    std::set<std::string> tmp;
                    for (auto &x : new_dom)
                        if (dom[p->name].count(x)) {
                            tmp.insert(x);
                        }
                    new_dom = std::move(tmp);
                }
            }

            new_dom.insert(b->name);

            if (new_dom != dom[b->name]) {
                dom[b->name] = std::move(new_dom);
                changed = true;
            }
        }
    }

    return dom;
}

struct BlockInfo {
    enum class Role {
        Normal,       // Sequential block, or if/else branch block
        WhileHeader,  // Has a Dispatch; backedge comes back to it from a latch
        DoWhileLatch, // Has a Dispatch; one arm is a backedge to a dominator
        InfLoopLatch, // Has an unconditional Jump that is a backedge
                      // (do-while(true))
    };

    Role role = Role::Normal;
    std::string loop_header; // for DoWhileLatch: which block is the header
    std::string loop_exit;   // for WhileHeader: which arm exits the loop
    std::string loop_body;   // for WhileHeader: which arm enters the body
};

using BlockInfoMap = std::map<std::string, BlockInfo>;

BlockInfoMap classify_blocks(const ssa::Function &func,
                             const BlockMap &block_map,
                             const DominatorMap &dom) {

    auto is_backedge = [&](const std::string &from, const std::string &to) {
        return dom.at(from).count(to) > 0;
    };

    BlockInfoMap info;
    for (auto &b : func.blocks)
        info[b->name] = BlockInfo{};

    // Pass 1: classify latches — blocks with outgoing backedges
    for (auto &b : func.blocks) {
        const std::string &name = b->name;
        std::visit(overloads{[&](const Terminator::Jump &j) {
                                 if (is_backedge(name, j.name)) {
                                     info[name].role =
                                         BlockInfo::Role::InfLoopLatch;
                                     info[name].loop_header = j.name;
                                 }
                             },
                             [&](const Terminator::Dispatch &d) {
                                 const std::string &t0 = d.targets[0].name;
                                 const std::string &t1 = d.targets[1].name;
                                 const bool t0_back = is_backedge(name, t0);
                                 const bool t1_back = is_backedge(name, t1);
                                 if (t0_back || t1_back) {
                                     internal_assert(!(t0_back && t1_back))
                                         << "Both arms backedges: " << name;
                                     info[name].role =
                                         BlockInfo::Role::DoWhileLatch;
                                     info[name].loop_header = t0_back ? t0 : t1;
                                     info[name].loop_exit = t0_back ? t1 : t0;
                                 }
                             },
                             [&](const auto &) {}},
                   b->terminator.data);
    }

    // Pass 2: classify while headers — Dispatch blocks that have a latch
    // pointing back to them. Use reachability to the latch to find body arm.
    for (auto &b : func.blocks) {
        const std::string &name = b->name;
        if (info.at(name).role != BlockInfo::Role::Normal)
            continue;

        auto *d = std::get_if<Terminator::Dispatch>(&b->terminator.data);
        if (!d)
            continue;

        // Find a latch whose header is this block
        std::string latch;
        for (auto &[bname, bi] : info) {
            if (bi.loop_header == name &&
                (bi.role == BlockInfo::Role::InfLoopLatch ||
                 bi.role == BlockInfo::Role::DoWhileLatch)) {
                latch = bname;
                break;
            }
        }
        if (latch.empty())
            continue;

        const std::string &t0 = d->targets[0].name;
        const std::string &t1 = d->targets[1].name;

        // Body arm is whichever can reach the latch
        const bool t0_is_body = reachable(t0, block_map).count(latch) > 0;
        const bool t1_is_body = reachable(t1, block_map).count(latch) > 0;

        internal_assert(t0_is_body ^ t1_is_body)
            << "Can't determine while body for header: " << name;

        info[name].role = BlockInfo::Role::WhileHeader;
        info[name].loop_body = t0_is_body ? t0 : t1;
        info[name].loop_exit = t0_is_body ? t1 : t0;
    }

    for (const auto &[name, i] : info) {
        std::cout << "name: " << name << " has type: ";
        switch (i.role) {
        case BlockInfo::Role::DoWhileLatch: {
            std::cout << "do-while latch";
            break;
        }
        case BlockInfo::Role::InfLoopLatch: {
            std::cout << "inf loop latch";
            break;
        }
        case BlockInfo::Role::Normal: {
            std::cout << "normal";
            break;
        }
        case BlockInfo::Role::WhileHeader: {
            std::cout << "while header";
            break;
        }
        }
        std::cout << std::endl;
    }

    return info;
}

Stmt structurize(const std::string &start, const std::string &exit,
                 const BlockMap &block_map, const DominatorMap &dom,
                 const BlockInfoMap &info, const ArgMutabilityMap &mut_map) {

    std::vector<Stmt> stmts;
    std::string name = start;

    auto emit_jump_args = [&](const std::string &target,
                              const std::vector<std::shared_ptr<Value>> &vals) {
        auto &target_block = block_map.at(target);
        if (target_block->args.empty())
            return;
        auto &muts = mut_map.at(target);
        internal_assert(vals.size() == target_block->args.size());

        for (size_t i = 0; i < vals.size(); i++) {
            if (muts[i]) {
                stmts.push_back(
                    Store::make(WriteLoc(target_block->args[i].name,
                                         target_block->args[i].type),
                                codegen_value(vals[i])));
            }
            /*else {
                // Immutable: bind at the jump site as a let
                stmts.push_back(
                    LetStmt::make(WriteLoc(target_block->args[i].name,
                                           target_block->args[i].type),
                                  codegen_value(vals[i])));
            }
            */
        }
    };

    while (name != exit) {
        auto block = block_map.at(name);
        const BlockInfo &bi = info.at(name);

        // Emit this block's instructions
        for (auto &instr : block->instrs) {
            stmts.push_back(codegen_instruction(*instr));
        }

        std::visit(
            overloads{

                [&](const std::monostate &) {
                    internal_error << "No terminator: " << name;
                },

                [&](const Terminator::Jump &j) {
                    emit_jump_args(j.name, j.args);
                    if (j.name == exit) {
                        // end of region
                        name = exit;
                    } else if (bi.role == BlockInfo::Role::InfLoopLatch) {
                        // Wrap everything accumulated so far as the loop body.
                        internal_assert(j.name == exit)
                            << "InfLoopLatch target " << j.name << " != exit "
                            << exit;
                        Stmt body = Sequence::make(std::move(stmts));
                        stmts = {DoWhile::make(std::move(body),
                                               BoolImm::make(true))};
                        name = exit; // terminate the while loop
                    } else {
                        // Normal sequential jump — just advance
                        name = j.name;
                    }
                },

                [&](const Terminator::Dispatch &d) {
                    Expr cond = codegen_value(d.cond);
                    const std::string &t0 = d.targets[0].name;
                    const std::string &t1 = d.targets[1].name;

                    if (bi.role == BlockInfo::Role::WhileHeader) {
                        // Recurse only for the body (bounded sub-region)

                        // Allocate mutable args before the while loop
                        /*
                        auto &muts = mut_map.at(name);
                        for (size_t i = 0; i < block->args.size(); i++) {
                            if (muts[i]) {
                                stmts.push_back(Allocate::make(
                                    WriteLoc(block->args[i].name,
                                             block->args[i].type),
                                    Var::make(block->args[i].type,
                                              block->args[i].name),
                                    Allocate::Stack));
                            }
                        }
                        */

                        Stmt body = structurize(bi.loop_body, name, block_map,
                                                dom, info, mut_map);

                        Expr loop_cond = inline_expr(d.cond);
                        if (bi.loop_body != t1) {
                            loop_cond = UnOp::make(UnOp::OpType::Not,
                                                   std::move(loop_cond));
                        }

                        stmts.push_back(
                            While::make(std::move(loop_cond), std::move(body)));
                        name = bi.loop_exit; // advance past the loop

                    } else if (bi.role == BlockInfo::Role::DoWhileLatch) {
                        // Wrap everything accumulated so far as the loop body
                        Expr loop_cond = (bi.loop_header == t1)
                                             ? cond
                                             : UnOp::make(UnOp::OpType::Not,
                                                          std::move(cond));

                        Stmt body = Sequence::make(std::move(stmts));
                        stmts = {DoWhile::make(std::move(body),
                                               std::move(loop_cond))};
                        name = bi.loop_exit; // advance past the loop

                    } else {
                        // If/else: recurse into both arms (bounded by merge)
                        std::string merge =
                            find_merge_block(t1, t0, block_map, {});

                        // Allocate mutable args of the merge block BEFORE the
                        // if/else
                        if (!merge.empty()) {
                            /*
                            auto &merge_block = block_map.at(merge);
                            auto &muts = mut_map.at(merge);
                            for (size_t i = 0; i < merge_block->args.size();
                                 i++) {
                                if (muts[i]) {
                                    stmts.push_back(Allocate::make(
                                        WriteLoc(merge_block->args[i].name,
                                                 merge_block->args[i].type),
                                        Var::make(merge_block->args[i].type,
                                                  merge_block->args[i].name),
                                        Allocate::Stack));
                                }
                            }
                            */
                        }

                        Stmt true_body = structurize(t1, merge, block_map, dom,
                                                     info, mut_map);
                        Stmt false_body = structurize(t0, merge, block_map, dom,
                                                      info, mut_map);

                        stmts.push_back(IfElse::make(std::move(cond),
                                                     std::move(true_body),
                                                     std::move(false_body)));
                        name = merge; // advance past the if/else
                    }
                },

                [&](const Terminator::Return &r) {
                    if (r.value)
                        stmts.push_back(
                            ir::Return::make(codegen_value(r.value)));
                    else
                        stmts.push_back(ir::Return::make());
                    name = exit; // terminate the while loop
                },

                [&](const Terminator::Yield &) {
                    stmts.push_back(Continue::make());
                    name = exit; // terminate the while loop
                },

                [&](const Terminator::ParFor &p) {
                    Expr begin = codegen_value(p.start);
                    Expr end = codegen_value(p.end);
                    Expr stride = codegen_value(p.stride);
                    ParFor::Slice slice{std::move(begin), std::move(end),
                                        std::move(stride)};

                    // Body is a genuinely separate sub-CFG, must recurse
                    Stmt body = structurize(p.body.name, "", block_map, dom,
                                            info, mut_map);
                    stmts.push_back(ir::ParFor::make(p.index, std::move(slice),
                                                     std::move(body)));
                    name = p.cont.name; // advance past the parfor
                },

                [&](const Terminator::Call &) {
                    internal_error << "TODO: Call lifting";
                },

            },
            block->terminator.data);
    }

    std::cout << start << " versus exit: " << exit << std::endl;
    return Sequence::make(std::move(stmts));
}

Stmt codegen_body(const ssa::Function &func) {
    std::cout << "codegen for: " << func.blocks[0]->name << std::endl;
    const auto block_map = make_block_map(func);
    const auto dom = compute_dominators(func, block_map);
    const auto info = classify_blocks(func, block_map, dom);
    const auto mut_map = get_mutability_map(func);
    return structurize(func.blocks[0]->name, "", block_map, dom, info, mut_map);
}

} // namespace

std::shared_ptr<ir::Function> codegen_stmt(const ssa::Function &func) {
    internal_assert(!func.blocks.empty());
    std::string name = func.blocks[0]->name;
    std::vector<ir::Function::Argument> args;
    args.reserve(func.blocks[0]->args.size());
    for (const auto &arg : func.blocks[0]->args) {
        // Figure out mutability / default values later.
        args.push_back(ir::Function::Argument(arg.name, arg.type));
    }

    Type ret_type = func.ret_type;

    Stmt body = codegen_body(func);

    func.dump(std::cout);

    std::cout << "\n\n ->\n\n";

    std::cout << body << std::endl;

    ir::Function::InterfaceList ilist; // always empty at this stage.

    std::vector<ir::Function::Attribute> attrs; // Figure this out later

    return std::make_shared<ir::Function>(std::move(name), std::move(args),
                                          std::move(ret_type), std::move(body),
                                          std::move(ilist), std::move(attrs));
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
