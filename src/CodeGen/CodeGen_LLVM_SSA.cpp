#include "CodeGen/CodeGen_LLVM.h"

#include "IR/Operators.h"
#include "SSA/Analysis.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <string>
#include <vector>

namespace bonsai {

using namespace ir;
using ir::ssa::Argument;
using ir::ssa::Block;
using ir::ssa::Constant;
using ir::ssa::Instruction;
using ir::ssa::Terminator;
using ir::ssa::Value;

// Lowering one SSA function to LLVM.
//
// The division of labour: control flow is handled here, because that is the
// part LLVM wants in exactly the form the SSA already has it -- blocks with
// arguments are blocks with phis. Everything within a block is rebuilt as the
// ir::Expr the relooper would have produced and handed to codegen_expr, so
// that an add, a gather or a masked store has one lowering rather than two.
//
// Names are how the two meet. An instruction's result is bound in the frame
// under its own name, which the function makes unique, and referring to it is
// an ir::Var of that name. A block argument is bound the same way as its block
// is entered, and overwritten as the next block is: block arguments
// deliberately share names across blocks -- that is how a value is threaded
// onwards -- so they are only meaningful within the block that declares them,
// which is also the only block whose instructions can refer to them.
struct CodeGen_LLVM::SSALowering {
    CodeGen_LLVM &cg;
    const ir::ssa::Function &func;
    llvm::Function *function;

    std::map<std::string, llvm::BasicBlock *> blocks;
    std::map<std::string, const Block *> by_name;
    // Per block, the phi standing for each of its arguments, in order.
    std::map<std::string, std::vector<llvm::PHINode *>> phis;

    SSALowering(CodeGen_LLVM &cg, const ir::ssa::Function &func,
                llvm::Function *function)
        : cg(cg), func(func), function(function) {}

    const std::string &entry() const { return func.blocks.front()->name; }

    // An operand, as an expression referring to whatever already holds it.
    Expr operand(const std::shared_ptr<Value> &v) {
        internal_assert(v) << "Null operand in " << func.blocks.front()->name;
        if (const auto *c = std::get_if<Constant>(&v->data)) {
            return constant(*c);
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            return Var::make(a->type, a->name);
        }
        const auto &instr = std::get<std::shared_ptr<Instruction>>(v->data);

        // A ramp is written out where it is used rather than referred to by
        // name. Whether an access is dense or a gather is read off the shape
        // of its index, so an index that has become an opaque name reads as a
        // gather -- which is correct but several times the work. The relooper
        // inlines these for the same reason; see codegen_value in
        // SSA/CodeGen_Stmt.cpp.
        if (instr->op == Instruction::Op::Ramp) {
            internal_assert(instr->operands.size() == 2)
                << "A ramp has a base and a stride";
            return Ramp::make(operand(instr->operands[0]),
                              operand(instr->operands[1]), instr->type.lanes());
        }
        // Likewise an address: most GEPs are the place a store goes, which
        // location() walks; one used as a value is the address-of it was made
        // from.
        if (instr->op == Instruction::Op::GEP) {
            internal_assert(instr->operands.size() == 2)
                << "A GEP has a base and an index";
            Expr base = operand(instr->operands[0]);
            internal_assert(base.type().is_reference())
                << "[unimplemented] the address of an element of " << base
                << ", which is not an array";
            return PtrTo::make(
                Extract::make(base, operand(instr->operands[1])));
        }
        return Var::make(instr->type, instr->name);
    }

    Expr constant(const Constant &c) {
        return std::visit(
            ir::ssa::overloads{
                [&](bool b) { return BoolImm::make(b); },
                [&](int64_t i) { return make_const(c.type, i); },
                [&](uint64_t u) { return make_const(c.type, int64_t(u)); },
                [&](double d) { return FloatImm::make(c.type, d); },
                [&](const std::string &s) -> Expr {
                    internal_error << "String constant " << s
                                   << " has no value form in SSA lowering";
                    return Expr();
                },
            },
            c.data);
    }

    std::vector<Expr> operands(const Instruction &instr) {
        std::vector<Expr> args;
        args.reserve(instr.operands.size());
        for (const auto &v : instr.operands) {
            args.push_back(operand(v));
        }
        return args;
    }

    // The place a GEP names. A GEP is not a value of its own here any more
    // than it is in the relooper: it is an address built for a store, and is
    // walked back to the thing it indexes into.
    WriteLoc location(const std::shared_ptr<Value> &v) {
        if (const auto *ptr =
                std::get_if<std::shared_ptr<Instruction>>(&v->data)) {
            const auto &instr = *ptr;
            if (instr->op == Instruction::Op::GEP) {
                internal_assert(instr->operands.size() == 2)
                    << "GEP takes a base and an index";
                WriteLoc loc = location(instr->operands[0]);
                loc.add_index_access(operand(instr->operands[1]));
                return loc;
            }
            return WriteLoc(instr->name, instr->type);
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            return WriteLoc(a->name, a->type);
        }
        internal_error << "A constant is not somewhere a store can go";
        return WriteLoc();
    }

    // The expression an instruction computes, or nothing if it is one that
    // only has an effect.
    Expr value_of(const Instruction &instr) {
        std::vector<Expr> args = operands(instr);
        const size_t n = args.size();

        auto binop = [&](BinOp::OpType op) {
            internal_assert(n == 2)
                << ir::ssa::op_name(instr.op) << " takes two";
            return BinOp::make(op, std::move(args[0]), std::move(args[1]));
        };

        switch (instr.op) {
        case Instruction::Op::Add:
            return binop(BinOp::OpType::Add);
        case Instruction::Op::Sub:
            return binop(BinOp::OpType::Sub);
        case Instruction::Op::Mul:
            return binop(BinOp::OpType::Mul);
        case Instruction::Op::Div:
            return binop(BinOp::OpType::Div);
        case Instruction::Op::Mod:
            return binop(BinOp::OpType::Mod);
        case Instruction::Op::Lt:
            return binop(BinOp::OpType::Lt);
        case Instruction::Op::Leq:
            return binop(BinOp::OpType::Le);
        case Instruction::Op::Eq:
            return binop(BinOp::OpType::Eq);
        case Instruction::Op::Ne:
            return binop(BinOp::OpType::Neq);
        case Instruction::Op::LAnd:
            return binop(BinOp::OpType::LAnd);
        case Instruction::Op::LOr:
            return binop(BinOp::OpType::LOr);
        case Instruction::Op::Min:
            internal_assert(n == 2) << "min takes two";
            return Intrinsic::make(Intrinsic::OpType::min, std::move(args));
        case Instruction::Op::Max:
            internal_assert(n == 2) << "max takes two";
            return Intrinsic::make(Intrinsic::OpType::max, std::move(args));
        case Instruction::Op::BwAnd:
            return binop(BinOp::OpType::BwAnd);
        case Instruction::Op::BwOr:
            return binop(BinOp::OpType::BwOr);
        case Instruction::Op::Xor:
            return binop(BinOp::OpType::Xor);
        case Instruction::Op::Shl:
            return binop(BinOp::OpType::Shl);
        case Instruction::Op::Shr:
            return binop(BinOp::OpType::Shr);

        case Instruction::Op::Select:
            internal_assert(n == 3) << "select takes three";
            return Select::make(std::move(args[0]), std::move(args[1]),
                                std::move(args[2]));
        case Instruction::Op::Bc: {
            internal_assert(n == 2) << "broadcast takes a value and a width";
            const auto lanes = get_constant_value<int64_t>(args[1]);
            internal_assert(lanes.has_value() && *lanes > 0)
                << "broadcast needs a constant width";
            return Broadcast::make(uint32_t(*lanes), std::move(args[0]));
        }
        case Instruction::Op::Ramp:
            internal_assert(n == 2) << "ramp takes a base and a stride";
            return Ramp::make(std::move(args[0]), std::move(args[1]),
                              instr.type.lanes());
        case Instruction::Op::ExtractIdx:
            internal_assert(n == 2) << "extract takes a value and an index";
            return Extract::make(std::move(args[0]), std::move(args[1]));
        case Instruction::Op::Load:
            internal_assert(n == 1) << "load takes a pointer";
            return Deref::make(std::move(args[0]));
        case Instruction::Op::Cast:
            internal_assert(n == 1) << "cast takes one value";
            return Cast::make(instr.type, std::move(args[0]));
        case Instruction::Op::Reinterpret:
            internal_assert(n == 1) << "reinterpret takes one value";
            return Cast::make(instr.type, std::move(args[0]),
                              Cast::Mode::Reinterpret);
        case Instruction::Op::Abs:
            return Intrinsic::make(Intrinsic::OpType::abs, std::move(args));
        case Instruction::Op::Intrinsic:
            return Intrinsic::make(instr.intrinsic, std::move(args));
        case Instruction::Op::Reduce:
            internal_assert(n == 1) << "a reduction takes one value";
            return VectorReduce::make(instr.reduce, std::move(args[0]));
        case Instruction::Op::Any:
            internal_assert(n == 1) << "any takes one value";
            // Before widening a gang holds one bool rather than a vector of
            // them, and "is any lane set" is then just that bool.
            if (args[0].type().is_vector()) {
                return VectorReduce::make(VectorReduce::Or, std::move(args[0]));
            }
            return std::move(args[0]);
        case Instruction::Op::AddressOf:
            internal_assert(n == 1) << "addressof takes one value";
            return PtrTo::make(std::move(args[0]));
        case Instruction::Op::SizeOf:
            internal_assert(args.empty()) << "sizeof takes no operands";
            internal_assert(instr.queried_type.defined())
                << "sizeof of nothing in " << instr.name;
            return SizeOf::make(instr.queried_type, instr.type);

        case Instruction::Op::GEP:
            internal_error << "GEP reached value lowering; it is an address "
                              "for a store, consumed by location()";
            return Expr();
        default:
            internal_error
                << "SSA-to-LLVM lowering does not handle "
                << ir::ssa::op_name(instr.op)
                << " yet. It is reached only by functions lowered straight "
                   "from SSA -- today the vectorized ones -- so an opcode "
                   "here means one of those grew a new kind of instruction.";
            return Expr();
        }
    }

    void emit_instruction(const std::shared_ptr<Instruction> &instr) {
        // Pure address and shape helpers, written out at their uses by
        // operand() rather than bound to a name of their own.
        if (instr->op == Instruction::Op::GEP ||
            instr->op == Instruction::Op::Ramp) {
            return;
        }
        if (instr->op == Instruction::Op::Store) {
            internal_assert(instr->operands.size() == 2 ||
                            instr->operands.size() == 3)
                << "store takes a place, a value and maybe a mask";
            WriteLoc loc = location(instr->operands[0]);
            Expr val = operand(instr->operands[1]);
            Expr mask = instr->operands.size() == 3
                            ? operand(instr->operands[2])
                            : Expr();
            cg.codegen_stmt(
                Store::make(std::move(loc), std::move(val), std::move(mask)));
            return;
        }
        Expr value = value_of(*instr);
        internal_assert(value.defined()) << "No value for " << instr->name
                                         << " (" << op_name(instr->op) << ")";
        bind(instr->name, cg.codegen_expr(value));
    }

    // Bind a name, whether or not it already stands for something.
    //
    // Block arguments share names across blocks on purpose -- that is how a
    // value is threaded onwards -- and only the block that declares one can
    // refer to it, so the binding that matters is always the most recent. An
    // instruction's name is unique to the function and never collides.
    void bind(const std::string &name, llvm::Value *value) {
        if (cg.frames.contains(name)) {
            cg.frames.replace(name, value);
        } else {
            cg.frames.add_to_frame(name, value);
        }
    }

    // Hand a block its arguments, as the values its predecessors pass.
    void bind_arguments(const Block &block) {
        const auto found = phis.find(block.name);
        if (found == phis.end()) {
            return;
        }
        for (size_t i = 0; i < block.args.size(); i++) {
            bind(block.args[i].name, found->second[i]);
        }
    }

    // What a jump hands its target, recorded against the block it leaves from.
    // `first` is where the target's argument list the jump's own values start:
    // a call continuation is handed the returned value before them.
    void supply(const Terminator::Jump &jump, llvm::BasicBlock *from,
                size_t first, llvm::Value *before = nullptr) {
        const auto found = phis.find(jump.name);
        if (found == phis.end()) {
            return;
        }
        const auto &target = found->second;
        if (before != nullptr) {
            internal_assert(!target.empty())
                << jump.name << " takes no value from the call before it";
            target[0]->addIncoming(before, from);
        }
        internal_assert(jump.args.size() + first == target.size())
            << "Jump to " << jump.name << " passes "
            << (jump.args.size() + first) << " arguments but it takes "
            << target.size();
        for (size_t i = 0; i < jump.args.size(); i++) {
            target[first + i]->addIncoming(
                cg.codegen_expr(operand(jump.args[i])), from);
        }
    }

    void run() {
        // In dominance order, and only what the entry can reach.
        //
        // A definition dominates its uses, so emitting in reverse postorder is
        // what makes a value bound by the time something reads it. The order
        // the blocks happen to be stored in is not that: a rewrite appends the
        // blocks it makes, so a join can sit in the list ahead of the block
        // whose value it selects. Unreachable blocks are left out entirely
        // rather than emitted empty, which would leave a block with no
        // terminator for the verifier to object to.
        for (const auto &block : func.blocks) {
            by_name[block->name] = block.get();
        }
        const std::vector<std::string> order = ir::ssa::reverse_postorder(
            entry(), ir::ssa::compute_successors(func));

        for (const std::string &name : order) {
            blocks[name] =
                llvm::BasicBlock::Create(*cg.context, name, function);
        }

        // The entry block's arguments are the function's own, so they are
        // already values rather than phis.
        cg.frames.push_frame();
        {
            const Block &head = *func.blocks.front();
            uint32_t i = 0;
            for (auto &arg : function->args()) {
                internal_assert(i < head.args.size())
                    << function->getName().str() << " takes more arguments "
                    << "than its entry block declares";
                arg.setName(head.args[i].name);
                bind(head.args[i].name, &arg);
                i++;
            }
        }

        // Every other block gets a phi per argument, made before anything
        // jumps to it so that the jumps have something to add themselves to.
        for (const std::string &name : order) {
            const Block &block = *by_name.at(name);
            if (name == entry() || block.args.empty()) {
                continue;
            }
            cg.builder->SetInsertPoint(blocks.at(name));
            std::vector<llvm::PHINode *> made;
            for (const Argument &arg : block.args) {
                made.push_back(cg.builder->CreatePHI(cg.codegen_type(arg.type),
                                                     0, arg.name));
            }
            phis[name] = std::move(made);
        }

        for (const std::string &name : order) {
            const Block &block = *by_name.at(name);
            cg.builder->SetInsertPoint(blocks.at(name));
            bind_arguments(block);
            for (const auto &instr : block.instrs) {
                emit_instruction(instr);
            }
            emit_terminator(block);
        }

        cg.frames.pop_frame();
    }

    void emit_terminator(const Block &block) {
        // The block a jump leaves from is wherever the builder has got to,
        // which is not the block it started in if something in between made
        // one of its own.
        llvm::BasicBlock *from = cg.builder->GetInsertBlock();
        std::visit(
            ir::ssa::overloads{
                [&](const std::monostate &) {
                    internal_error << block.name << " has no terminator";
                },
                [&](const Terminator::Jump &j) {
                    supply(j, from, 0);
                    cg.builder->CreateBr(blocks.at(j.name));
                },
                [&](const Terminator::Dispatch &d) {
                    internal_assert(d.targets.size() == 2)
                        << "A dispatch goes two ways";
                    llvm::Value *cond = cg.codegen_expr(operand(d.cond));
                    // targets[0] is where a false condition goes.
                    supply(d.targets[0], from, 0);
                    supply(d.targets[1], from, 0);
                    cg.builder->CreateCondBr(cond, blocks.at(d.targets[1].name),
                                             blocks.at(d.targets[0].name));
                },
                [&](const Terminator::Return &r) {
                    if (r.value) {
                        cg.builder->CreateRet(
                            cg.codegen_expr(operand(r.value)));
                    } else {
                        cg.builder->CreateRetVoid();
                    }
                },
                [&](const Terminator::Call &c) {
                    llvm::Function *callee =
                        cg.module->getFunction(c.call.name);
                    internal_assert(callee)
                        << "Call to undeclared function " << c.call.name;
                    std::vector<llvm::Value *> args;
                    for (const auto &a : c.call.args) {
                        args.push_back(cg.codegen_expr(operand(a)));
                    }
                    llvm::Value *result = cg.builder->CreateCall(callee, args);
                    llvm::BasicBlock *after = cg.builder->GetInsertBlock();
                    supply(c.cont, after, c.drop ? 0 : 1,
                           c.drop ? nullptr : result);
                    cg.builder->CreateBr(blocks.at(c.cont.name));
                },
                [&](const Terminator::ParFor &p) {
                    internal_error
                        << block.name << " still has a parfor over " << p.index
                        << ". Only vectorized functions are lowered from SSA "
                           "today, and vectorize() folds the loop it widens "
                           "into a single gang.";
                },
                [&](const Terminator::Yield &) {
                    internal_error << block.name
                                   << " still yields, which only a parfor "
                                      "body does, and there is no parfor here.";
                },
            },
            block.terminator.data);
    }
};

void CodeGen_LLVM::compile_function(const ir::ssa::Function &func,
                                    llvm::Function *function) {
    internal_assert(current_function == nullptr);
    internal_assert(function);
    internal_assert(!func.blocks.empty())
        << function->getName().str() << " has no blocks";
    current_function = function;

    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    SSALowering(*this, func, function).run();
    builder->restoreIP(here);

    internal_assert(!llvm::verifyFunction(*function, &llvm::errs()))
        << "Function verification failed for " << function->getName().str()
        << ", lowered straight from SSA";

    current_function = nullptr;
}

} // namespace bonsai
