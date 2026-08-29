#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Mutator.h"
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
            [](const std::shared_ptr<Instruction> &i) -> Expr {
                if (i->op == Instruction::Op::Ramp) {
                    // Inlined rather than bound to a name: whether a memory
                    // access is dense or a gather is read off the shape of
                    // its index (see CodeGen_LLVM's Ramp handling), so a Ramp
                    // has to be visible at its use.
                    internal_assert(i->operands.size() == 2);
                    return Ramp::make(codegen_value(i->operands[0]),
                                      codegen_value(i->operands[1]),
                                      i->type.lanes());
                }
                if (i->op == Instruction::Op::GEP) {
                    // Also inlined, and for the same reason: an address is
                    // written where it is used. Most of them are the location
                    // of a store, which codegen_gep turns into a WriteLoc;
                    // one used as a value -- an element's address handed to a
                    // callee that takes a pointer -- is the address-of it was
                    // made from (see the PtrTo visitor in SSA/Convert.cpp).
                    internal_assert(i->operands.size() == 2)
                        << "Malformed GEP: expected 2 operands";
                    Expr base = codegen_value(i->operands[0]);
                    internal_assert(base.type().is_reference())
                        << "[unimplemented] the address of an element of "
                        << base << ", which is not an array";
                    return PtrTo::make(
                        Extract::make(base, codegen_value(i->operands[1])));
                }
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
    case Instruction::Op::Print:
    case Instruction::Op::Store:
    case Instruction::Op::Alloc:
    case Instruction::Op::Alloca:
        return true;
    case Instruction::Op::Abs:
    case Instruction::Op::Add:
    case Instruction::Op::AddressOf:
    case Instruction::Op::Any:
    case Instruction::Op::Bc:
    case Instruction::Op::BwAnd:
    case Instruction::Op::BwOr:
    case Instruction::Op::Shl:
    case Instruction::Op::Shr:
    case Instruction::Op::SizeOf:
    case Instruction::Op::Xor:
    case Instruction::Op::Cast:
    case Instruction::Op::Div:
    case Instruction::Op::Eps:
    case Instruction::Op::Inf:
    // `rand` updates the generator's state, so it is not pure -- but it does
    // produce a value, and this says whether an instruction is emitted as a
    // bare statement rather than bound to a name. It is bound, in the order
    // the block puts it in, which is what makes it happen exactly once.
    // Nothing here duplicates it: inline_expr has no case for Op::Intrinsic
    // and so refers to the binding.
    case Instruction::Op::Intrinsic:
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
    case Instruction::Op::Ne:
    case Instruction::Op::Ramp:
    case Instruction::Op::Reduce:
    case Instruction::Op::Reinterpret:
    case Instruction::Op::Select:
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

        // Base case: non-GEP instruction -> variable. Every base a
        // Store/Accumulate GEP chain bottoms out at (an Alloc/Alloca; see
        // SSA/Convert.cpp) is pointer-typed, since only mutable
        // args/locals are ever looked up under a pointer type there.
        // WriteLoc::base_type/type must be the raw pointee type (it's
        // narrowed by add_index_access/add_struct_access as accesses are
        // appended), so unwrap it here.
        internal_assert(!i->name.empty())
            << "Cannot form WriteLoc from unnamed instruction";
        // An array handle is already the address of its elements, so it is
        // not wrapped in a Ptr_t (see Type::is_reference) and is its own
        // base.
        if (i->type.is_reference()) {
            return WriteLoc(i->name, i->type);
        }
        const Ptr_t *ptr_t = i->type.as<Ptr_t>();
        internal_assert(ptr_t)
            << "GEP base instruction: " << i->name
            << " expected to be pointer-typed, got: " << i->type;

        return WriteLoc(i->name, ptr_t->etype);
    }

    // Argument base case: same pointer-typed assumption as above, since
    // mutable arguments are registered under Ptr_t(original type) (see
    // `mut_names` in SSA/Convert.cpp).
    if (auto arg = std::get_if<Argument>(&v->data)) {
        if (arg->type.is_reference()) {
            return WriteLoc(arg->name, arg->type);
        }
        const Ptr_t *ptr_t = arg->type.as<Ptr_t>();
        internal_assert(ptr_t)
            << "GEP base argument: " << arg->name
            << " expected to be pointer-typed, got: " << arg->type;

        return WriteLoc(arg->name, ptr_t->etype);
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
        case Instruction::Op::Print:
            return Print::make(codegen_values(instr.operands));
        case Instruction::Op::Append:
            internal_error << "TODO: Append codegen!\n";
        case Instruction::Op::Store: {
            // A third operand is the execution mask of a vectorized store:
            // only the lanes it enables are written.
            internal_assert(instr.operands.size() == 2 ||
                            instr.operands.size() == 3)
                << instr.operands.size();
            WriteLoc loc = codegen_gep(instr.operands[0]);
            Expr val = codegen_value(instr.operands[1]);
            Expr mask = instr.operands.size() == 3
                            ? codegen_value(instr.operands[2])
                            : Expr();
            return Store::make(std::move(loc), std::move(val),
                               std::move(mask));
        }
        case Instruction::Op::Alloc:
        case Instruction::Op::Alloca: {
            internal_assert(instr.operands.empty()) << instr.operands.size();
            // An array handle is registered under its own type, everything
            // else under a pointer to what it allocates (see the Allocate
            // visitor in SSA/Convert.cpp and Type::is_reference).
            Type allocated = instr.type;
            if (!allocated.is_reference()) {
                const Ptr_t *ptr_t = instr.type.as<Ptr_t>();
                internal_assert(ptr_t)
                    << "Alloc(a) instruction must have pointer type: "
                    << instr.type;
                allocated = ptr_t->etype;
            }
            Allocate::Memory memory = (instr.op == Instruction::Op::Alloca)
                                          ? Allocate::Stack
                                          : Allocate::Heap;
            // The initial value (if any) was split into a separate Store
            // instruction by the SSA builder (see SSA/Convert.cpp), so this
            // just declares storage.
            return Allocate::make(WriteLoc(instr.name, std::move(allocated)),
                                  memory);
        }
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
    case Instruction::Op::AddressOf: {
        internal_assert(args.size() == 1) << args.size();
        // Straight back to the form the backends already handle, which is the
        // one the Stmt pipeline hands them too: both now ask for the address
        // of a value and let code generation decide whether that is a GEP, a
        // pointer something was already loaded through, or -- only when the
        // value really is not anywhere -- a stack slot to copy it into.
        value = PtrTo::make(std::move(args[0]));
        break;
    }
    case Instruction::Op::Any: {
        internal_assert(args.size() == 1) << args.size();
        // A gang that has not been widened yet holds one bool, not a vector
        // of them, and "is any lane set" is then just that bool. Only the
        // widened form is a real reduction.
        if (args[0].type().is_vector()) {
            value = VectorReduce::make(VectorReduce::Or, std::move(args[0]));
        } else {
            value = std::move(args[0]);
        }
        break;
    }
    case Instruction::Op::Bc: {
        internal_assert(args.size() == 2) << args.size();
        const uint64_t lanes = get_const_u64(args[1]);
        value = Broadcast::make(lanes, std::move(args[0]));
        break;
    }
    case Instruction::Op::Intrinsic: {
        value = ir::Intrinsic::make(instr.intrinsic, std::move(args));
        break;
    }
    case Instruction::Op::Reduce: {
        internal_assert(args.size() == 1) << args.size();
        value = VectorReduce::make(instr.reduce, std::move(args[0]));
        break;
    }
    case Instruction::Op::BwAnd: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::BwAnd, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::BwOr: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::BwOr, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Shl: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Shl, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::SizeOf: {
        internal_assert(args.empty()) << args.size();
        internal_assert(instr.queried_type.defined())
            << "sizeof of nothing in " << instr.name;
        value = SizeOf::make(instr.queried_type, instr.type);
        break;
    }
    case Instruction::Op::Shr: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Shr, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Xor: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Xor, std::move(args[0]),
                            std::move(args[1]));
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
    case Instruction::Op::Inf: {
        internal_assert(args.size() == 0) << args.size();
        value = Extrema::make(instr.type, Extrema::inf);
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
        // GEP has no standalone codegen: it's a pure address-computation
        // helper for a Store/Accumulate, consumed inline by codegen_gep()
        // above, which walks the operand chain directly rather than by
        // name. The per-instruction loop below skips GEP instructions for
        // this reason. Reaching here means something referenced a GEP's
        // result outside of a Store/Accumulate address, which the SSA
        // builder never constructs and this codegen doesn't support.
        internal_error << "GEP instruction: " << instr.name
                       << " was codegen'd standalone instead of being "
                          "consumed via codegen_gep() by its owning "
                          "Store/Accumulate.";
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
    case Instruction::Op::Load: {
        internal_assert(args.size() == 1) << args.size();
        value = Deref::make(std::move(args[0]));
        break;
    }
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
    case Instruction::Op::Ne: {
        internal_assert(args.size() == 2) << args.size();
        value = BinOp::make(BinOp::OpType::Neq, std::move(args[0]),
                            std::move(args[1]));
        break;
    }
    case Instruction::Op::Ramp: {
        internal_assert(args.size() == 2) << args.size();
        value = Ramp::make(std::move(args[0]), std::move(args[1]),
                           instr.type.lanes());
        break;
    }
    case Instruction::Op::Reinterpret: {
        internal_assert(args.size() == 1) << args.size();
        value =
            Cast::make(instr.type, std::move(args[0]), Cast::Mode::Reinterpret);
        break;
    }
    case Instruction::Op::Select: {
        internal_assert(args.size() == 3) << args.size();
        value = Select::make(std::move(args[0]), std::move(args[1]), std::move(args[2]));
        break;
    }
    case Instruction::Op::Set: {
        internal_assert(args.size() == 1) << args.size();
        internal_assert(!instr.name.starts_with("@")) << instr.name;
        // Giving a value a name -- what the SSA builder makes of a `let`
        // whose right-hand side is not an instruction it could have renamed
        // in place (see Block::make_instruction). A binding, not storage: an
        // allocation would mean the name held the address of the value
        // instead of the value, and every read of it would have to say so.
        return LetStmt::make(WriteLoc(instr.name, instr.type), args[0]);
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

// Recursively inline a pure SSA value as an expression, following def-use
// chains instead of emitting let-bindings.
//
// `unbound`, if given, is a block whose instructions are not emitted as
// statements, so that referring to one of them by name would name something
// that does not exist -- it has to be inlined or reported. A while header is
// such a block: its instructions compute the condition, which has to be redone
// on every iteration and so becomes part of the condition expression. Values
// from anywhere else are already bound and can simply be named.
Expr inline_expr(const std::shared_ptr<Value> &v,
                 const Block *unbound = nullptr) {
    auto *ip = std::get_if<std::shared_ptr<Instruction>>(&v->data);
    if (!ip)
        return codegen_value(v); // Constant or Argument — use as-is

    const Instruction &instr = **ip;
    const bool must_inline =
        unbound != nullptr && instr.owner.lock().get() == unbound;

    if (is_side_effecty(instr.op)) {
        internal_assert(!must_inline)
            << "Cannot inline " << instr.name << ", which has an effect, into "
            << "an expression that has nowhere to be bound";
        return codegen_value(v); // don't inline side effects
    }

    // Recursively inline operands
    std::vector<Expr> args;
    for (auto &op : instr.operands)
        args.push_back(inline_expr(op, unbound));

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
    case Instruction::Op::Ne:
        return BinOp::make(BinOp::OpType::Neq, args[0], args[1]);
    case Instruction::Op::Select:
        return Select::make(args[0], args[1], args[2]);
    case Instruction::Op::Load:
        // Reading storage, which for a loop condition is the point: what the
        // body wrote is what decides whether to go round again.
        return Deref::make(args[0]);
    case Instruction::Op::Any:
        // A while loop's test is re-evaluated on every iteration, so a
        // uniformized loop's "is any lane still live" has to be part of the
        // condition rather than a value computed once ahead of it.
        return args[0].type().is_vector()
                   ? VectorReduce::make(VectorReduce::Or, args[0])
                   : args[0];
    default:
        // Not inlineable — fall back to variable reference
        internal_assert(!must_inline)
            << "Cannot inline " << instr.name
            << " into an expression that has nowhere to be bound";
        return codegen_value(v);
    }
}

// The blocks reachable from `name`. `stop` is not followed past: reachability
// that is allowed to go round an enclosing loop reaches nearly everything, so
// asking a question about one loop means stopping at its header.
std::set<std::string> reachable(const std::string &name,
                                const BlockMap &block_map,
                                const std::string &stop = "") {
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
            if (name == stop && name != start)
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
        DoWhileHeader, // A DoWhileLatch jumps back to it; it is not the test
        DoWhileLatch, // Has a Dispatch; one arm is a backedge to a dominator
        InfLoopLatch, // Has an unconditional Jump that is a backedge
                      // (do-while(true))
    };

    Role role = Role::Normal;
    std::string loop_header; // for DoWhileLatch: which block is the header
    // for WhileHeader and DoWhileHeader: where control goes after the loop
    std::string loop_exit;
    std::string loop_body; // for WhileHeader: which arm enters the body
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

        // Body arm is whichever can reach the latch without going round the
        // loop again -- stopping at the header, since a path that comes back
        // through it can reach the latch from either arm.
        const bool t0_is_body = reachable(t0, block_map, name).count(latch) > 0;
        const bool t1_is_body = reachable(t1, block_map, name).count(latch) > 0;

        internal_assert(t0_is_body ^ t1_is_body)
            << "Can't determine while body for header: " << name
            << " (neither or both of " << t0 << " and " << t1
            << " reach its latch " << latch << ")";

        info[name].role = BlockInfo::Role::WhileHeader;
        info[name].loop_body = t0_is_body ? t0 : t1;
        info[name].loop_exit = t0_is_body ? t1 : t0;
    }

    // Pass 3: the headers of do-while loops -- the block a DoWhileLatch jumps
    // back to, when that block is not itself the loop's test.
    //
    // A do-while is emitted by wrapping everything accumulated so far as the
    // body once the latch is reached, so the walk has to start a fresh region
    // at the header. Otherwise whatever came before the loop, the values it
    // is entered with above all, ends up inside it and is redone on every
    // iteration.
    for (auto &[name, bi] : info) {
        if (bi.role != BlockInfo::Role::DoWhileLatch) {
            continue;
        }
        BlockInfo &head = info.at(bi.loop_header);
        if (head.role != BlockInfo::Role::Normal) {
            continue;
        }
        head.role = BlockInfo::Role::DoWhileHeader;
        head.loop_exit = bi.loop_exit;
    }

    return info;
}

// `loop_header` is the header of the innermost loop this region is inside, if
// any. A jump to it from anywhere other than the end of the region is the
// loop going round again from the middle -- a `continue` -- which is how a
// traversal that has finished with one node gets on to the next.
Stmt structurize(const std::string &start, const std::string &exit,
                 const BlockMap &block_map, const DominatorMap &dom,
                 const BlockInfoMap &info, const ArgMutabilityMap &mut_map,
                 const TypeMap &func_type_map,
                 const std::string &loop_header = "") {

    std::vector<Stmt> stmts;
    std::string name = start;

    auto append = [&](const Stmt &stmt) {
        if (stmt.defined()) {
            // guard against empty blocks.
            stmts.push_back(stmt);
        }
    };

    auto emit_jump_args = [&](const std::string &target,
                              const std::vector<std::shared_ptr<Value>> &vals) {
        auto &target_block = block_map.at(target);
        if (target_block->args.empty())
            return;
        auto &muts = mut_map.at(target);
        internal_assert(vals.size() == target_block->args.size())
            << "Jump to " << target << " passes " << vals.size()
            << " arguments but it takes " << target_block->args.size();

        // Is this the same definition the argument already stands for? Nearly
        // every block argument is: the SSA builder threads a value through the
        // blocks that use it under its own name, so the jump passes the name
        // straight back and there is nothing to bind.
        auto passes_itself = [&](size_t i) {
            const std::string &param = target_block->args[i].name;
            if (const auto *a = std::get_if<Argument>(&vals[i]->data)) {
                return a->name == param;
            }
            if (const auto *in =
                    std::get_if<std::shared_ptr<Instruction>>(&vals[i]->data)) {
                return (*in)->name == param;
            }
            return false;
        };

        for (size_t i = 0; i < vals.size(); i++) {
            if (passes_itself(i)) {
                continue;
            }
            if (muts[i]) {
                // Several predecessors disagree about the value, so the
                // argument is a variable and each of them assigns to it.
                append(Store::make(WriteLoc(target_block->args[i].name,
                                            target_block->args[i].type),
                                   codegen_value(vals[i])));
            } else {
                // One definition reaches the argument, under some other name:
                // bind it here, where the jump is, so that the block below can
                // refer to it. This is what a block argument means when it is
                // not a phi.
                append(LetStmt::make(WriteLoc(target_block->args[i].name,
                                              target_block->args[i].type),
                                     codegen_value(vals[i])));
            }
        }
    };

    while (name != exit) {
        auto block = block_map.at(name);
        const BlockInfo &bi = info.at(name);

        // A do-while is built by wrapping up everything emitted since the
        // region began, so the loop gets a region of its own -- starting here,
        // at its header -- and what came before it stays outside.
        if (bi.role == BlockInfo::Role::DoWhileHeader && name != start) {
            append(structurize(name, bi.loop_exit, block_map, dom, info,
                               mut_map, func_type_map, loop_header));
            name = bi.loop_exit;
            continue;
        }

        // Emit this block's instructions. A while header's are not emitted as
        // statements at all: they compute the test, which is inlined into the
        // condition below so that it is redone on every iteration. Emitting
        // them here would compute it once, before the loop.
        for (auto &instr : block->instrs) {
            if (bi.role == BlockInfo::Role::WhileHeader) {
                internal_assert(!is_side_effecty(instr->op))
                    << "While header " << name << " has a side effect in it, "
                    << "which cannot be moved into the loop condition";
                continue;
            }
            if (instr->op == Instruction::Op::GEP ||
                instr->op == Instruction::Op::Ramp) {
                // Pure address-computation helpers, consumed inline by
                // codegen_gep()/codegen_value() when codegening the owning
                // Store/Accumulate below; they have no standalone Stmt
                // representation.
                continue;
            }
            append(codegen_instruction(*instr));
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
                    } else if (j.name == loop_header) {
                        // Round the enclosing loop again, from the middle of
                        // it. The values it carries have been assigned above,
                        // the same as on the edge that closes it.
                        append(Continue::make());
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
                                append(Allocate::make(
                                    WriteLoc(block->args[i].name,
                                             block->args[i].type),
                                    Var::make(block->args[i].type,
                                              block->args[i].name),
                                    Allocate::Stack));
                            }
                        }
                        */

                        // Inside the body, this block is the loop to continue.
                        Stmt body =
                            structurize(bi.loop_body, name, block_map, dom,
                                        info, mut_map, func_type_map, name);

                        Expr loop_cond = inline_expr(d.cond, block.get());
                        if (bi.loop_body != t1) {
                            loop_cond = UnOp::make(UnOp::OpType::Not,
                                                   std::move(loop_cond));
                        }

                        append(
                            While::make(std::move(loop_cond), std::move(body)));
                        name = bi.loop_exit; // advance past the loop

                    } else if (bi.role == BlockInfo::Role::DoWhileLatch) {
                        // Wrap everything accumulated so far as the loop body
                        Expr loop_cond = (bi.loop_header == t1)
                                             ? cond
                                             : UnOp::make(UnOp::OpType::Not,
                                                          std::move(cond));

                        // What the next iteration is handed, assigned at the
                        // end of the body -- the copies a phi becomes. They
                        // read values the body computed, so they cannot be
                        // hoisted out of it, and they are what carries a loop
                        // forward at all.
                        const size_t back = bi.loop_header == t0 ? 0 : 1;
                        emit_jump_args(bi.loop_header, d.targets[back].args);

                        Stmt body = Sequence::make(std::move(stmts));
                        stmts = {DoWhile::make(std::move(body),
                                               std::move(loop_cond))};

                        // And what the block after the loop is handed.
                        emit_jump_args(bi.loop_exit, d.targets[1 - back].args);
                        name = bi.loop_exit; // advance past the loop

                    } else {
                        // If/else: recurse into both arms (bounded by merge).
                        // The search stops at the enclosing loop's header, so
                        // that a path which goes round the loop is not taken
                        // for a way the arms come back together -- otherwise
                        // no merge is found and both arms are emitted all the
                        // way to the end of the region, twice.
                        std::set<std::string> stop;
                        if (!loop_header.empty()) {
                            stop.insert(loop_header);
                        }
                        std::string merge =
                            find_merge_block(t1, t0, block_map, stop);

                        // Allocate mutable args of the merge block BEFORE the
                        // if/else
                        if (!merge.empty()) {
                            /*
                            auto &merge_block = block_map.at(merge);
                            auto &muts = mut_map.at(merge);
                            for (size_t i = 0; i < merge_block->args.size();
                                 i++) {
                                if (muts[i]) {
                                    append(Allocate::make(
                                        WriteLoc(merge_block->args[i].name,
                                                 merge_block->args[i].type),
                                        Var::make(merge_block->args[i].type,
                                                  merge_block->args[i].name),
                                        Allocate::Stack));
                                }
                            }
                            */
                        }

                        // Arms that never come back together -- one of them
                        // returns, say, or goes round the enclosing loop
                        // again -- each run to the end of the region this
                        // if/else is part of, and there is nothing after it.
                        const std::string &arm_exit = merge.empty() ? exit
                                                                    : merge;
                        Stmt true_body =
                            structurize(t1, arm_exit, block_map, dom, info,
                                        mut_map, func_type_map, loop_header);
                        Stmt false_body =
                            structurize(t0, arm_exit, block_map, dom, info,
                                        mut_map, func_type_map, loop_header);

                        // An arm can be empty -- a branch whose taken side
                        // goes straight to where the other one ends up, which
                        // is what a `return` in the middle of a loop becomes
                        // once the return is the loop going round again. An
                        // `if` needs something to do, so the sides are
                        // swapped rather than filled with nothing, and a
                        // branch with nothing on either side is not emitted at
                        // all.
                        if (true_body.defined()) {
                            append(IfElse::make(std::move(cond),
                                                std::move(true_body),
                                                std::move(false_body)));
                        } else if (false_body.defined()) {
                            append(IfElse::make(
                                UnOp::make(UnOp::OpType::Not, std::move(cond)),
                                std::move(false_body), Stmt()));
                        }
                        name = merge.empty() ? exit : merge;
                    }
                },

                [&](const Terminator::Return &r) {
                    if (r.value) {
                        append(ir::Return::make(codegen_value(r.value)));
                    } else {
                        append(ir::Return::make());
                    }
                    name = exit; // terminate the while loop
                },

                [&](const Terminator::Yield &) {
                    append(Continue::make());
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
                                            info, mut_map, func_type_map);
                    append(ir::ParFor::make(p.index, std::move(slice),
                                            std::move(body), p.binding));
                    name = p.cont.name; // advance past the parfor
                },

                [&](const Terminator::Call &c) {
                    auto &cont_block = block_map.at(c.cont.name);

                    // A call continuation always has exactly one predecessor —
                    // the call site.
                    internal_assert(cont_block->preds.size() == 1)
                        << "Call continuation " << c.cont.name << " has "
                        << cont_block->preds.size()
                        << " predecessors, expected exactly 1";

                    // Since there is only one predecessor, no arg can be
                    // mutable — there is nothing to merge.
                    auto &muts = mut_map.at(c.cont.name);
                    for (size_t i = 0; i < muts.size(); i++) {
                        internal_assert(!muts[i])
                            << "Call continuation " << c.cont.name << " arg "
                            << i << " (" << cont_block->args[i].name
                            << ") is mutable, but continuations with one "
                               "predecessor "
                               "should never have mutable args";
                    }

                    std::vector<Expr> call_args;
                    for (auto &arg : c.call.args) {
                        call_args.push_back(codegen_value(arg));
                    }

                    internal_assert(func_type_map.contains(c.call.name))
                        << c.call.name;
                    Type func_t = func_type_map.at(c.call.name);
                    Expr call_func = Var::make(func_t, c.call.name);

                    if (c.drop) {
                        append(CallStmt::make(std::move(call_func),
                                              std::move(call_args)));
                    } else {
                        internal_assert(!cont_block->args.empty())
                            << "Call continuation " << c.cont.name
                            << " has no args, but drop=false requires a return "
                               "value binding";
                        auto &ret_arg = cont_block->args[0];
                        append(
                            LetStmt::make(WriteLoc(ret_arg.name, ret_arg.type),
                                          Call::make(std::move(call_func),
                                                     std::move(call_args))));
                    }

                    // Bind all continuation args (immutable, so all become
                    // let-bindings)
                    // emit_jump_args(c.cont.name, c.cont.args);

                    name = c.cont.name;
                },

            },
            block->terminator.data);
    }

    if (stmts.empty()) {
        return Stmt();
    }
    return Sequence::make(std::move(stmts));
}

// Turns every reference to one of `storage` into a load from it.
//
// Those names were block arguments, which are values. The structured form
// binds them to stack slots instead, because a loop has to be able to assign
// to what it carries, and this IR reads a slot explicitly rather than
// implicitly. Assignments are unaffected: a store names its destination
// instead of evaluating it, so the base of a WriteLoc is not an expression
// this can reach.
struct LoadMaterialized : public Mutator {
    const std::map<std::string, Type> &storage;

    explicit LoadMaterialized(const std::map<std::string, Type> &storage)
        : storage(storage) {}

    Expr visit(const Var *node) override {
        const auto it = storage.find(node->name);
        if (it == storage.end()) {
            return node;
        }
        return Deref::make(Var::make(Ptr_t::make(it->second), node->name));
    }
};

Stmt load_materialized(const Stmt &body,
                       const std::map<std::string, Type> &storage) {
    if (storage.empty()) {
        return body;
    }
    LoadMaterialized loads(storage);
    return loads.mutate(body);
}

Stmt codegen_body(const ssa::Function &func, const TypeMap &func_type_map) {
    const auto block_map = make_block_map(func);
    const auto dom = compute_dominators(func, block_map);
    const auto info = classify_blocks(func, block_map, dom);
    const auto mut_map = get_mutability_map(func);

    // A block argument that several predecessors pass different values to
    // becomes a variable, assigned at each of those jumps. Most of them are
    // already backed by storage: they carry the name of the mutable local
    // whose alloca is still in the entry block, and were only ever arguments
    // because the SSA builder threads a local through the blocks that use it.
    // A loop's carried values are not -- nothing declared them -- so they are
    // declared here, at the top of the function, where their first assignment
    // is sure to be dominated by the declaration.
    std::set<std::string> declared;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if (instr->op == Instruction::Op::Alloca ||
                instr->op == Instruction::Op::Alloc) {
                declared.insert(instr->name);
            }
        }
    }
    for (const auto &arg : func.blocks[0]->args) {
        declared.insert(arg.name);
    }

    std::vector<Stmt> stmts;
    std::map<std::string, Type> materialized;
    for (const auto &block : func.blocks) {
        const auto muts = mut_map.find(block->name);
        if (muts == mut_map.end()) {
            continue;
        }
        for (size_t i = 0; i < block->args.size(); i++) {
            if (!muts->second[i] || !declared.insert(block->args[i].name).second) {
                continue;
            }
            materialized[block->args[i].name] = block->args[i].type;
            stmts.push_back(Allocate::make(
                WriteLoc(block->args[i].name, block->args[i].type),
                Allocate::Memory::Stack));
        }
    }

    Stmt body = structurize(func.blocks[0]->name, "", block_map, dom, info,
                            mut_map, func_type_map);
    if (stmts.empty()) {
        return body;
    }

    // Those names are storage now, and this IR reads storage explicitly: a
    // reference to one is a load, not the value itself. The assignments to
    // them are unaffected, since a store names its destination rather than
    // evaluating it.
    body = load_materialized(body, materialized);

    stmts.push_back(std::move(body));
    return Sequence::make(std::move(stmts));
}

} // namespace

std::shared_ptr<ir::Function> codegen_stmt(const ssa::Function &func,
                                           const TypeMap &func_type_map) {
    internal_assert(!func.blocks.empty());
    std::string name = func.blocks[0]->name;
    std::vector<ir::Function::Argument> args;
    args.reserve(func.blocks[0]->args.size());
    for (const auto &arg : func.blocks[0]->args) {
        // TODO: default values aren't preserved through SSA.
        args.push_back(ir::Function::Argument(arg.name, arg.type, Expr(),
                                              arg.mutating, arg.unaliased));
    }

    Type ret_type = func.ret_type;

    Stmt body = codegen_body(func, func_type_map);

    ir::Function::InterfaceList ilist; // always empty at this stage.

    return std::make_shared<ir::Function>(
        std::move(name), std::move(args), std::move(ret_type),
        std::move(body), std::move(ilist), func.attributes);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
