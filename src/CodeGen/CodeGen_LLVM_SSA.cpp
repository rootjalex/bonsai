#include "CodeGen/CodeGen_LLVM.h"

#include "IR/Operators.h"
#include "Lower/Random.h"
#include "SSA/Analysis.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
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
    // A block argument as phis: one per scalar leaf of its type, and the
    // aggregate rebuilt from them, which is what the argument's name is
    // bound to. Per field rather than one phi of the struct because LLVM
    // neither splits an aggregate phi by field nor if-converts one, so a
    // struct-typed argument merged at a join would keep the branch that a
    // struct held in memory -- which SROA splits by field and mem2reg then
    // makes scalar phis of -- loses to a select. The phis here stand for
    // exactly that memory (SSA/PromoteAllocas.h), so they are made as SROA
    // would have made them; the insertvalues rebuilding the aggregate and
    // the extractvalues feeding the leaves fold away against each other.
    // create_select does the same for a select of aggregate type.
    struct ArgPhis {
        llvm::Value *value = nullptr;
        // Each leaf's phi, with the path of indices to the leaf in the
        // aggregate; empty for an argument that is itself a leaf.
        std::vector<std::pair<llvm::PHINode *, std::vector<unsigned>>> leaves;
    };
    // Per block, the phis standing for each of its arguments, in order.
    std::map<std::string, std::vector<ArgPhis>> phis;

    void make_leaf_phis(
        llvm::Type *type, const std::string &name, std::vector<unsigned> &path,
        std::vector<std::pair<llvm::PHINode *, std::vector<unsigned>>> &leaves) {
        if (auto *st = llvm::dyn_cast<llvm::StructType>(type)) {
            for (unsigned i = 0; i < st->getNumElements(); i++) {
                path.push_back(i);
                make_leaf_phis(st->getElementType(i),
                               name + "." + std::to_string(i), path, leaves);
                path.pop_back();
            }
            return;
        }
        leaves.emplace_back(cg.builder->CreatePHI(type, 0, name), path);
    }

    // The argument's value from its leaves: the leaf itself, or the
    // aggregate with each leaf inserted at its path.
    llvm::Value *rebuild(llvm::Type *type, const ArgPhis &arg,
                         const std::string &name) {
        if (!llvm::isa<llvm::StructType>(type)) {
            internal_assert(arg.leaves.size() == 1 && arg.leaves[0].second.empty())
                << "a non-aggregate block argument with " << arg.leaves.size()
                << " leaves";
            return arg.leaves[0].first;
        }
        llvm::Value *value = llvm::PoisonValue::get(type);
        for (const auto &[phi, path] : arg.leaves) {
            value = cg.builder->CreateInsertValue(value, phi, path);
        }
        if (!arg.leaves.empty()) {
            value->setName(name);
        }
        return value;
    }
    // Where a `Yield` (the end of one parfor iteration) goes, innermost last:
    // the latch of the loop a serial parfor became, or null in a parallel
    // parfor's kernel, where the iteration is a function call that returns.
    std::vector<llvm::BasicBlock *> yield_targets;

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
        // A field's address, read as a value: dereference the base, name the
        // field, take its address -- the same shape the relooper builds. The
        // field index is a constant by construction.
        if (instr->op == Instruction::Op::FieldPtr) {
            internal_assert(instr->operands.size() == 2)
                << "A FieldPtr has a base and a field index";
            Expr base = operand(instr->operands[0]);
            // What the base points at: through one pointer, or through one
            // pointer per lane, whose field's address is then one per lane
            // too (see Deref::make and PtrTo::make).
            const Type &base_type = base.type();
            const Type pointee =
                base_type.is<Ptr_t>() ? base_type.as<Ptr_t>()->etype
                : base_type.is<Vector_t>() &&
                        base_type.element_of().is<Ptr_t>()
                    ? base_type.element_of().as<Ptr_t>()->etype
                    : base_type;
            const auto *c =
                std::get_if<Constant>(&instr->operands[1]->data);
            internal_assert(c && std::holds_alternative<uint64_t>(c->data))
                << "FieldPtr index is not a constant";
            const uint64_t idx = std::get<uint64_t>(c->data);
            std::string field;
            if (const Struct_t *s = pointee.as<Struct_t>()) {
                internal_assert(idx < s->fields.size())
                    << "FieldPtr index " << idx << " past the end of "
                    << pointee;
                field = s->fields[idx].name;
            } else {
                internal_error << "the address of a field of " << pointee
                               << ", which is not a struct";
            }
            return PtrTo::make(Access::make(field, Deref::make(base)));
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
                [&](const ssa::Undefined &) { return Undef::make(c.type); },
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
            if (instr->op == Instruction::Op::FieldPtr) {
                internal_assert(instr->operands.size() == 2)
                    << "FieldPtr takes a base and a field index";
                WriteLoc loc = location(instr->operands[0]);
                const Struct_t *struct_t = loc.type.as<Struct_t>();
                internal_assert(struct_t)
                    << "FieldPtr into non-struct type " << loc.type;
                const Expr index = operand(instr->operands[1]);
                const auto *imm = index.as<UIntImm>();
                internal_assert(imm) << "FieldPtr with a non-constant index";
                internal_assert(imm->value < struct_t->fields.size())
                    << "FieldPtr index " << imm->value << " past the end of "
                    << loc.type;
                loc.add_struct_access(struct_t->fields[imm->value].name);
                return loc;
            }
            // The base a chain of accesses bottoms out at: an array handle is
            // its own address, anything else is a pointer whose pointee is what
            // a store's location records (see the relooper's codegen_gep).
            internal_assert(!instr->name.empty())
                << "cannot form a store location from an unnamed instruction";
            if (instr->type.is_reference()) {
                return WriteLoc(instr->name, instr->type);
            }
            const Ptr_t *ptr_t = instr->type.as<Ptr_t>();
            internal_assert(ptr_t) << "a store's base " << instr->name
                                   << " is not pointer-typed: " << instr->type;
            return WriteLoc(instr->name, ptr_t->etype);
        }
        if (const auto *a = std::get_if<Argument>(&v->data)) {
            if (a->type.is_reference()) {
                return WriteLoc(a->name, a->type);
            }
            const Ptr_t *ptr_t = a->type.as<Ptr_t>();
            internal_assert(ptr_t) << "a store's base argument " << a->name
                                   << " is not pointer-typed: " << a->type;
            return WriteLoc(a->name, ptr_t->etype);
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
        case Instruction::Op::Not:
            internal_assert(n == 1) << "not takes one";
            return UnOp::make(UnOp::Not, std::move(args[0]));

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
        case Instruction::Op::AtomicAdd:
            // A fetch-and-add on the place the first operand addresses,
            // whose value is what the place held before: a queue's count
            // claiming slots (see lower_pushes in SSA/Defer.cpp).
            internal_assert(n == 2) << "atomic add takes a place and a value";
            return AtomicAdd::make(std::move(args[0]), std::move(args[1]));
        case Instruction::Op::ExtractIdx:
            // A third operand is the execution mask of a gather (see
            // Extract::mask): the lanes that read at all.
            internal_assert(n == 2 || n == 3)
                << "extract takes a value, an index and maybe a mask";
            return Extract::make(std::move(args[0]), std::move(args[1]),
                                 n == 3 ? std::move(args[2]) : Expr());
        case Instruction::Op::LoadField: {
            // A field of a struct value, by index -- the read an inlined body
            // makes of a uniform aggregate it was handed, a box's corners say.
            internal_assert(n == 2)
                << "load_field takes a struct and a field index";
            const Struct_t *struct_t = args[0].type().as<Struct_t>();
            internal_assert(struct_t)
                << "load_field of a non-struct " << args[0].type() << " in "
                << instr.name;
            const auto idx = get_constant_value<uint64_t>(args[1]);
            internal_assert(idx.has_value() && *idx < struct_t->fields.size())
                << "load_field of field " << args[1] << " of "
                << args[0].type();
            return Access::make(struct_t->fields[*idx].name,
                                std::move(args[0]));
        }
        case Instruction::Op::MakeStruct:
            return Build::make(instr.type, std::move(args));
        case Instruction::Op::Shuffle:
            return Shuffle::make(std::move(args), instr.shuffle);
        case Instruction::Op::Eps:
            internal_assert(args.empty()) << "eps takes no operands";
            return Extrema::make(instr.type, Extrema::eps);
        case Instruction::Op::Inf:
            internal_assert(args.empty()) << "inf takes no operands";
            return Extrema::make(instr.type, Extrema::inf);
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
        case Instruction::Op::Set:
            // A copy: `let b = a` where `a` is not an instruction the builder
            // could rename, an argument or a constant.
            internal_assert(n == 1) << "set takes one value";
            return std::move(args[0]);
        case Instruction::Op::Abs:
            return Intrinsic::make(Intrinsic::OpType::abs, std::move(args));
        case Instruction::Op::Intrinsic:
            // `rand` carries the generator's state as its last operand in SSA
            // (see SSA/Convert.cpp); the expression form names it instead.
            if (instr.intrinsic == Intrinsic::rand) {
                internal_assert(n >= 1) << "rand without its state";
                args.pop_back();
            }
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
        case Instruction::Op::Popcount: {
            internal_assert(n == 1) << "popcount takes one value";
            // The lanes that are on, each as a one, summed: `kmov` and
            // `popcnt` once the x86 backend has seen it. Before widening the
            // gang is one bool, and its count is that bool as a number.
            const Type count = UInt_t::make(32);
            const Type held = args[0].type();
            if (held.is_vector()) {
                return VectorReduce::make(
                    VectorReduce::Add,
                    Cast::make(Vector_t::make(count, held.lanes()),
                               std::move(args[0])));
            }
            return Cast::make(count, std::move(args[0]));
        }
        case Instruction::Op::Vote:
            internal_assert(n == 1) << "vote takes one value";
            // Settled by lower_votes (SSA/Vectorize.cpp) wherever a gang
            // holds it; one that is still here is one visitor's own decision.
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

    // A side effect on memory or the outside world, as opposed to a value:
    // what a kernel makes once per block where the block's threads all run
    // the same code (see CodeGen_LLVM::effects_once_guard).
    static bool is_effect(Instruction::Op op) {
        switch (op) {
        case Instruction::Op::AccAdd:
        case Instruction::Op::AccMul:
        case Instruction::Op::AccSub:
        case Instruction::Op::AccMin:
        case Instruction::Op::AccMax:
        case Instruction::Op::AccArgmin:
        case Instruction::Op::AccArgmax:
        case Instruction::Op::Store:
        case Instruction::Op::Print:
            return true;
        default:
            return false;
        }
    }

    // The value a chain of addresses -- GEPs, field pointers, an address-of
    // -- bottoms out at: what a store through it writes into.
    static const Value &base_of(const std::shared_ptr<Value> &v) {
        const auto *instr = std::get_if<std::shared_ptr<Instruction>>(&v->data);
        if (instr != nullptr && !(*instr)->operands.empty() &&
            ((*instr)->op == Instruction::Op::GEP ||
             (*instr)->op == Instruction::Op::FieldPtr ||
             (*instr)->op == Instruction::Op::AddressOf)) {
            return base_of((*instr)->operands[0]);
        }
        return *v;
    }

    // The locals made where every thread of a block runs the same code (see
    // CodeGen_LLVM::effects_run_once): one per thread, as a thread's locals
    // are, so a store into one is that thread's own and is not an effect
    // made once for the block. A thread loop that reads one is refused (see
    // CodeGen_PTX::emit_bound_parfor), since that would be memory the block
    // shares.
    std::set<std::string> thread_private;

    void emit_instruction(const std::shared_ptr<Instruction> &instr) {
        // Where one thread of many makes the effects, an effect goes under
        // that thread's guard; everything else is a value every thread
        // computes for itself, a store into a thread's own local included.
        if (is_effect(instr->op)) {
            bool own_memory = false;
            if (instr->op != Instruction::Op::Print) {
                const Value &base = base_of(instr->operands[0]);
                const auto *base_instr =
                    std::get_if<std::shared_ptr<Instruction>>(&base.data);
                own_memory = base_instr != nullptr &&
                             thread_private.count((*base_instr)->name) != 0;
            }
            if (!own_memory) {
                if (llvm::Value *guard = cg.effects_once_guard()) {
                    cg.emit_if(guard,
                               [&] { emit_instruction_unguarded(instr); });
                    return;
                }
            }
        }
        emit_instruction_unguarded(instr);
    }

    void emit_instruction_unguarded(const std::shared_ptr<Instruction> &instr) {
        // Pure address and shape helpers, written out at their uses by
        // operand() rather than bound to a name of their own.
        if (instr->op == Instruction::Op::GEP ||
            instr->op == Instruction::Op::FieldPtr ||
            instr->op == Instruction::Op::Ramp) {
            return;
        }
        if (cg.effects_run_once()) {
            if (instr->op == Instruction::Op::Alloca ||
                instr->op == Instruction::Op::Alloc) {
                thread_private.insert(instr->name);
            } else if (instr->op == Instruction::Op::AtomicAdd ||
                       instr->op == Instruction::Op::Append) {
                // A value fetched here would exist in the thread that fetched
                // it only; handing it to the rest is shared memory, which is
                // how a block's threads hold one thing between them.
                internal_error
                    << "[unimplemented] " << op_name(instr->op) << " `"
                    << instr->name << "` in the body of a loop bound to "
                    << "GPUBlock, outside its thread loop. A block's threads "
                    << "all run this code, and what this fetches has to be one "
                    << "thing the block shares -- shared memory, which the PTX "
                    << "backend does not allocate yet. Move it into the thread "
                    << "loop, or out of the block loop.";
            }
        }
        if (instr->op == Instruction::Op::Print) {
            cg.codegen_stmt(Print::make(operands(*instr)));
            return;
        }
        if (instr->op == Instruction::Op::Alloca ||
            instr->op == Instruction::Op::Alloc) {
            // A local the promotion pass could not lift to a register -- an
            // address-taken or dynamically-indexed one -- kept as storage. It
            // is declared the way the relooper declares it (CodeGen_Stmt.cpp):
            // the type is what the pointer points at, the memory is the stack
            // for Alloca and the heap for Alloc, and the Allocate binds the
            // name to the storage itself, so nothing binds it here. Its initial
            // value, if any, is a separate Store the builder split off.
            internal_assert(instr->operands.empty()) << instr->operands.size();
            Type allocated = instr->type;
            if (!allocated.is_reference()) {
                const Ptr_t *ptr_t = instr->type.as<Ptr_t>();
                internal_assert(ptr_t)
                    << "Alloc(a) must have pointer type: " << instr->type;
                allocated = ptr_t->etype;
            }
            const Allocate::Memory memory =
                instr->op == Instruction::Op::Alloca ? Allocate::Stack
                                                     : Allocate::Heap;
            cg.codegen_stmt(
                Allocate::make(WriteLoc(instr->name, allocated), memory));
            return;
        }
        if (instr->op == Instruction::Op::AccAdd ||
            instr->op == Instruction::Op::AccMul ||
            instr->op == Instruction::Op::AccSub ||
            instr->op == Instruction::Op::AccMin ||
            instr->op == Instruction::Op::AccMax ||
            instr->op == Instruction::Op::AccArgmin ||
            instr->op == Instruction::Op::AccArgmax) {
            // A read-modify-write of a place, the same as the relooper builds
            // (CodeGen_Stmt.cpp): its atomicity is carried on the instruction,
            // and a vector value written into a scalar place is a cross-lane
            // reduction the Accumulate lowering performs.
            internal_assert(instr->operands.size() == 2 ||
                            instr->operands.size() == 3)
                << "accumulate takes a place, a value and maybe a mask";
            WriteLoc loc = location(instr->operands[0]);
            Expr val = operand(instr->operands[1]);
            const Accumulate::OpType op =
                instr->op == Instruction::Op::AccAdd ? Accumulate::OpType::Add
                : instr->op == Instruction::Op::AccMul
                    ? Accumulate::OpType::Mul
                : instr->op == Instruction::Op::AccSub
                    ? Accumulate::OpType::Sub
                : instr->op == Instruction::Op::AccMin
                    ? Accumulate::OpType::Min
                : instr->op == Instruction::Op::AccMax
                    ? Accumulate::OpType::Max
                : instr->op == Instruction::Op::AccArgmin
                    ? Accumulate::OpType::Argmin
                    : Accumulate::OpType::Argmax;
            if (instr->operands.size() == 3) {
                // Made by some lanes only. Into per-lane memory -- the value
                // is then gang-wide, one per slot -- it is a read, a combine
                // and a store of the lanes that are on; into shared memory it
                // is the one accumulate, made if any lane is on, of a value
                // that already folded the identity in for the lanes that are
                // not (see widen_region in SSA/Vectorize.cpp).
                Expr mask = operand(instr->operands[2]);
                if (!mask.type().is_vector()) {
                    // The gang's one accumulate into shared memory, of a
                    // value the vectorizer has already folded the off lanes
                    // out of; the mask is the one bool saying some lane was
                    // on (see widen_region in SSA/Vectorize.cpp). The value
                    // may well be a short vector, and is not a lane per
                    // component.
                    cg.emit_if(cg.codegen_expr(mask), [&] {
                        cg.codegen_stmt(Accumulate::make(
                            std::move(loc), op, std::move(val), instr->atomic));
                    });
                    return;
                }
                if (val.type().is_vector() || val.type().is<Struct_t>()) {
                    internal_assert(op != Accumulate::OpType::Argmin &&
                                    op != Accumulate::OpType::Argmax)
                        << "[unimplemented] a masked argmin/argmax into "
                        << "per-lane memory: " << instr->name;
                    Expr place = operand(instr->operands[0]);
                    if (instr->atomic) {
                        // Indivisible per lane: each lane that is on updates
                        // its own place with its own value, as a scatter of
                        // atomics (see CodeGen_LLVM::emit_atomic_lanes).
                        cg.emit_atomic_lanes(op, val.type(),
                                             cg.codegen_expr(place),
                                             cg.codegen_expr(val),
                                             cg.codegen_expr(mask));
                        return;
                    }
                    Expr current = Deref::make(
                        place, place.type().is_vector() ? mask : Expr());
                    Expr combined =
                        op == Accumulate::OpType::Add ? current + val
                        : op == Accumulate::OpType::Sub ? current - val
                        : op == Accumulate::OpType::Mul ? current * val
                        : op == Accumulate::OpType::Min
                            ? Intrinsic::make(Intrinsic::OpType::min,
                                              {current, val})
                            : Intrinsic::make(Intrinsic::OpType::max,
                                              {current, val});
                    cg.codegen_stmt(Store::make(std::move(loc),
                                                std::move(combined),
                                                std::move(mask)));
                    return;
                }
                cg.emit_if_any_lane(cg.codegen_expr(mask), [&] {
                    cg.codegen_stmt(Accumulate::make(std::move(loc), op,
                                                     std::move(val),
                                                     instr->atomic));
                });
                return;
            }
            cg.codegen_stmt(Accumulate::make(std::move(loc), op, std::move(val),
                                             instr->atomic));
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
            cg.codegen_stmt(Store::make(std::move(loc), std::move(val),
                                        std::move(mask), instr->compact));
            return;
        }
        Expr value = value_of(*instr);
        internal_assert(value.defined()) << "No value for " << instr->name
                                         << " (" << op_name(instr->op) << ")";
        bind(instr->name, cg.codegen_expr(value));
    }

    // Bind a name in the current frame, shadowing any outer one rather than
    // overwriting it in place.
    //
    // Block arguments share names across blocks on purpose -- that is how a
    // value is threaded onwards -- and only the block that declares one can
    // refer to it, so the binding that matters is always the most recent. An
    // instruction's name is unique to the function and never collides. Adding
    // to the current frame (rather than replacing wherever the name already is)
    // is what lets a nested region -- the body of a serial loop, a kernel --
    // rebind those shared names for its own blocks and, when it is popped,
    // restore what the enclosing region had. Overwriting in place instead left
    // a loop's block arguments standing for the continuation after it, which
    // referred to values defined inside the loop and did not dominate their
    // uses.
    void bind(const std::string &name, llvm::Value *value) {
        cg.frames.set_local(name, value);
    }

    // Hand a block its arguments, as the values its predecessors pass.
    void bind_arguments(const Block &block) {
        const auto found = phis.find(block.name);
        if (found == phis.end()) {
            return;
        }
        for (size_t i = 0; i < block.args.size(); i++) {
            bind(block.args[i].name, found->second[i].value);
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
        // Said here, with the block and the argument named, rather than by
        // LLVM's assertion that a phi's operands must be of its type.
        auto matching = [&](llvm::PHINode *phi, llvm::Value *value, size_t i,
                            const std::string &what) {
            if (phi->getType() == value->getType()) {
                return;
            }
            std::string want, got;
            llvm::raw_string_ostream want_os(want), got_os(got);
            phi->getType()->print(want_os);
            value->getType()->print(got_os);
            internal_error << "Argument " << i << " of " << jump.name
                           << " is " << want_os.str() << " but " << what
                           << " from " << from->getName().str() << " is "
                           << got_os.str();
        };
        // Each leaf of the value to the leaf's phi, extracted here in the
        // block the edge leaves from.
        auto feed = [&](const ArgPhis &arg, llvm::Value *value, size_t i,
                        const std::string &what) {
            for (const auto &[phi, path] : arg.leaves) {
                llvm::Value *leaf =
                    path.empty() ? value
                                 : cg.builder->CreateExtractValue(value, path);
                matching(phi, leaf, i, what);
                phi->addIncoming(leaf, from);
            }
        };
        if (before != nullptr) {
            internal_assert(!target.empty())
                << jump.name << " takes no value from the call before it";
            feed(target[0], before, 0, "the value the call returns");
        }
        internal_assert(jump.args.size() + first == target.size())
            << "Jump to " << jump.name << " passes "
            << (jump.args.size() + first) << " arguments but it takes "
            << target.size();
        for (size_t i = 0; i < jump.args.size(); i++) {
            llvm::Value *value = cg.codegen_expr(operand(jump.args[i]));
            feed(target[first + i], value, first + i, "the value passed");
        }
    }

    // The blocks a terminator leads to at this nesting level. A parfor's body
    // is a nested region, emitted when the parfor itself is reached, so only
    // its continuation is followed here; a Yield ends an iteration and leads
    // nowhere.
    std::vector<std::string> level_successors(const std::string &name) {
        std::vector<std::string> out;
        std::visit(ir::ssa::overloads{
                       [&](const std::monostate &) {},
                       [&](const Terminator::Jump &j) { out.push_back(j.name); },
                       [&](const Terminator::Dispatch &d) {
                           for (const auto &t : d.targets) {
                               out.push_back(t.name);
                           }
                       },
                       [&](const Terminator::Return &) {},
                       [&](const Terminator::ParFor &p) {
                           out.push_back(p.cont.name);
                       },
                       [&](const Terminator::Yield &) {},
                       [&](const Terminator::Call &c) {
                           out.push_back(c.cont.name);
                       },
                       [&](const Terminator::MultiCall &c) {
                           out.push_back(c.cont.name);
                       },
                   },
                   by_name.at(name)->terminator.data);
        return out;
    }

    // Every block at one nesting level reachable from `entry`: the top of a
    // function, or one parfor body. Nested parfor bodies are their own regions
    // and are not included.
    std::set<std::string> region_of(const std::string &entry) {
        std::set<std::string> region;
        std::vector<std::string> work{entry};
        while (!work.empty()) {
            const std::string n = work.back();
            work.pop_back();
            if (!region.insert(n).second) {
                continue;
            }
            for (const std::string &s : level_successors(n)) {
                work.push_back(s);
            }
        }
        return region;
    }

    // Reverse postorder within one region, so a definition is emitted before
    // its uses the way the whole function used to be walked.
    std::vector<std::string> region_rpo(const std::string &entry,
                                        const std::set<std::string> &region) {
        std::vector<std::string> post;
        std::set<std::string> visited{entry};
        std::vector<std::pair<std::string, size_t>> stack{{entry, 0}};
        while (!stack.empty()) {
            const std::string name = stack.back().first;
            const std::vector<std::string> succ = level_successors(name);
            if (stack.back().second < succ.size()) {
                const std::string s = succ[stack.back().second++];
                if (region.count(s) && visited.insert(s).second) {
                    stack.push_back({s, 0});
                }
            } else {
                post.push_back(name);
                stack.pop_back();
            }
        }
        std::reverse(post.begin(), post.end());
        return post;
    }

    // Emits one region into the current LLVM function. `entry_bb` is the block
    // its entry is emitted into, already created and with the entry's arguments
    // already bound by the caller; every other block of the region is made
    // here. `yield_to` is where a Yield in the region branches, or null to
    // return -- see `yield_targets`. The block and phi maps are swapped for the
    // duration, since a region emitted into a kernel has its own LLVM blocks.
    void emit_region(const std::string &entry_name,
                     const std::set<std::string> &region,
                     llvm::BasicBlock *entry_bb, llvm::BasicBlock *yield_to) {
        auto saved_blocks = std::move(blocks);
        auto saved_phis = std::move(phis);
        blocks.clear();
        phis.clear();
        blocks[entry_name] = entry_bb;
        // A frame of this region's own, so its block arguments and instruction
        // results -- shared names and all -- shadow the enclosing region's and
        // are gone when it returns. See bind().
        cg.frames.push_frame();

        const std::vector<std::string> order = region_rpo(entry_name, region);
        for (const std::string &name : order) {
            if (name == entry_name) {
                continue;
            }
            blocks[name] = llvm::BasicBlock::Create(*cg.context, name,
                                                    cg.current_function);
        }
        for (const std::string &name : order) {
            const Block &block = *by_name.at(name);
            if (name == entry_name || block.args.empty()) {
                continue;
            }
            cg.builder->SetInsertPoint(blocks.at(name));
            std::vector<ArgPhis> made(block.args.size());
            // Every phi first -- a block's phis lead it -- and then the
            // aggregates rebuilt from theirs.
            for (size_t i = 0; i < block.args.size(); i++) {
                std::vector<unsigned> path;
                make_leaf_phis(cg.codegen_type(block.args[i].type),
                               block.args[i].name, path, made[i].leaves);
            }
            for (size_t i = 0; i < block.args.size(); i++) {
                made[i].value = rebuild(cg.codegen_type(block.args[i].type),
                                        made[i], block.args[i].name);
            }
            phis[name] = std::move(made);
        }

        // Emitted along the dominator tree, with a frame per block that is
        // popped once the blocks it dominates are done. A block argument is
        // referred to by name, and two blocks may declare the same name -- an
        // inner loop's header carries the outer's `a!loop` onwards under that
        // very name -- so a reference has to resolve to the declaration in the
        // nearest block that dominates the use, which is what a block reached
        // by a different path from the inner loop's exit must see. Walking the
        // tree makes the most recent binding exactly that declaration, where
        // walking in reverse postorder made it whichever block came later.
        // Every definition still precedes its uses: a definition dominates
        // them, so its block is an ancestor of theirs.
        // The region's graph over dense ids, in name order (see
        // SSA/Analysis.h), with the level's edges rather than the blocks'
        // own.
        const std::vector<std::string> names(region.begin(), region.end());
        std::unordered_map<std::string, ir::ssa::BlockId> ids;
        for (size_t i = 0; i < names.size(); i++) {
            ids[names[i]] = ir::ssa::BlockId(i);
        }
        std::vector<std::vector<ir::ssa::BlockId>> succs(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            for (const std::string &s : level_successors(names[i])) {
                const auto to = ids.find(s);
                if (to != ids.end()) {
                    succs[i].push_back(to->second);
                }
            }
        }
        const ir::ssa::DomTree dom = ir::ssa::compute_dominator_tree(
            ir::ssa::Graph::from_successors(std::move(succs),
                                            ids.at(entry_name)));

        yield_targets.push_back(yield_to);
        std::function<void(ir::ssa::BlockId)> emit =
            [&](ir::ssa::BlockId b) {
                cg.frames.push_frame();
                const std::string &name = names[b];
                const Block &block = *by_name.at(name);
                cg.builder->SetInsertPoint(blocks.at(name));
                bind_arguments(block);
                for (const auto &instr : block.instrs) {
                    emit_instruction(instr);
                }
                emit_terminator(block);
                for (ir::ssa::BlockId child : dom.children(b)) {
                    emit(child);
                }
                cg.frames.pop_frame();
            };
        emit(ids.at(entry_name));
        yield_targets.pop_back();

        cg.frames.pop_frame();
        blocks = std::move(saved_blocks);
        phis = std::move(saved_phis);
    }

    // A parfor a schedule left serial: an ordinary counted loop, with the body
    // region as the loop body and a Yield as the back edge. The captures the
    // body takes as its uniform arguments are loop-invariant, so they are bound
    // once to the values the parent passes.
    void emit_serial_parfor(const Terminator::ParFor &p,
                            llvm::BasicBlock *preheader,
                            const std::set<std::string> &body_region,
                            const Block &body_head) {
        const Expr begin_e = operand(p.start), end_e = operand(p.end),
                   stride_e = operand(p.stride);
        llvm::Value *begin_v = cg.codegen_expr(begin_e);
        llvm::Type *idx_t = cg.codegen_type(begin_e.type());
        const std::string &idx_name = body_head.args[0].name;
        const std::string id =
            p.index + std::to_string(cg.forall_loop_id++) + "_pf";

        llvm::BasicBlock *header = llvm::BasicBlock::Create(
            *cg.context, id + "_head", cg.current_function);
        llvm::BasicBlock *body_bb = llvm::BasicBlock::Create(
            *cg.context, p.body.name, cg.current_function);
        llvm::BasicBlock *latch = llvm::BasicBlock::Create(
            *cg.context, id + "_latch", cg.current_function);
        llvm::BasicBlock *exit = llvm::BasicBlock::Create(
            *cg.context, id + "_exit", cg.current_function);

        cg.builder->CreateBr(header);
        cg.builder->SetInsertPoint(header);
        llvm::PHINode *phi = cg.builder->CreatePHI(idx_t, 2, idx_name);
        phi->addIncoming(begin_v, preheader);

        cg.frames.push_frame();
        bind(idx_name, phi);
        // The body's arguments are its index (arg 0, supplied by the loop) and
        // then its uniform captures; the body jump passes only the captures, so
        // it is one shorter and its i-th value feeds argument i + 1.
        for (size_t i = 1; i < body_head.args.size(); i++) {
            bind(body_head.args[i].name,
                 cg.codegen_expr(operand(p.body.args[i - 1])));
        }
        // A loop is expected to be entered, as the statement path's counted
        // loop says of itself.
        llvm::Value *test =
            cg.codegen_expr(Var::make(begin_e.type(), idx_name) < end_e);
        cg.builder->CreateCondBr(test, body_bb, exit, cg.very_likely_branch);

        cg.builder->SetInsertPoint(body_bb);
        emit_region(p.body.name, body_region, body_bb, latch);

        cg.builder->SetInsertPoint(latch);
        llvm::Value *next =
            cg.codegen_expr(Var::make(begin_e.type(), idx_name) + stride_e);
        phi->addIncoming(next, latch);
        cg.builder->CreateBr(header);
        cg.frames.pop_frame();

        cg.builder->SetInsertPoint(exit);
    }

    // The loop `p` as the code generator placing it sees it (see
    // CodeGen_LLVM::BoundLoop): its blocks, and this lowering's operand
    // reading, name binding and region emission, lent out as closures.
    CodeGen_LLVM::BoundLoop bound_loop(const Terminator::ParFor &p) {
        const std::set<std::string> body_region = region_of(p.body.name);
        const Block &body_head = *by_name.at(p.body.name);
        internal_assert(!body_head.args.empty())
            << "parfor body " << p.body.name << " takes no index argument";
        return CodeGen_LLVM::BoundLoop{
            func,
            p,
            body_head,
            [this](const std::shared_ptr<Value> &v) { return operand(v); },
            [this](const std::string &name, llvm::Value *value) {
                bind(name, value);
            },
            [this, p, body_region](llvm::BasicBlock *entry,
                                   llvm::BasicBlock *yield_to) {
                emit_region(p.body.name, body_region, entry, yield_to);
            },
        };
    }

    // A parfor terminator: the loop it stands for, then its continuation.
    void emit_parfor(const Terminator::ParFor &p) {
        llvm::BasicBlock *preheader = cg.builder->GetInsertBlock();

        if (p.binding.has_value()) {
            // Placed on hardware: what that means is the code generator's
            // to say -- a call into the thread runtime, a kernel launch, or
            // the loop of a kernel's threads (see emit_bound_parfor).
            CodeGen_LLVM::BoundLoop loop = bound_loop(p);
            cg.emit_bound_parfor(loop);
        } else {
            const std::set<std::string> body_region = region_of(p.body.name);
            const Block &body_head = *by_name.at(p.body.name);
            internal_assert(!body_head.args.empty())
                << "parfor body " << p.body.name << " takes no index argument";
            emit_serial_parfor(p, preheader, body_region, body_head);
        }

        llvm::BasicBlock *from = cg.builder->GetInsertBlock();
        supply(p.cont, from, 0);
        cg.builder->CreateBr(blocks.at(p.cont.name));
    }

    // Emits the body of `p` as the whole of the current function, `kernel`:
    // the entry block, `prologue` binding the body's arguments, then the
    // body's blocks. See CodeGen_LLVM::compile_kernel_body.
    void run_kernel(const Terminator::ParFor &p,
                    const std::function<void(CodeGen_LLVM::BoundLoop &)>
                        &prologue) {
        for (const auto &block : func.blocks) {
            by_name[block->name] = block.get();
        }
        cg.frames.push_frame();
        cg.current_sret = nullptr;
        llvm::BasicBlock *entry_bb =
            llvm::BasicBlock::Create(*cg.context, p.body.name, function);
        cg.builder->SetInsertPoint(entry_bb);
        CodeGen_LLVM::BoundLoop loop = bound_loop(p);
        prologue(loop);
        loop.emit_body(entry_bb, nullptr);
        cg.frames.pop_frame();
    }

    // Where the function's parameters are used: on the host -- in any block
    // that is not inside a loop bound to the GPU -- or on the device, as a
    // capture of such a loop's body. By name: a parameter threaded onwards
    // as a block argument keeps its name, so the name is the value. A
    // parameter may be in both sets; one in neither is unused.
    struct Uses {
        std::set<std::string> host;
        std::set<std::string> device;
    };
    Uses classify_uses() {
        Uses uses;
        const auto note_host = [&](const std::shared_ptr<Value> &v) {
            if (const auto *a = std::get_if<Argument>(&v->data)) {
                uses.host.insert(a->name);
            }
        };
        // A jump's arguments: a parameter handed to the target under its own
        // name is threading -- the value flowing past a branch or a join,
        // which the builder does for every value the code beyond needs --
        // and not a read of it on the host. Counting it as one made every
        // buffer of a function with a branch before its launch "both", and
        // so marked the kernel's outputs host-dirty on every call. Handed
        // under another name it is a use like any other, since the reads
        // beyond go by that name. `first` is where the target's argument
        // list the jump's own values start (a call continuation is handed
        // the returned value before them); a jump to a block this function
        // does not have -- a call's jump to its callee -- is all uses.
        const auto note_jump = [&](const Terminator::Jump &j, size_t first) {
            const auto target = by_name.find(j.name);
            for (size_t k = 0; k < j.args.size(); k++) {
                const auto *a = std::get_if<Argument>(&j.args[k]->data);
                if (a == nullptr) {
                    continue;
                }
                const size_t at = k + first;
                const bool threaded = target != by_name.end() &&
                                      at < target->second->args.size() &&
                                      target->second->args[at].name == a->name;
                if (!threaded) {
                    uses.host.insert(a->name);
                }
            }
        };
        std::set<std::string> seen;
        std::vector<std::string> work{entry()};
        while (!work.empty()) {
            const std::string name = work.back();
            work.pop_back();
            if (!seen.insert(name).second) {
                continue;
            }
            const auto found = by_name.find(name);
            if (found == by_name.end()) {
                continue;
            }
            const Block &block = *found->second;
            for (const auto &instr : block.instrs) {
                for (const auto &operand : instr->operands) {
                    note_host(operand);
                }
            }
            std::visit(
                ir::ssa::overloads{
                    [](const std::monostate &) {},
                    [&](const Terminator::Jump &j) {
                        note_jump(j, 0);
                        work.push_back(j.name);
                    },
                    [&](const Terminator::Dispatch &d) {
                        note_host(d.cond);
                        for (const auto &t : d.targets) {
                            note_jump(t, 0);
                            work.push_back(t.name);
                        }
                    },
                    [&](const Terminator::Return &r) {
                        if (r.value) {
                            note_host(r.value);
                        }
                    },
                    [&](const Terminator::Call &c) {
                        for (const auto &a : c.call.args) {
                            note_host(a);
                        }
                        note_jump(c.cont, c.drop ? 0 : 1);
                        work.push_back(c.cont.name);
                    },
                    [&](const Terminator::MultiCall &c) {
                        for (const auto &a : c.call.args) {
                            note_host(a);
                        }
                        for (const auto &one : c.varying) {
                            for (const auto &a : one) {
                                note_host(a);
                            }
                        }
                        note_jump(c.cont, c.drop ? 0 : 1);
                        work.push_back(c.cont.name);
                    },
                    [&](const Terminator::ParFor &p) {
                        note_host(p.start);
                        note_host(p.end);
                        note_host(p.stride);
                        note_jump(p.cont, 0);
                        work.push_back(p.cont.name);
                        const bool gpu =
                            p.binding.has_value() &&
                            (*p.binding == Resource::GPUBlock ||
                             *p.binding == Resource::GPUThread);
                        if (gpu) {
                            // The body is a kernel; what it is handed is
                            // read on the device, by the launch.
                            for (const auto &a : p.body.args) {
                                if (const auto *arg =
                                        std::get_if<Argument>(&a->data)) {
                                    uses.device.insert(arg->name);
                                }
                            }
                            return;
                        }
                        note_jump(p.body, 1);
                        work.push_back(p.body.name);
                    },
                    [](const Terminator::Yield &) {},
                },
                block.terminator.data);
        }
        return uses;
    }

    void run() {
        // Only what the entry can reach, in dominance order: a definition
        // dominates its uses, so reverse postorder binds a value before it is
        // read. The blocks inside a parfor body are their own region, emitted
        // as the loop or kernel that parfor becomes, so the top level walks
        // only the blocks outside every parfor.
        for (const auto &block : func.blocks) {
            by_name[block->name] = block.get();
        }

        cg.frames.push_frame();
        llvm::BasicBlock *entry_bb =
            llvm::BasicBlock::Create(*cg.context, entry(), function);
        cg.builder->SetInsertPoint(entry_bb);

        // The hidden return pointer, when there is one, comes first and is no
        // parameter of the entry block's. The entry's other arguments are the
        // function's own, so they are already values rather than phis.
        cg.current_sret = nullptr;
        if (function->hasParamAttribute(0, llvm::Attribute::StructRet)) {
            cg.current_sret = function->getArg(0);
            cg.current_sret->setName("_sret");
        }
        const Block &head = *func.blocks.front();
        // An exported function's arrays come in as buffer descriptors (see
        // CodeGen_LLVM::ExportedBuffer): the prologue asks for each one the
        // host-side code uses on the host, and binds the parameter's name
        // to the pointer it gets back. One only a kernel reads is left to
        // the launch, which asks for it on the device; its name is bound to
        // poison, so that a host use the analysis missed is a visible fault
        // in the IR rather than a read of the descriptor as data.
        const bool exported =
            std::find(func.attributes.begin(), func.attributes.end(),
                      ir::Function::Attribute::exported) !=
            func.attributes.end();
        const Uses uses = exported ? classify_uses() : Uses{};
        std::vector<uint8_t> sides;
        cg.exported_buffers.clear();
        uint32_t i = 0;
        for (auto &arg : function->args()) {
            if (&arg == cg.current_sret) {
                continue;
            }
            internal_assert(i < head.args.size())
                << function->getName().str() << " takes more arguments "
                << "than its entry block declares";
            const Argument &declared = head.args[i];
            i++;
            const Struct_t *layout =
                exported ? CodeGen_LLVM::layout_struct_of(declared.type)
                         : nullptr;
            if (!exported ||
                (!declared.type.is<Array_t>() && layout == nullptr)) {
                arg.setName(declared.name);
                bind(declared.name, &arg);
                continue;
            }
            arg.setName(declared.name + "_buffer");
            CodeGen_LLVM::ExportedBuffer buffer;
            buffer.descriptor = &arg;
            buffer.type = declared.type;
            buffer.mutating = declared.mutating;
            buffer.host_used = uses.host.count(declared.name) != 0;
            buffer.layout = layout;
            const bool device_used = uses.device.count(declared.name) != 0;
            const uint8_t side =
                (buffer.host_used ? 1 : 0) | (device_used ? 2 : 0);
            // A layout struct is as many buffers as it has array fields, all
            // needed where the struct is.
            sides.insert(sides.end(),
                         layout ? CodeGen_LLVM::layout_buffer_count(layout) : 1,
                         side);
            llvm::Value *bound = nullptr;
            if (buffer.host_used && layout != nullptr) {
                bound = cg.unwrap_layout(&arg, layout, /*device=*/false,
                                         buffer.mutating, declared.name);
            } else if (buffer.host_used) {
                bound = cg.buffer_require(&arg, /*device=*/false);
                bound->setName(declared.name);
                if (buffer.mutating) {
                    // The host may write it from here on; a device copy, if
                    // one exists, is stale until the next require there.
                    cg.buffer_mark_dirty(&arg, /*device=*/false);
                }
            } else {
                bound = llvm::PoisonValue::get(arg.getType());
            }
            bind(declared.name, bound);
            cg.exported_buffers[declared.name] = buffer;
        }
        if (exported) {
            cg.sides_of_exported[function->getName().str()] = sides;
        }
        // A function that seeds the random generator does so first, as the
        // statement path does: the state is a local the body's `rand` reads.
        if (std::find(func.attributes.begin(), func.attributes.end(),
                      ir::Function::Attribute::setup_rng) !=
            func.attributes.end()) {
            cg.emit_rng_setup();
        }

        emit_region(entry(), region_of(entry()), entry_bb, nullptr);

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
                    llvm::Value *cond = cg.codegen_expr(operand(d.cond));
                    // Once per target rather than once per distinct block:
                    // two targets on the same block are two edges, and a phi
                    // there takes a value along each.
                    for (const Terminator::Jump &target : d.targets) {
                        supply(target, from, 0);
                    }
                    if (cond->getType()->isIntegerTy(1)) {
                        internal_assert(d.targets.size() == 2)
                            << "A dispatch on a bool goes two ways";
                        // targets[0] is where a false condition goes.
                        cg.builder->CreateCondBr(cond,
                                                 blocks.at(d.targets[1].name),
                                                 blocks.at(d.targets[0].name));
                        return;
                    }
                    // A switch: target k on k, the last target on anything
                    // else (see ir::SwitchStmt).
                    auto *int_type =
                        llvm::dyn_cast<llvm::IntegerType>(cond->getType());
                    internal_assert(int_type)
                        << "Dispatch on a non-integer in " << from->getName().str();
                    llvm::SwitchInst *sw = cg.builder->CreateSwitch(
                        cond, blocks.at(d.targets.back().name),
                        static_cast<unsigned>(d.targets.size() - 1));
                    for (size_t k = 0; k + 1 < d.targets.size(); k++) {
                        sw->addCase(llvm::ConstantInt::get(int_type, k),
                                    blocks.at(d.targets[k].name));
                    }
                },
                [&](const Terminator::Return &r) {
                    if (!r.value) {
                        cg.builder->CreateRetVoid();
                        return;
                    }
                    llvm::Value *value = cg.codegen_expr(operand(r.value));
                    if (cg.current_sret) {
                        // Returned through the hidden pointer; see
                        // indirect_return_type.
                        cg.builder->CreateStore(value, cg.current_sret);
                        cg.builder->CreateRetVoid();
                    } else {
                        cg.builder->CreateRet(value);
                    }
                },
                [&](const Terminator::Call &c) {
                    llvm::Function *callee =
                        cg.module->getFunction(c.call.name);
                    internal_assert(callee)
                        << "Call to undeclared function " << c.call.name;
                    cg.check_block_level_call(c.call.name);
                    std::vector<llvm::Value *> args;
                    for (const auto &a : c.call.args) {
                        args.push_back(cg.codegen_expr(operand(a)));
                    }
                    // A call made under a mask is made only if some lane is
                    // on, and that is settled in the SSA: the call sits
                    // behind a test of its mask, either the linearizer's
                    // guard around the arm it is in or one of its own (see
                    // specialize_calls in SSA/Vectorize.cpp), so here it is
                    // simply made.
                    llvm::Value *result = cg.emit_call(callee, std::move(args));
                    llvm::BasicBlock *after = cg.builder->GetInsertBlock();
                    supply(c.cont, after, c.drop ? 0 : 1,
                           c.drop ? nullptr : result);
                    cg.builder->CreateBr(blocks.at(c.cont.name));
                },
                [&](const Terminator::MultiCall &c) {
                    // Where a run of recursive calls finally becomes several
                    // calls. Nothing after this point can reorder them, which
                    // is the whole reason the run was kept together until now:
                    // the schedule has already decided the order, and this
                    // just spells it out.
                    llvm::Function *callee =
                        cg.module->getFunction(c.call.name);
                    internal_assert(callee)
                        << "Call to undeclared function " << c.call.name;
                    cg.check_block_level_call(c.call.name);
                    internal_assert(c.drop || c.varying.size() == 1)
                        << block.name << " keeps the result of a run of "
                        << c.varying.size()
                        << " calls, but a run has one continuation and so at "
                           "most one result to give it.";
                    // Under a mask the run is made only if some lane is on,
                    // and as with a Call (above) the test is the SSA's:
                    // every call of the run shares the mask, so one test in
                    // front of the run guards them all.
                    llvm::Value *result = nullptr;
                    for (size_t i = 0; i < c.varying.size(); i++) {
                        std::vector<llvm::Value *> args;
                        for (const auto &a : c.call_args(i)) {
                            args.push_back(cg.codegen_expr(operand(a)));
                        }
                        result = cg.emit_call(callee, std::move(args));
                    }
                    llvm::BasicBlock *after = cg.builder->GetInsertBlock();
                    supply(c.cont, after, c.drop ? 0 : 1,
                           c.drop ? nullptr : result);
                    cg.builder->CreateBr(blocks.at(c.cont.name));
                },
                [&](const Terminator::ParFor &p) { emit_parfor(p); },
                [&](const Terminator::Yield &) {
                    // The end of one parfor iteration: the back edge of the
                    // loop it became, or a return from its kernel (the null
                    // target). See yield_targets.
                    internal_assert(!yield_targets.empty())
                        << block.name << " yields outside any parfor body";
                    llvm::BasicBlock *target = yield_targets.back();
                    if (target != nullptr) {
                        cg.builder->CreateBr(target);
                    } else {
                        cg.builder->CreateRetVoid();
                    }
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
    lowering_from_ssa = true;

    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    SSALowering(*this, func, function).run();
    builder->restoreIP(here);

    lowering_from_ssa = false;

    internal_assert(!llvm::verifyFunction(*function, &llvm::errs()))
        << "Function verification failed for " << function->getName().str()
        << ", lowered straight from SSA";

    current_function = nullptr;
    current_sret = nullptr;
}

void CodeGen_LLVM::compile_kernel_body(
    const ir::ssa::Function &func, const Terminator::ParFor &loop,
    llvm::Function *kernel,
    const std::function<void(BoundLoop &)> &prologue) {
    internal_assert(current_function == nullptr);
    internal_assert(kernel);
    current_function = kernel;
    lowering_from_ssa = true;

    llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
    SSALowering(*this, func, kernel).run_kernel(loop, prologue);
    builder->restoreIP(here);

    lowering_from_ssa = false;

    internal_assert(!llvm::verifyFunction(*kernel, &llvm::errs()))
        << "Function verification failed for " << kernel->getName().str()
        << ", the body of the loop over " << loop.index << " as a kernel";

    current_function = nullptr;
    current_sret = nullptr;
}

void CodeGen_LLVM::emit_bound_parfor(BoundLoop &loop) {
    switch (*loop.loop.binding) {
    case Resource::CPUThread:
        emit_cpu_parfor(loop);
        return;
    case Resource::GPUBlock:
    case Resource::GPUThread:
        // A program with a loop on the GPU is compiled by CodeGen_GPU_Host,
        // which answers this; reaching here means make_llvm_codegen was not
        // asked, or a device generator met a loop it does not run.
        internal_error << "bind(" << loop.loop.index << ", "
                       << to_string(*loop.loop.binding)
                       << "): this code generator has no GPU to place the "
                       << "loop on.";
    case Resource::RTCore:
    case Resource::OptixThread:
        internal_error << "bind(" << loop.loop.index << ", "
                       << to_string(*loop.loop.binding)
                       << "): the OptiX backend is not built yet.";
    case Resource::TextureUnit:
        internal_error << "bind(" << loop.loop.index << ", TextureUnit): a "
                       << "texture unit runs a function, not a loop (see "
                       << "ir::Bind::lambda).";
    }
}

Expr CodeGen_LLVM::trip_count(const Expr &begin, const Expr &end,
                              const Expr &stride) {
    return (end - begin + (stride - make_const(stride.type(), 1))) / stride;
}

std::vector<size_t> CodeGen_LLVM::launch_captures(const Block &body_head) {
    std::vector<size_t> captures;
    for (size_t i = 1; i < body_head.args.size(); i++) {
        if (body_head.args[i].name != lower::rng_state_name) {
            captures.push_back(i);
        }
    }
    return captures;
}

bool CodeGen_LLVM::seeds_rng(const Block &body_head) {
    return std::any_of(body_head.args.begin() + 1, body_head.args.end(),
                       [](const ir::ssa::Argument &arg) {
                           return arg.name == lower::rng_state_name;
                       });
}

// A parfor a schedule bound to CPU threads: the body region is compiled as
// a kernel taking a context pointer and an iteration number, and the parent
// packs the captures into that context and calls `bonsai_parallel_for`. The
// captures are the body's uniform arguments, which the SSA already threads
// for us, plus the loop's begin and stride, which the kernel needs to turn
// its iteration number into the index.
void CodeGen_LLVM::emit_cpu_parfor(BoundLoop &loop) {
    const Terminator::ParFor &p = loop.loop;
    const Block &body_head = loop.body_head;
    const Expr begin_e = loop.operand(p.start), end_e = loop.operand(p.end),
               stride_e = loop.operand(p.stride);

    // The context: the loop's begin and stride, unless they are constants
    // the kernel can be given outright -- which they nearly always are, and
    // a kernel that loaded `0` and `1` from memory to multiply by would be
    // doing per iteration what the statement path never did -- and then the
    // body's uniform arguments.
    const auto is_constant = [](const Expr &e) {
        return e.as<IntImm>() != nullptr || e.as<UIntImm>() != nullptr;
    };
    const bool begin_constant = is_constant(begin_e);
    const bool stride_constant = is_constant(stride_e);
    std::vector<Expr> cap_exprs;
    Struct_t::Map fields;
    std::optional<size_t> begin_slot, stride_slot;
    if (!begin_constant) {
        begin_slot = cap_exprs.size();
        cap_exprs.push_back(begin_e);
        fields.push_back({"_begin", begin_e.type()});
    }
    if (!stride_constant) {
        stride_slot = cap_exprs.size();
        cap_exprs.push_back(stride_e);
        fields.push_back({"_stride", stride_e.type()});
    }
    // The body jump passes only the captures, not the index it takes as its
    // first argument, so its i-th value feeds body argument i + 1. The
    // random generator's state is not among them: it is thread-specific,
    // and each iteration seeds its own below (see emit_rng_setup(index)).
    const size_t first_capture = cap_exprs.size();
    const std::vector<size_t> captures = launch_captures(body_head);
    for (const size_t i : captures) {
        cap_exprs.push_back(loop.operand(p.body.args[i - 1]));
        fields.push_back({body_head.args[i].name, body_head.args[i].type});
    }
    const Type ctx_t =
        Struct_t::make("_pfctx" + std::to_string(forall_loop_id++), fields);
    auto *ctx_ll = llvm::cast<llvm::StructType>(codegen_type(ctx_t));

    // Parent side: pack the context and count the iterations.
    llvm::Value *ctx = create_alloca_at_entry(ctx_ll, "_pfctx");
    for (size_t i = 0; i < cap_exprs.size(); i++) {
        llvm::Value *slot = builder->CreateStructGEP(ctx_ll, ctx, i);
        builder->CreateStore(codegen_expr(cap_exprs[i]), slot);
    }
    llvm::Value *count = builder->CreateIntCast(
        codegen_expr(trip_count(begin_e, end_e, stride_e)), i64_t,
        begin_e.type().is_int());

    // The kernel: void(context*, i64 index).
    llvm::Type *ptr_t = llvm::PointerType::getUnqual(*context);
    llvm::FunctionType *kern_ty =
        llvm::FunctionType::get(void_t, {ptr_t, i64_t}, false);
    llvm::Function *kern = llvm::Function::Create(
        kern_ty, llvm::Function::InternalLinkage,
        "_pfkernel" + std::to_string(forall_loop_id++), module.get());
    // The kernel is reached only through the runtime's function pointer, so
    // it is entered on the sixteen-byte-aligned stack the C ABI promises and
    // nothing more. A gang is eight lanes wide, which is a 256-bit vector;
    // left to itself the optimizer widened the kernel's body to AVX-512 and
    // spilled 512-bit vectors, whose sixty-four-byte alignment the function
    // would have had to realign its own stack to provide and did not, so an
    // aligned store of one to a merely sixteen-aligned slot faulted. Capping
    // the width at the gang's own keeps the spills sixteen-aligned, which is
    // what the entry stack already is.
    kern->addFnAttr("frame-pointer", "all");

    {
        llvm::IRBuilderBase::InsertPoint here = builder->saveIP();
        llvm::Function *saved_fn = current_function;
        llvm::Value *saved_sret = current_sret;
        current_function = kern;
        current_sret = nullptr;
        frames.push_frame();

        llvm::BasicBlock *kentry =
            llvm::BasicBlock::Create(*context, p.body.name, kern);
        builder->SetInsertPoint(kentry);
        llvm::Value *kctx = kern->getArg(0);
        const auto load_field = [&](size_t i, const Type &t) {
            llvm::Value *slot = builder->CreateStructGEP(ctx_ll, kctx, i);
            return builder->CreateLoad(codegen_type(t), slot);
        };
        // The index: begin + n * stride, from the constants or the context.
        llvm::Value *begin_k = begin_slot ? load_field(*begin_slot, begin_e.type())
                                          : codegen_expr(begin_e);
        llvm::Value *stride_k = stride_slot
                                    ? load_field(*stride_slot, stride_e.type())
                                    : codegen_expr(stride_e);
        llvm::Value *idx_cast = builder->CreateIntCast(
            kern->getArg(1), codegen_type(begin_e.type()), false);
        llvm::Value *index = builder->CreateAdd(
            begin_k, builder->CreateMul(idx_cast, stride_k),
            body_head.args[0].name);
        loop.bind(body_head.args[0].name, index);
        for (size_t k = 0; k < captures.size(); k++) {
            const ir::ssa::Argument &arg = body_head.args[captures[k]];
            loop.bind(arg.name, load_field(first_capture + k, arg.type));
        }
        if (seeds_rng(body_head)) {
            emit_rng_setup(index, /*outer_state=*/nullptr);
        }
        loop.emit_body(kentry, nullptr);

        frames.pop_frame();
        current_sret = saved_sret;
        current_function = saved_fn;
        builder->restoreIP(here);
    }

    // One call: run the kernel over the iterations. See
    // runtime/bonsai_parallel.h and CodeGen_LLVM::visit(const Launch *).
    llvm::FunctionType *pf_ty =
        llvm::FunctionType::get(void_t, {i64_t, ptr_t, ptr_t}, false);
    llvm::Function *pf = module->getFunction("bonsai_parallel_for");
    if (!pf) {
        pf = llvm::Function::Create(pf_ty, llvm::Function::ExternalLinkage,
                                    "bonsai_parallel_for", module.get());
    }
    builder->CreateCall(pf, {count, ctx, kern});
}

} // namespace bonsai
