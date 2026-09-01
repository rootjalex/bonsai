#include "SSA/Convert.h"

#include "SSA/CodeGen_Stmt.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"
#include "Lower/Random.h"

#include "Utils.h"

#include <iostream>
#include <set>

namespace bonsai {
namespace ir {
namespace ssa {

struct FunctionBuilder : Visitor {
    std::shared_ptr<Value> value = nullptr;
    std::shared_ptr<Block> block = nullptr;

    std::shared_ptr<ssa::Function> function;

    // Names of mutable function arguments and locals (populated from `func`'s
    // args below, and as `Allocate` nodes are visited). Both are registered
    // in `lookups` under a pointer type -- function args because `arg.type`
    // is already `Ptr_t(original type)` by the time Mutability has run,
    // locals because `visit(const Allocate *)` below allocates a real
    // pointer to match how Lower/Mutability.cpp rewrites reads of them into
    // Deref(Var(Ptr_t(...), name)). WriteLoc::base_type in Store/Accumulate
    // nodes still refers to the pre-pointer element type, so lookups for
    // these names must ask for the pointer type instead.
    std::set<std::string> mut_names;

    FunctionBuilder(const ir::Function &func) {
        function = std::make_shared<Function>();
        block = std::make_shared<Block>();
        block->name = func.name;
        block->owner = function;

        for (const auto &arg : func.args) {
            internal_assert(!arg.default_value.defined())
                << "TODO: handle default values: " << func.name << " has "
                << arg.name << " = " << arg.default_value;
            // TODO: Ref_t???
            if (arg.mutating) {
                mut_names.insert(arg.name);
            }
            Argument a = {arg.type, arg.name, arg.mutating, arg.unaliased};
            block->args.push_back(a);
            auto [_, inserted] = block->lookups.insert(
                {arg.name, std::make_shared<Value>(std::move(a))});
            internal_assert(inserted)
                << "Failed to insert argument: " << arg.name
                << " of function: " << func.name;
        }

        // A function that seeds the random generator is handed its state by
        // its own prologue rather than by a caller (Lower/Random.cpp gives it
        // the setup_rng attribute instead of an extra parameter, and the
        // backend allocates the state on entry). So the name exists without
        // being an argument: bind it, so that the calls which pass it on can
        // find it, but leave it out of the signature.
        if (func.must_setup_rng()) {
            const Argument state{Ptr_t::make(Rand_State_t::make()),
                                 lower::rng_state_name, /*mutating=*/true};
            mut_names.insert(state.name);
            block->lookups.insert({state.name, std::make_shared<Value>(state)});
        }

        function->blocks.push_back(block);
        function->ret_type = func.ret_type;
        function->attributes = func.attributes;

        func.body.accept(this);
    }

    // Returns the type to look up `loc.base` under: the pointer type if
    // `loc.base` names a mutable argument or local (see `mut_names` above),
    // otherwise `loc.base_type` unchanged.
    Type base_lookup_type(const WriteLoc &loc) const {
        if (mut_names.contains(loc.base) && !loc.base_type.is_reference()) {
            return Ptr_t::make(loc.base_type);
        }
        return loc.base_type;
    }

    // TODO: cache for unmutable expressions!
    std::shared_ptr<Value> get_value(const Expr &expr) {
        value = nullptr;
        expr.accept(this);
        internal_assert(value) << expr << " failed to produce SSA value";
        return std::move(value);
    }

    uint64_t block_counter = 0;
    std::string get_block_name(const std::string &prefix) {
        return "!" + prefix + "_" + std::to_string(block_counter++);
    }

    std::shared_ptr<Block> make_block(const std::string &prefix) {
        std::shared_ptr<Block> new_block = std::make_shared<Block>();
        new_block->name = get_block_name(prefix);
        new_block->owner = function;
        function->blocks.push_back(new_block);
        return new_block;
    }

    void set_block_jump(const std::string &name) {
        internal_assert(!block->terminator.defined())
            << "set_block_jmp(" << name << ") on block " << block->name
            << " that already has terminator";
        block->terminator.data = Terminator::Jump{.name = name};
    }

    void visit(const IfElse *node) override {
        auto v = get_value(node->cond); // in current block
        internal_assert(!block->terminator.defined());
        std::shared_ptr<Block> then_case = make_block("then");
        std::shared_ptr<Block> else_case =
            node->else_body.defined() ? make_block("else") : nullptr;

        const bool then_needs_merge = !always_returns(node->then_body);
        const bool else_needs_merge =
            !node->else_body.defined() || !always_returns(node->else_body);
        const bool needs_merge = then_needs_merge || else_needs_merge;
        std::shared_ptr<Block> merge_block =
            needs_merge ? make_block("merge") : nullptr;

        block->terminator.data = Terminator::Dispatch{
            .cond = std::move(v),
            // v == 0
            .targets = {Terminator::Jump{.name = node->else_body.defined()
                                                     ? else_case->name
                                                     : merge_block->name,
                                         .args = {}},
                        // v != 0
                        Terminator::Jump{.name = then_case->name, .args = {}}}};

        auto curr_block = std::move(block);

        then_case->preds.push_back(curr_block);
        block = then_case;
        node->then_body.accept(this);

        if (then_needs_merge) {
            // *current* insert block is predecessor to merge.
            merge_block->preds.push_back(block);
            set_block_jump(merge_block->name);
        }

        if (node->else_body.defined()) {
            else_case->preds.push_back(curr_block);
            block = else_case;
            node->else_body.accept(this);

            if (else_needs_merge) {
                merge_block->preds.push_back(block);
                set_block_jump(merge_block->name);
            } // otherwise returns
        } else {
            // Dispatch goes to merge block if condition is false
            merge_block->preds.push_back(curr_block);
        }

        if (merge_block) {
            block = merge_block;
        }
    }

    void visit(const DoWhile *node) override {
        std::shared_ptr<Block> loop_head = make_block("do_while");
        internal_assert(!block->terminator.defined());
        block->terminator.data = Terminator::Jump{.name = loop_head->name};
        loop_head->preds.push_back(block);
        auto entry_block = std::move(block);
        block = loop_head;
        node->body.accept(this);

        // If the block terminator is defined, it should be a return,
        // (should never happen?), otherwise need to make a dispatch terminator.
        // Should the condition be made in the body though? I guess so. That's
        // kinda weird.
        auto v = get_value(node->cond);
        internal_assert(!block->terminator.defined());

        std::shared_ptr<Block> loop_end = make_block("do_while_end");

        block->terminator.data = Terminator::Dispatch{
            .cond = std::move(v),
            // v == 0
            .targets = {Terminator::Jump{.name = loop_end->name, .args = {}},
                        // v != 0
                        //  TODO: copy arguments?
                        Terminator::Jump{.name = loop_head->name, .args = {}}}};
        loop_end->preds.push_back(block);
        loop_head->preds.push_back(block); // possible self-cycle?
        block = loop_end;
    }

    void visit(const While *node) override {
        internal_error << "TODO: convert While to SSA\n";
    }

    void visit(const LetStmt *node) override {
        auto v = get_value(node->value);
        block->make_instruction(node->loc.base, node->loc.base_type,
                                std::move(v));
    }

    void visit(const Return *node) override {
        std::shared_ptr<Value> v = nullptr;
        if (node->value.defined()) {
            v = get_value(node->value);
        }
        block->terminator.data = Terminator::Return{v};
    }

    std::string get_call_name(const std::shared_ptr<Value> &v) const {
        if (const auto *c = std::get_if<Constant>(&(v->data))) {
            if (const auto *s = std::get_if<std::string>(&(c->data))) {
                return *s;
            }
        }
        v->dump(std::cerr);
        internal_error << "In SSA call lowering ^ is not a string call.";
    }

    // Make args consistent in current block.
    void rethread(std::vector<std::shared_ptr<Value>> &args) {
        for (auto &arg : args) {
            std::visit(overloads{
                           [&](const Argument &a) {
                               auto it = block->lookups.find(a.name);
                               if (it != block->lookups.end()) {
                                   arg = it->second;
                               } else {
                                   arg = block->get_value(a.name, a.type);
                               }
                           },
                           [&](const std::shared_ptr<Instruction> &i) {
                               if (i->owner.lock().get() != block.get()) {
                                   arg = block->get_value(i->name, i->type);
                               }
                           },
                           [](const Constant &) {},
                       },
                       arg->data);
        }
    }

    void visit(const CallStmt *node) override {
        auto func = get_value(node->func);
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());

        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }
        // Make args consistent in current block.
        rethread(args);

        auto call_name = get_call_name(func);

        // Create continuation block.
        auto cont_block = make_block(call_name + "_call_cont");
        cont_block->preds.push_back(block);

        // End the current block with a Call
        internal_assert(!block->terminator.defined());
        auto call_block = std::move(block);

        call_block->terminator.data = Terminator::Call{
            .call =
                Terminator::Jump{.name = call_name, .args = std::move(args)},
            // 'args' is empty for now (implicit capture via CFG lookup)
            // Does not receive output of call, because value is dropped.
            .cont = Terminator::Jump{.name = cont_block->name, .args = {}},
            .drop = true};

        block = cont_block;
    }

    void visit(const Print *node) override {
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());
        for (const Expr &arg : node->args) {
            args.push_back(get_value(arg));
        }
        block->make_side_effect(Instruction::Op::Print, std::move(args));
    }

    void visit(const Append *node) override {
        auto v = get_value(node->value);
        auto loc = get_value(node->loc.to_expr());
        std::vector<std::shared_ptr<Value>> args = {std::move(loc),
                                                    std::move(v)};
        block->make_side_effect(Instruction::Op::Append, std::move(args));
    }

    void visit(const Allocate *node) override {
        // Allocate is only ever emitted for `mut` locals (see
        // Parser::parse_assign / parse declarations), so any read of this
        // name later in the body was rewritten by Lower/Mutability.cpp into
        // Deref(Var(Ptr_t(base_type), name)). Register a real pointer here
        // (matching mut function arguments, see `mut_names` above) so those
        // reads and any later Store/Accumulate agree on its type.
        auto op = (node->memory == Allocate::Stack) ? Instruction::Op::Alloca
                                                    : Instruction::Op::Alloc;

        mut_names.insert(node->loc.base);

        // An array handle already refers to storage, so it is registered
        // under its own type rather than a pointer to it -- the same
        // convention Lower/Mutability.cpp uses (see Type::is_reference), and
        // what `base_lookup_type` below expects to find.
        const Type &base_type = node->loc.base_type;
        Type alloc_type =
            base_type.is_reference() ? base_type : Ptr_t::make(base_type);
        std::vector<std::shared_ptr<Value>> args;
        std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
            node->loc.base, alloc_type, op, args, block);
        block->instrs.push_back(instr);

        auto [_, inserted] = block->lookups.insert(
            {node->loc.base, std::make_shared<Value>(instr)});
        internal_assert(inserted)
            << node->loc.base << "already exists in block!\n";

        if (node->value.defined()) {
            auto v = get_value(node->value);
            block->make_side_effect(
                Instruction::Op::Store,
                {std::make_shared<Value>(instr), std::move(v)});
        }
    }

    void visit(const Store *node) override {
        if (node->loc.accesses.empty()) {
            // Assigning a whole name. Either it is backed by storage, and the
            // value is written through the pointer, or it is not, and the
            // name is simply rebound -- neither of which cares what the type
            // is, so a struct or a tuple is written the same way a float is.
            //
            // An array is the exception: its name is bound to its elements
            // rather than to a slot holding a handle (see Type::is_reference),
            // so there is nothing to write a handle into.
            internal_assert(!node->loc.type.is_reference())
                << "TODO: assign a whole array in SSA, whose name is bound to "
                << "its elements and not to a slot holding it: " << Stmt(node);
            auto v = get_value(node->value);

            if (mut_names.contains(node->loc.base)) {
                // `node->loc.base` is a pointer-backed mutable argument (see
                // `mut_names` above); the write must go through a real Store
                // so it's visible to the caller, not just renamed in this
                // function's local SSA lookups. Also, overwriting the lookup
                // with the raw (unwrapped) value here would stomp the
                // pointer entry needed by any later access to this argument.
                auto ptr = block->get_value(node->loc.base,
                                            base_lookup_type(node->loc));
                block->make_side_effect(Instruction::Op::Store,
                                        {std::move(ptr), std::move(v)});
                return;
            }

            // Overwrite the name with v (insert if missing).
            // All successors of the current block will receive v.
            block->lookups[node->loc.base] = v;
            return;
        }

        // Get the stored value *first*.
        // Before evaluating the lhs.
        auto v = get_value(node->value);

        // Create GEP
        auto var =
            block->get_value(node->loc.base, base_lookup_type(node->loc));

        for (const auto &value : node->loc.accesses) {
            if (std::holds_alternative<std::string>(value)) {
                internal_error << "TODO: handle stores to field vars in SSA: "
                               << Stmt(node);
            } else {
                Expr idx = std::get<Expr>(value);
                auto i = get_value(idx);

                // TODO: FIGURE OUT TYPE!!
                // TODO: aligned load first?? not sure how this works.
                // LLVM is weird here.
                var = block->make_instruction(Type(), Instruction::Op::GEP,
                                              {std::move(var), std::move(i)});
            }
        }

        std::vector<std::shared_ptr<Value>> args = {std::move(var),
                                                    std::move(v)};
        block->make_side_effect(Instruction::Op::Store, std::move(args));
    }

    Instruction::Op get_acc_op(const Accumulate::OpType op) {
        switch (op) {
        case Accumulate::Add: {
            return Instruction::Op::AccAdd;
        }
        case Accumulate::Mul: {
            return Instruction::Op::AccMul;
        }
        case Accumulate::Sub: {
            return Instruction::Op::AccSub;
        }
        case Accumulate::Min: {
            return Instruction::Op::AccMin;
        }
        case Accumulate::Max: {
            return Instruction::Op::AccMax;
        }
        default: {
            internal_error << "TODO: handle all Accumulate ops -> SSA ops: "
                           << (int)op;
            return Instruction::Op::AccAdd;
        }
        }
    }

    // TODO: dedup with Store visitor
    void visit(const Accumulate *node) override {
        auto v = get_value(node->value);
        auto op = get_acc_op(node->op);

        std::shared_ptr<Value> ptr = nullptr;

        // Handle local variables
        if (node->loc.accesses.empty()) {
            internal_assert(node->loc.type.is_stack_allocatable())
                << "TODO: handle non-primitive (heap) accumulates in SSA: "
                << Stmt(node);

            // Get current value from lookup
            ptr = block->get_value(node->loc.base, base_lookup_type(node->loc));
        } else {
            // Memory Access (GEP -> AccOp)

            // Calculate address (TODO: dedup with Store)
            ptr = block->get_value(node->loc.base, base_lookup_type(node->loc));

            for (const auto &value : node->loc.accesses) {
                if (std::holds_alternative<std::string>(value)) {
                    internal_error
                        << "TODO: handle accumulates to field vars in SSA: "
                        << Stmt(node);
                } else {
                    Expr idx = std::get<Expr>(value);
                    auto i = get_value(idx);

                    // Update ptr with GEP result
                    ptr =
                        block->make_instruction(Type(), Instruction::Op::GEP,
                                                {std::move(ptr), std::move(i)});
                }
            }
        }

        std::vector<std::shared_ptr<Value>> args = {std::move(ptr),
                                                    std::move(v)};
        block->make_side_effect(op, std::move(args), node->atomic);
    }

    void visit(const ParFor *node) override {
        auto start = get_value(node->slice.begin);
        auto end = get_value(node->slice.end);
        auto stride = get_value(node->slice.stride);

        // Create body and continuation blocks.
        auto body_block = make_block("par_body");
        body_block->preds.push_back(block);
        auto cont_block = make_block("par_cont");
        cont_block->preds.push_back(block);

        // Save the current block
        internal_assert(!block->terminator.defined());
        auto header_block = std::move(block);

        // Terminate header with parfor
        header_block->terminator.data = Terminator::ParFor{
            .index = node->index,
            .start = std::move(start),
            .end = std::move(end),
            .stride = std::move(stride),
            // 'args' is empty for now (implicit capture via CFG lookup)
            .body = Terminator::Jump{.name = body_block->name},
            .cont = Terminator::Jump{.name = cont_block->name}};

        // Set up loop index (MUST BE FIRST ARG).
        Argument loop_idx{.type = node->slice.begin.type(),
                          .name = node->index};
        body_block->args.push_back(loop_idx);
        auto [_, inserted] = body_block->lookups.insert(
            {node->index, std::make_shared<Value>(std::move(loop_idx))});
        internal_assert(inserted)
            << node->index << "already exists in block!\n";

        block = body_block;

        node->body.accept(this);
        internal_assert(!block->terminator.defined())
            << "ParFor block for: " << node->body << " has terminator.";
        block->terminator.data = Terminator::Yield{};

        // Continue compilation in the continuation block
        block = cont_block;
    }

    // A sequential loop, as a loop in the control-flow graph: a header that
    // carries the index and decides whether to go round again, a body, and a
    // latch that steps the index and closes the back edge.
    //
    //     preheader:  jmp head(begin)
    //     head(i):    c = i < end;  dispatch c [exit, body]
    //     body/latch: ...body...;   n = i + stride;  jmp head(n)
    //     exit:
    //
    // The test is at the top, so a loop whose range is empty runs zero times.
    void visit(const ForAll *node) override {
        const Type index_type = node->index_type();

        auto end = get_value(node->slice.end);
        auto stride = get_value(node->slice.stride);
        auto begin = get_value(node->slice.begin);

        auto head_block = make_block("for_" + node->index);
        auto body_block = make_block("for_" + node->index + "_body");
        auto exit_block = make_block("for_" + node->index + "_end");

        internal_assert(!block->terminator.defined());
        auto preheader = std::move(block);
        preheader->terminator.data =
            Terminator::Jump{.name = head_block->name, .args = {begin}};
        head_block->preds.push_back(preheader);

        // The index is the header's first argument -- the phi between where
        // the loop starts and what the latch hands back.
        Argument loop_idx{.type = index_type, .name = node->index};
        head_block->args.push_back(loop_idx);
        auto [_, inserted] = head_block->lookups.insert(
            {node->index, std::make_shared<Value>(loop_idx)});
        internal_assert(inserted)
            << node->index << " already exists in block!\n";

        block = head_block;
        // Building this threads `end` in from the preheader, which is what
        // gives the header the rest of its arguments.
        auto cond = block->make_instruction(
            Bool_t::make(), Instruction::Op::Lt,
            {block->lookups.at(node->index), std::move(end)});
        block->terminator.data = Terminator::Dispatch{
            .cond = std::move(cond),
            .targets = {Terminator::Jump{.name = exit_block->name},
                        Terminator::Jump{.name = body_block->name}}};
        body_block->preds.push_back(head_block);
        exit_block->preds.push_back(head_block);

        block = body_block;
        node->body.accept(this);
        internal_assert(!block->terminator.defined())
            << "The body of loop " << node->index << " ends in " << block->name
            << ", which already has a terminator: a loop body that leaves "
            << "early is not supported here";

        // Whatever block the body ended in is the latch.
        auto latch = std::move(block);
        auto next = latch->make_instruction(
            index_type, Instruction::Op::Add,
            {latch->get_value(node->index, index_type), std::move(stride)});

        // Snapshot after the step, which may have threaded `stride` in and so
        // given the header another argument.
        const std::vector<Argument> carried = head_block->args;
        std::vector<std::shared_ptr<Value>> back{next};
        for (size_t i = 1; i < carried.size(); i++) {
            back.push_back(latch->get_value(carried[i].name, carried[i].type));
        }
        internal_assert(head_block->args.size() == carried.size())
            << "Closing the back edge of loop " << node->index
            << " gave its header more arguments";
        latch->terminator.data =
            Terminator::Jump{.name = head_block->name, .args = std::move(back)};
        head_block->preds.push_back(latch);

        block = exit_block;
    }

    template <typename T>
    std::shared_ptr<Value> make_constant(ir::Type type, T data) {
        Constant c{.type = std::move(type), .data = data};
        return std::make_shared<Value>(std::move(c));
    }

    void visit(const IntImm *node) override {
        value = make_constant(node->type, node->value);
    }

    void visit(const UIntImm *node) override {
        value = make_constant(node->type, node->value);
    }

    void visit(const FloatImm *node) override {
        value = make_constant(node->type, node->value);
    }

    void visit(const BoolImm *node) override {
        value = make_constant(node->type, node->value);
    }

    void visit(const Extrema *node) override {
        const Instruction::Op op = node->op == Extrema::eps
                                       ? Instruction::Op::Eps
                                       : Instruction::Op::Inf;
        value = block->make_instruction(node->type, op, {});
    }

    void visit(const Access *node) override {
        auto v = get_value(node->value);
        static const Type u32 = UInt_t::make(32);

        // A union's members all begin at the same address, so reading one is
        // reading those bytes at that member's type. There is no field index
        // to load -- LoadField takes one into a struct -- so this goes through
        // the address, which is what CodeGen_LLVM does for the same Access.
        if (node->value.type().is<Union_t>()) {
            auto addr = block->make_instruction(Ptr_t::make(node->type),
                                                Instruction::Op::AddressOf,
                                                {std::move(v)});
            value = block->make_instruction(node->type, Instruction::Op::Load,
                                            {std::move(addr)});
            return;
        }

        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        internal_assert(struct_t) << node->value.type() << " of " << Expr(node);
        auto idx = find_struct_index(node->field, struct_t->fields);
        auto vidx = make_constant(u32, (uint64_t)idx);
        value = block->make_instruction(node->type, Instruction::Op::LoadField,
                                        {std::move(v), std::move(vidx)});
    }

    // A union holding one of its members: the member's bytes, read as the
    // union. The mirror of reading one out above, but not simply the address
    // of the member -- a union is as wide as its largest member, and loading
    // one out of storage only big enough for this member would read past the
    // end of it. So the storage is the union's and the member is written into
    // it, which is the same shape CodeGen_LLVM::visit(const UnionOf *) has.
    void visit(const UnionOf *node) override {
        auto v = get_value(node->value);
        auto slot = block->make_instruction(Ptr_t::make(node->type),
                                            Instruction::Op::Alloca, {});
        block->make_side_effect(Instruction::Op::Store, {slot, std::move(v)});
        value = block->make_instruction(node->type, Instruction::Op::Load,
                                        {std::move(slot)});
    }

    Instruction::Op get_binop(BinOp::OpType op) {
        switch (op) {
        case BinOp::Add:
            return Instruction::Op::Add;
        case BinOp::Mul:
            return Instruction::Op::Mul;
        case BinOp::Div:
            return Instruction::Op::Div;
        case BinOp::Sub:
            return Instruction::Op::Sub;
        case BinOp::Mod:
            return Instruction::Op::Mod;
        case BinOp::Neq:
            return Instruction::Op::Ne;
        case BinOp::Eq:
            return Instruction::Op::Eq;
        case BinOp::Le:
            return Instruction::Op::Leq;
        case BinOp::Lt:
            return Instruction::Op::Lt;
        case BinOp::LAnd:
            // TODO: SHORT CIRCUITING!!!
            return Instruction::Op::LAnd;
        case BinOp::LOr:
            // TODO: SHORT CIRCUITING!!!
            return Instruction::Op::LOr;
        case BinOp::Xor:
            return Instruction::Op::Xor;
        case BinOp::BwAnd:
            return Instruction::Op::BwAnd;
        case BinOp::BwOr:
            return Instruction::Op::BwOr;
        // Simplify turns a multiply or divide by a power of two into one of
        // these, so integer arithmetic reaches them whether or not the source
        // ever mentions a shift.
        case BinOp::Shl:
            return Instruction::Op::Shl;
        case BinOp::Shr:
            return Instruction::Op::Shr;
        default: {
            internal_error << "TODO: handle: " << to_string(op);
        }
        }
    }

    void visit(const BinOp *node) override {
        auto a = get_value(node->a);
        auto b = get_value(node->b);
        auto op = get_binop(node->op);
        value = block->make_instruction(node->type, op,
                                        {std::move(a), std::move(b)});
    }

    void visit(const Broadcast *node) override {
        auto v = get_value(node->value);
        static const Type u32 = UInt_t::make(32);
        auto lanes = make_constant(u32, (uint64_t)node->lanes);
        value = block->make_instruction(node->type, Instruction::Op::Bc,
                                        {std::move(v), std::move(lanes)});
    }

    // A shuffle is a vector built from lanes picked out of another one, which
    // is how the backends lower it too -- an extract per index, gathered into
    // a build. Doing it here rather than carrying a shuffle opcode keeps the
    // shuffled value to a single evaluation and needs nothing the SSA form
    // does not already have.
    void visit(const VectorShuffle *node) override {
        auto v = get_value(node->value);
        const Type lane_type = node->value.type().element_of();
        std::vector<std::shared_ptr<Value>> lanes;
        lanes.reserve(node->idxs.size());
        for (const auto &idx : node->idxs) {
            auto i = get_value(idx);
            lanes.push_back(block->make_instruction(
                lane_type, Instruction::Op::ExtractIdx, {v, std::move(i)}));
        }
        value = block->make_instruction(node->type, Instruction::Op::MakeStruct,
                                        std::move(lanes));
    }

    // A vector literal is a vector built from its components, which is what
    // Build already is -- `Build` of a vector type and `VecImm` differ only in
    // spelling (compare lower::cross_product, which builds a vector that way).
    // Rewriting it here means the SSA form has one way to make a vector.
    void visit(const VecImm *node) override {
        ir::Build::make(node->type, node->values).accept(this);
    }

    void visit(const Build *node) override {
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->values.size());
        for (const auto &arg : node->values) {
            args.emplace_back(get_value(arg));
        }
        value = block->make_instruction(node->type, Instruction::Op::MakeStruct,
                                        std::move(args));
    }

    void visit(const Cast *node) override {
        auto v = get_value(node->value);
        Instruction::Op op = (node->mode == Cast::Mode::Reinterpret)
                                 ? Instruction::Op::Reinterpret
                                 : Instruction::Op::Cast;
        value = block->make_instruction(node->type, op, {std::move(v)});
    }

    void visit(const Call *node) override {
        auto func = get_value(node->func);
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());

        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }
        // Make args consistent in current block.
        rethread(args);

        auto call_name = get_call_name(func);

        // Create continuation block.
        auto cont_block = make_block(call_name + "_call_cont");
        cont_block->preds.push_back(block);

        // End the current block with a Call
        internal_assert(!block->terminator.defined());
        auto call_block = std::move(block);

        call_block->terminator.data =
            Terminator::Call{.call = Terminator::Jump{.name = call_name,
                                                      .args = std::move(args)},
                             // cont `args` takes an argument that is the result
                             // of the call. Pass any live vars!
                             .cont = Terminator::Jump{.name = cont_block->name},
                             .drop = false};

        block = cont_block;

        std::string name = function->get_unique_name(); // name of returned item

        Argument arg{.type = node->type, .name = name};
        auto arg_value = std::make_shared<Value>(arg);
        // Insert call arg into block args and block lookups!
        block->args.push_back(arg);
        block->lookups.insert({name, arg_value});
        value = arg_value;
    }

    void visit(const Extract *node) override {
        auto vec = get_value(node->vec);
        auto idx = get_value(node->idx);
        value = block->make_instruction(node->type, Instruction::Op::ExtractIdx,
                                        {std::move(vec), std::move(idx)});
    }

    // The intrinsics that have an SSA opcode of their own, because the passes
    // downstream reason about them rather than only passing them on. The rest
    // ride on Op::Intrinsic (see the enum in SSA/SSA.h).
    std::optional<Instruction::Op> get_intrinsic(const Intrinsic::OpType &op) {
        switch (op) {
        case Intrinsic::abs:
            return Instruction::Op::Abs;
        case Intrinsic::max:
            return Instruction::Op::Max;
        case Intrinsic::min:
            return Instruction::Op::Min;
        default:
            return std::nullopt;
        }
    }

    void visit(const Intrinsic *node) override {
        // A few are defined in terms of the others rather than being
        // primitive. Rewriting them here means the SSA form -- and every pass
        // that reads it -- only ever sees the pieces.
        if (node->op == Intrinsic::dot) {
            internal_assert(node->args.size() == 2);
            ir::Expr e = lower::dot_product(node->args[0], node->args[1]);
            e.accept(this);
            return;
        } else if (node->op == Intrinsic::cross) {
            internal_assert(node->args.size() == 2);
            ir::Expr e = lower::cross_product(node->args[0], node->args[1]);
            e.accept(this);
            return;
        } else if (node->op == Intrinsic::norm) {
            internal_assert(node->args.size() == 1);
            ir::Expr e = lower::norm(node->args[0]);
            e.accept(this);
            return;
        } else if (node->op == Intrinsic::sqr) {
            internal_assert(node->args.size() == 1);
            // Squaring is a multiplication, and naming the operand first
            // keeps it to one evaluation.
            auto a = get_value(node->args[0]);
            value = block->make_instruction(node->type, Instruction::Op::Mul,
                                            {a, a});
            return;
        }

        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());
        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }

        if (const auto op = get_intrinsic(node->op)) {
            value = block->make_instruction(node->type, *op, std::move(args));
            return;
        }

        // Everything else is carried through as it is, for the backend to
        // lower the same way it would have without the SSA form in between.
        value = block->make_instruction(node->type, Instruction::Op::Intrinsic,
                                        std::move(args));
        std::get<std::shared_ptr<Instruction>>(value->data)->intrinsic =
            node->op;
    }

    void visit(const Var *node) override {
        internal_assert(block);
        if (node->type.is_func()) {
            // TODO: what about named lambdas??
            value = make_constant(node->type, node->name);
        } else {
            value = block->get_value(node->name, node->type);
        }
    }

    void visit(const UnOp *node) override {
        // Neither negation nor logical not has its own SSA opcode: negation
        // is a subtraction from zero, and a not is a select between the two
        // boolean constants, both of which the backends already lower.
        auto a = get_value(node->a);

        // A constant holds one value, so the constants below are of the
        // element type and broadcast when the operand is a vector. Giving a
        // constant a vector type instead makes something that says it is
        // three floats while holding one.
        const Type elem =
            node->type.is_vector() ? node->type.element_of() : node->type;
        auto splat = [&](Constant c) {
            auto v = std::make_shared<Value>(std::move(c));
            if (!node->type.is_vector()) {
                return v;
            }
            auto lanes = std::make_shared<Value>(
                Constant{UInt_t::make(32), uint64_t(node->type.lanes())});
            return block->make_instruction(node->type, Instruction::Op::Bc,
                                           {std::move(v), std::move(lanes)});
        };

        switch (node->op) {
        case UnOp::Neg: {
            auto zero = splat(elem.is_float()
                                  ? Constant{elem, 0.0}
                                  : Constant{elem, static_cast<int64_t>(0)});
            value = block->make_instruction(node->type, Instruction::Op::Sub,
                                            {std::move(zero), std::move(a)});
            return;
        }
        case UnOp::Not: {
            auto true_ = splat(Constant{elem, true});
            auto false_ = splat(Constant{elem, false});
            value = block->make_instruction(
                node->type, Instruction::Op::Select,
                {std::move(a), std::move(false_), std::move(true_)});
            return;
        }
        }
    }

    void visit(const PtrTo *node) override {
        // The address of a value. Lower/Mutability.cpp introduces these at
        // call sites, for arguments a callee takes by pointer.
        internal_assert(node->expr.defined()) << "PtrTo of nothing";

        // Addressing what a pointer already points at is that pointer. (The
        // IR folds this away when it builds the node, so this only catches
        // what survives.)
        if (const Deref *deref = node->expr.as<Deref>()) {
            value = get_value(deref->expr);
            return;
        }

        // The random generator's state is storage from the start -- the
        // prologue allocates it -- so its address is that storage, not a copy
        // of what it holds. The backends make the same exception.
        if (const Var *var = node->expr.as<Var>();
            var != nullptr && var->name == lower::rng_state_name) {
            value = block->get_value(var->name, Ptr_t::make(var->type));
            return;
        }

        // An element of an array: the address is an offset from the array,
        // which is what GEP computes.
        if (const Extract *extract = node->expr.as<Extract>();
            extract != nullptr && extract->vec.type().is_reference()) {
            auto base = get_value(extract->vec);
            auto index = get_value(extract->idx);
            value =
                block->make_instruction(node->type, Instruction::Op::GEP,
                                        {std::move(base), std::move(index)});
            return;
        }

        // Anything else is carried as what it is -- the address of a value --
        // and where that address comes from is settled when the code is
        // generated, not here.
        //
        // This used to allocate a stack slot and store the value into it,
        // which is one of the answers but rarely the right one. `&(*p).field`
        // names a field of a struct that is already in memory, and the
        // backends turn that into a GEP; a value that came from a load can
        // reuse the pointer it was loaded through. Only a value with no
        // storage anywhere actually needs a copy. Committing to the copy here
        // hid the other cases, and it could not be undone downstream, because
        // the pointer escapes into the call it was made for and
        // SSA/PromoteAllocas.h will not promote an allocation that escapes.
        auto v = get_value(node->expr);
        value =
            block->make_instruction(Ptr_t::make(node->expr.type()),
                                    Instruction::Op::AddressOf, {std::move(v)});
    }

    void visit(const Deref *node) override {
        // `node->expr` is a pointer (e.g. a `mut` argument/local, wrapped by
        // Lower/Mutability.cpp); Load reads through it to produce a value of
        // the pointee type (node->type).
        auto ptr = get_value(node->expr);
        value = block->make_instruction(node->type, Instruction::Op::Load,
                                        {std::move(ptr)});
    }

    // A reduction over the lanes of one value. It used to be mapped onto the
    // binary opcodes -- Add for a sum, Min for a minimum -- which gave those a
    // one-operand form that nothing downstream expected; a reduction is its
    // own operation and is carried as one.
    void visit(const VectorReduce *node) override {
        auto a = get_value(node->value);
        value = block->make_instruction(node->type, Instruction::Op::Reduce,
                                        {std::move(a)});
        std::get<std::shared_ptr<Instruction>>(value->data)->reduce = node->op;
    }

    void visit(const Select *node) override {
        auto cond = get_value(node->cond);
        auto true_val = get_value(node->tvalue);
        auto false_val = get_value(node->fvalue);
        value = block->make_instruction(
            node->type, Instruction::Op::Select,
            {std::move(cond), std::move(true_val), std::move(false_val)});
    }

    // RESTRICT_VISITOR(IntImm);
    // RESTRICT_VISITOR(UIntImm);
    // RESTRICT_VISITOR(FloatImm);
    // RESTRICT_VISITOR(BoolImm);
    // RESTRICT_VISITOR(VecImm);
    RESTRICT_VISITOR(StringImm);
    // RESTRICT_VISITOR(Extrema);
    // RESTRICT_VISITOR(Var);
    // RESTRICT_VISITOR(BinOp);
    // RESTRICT_VISITOR(UnOp);
    // RESTRICT_VISITOR(Select);
    // RESTRICT_VISITOR(Cast);
    // RESTRICT_VISITOR(Broadcast);
    // RESTRICT_VISITOR(VectorReduce);
    // RESTRICT_VISITOR(VectorShuffle);
    RESTRICT_VISITOR(Ramp);
    // RESTRICT_VISITOR(Extract);
    // RESTRICT_VISITOR(Build);
    // RESTRICT_VISITOR(Access);
    RESTRICT_VISITOR(Unwrap);
    // RESTRICT_VISITOR(Intrinsic);
    RESTRICT_VISITOR(Generator);
    RESTRICT_VISITOR(Lambda);
    RESTRICT_VISITOR(GeomOp);
    RESTRICT_VISITOR(SetOp);
    RESTRICT_VISITOR(AggOp);
    // RESTRICT_VISITOR(Call);
    RESTRICT_VISITOR(Instantiate);
    // RESTRICT_VISITOR(PtrTo);
    // RESTRICT_VISITOR(Deref);
    RESTRICT_VISITOR(AtomicAdd);

    // RESTRICT_VISITOR(CallStmt);
    // RESTRICT_VISITOR(Print);
    // RESTRICT_VISITOR(Return);
    // RESTRICT_VISITOR(LetStmt);
    // RESTRICT_VISITOR(IfElse);
    // RESTRICT_VISITOR(DoWhile);
    // RESTRICT_VISITOR(Sequence); // default behavior is fine.
    // RESTRICT_VISITOR(Allocate);
    RESTRICT_VISITOR(Free);
    // RESTRICT_VISITOR(Store);
    // RESTRICT_VISITOR(Accumulate);
    RESTRICT_VISITOR(Label);
    RESTRICT_VISITOR(RecLoop);
    RESTRICT_VISITOR(Match);
    RESTRICT_VISITOR(Yield);
    RESTRICT_VISITOR(Iterate);
    RESTRICT_VISITOR(Scan);
    RESTRICT_VISITOR(YieldFrom);
    // RESTRICT_VISITOR(ForAll);
    RESTRICT_VISITOR(ForEach);
    RESTRICT_VISITOR(Continue);
    RESTRICT_VISITOR(Launch);
    // RESTRICT_VISITOR(Append);
};

std::shared_ptr<ssa::Function>
build(const std::shared_ptr<ir::Function> &func) {
    FunctionBuilder builder(*func);
    if (!builder.block->terminator.defined()) {
        builder.block->terminator.data = Terminator::Return{};
    }
    return builder.function;
}

namespace {

// Where a loop a schedule names actually is.
struct LoopSite {
    std::string func;  // the function holding it
    std::string index; // the name it goes by there
};

// Every parfor reachable from `start`, by function.
//
// A schedule names a function and a loop in it, but neither need be where the
// loop ends up: lowering moves a `map` into a `_traverse_arrayN` helper of its
// own, so the loop a schedule calls `process`'s is really in a function
// `process` calls. The Stmt-level pass follows the same calls, by name and
// with a "this is hacky" note; following the call graph is the general form of
// that.
std::map<std::string, std::set<std::string>>
parfors_reachable_from(const FuncMap &fmap, const std::string &start) {
    std::map<std::string, std::set<std::string>> found;
    std::set<std::string> seen;
    std::vector<std::string> work{start};
    while (!work.empty()) {
        const std::string name = work.back();
        work.pop_back();
        if (!seen.insert(name).second) {
            continue;
        }
        const auto it = fmap.find(name);
        if (it == fmap.end()) {
            continue; // an extern, or something not compiled here
        }
        for (const auto &block : it->second->blocks) {
            if (const auto *p =
                    std::get_if<Terminator::ParFor>(&block->terminator.data)) {
                found[name].insert(p->index);
            }
            if (const auto *c =
                    std::get_if<Terminator::Call>(&block->terminator.data)) {
                work.push_back(c->call.name);
            }
        }
    }
    return found;
}

// The loop a schedule means. A loop the program wrote is called what the
// program called it; one lowering generated is called `_` followed by that,
// because lowering labels what it invents, while a schedule names it without
// the underscore either way.
LoopSite resolve_loop(const FuncMap &fmap, const std::string &start,
                      const std::string &wanted, const std::string &transform) {
    const auto found = parfors_reachable_from(fmap, start);
    for (const std::string &candidate : {wanted, "_" + wanted}) {
        // The function the schedule named wins over one it merely reaches.
        const auto here = found.find(start);
        if (here != found.end() && here->second.count(candidate)) {
            return {start, candidate};
        }
        for (const auto &[fname, loops] : found) {
            if (loops.count(candidate)) {
                return {fname, candidate};
            }
        }
    }

    std::string all;
    for (const auto &[fname, loops] : found) {
        for (const auto &loop : loops) {
            all += (all.empty() ? "" : ", ") + fname + ":" + loop;
        }
    }
    internal_error << transform << "() on " << start << ": no parfor named "
                   << wanted << ". "
                   << (all.empty()
                           ? "Nothing it reaches has a parfor at all; only a "
                             "parfor can be transformed, since a sequential "
                             "loop has an order to keep."
                           : "The parfor loops it reaches are: " + all);
    return {start, wanted};
}

// Applies the SSA-level schedule `transforms` and builds/codegens `funcs`
// through the SSA representation. A transform this pipeline cannot apply is
// reported, not skipped -- see the visit below.
ir::FuncMap convert(ir::FuncMap funcs, const ir::TransformMap &transforms,
                    const CompilerOptions &options,
                    ir::Program *keep_ssa = nullptr) {
    FuncMap fmap;

    TypeMap func_type_map;

    for (const auto &[name, func] : funcs) {
        func_type_map[name] = func->call_type();
        auto f = build(func);
        fmap[name] = std::move(f);
    }

    // Loopify first, whichever order the schedule names the functions in: it
    // turns a function's self tail-calls into a loop, and vectorizing a caller
    // has to see the callee in its final shape -- as a loop it can make
    // uniform, rather than as a recursion it would specialize forever.
    for (const auto &[name, ts] : transforms) {
        if (!fmap.contains(name)) {
            continue;
        }
        for (const auto &t : ts) {
            if (const auto *l = std::get_if<ir::Loopify>(&t)) {
                int size = 0;
                if (l->queue_size.has_value()) {
                    const auto n = get_constant_value<int64_t>(*l->queue_size);
                    internal_assert(n.has_value() && *n > 0)
                        << "loopify(" << *l->queue_size << ") on " << name
                        << " needs a constant, positive stack depth";
                    size = int(*n);
                }
                loopify(fmap, name, size);
            }
        }
    }

    for (const auto &[name, ts] : transforms) {
        if (!fmap.contains(name)) {
            continue;
        }
        const auto unimplemented = [&name](const std::string &what) {
            internal_error
                << what << "() is in the schedule for " << name
                << ", but the SSA pipeline does not apply it yet. Ignoring it "
                   "would compile a different program than the schedule asks "
                   "for, so it is an error instead. Compile without `-p ssa` "
                   "to use it today.";
        };

        for (const auto &t : ts) {
            // Deliberately no catch-all arm. A transform this pipeline does
            // not apply has to say so: the Stmt-level LoopTransforms pass
            // that used to pick up the rest does not run here, so anything
            // quietly ignored is a schedule the program was compiled without
            // -- a `cpu_thread` that never threads, a `split` that leaves one
            // loop where the schedule asked for two. Listing every kind also
            // means a new one will not compile until someone decides which of
            // these it is.
            std::visit(
                overloads{
                    [&](const ir::Vectorize &v) {
                        internal_assert(!v.i.names.empty())
                            << "vectorize() requires a loop name for: " << name;
                        const LoopSite at = resolve_loop(
                            fmap, name, v.i.names.back(), "vectorize");
                        vectorize(fmap, at.func, at.index);
                    },
                    // Applied by the loop above, before anything else
                    // sees the function.
                    [&](const ir::Loopify &) {},
                    // Applied earlier in lowering, by the pass named.
                    [&](const ir::Defer &) {},     // Lower/Defers.cpp
                    [&](const ir::MakeQueue &) {}, // Lower/Defers.cpp
                    [&](const ir::Sort &) {},      // Lower/Sorts.cpp
                    [&](const ir::Split &s) {
                        internal_assert(!s.i.names.empty() &&
                                        !s.io.names.empty() &&
                                        !s.ii.names.empty())
                            << "split() requires loop names for: " << name;
                        const auto factor =
                            get_constant_value<int64_t>(s.factor);
                        internal_assert(factor.has_value() && *factor > 0)
                            << "split(" << s.factor << ") on " << name
                            << " needs a constant, positive factor";
                        const LoopSite at =
                            resolve_loop(fmap, name, s.i.names.back(), "split");
                        split(fmap, at.func, at.index, int(*factor),
                              s.io.names.back(), s.ii.names.back(),
                              !s.generate_tail);
                    },
                    [&](const ir::Collapse &c) {
                        internal_assert(!c.io.names.empty() &&
                                        !c.ii.names.empty() &&
                                        !c.i.names.empty())
                            << "collapse() requires loop names for: " << name;
                        const LoopSite at = resolve_loop(
                            fmap, name, c.io.names.back(), "collapse");
                        const LoopSite in = resolve_loop(
                            fmap, at.func, c.ii.names.back(), "collapse");
                        internal_assert(in.func == at.func)
                            << "collapse(" << c.io.names.back() << ", "
                            << c.ii.names.back() << ") on " << name
                            << ": those loops are in different "
                            << "functions (" << at.func << " and " << in.func
                            << "), so they are not nested";
                        collapse(fmap, at.func, at.index, in.index,
                                 c.i.names.back());
                    },
                    [&](const ir::Bind &b) {
                        internal_assert(!b.i.names.empty())
                            << "bind() requires a cursor for: " << name;
                        // Not finished, and a bind that silently did
                        // nothing would be a program that runs
                        // somewhere other than it was told to.
                        internal_assert(b.resource != ir::Resource::RTCore &&
                                        b.resource != ir::Resource::OptixThread)
                            << "bind(" << b.i.names.back() << ", "
                            << to_string(b.resource) << ") on " << name
                            << ": that backend is not built yet.";
                        const LoopSite at =
                            resolve_loop(fmap, name, b.i.names.back(), "bind");
                        bind(fmap, at.func, at.index, b.resource);
                    },
                },
                t);
        }
    }

    // A transform may have added functions -- vectorize() specializes the
    // callees of a gang -- whose types nothing has recorded yet. Their
    // signature is whatever their entry block takes and their return says.
    for (const auto &[name, f] : fmap) {
        if (func_type_map.contains(name)) {
            continue;
        }
        internal_assert(!f->blocks.empty()) << name << " has no blocks";
        std::vector<Function_t::ArgSig> args;
        for (const auto &arg : f->blocks.front()->args) {
            args.push_back(Function_t::ArgSig{arg.type, arg.mutating});
        }
        func_type_map[name] = Function_t::make(f->ret_type, std::move(args));
    }

    // The SSA form the schedule left behind, before it is turned back into
    // statements. This is the only place it can be seen: what `-p ssa` prints
    // is the result of the relooper, by which point the block graph the
    // rewrites actually worked on is gone.
    if (options.is_verbose) {
        for (const auto &[fname, f] : fmap) {
            const bool direct =
                std::find(f->attributes.begin(), f->attributes.end(),
                          ir::Function::Attribute::vectorized) !=
                f->attributes.end();
            std::cerr << "; --- ssa: " << fname
                      << (direct ? " (lowered from here)"
                                 : " (lowered via statements)")
                      << " ---\n";
            f->dump(std::cerr);
        }
    }

    ir::FuncMap new_funcs;

    for (const auto &[fname, f] : fmap) {
        // The relooper runs for every function regardless of what generates
        // code for it: reading a schedule's work as ordinary statements is
        // worth the pass on its own.
        new_funcs[fname] = codegen_stmt(*f, func_type_map);

        // A vectorized function is generated from the SSA instead, which is
        // the shape its control flow is actually in -- what partial
        // linearization leaves behind fits structured statements worst.
        // Asked of the SSA function, which is where vectorize() recorded it.
        const bool direct =
            std::find(f->attributes.begin(), f->attributes.end(),
                      ir::Function::Attribute::vectorized) !=
            f->attributes.end();
        if (keep_ssa != nullptr && direct) {
            keep_ssa->ssa_funcs[fname] = f;
            if (options.is_verbose) {
                std::cerr << "; keeping the SSA of " << fname
                          << " for direct lowering\n";
            }
        }
    }

    return new_funcs;
}

} // namespace

ir::Program ConvertToSSA::run(ir::Program program,
                              const CompilerOptions &options) const {
    ir::TransformMap transforms;
    if (const auto it = program.schedules.find(ir::Target::Host);
        it != program.schedules.end()) {
        transforms = it->second.func_transforms;
    }

    ir::Program new_program;
    new_program.types = program.types;
    new_program.externs = program.externs;
    new_program.schedules = program.schedules;
    new_program.funcs =
        convert(std::move(program.funcs), transforms, options, &new_program);
    return new_program;
}

ir::FuncMap ConvertToSSA::run(ir::FuncMap funcs,
                              const CompilerOptions &options) const {
    return convert(std::move(funcs), {}, options);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
