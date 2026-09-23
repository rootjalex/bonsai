#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/CloseBodies.h"
#include "SSA/CodeGen_Stmt.h"
#include "SSA/Contract.h"
#include "SSA/Defer.h"
#include "SSA/BlockAccumulates.h"
#include "SSA/DemoteAtomics.h"
#include "SSA/InvariantDivision.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/Simplify.h"
#include "SSA/SortRecursion.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"
#include "Lower/Random.h"

#include "Utils.h"

#include <chrono>
#include <cstdlib>
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

    // A switch is a Dispatch with a target per arm: the k-th target is taken
    // on k, the last on anything else, which is what the statement means (see
    // ir::SwitchStmt). The arms meet at a merge block, as an if's do, unless
    // every one of them returns; an arm with nothing in it goes straight
    // there.
    void visit(const SwitchStmt *node) override {
        auto v = get_value(node->value); // in current block
        internal_assert(!block->terminator.defined());

        bool needs_merge = false;
        for (const Stmt &arm : node->arms) {
            needs_merge = needs_merge || !arm.defined() || !always_returns(arm);
        }
        std::shared_ptr<Block> merge_block =
            needs_merge ? make_block("merge") : nullptr;

        std::vector<std::shared_ptr<Block>> cases;
        Terminator::Dispatch dispatch{.cond = std::move(v), .targets = {}};
        for (size_t k = 0; k < node->arms.size(); k++) {
            const Stmt &arm = node->arms[k];
            if (!arm.defined()) {
                cases.push_back(nullptr);
                dispatch.targets.push_back(
                    Terminator::Jump{.name = merge_block->name, .args = {}});
                continue;
            }
            cases.push_back(make_block("case"));
            // The arm's provenance rides on the block the arm begins with,
            // which is what a schedule's cursor finds (see ir::Provenance).
            if (!node->provenance.empty()) {
                cases.back()->provenance = node->provenance[k];
            }
            dispatch.targets.push_back(
                Terminator::Jump{.name = cases.back()->name, .args = {}});
        }
        block->terminator.data = std::move(dispatch);

        auto curr_block = std::move(block);
        bool merges_from_here = false;
        for (size_t k = 0; k < node->arms.size(); k++) {
            if (cases[k] == nullptr) {
                merges_from_here = true;
                continue;
            }
            cases[k]->preds.push_back(curr_block);
            block = cases[k];
            node->arms[k].accept(this);
            if (!always_returns(node->arms[k])) {
                // *current* insert block is predecessor to merge.
                merge_block->preds.push_back(block);
                set_block_jump(merge_block->name);
            }
        }
        if (merges_from_here) {
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
        // The body may have branched, so the block the back edge leaves from
        // is whichever one it ended in rather than the header itself.
        auto latch = block;

        block->terminator.data = Terminator::Dispatch{
            .cond = std::move(v),
            // v == 0
            .targets = {Terminator::Jump{.name = loop_end->name, .args = {}},
                        // v != 0
                        Terminator::Jump{.name = loop_head->name,
                                         .args = close_back_edge(latch,
                                                                 loop_head)}}};
        loop_end->preds.push_back(block);
        loop_head->preds.push_back(block); // possible self-cycle?
        block = loop_end;
    }

    // What a latch has to hand back to the header it jumps to.
    //
    // Reading a variable the header does not define gives the header an
    // argument and threads the value in from the preds it has -- which, while
    // the loop is being built, is only the block before it. Everything the body
    // read from outside the loop is therefore a header argument by the time the
    // body is done, and closing the cycle means passing each one on as the body
    // left it.
    //
    // The list is walked by index rather than copied first, because fetching
    // one value can thread another in: the latch is downstream of the header,
    // so asking it for a name neither has adds an argument to both. New
    // arguments are appended, so the answers stay lined up with them.
    std::vector<std::shared_ptr<Value>>
    close_back_edge(const std::shared_ptr<Block> &latch,
                    const std::shared_ptr<Block> &header) {
        std::vector<std::shared_ptr<Value>> back;
        for (size_t i = 0; i < header->args.size(); i++) {
            const Argument carried = header->args[i];
            back.push_back(latch->get_value(carried.name, carried.type));
        }
        return back;
    }

    void visit(const While *node) override {
        // The same three blocks as a do-while, except that the test is at the
        // top rather than the bottom: a header holding the condition, a body,
        // and the block after. The header is its own block because control
        // comes back to it, and it must hold the condition rather than merely
        // dispatch on one computed before the loop -- the value is recomputed
        // every time round.
        std::shared_ptr<Block> header = make_block("while");
        internal_assert(!block->terminator.defined());
        block->terminator.data = Terminator::Jump{.name = header->name};
        header->preds.push_back(block);

        block = header;
        auto v = get_value(node->cond);
        // The condition may itself have branched, in which case the dispatch
        // belongs at the end of wherever evaluating it left us.
        std::shared_ptr<Block> test = block;

        std::shared_ptr<Block> body = make_block("while_body");
        std::shared_ptr<Block> end = make_block("while_end");
        internal_assert(!test->terminator.defined());
        test->terminator.data = Terminator::Dispatch{
            .cond = std::move(v),
            // v == 0 leaves the loop; v != 0 enters the body.
            .targets = {Terminator::Jump{.name = end->name, .args = {}},
                        Terminator::Jump{.name = body->name, .args = {}}}};
        end->preds.push_back(test);
        body->preds.push_back(test);

        block = body;
        node->body.accept(this);
        // A body that returned has already terminated, and there is no back
        // edge from it.
        if (!block->terminator.defined()) {
            auto latch = block;
            latch->terminator.data = Terminator::Jump{
                .name = header->name, .args = close_back_edge(latch, header)};
            header->preds.push_back(latch);
        }
        block = end;
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

    void visit(const MultiRecurse *node) override {
        auto func = get_value(node->func);
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());
        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }

        std::vector<std::vector<std::shared_ptr<Value>>> varying;
        varying.reserve(node->varying.size());
        for (const auto &vs : node->varying) {
            std::vector<std::shared_ptr<Value>> values;
            values.reserve(vs.size());
            for (const auto &v : vs) {
                values.emplace_back(get_value(v));
            }
            varying.push_back(std::move(values));
        }

        std::vector<std::shared_ptr<Value>> keys;
        keys.reserve(node->keys.size());
        for (const auto &key : node->keys) {
            keys.emplace_back(get_value(key));
        }

        // Every value the run uses has to reach this block, including the ones
        // that only one of the calls passes: the run is a single terminator,
        // so there is nowhere later for them to be threaded in.
        rethread(args);
        for (auto &vs : varying) {
            rethread(vs);
        }
        rethread(keys);

        auto call_name = get_call_name(func);

        auto cont_block = make_block(call_name + "_multicall_cont");
        cont_block->preds.push_back(block);

        internal_assert(!block->terminator.defined());
        auto call_block = std::move(block);

        call_block->terminator.data = Terminator::MultiCall{
            .call =
                Terminator::Jump{.name = call_name, .args = std::move(args)},
            .cont = Terminator::Jump{.name = cont_block->name, .args = {}},
            .varying_at = node->varying_at,
            .varying = std::move(varying),
            .keys = std::move(keys),
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

        var = walk_accesses(std::move(var), node->loc);

        std::vector<std::shared_ptr<Value>> args = {std::move(var),
                                                    std::move(v)};
        block->make_side_effect(Instruction::Op::Store, std::move(args));
    }

    // Walk a write location's accesses, turning each into the address it
    // names: an index into an array is a GEP, and a field of a struct is a
    // FieldPtr carrying that field's position.
    //
    // Shared by Store and Accumulate because they ask the same question. They
    // used to have a copy each, and both copies gave up on a field -- so
    // `s.field = v` for any struct held in memory had no lowering at all,
    // which is the ordinary way to write to a mutable struct.
    std::shared_ptr<Value> walk_accesses(std::shared_ptr<Value> ptr,
                                         const WriteLoc &loc) {
        static const Type u32 = UInt_t::make(32);
        // The type as each access narrows it, which is what says where a
        // field sits. WriteLoc keeps this in step for its own `type`, but only
        // for the whole chain, so it is tracked again here step by step.
        Type current = loc.base_type;
        for (const auto &value : loc.accesses) {
            if (std::holds_alternative<std::string>(value)) {
                const std::string &field = std::get<std::string>(value);
                const Struct_t *struct_t = current.as<Struct_t>();
                internal_assert(struct_t)
                    << "field access `" << field << "` on non-struct type "
                    << current << " in " << loc.base;
                const auto idx = find_struct_index(field, struct_t->fields);
                current = get_field_type(current, field);
                ptr = block->make_instruction(
                    Ptr_t::make(current), Instruction::Op::FieldPtr,
                    {std::move(ptr), make_constant(u32, (uint64_t)idx)});
            } else {
                Expr idx = std::get<Expr>(value);
                auto i = get_value(idx);
                // The address of one element, typed as such -- the same way
                // the PtrTo visitor below types the GEPs it builds. The type
                // is what tells a later pass what a store through it writes,
                // and what a vectorized index turns it into: a per-lane index
                // makes it a vector of element addresses, a scatter.
                Type element;
                if (current.defined()) {
                    current = current.element_of();
                    element = Ptr_t::make(current);
                }
                ptr = block->make_instruction(element, Instruction::Op::GEP,
                                              {std::move(ptr), std::move(i)});
            }
        }
        return ptr;
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

            ptr = walk_accesses(std::move(ptr), node->loc);
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

    void visit(const Undef *node) override {
        value = std::make_shared<Value>(Constant{node->type, Undefined{}});
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

        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        internal_assert(struct_t) << node->value.type() << " of " << Expr(node);
        auto idx = find_struct_index(node->field, struct_t->fields);
        auto vidx = make_constant(u32, (uint64_t)idx);
        value = block->make_instruction(node->type, Instruction::Op::LoadField,
                                        {std::move(v), std::move(vidx)});
    }

    // The constant indices ride on the instruction, as a reduction's kind
    // does; the vectors are its operands.
    void visit(const Shuffle *node) override {
        std::vector<std::shared_ptr<Value>> vectors;
        vectors.reserve(node->vectors.size());
        for (const Expr &v : node->vectors) {
            vectors.push_back(get_value(v));
        }
        value = block->make_instruction(node->type, Instruction::Op::Shuffle,
                                        std::move(vectors));
        std::get<std::shared_ptr<Instruction>>(value->data)->shuffle =
            node->indices;
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
        // The element's type, from the container's when the expression arrived
        // without one -- a loop bound written as an element of an array is one
        // such (see Extract::make, which infers the type only when the
        // container's is known as it is built). Every instruction here has to
        // have a type: the passes downstream read the shape of a value off it,
        // and a vectorized one is widened by rewriting it.
        Type type = node->type;
        if (!type.defined()) {
            const Type &container = vec->get_type();
            internal_assert(container.defined())
                << "Element read of a container of unknown type: " << Expr(node);
            type = container.element_of();
        }
        value = block->make_instruction(std::move(type),
                                        Instruction::Op::ExtractIdx,
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
        args.reserve(node->args.size() + 1);
        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }
        // `rand` reads and advances the generator's state, which the Stmt
        // form leaves implicit: the backend finds it by name. In SSA a
        // block's values are what it is passed, so the state is an operand,
        // last, after the count when there is one -- which is what threads
        // it into a parfor body's arguments and so into the captures of the
        // kernel the body becomes (see CodeGen_LLVM::launch_captures, which
        // then leaves it out of the launch: each thread seeds its own). The
        // paths back to an expression drop the operand again.
        if (node->op == Intrinsic::rand) {
            args.emplace_back(block->get_value(
                lower::rng_state_name, Ptr_t::make(Rand_State_t::make())));
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
            value = block->make_instruction(node->type, Instruction::Op::Not,
                                            {std::move(a)});
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

        // A place named by a chain of accesses -- `vs[i].w`, `(*h).c`,
        // `prims[k].payload.Geom.g` -- has an address that is an offset from
        // what the chain is rooted at: a dereferenced pointer, or an array,
        // which is already the address of its elements (see
        // Type::is_reference). The chain is walked in to its root and the
        // address composed back out, an index as a GEP and a field as a
        // FieldPtr, the way walk_accesses does for the destination of a
        // store.
        //
        // Without this a field of an element was loaded and the address taken
        // *of the loaded value*, so a callee taking it by `mut` wrote into a
        // copy; `next_float(st.rng)`, a mutable field of a mutable parameter,
        // was the first case (tests/bonsai/correctness/llvm/
        // mut-field-argument.bonsai), and `bump(vs[i].w)`, a field of an
        // array's element, the second (mut-element-field-argument.bonsai).
        // An argmin that keeps a reference to its best element rather than a
        // copy of it (Lower/Trees.cpp) is made of exactly these addresses.
        if (auto place = address_of_place(node->expr)) {
            value = std::move(place);
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

    // The address of `expr` when it names a place, or null when it does not.
    //
    // From the outside in, the expression is peeled of its field and index
    // accesses down to a root that is in memory: a `Deref`, whose pointer is
    // the base, or an expression of array type, whose value is the address of
    // its elements. Then from the inside out each access becomes an offset:
    // FieldPtr for a field of the struct the pointer names, GEP for an index
    // into an array. An index into a vector, or a chain rooted at a value
    // that is nowhere in memory, names no place.
    std::shared_ptr<Value> address_of_place(const Expr &expr) {
        static const Type u32 = UInt_t::make(32);
        std::vector<std::variant<std::string, Expr>> accesses; // outermost first
        Expr root = expr;
        std::shared_ptr<Value> base;
        Type current; // what `base` points at
        while (true) {
            if (!accesses.empty() && root.type().is_reference()) {
                base = get_value(root);
                current = root.type();
                break;
            }
            if (const Deref *deref = root.as<Deref>(); deref != nullptr &&
                                                        !accesses.empty()) {
                base = get_value(deref->expr);
                current = deref->type;
                break;
            }
            if (const Access *access = root.as<Access>()) {
                accesses.emplace_back(access->field);
                root = access->value;
                continue;
            }
            if (const Extract *extract = root.as<Extract>()) {
                accesses.emplace_back(extract->idx);
                root = extract->vec;
                continue;
            }
            return nullptr;
        }
        for (auto it = accesses.rbegin(); it != accesses.rend(); ++it) {
            if (const auto *field = std::get_if<std::string>(&*it)) {
                size_t idx = 0;
                if (const Struct_t *struct_t = current.as<Struct_t>()) {
                    idx = find_struct_index(*field, struct_t->fields);
                    current = get_field_type(current, *field);
                } else {
                    return nullptr;
                }
                base = block->make_instruction(
                    Ptr_t::make(current), Instruction::Op::FieldPtr,
                    {std::move(base), make_constant(u32, (uint64_t)idx)});
            } else {
                if (!current.is_reference()) {
                    return nullptr; // a lane of a vector, which is a value
                }
                current = current.element_of();
                base = block->make_instruction(
                    Ptr_t::make(current), Instruction::Op::GEP,
                    {std::move(base), get_value(std::get<Expr>(*it))});
            }
        }
        return base;
    }

    void visit(const AtomicAdd *node) override {
        // Fetch-and-add. The pointer is already a pointer -- Lower/ADTs.cpp
        // and Lower/Defers.cpp both build one with a PtrTo, which the visitor
        // above turns into a GEP or an AddressOf -- so there is nothing to
        // address here, only the two operands to hand over.
        auto ptr = get_value(node->ptr);
        auto amount = get_value(node->value);
        value = block->make_instruction(node->type, Instruction::Op::AtomicAdd,
                                        {std::move(ptr), std::move(amount)});
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
    RESTRICT_VISITOR(MatchExpr);
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
    // RESTRICT_VISITOR(AtomicAdd);

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

// The block graph itself, which nothing downstream can be asked about: the
// relooper's reading of it is what `-p ssa` prints, and two different graphs
// can reloop to the same statements. See
// CompilerOptions::dump_ssa_preschedule.
//
// The program's own functions and not the library's, unless asked for all of
// them -- the same rule, and the same meaning of verbose, that the program
// printer applies. A golden of a schedule's work should not carry every
// geometric predicate the query happened to reach.
//
// `std::map` iteration is ordered, so the dump is deterministic.
void dump_ssa(std::ostream &os, const std::string &when, const FuncMap &fmap,
              bool include_imported) {
    const auto has = [](const std::shared_ptr<Function> &f,
                        ir::Function::Attribute a) {
        return std::find(f->attributes.begin(), f->attributes.end(), a) !=
               f->attributes.end();
    };
    os << "; === ssa " << when << " ===\n";
    for (const auto &[fname, f] : fmap) {
        if (!include_imported && has(f, ir::Function::Attribute::imported)) {
            continue;
        }
        os << "; --- " << fname << " ---\n";
        f->dump(os);
    }
}

// Applies the SSA-level schedule `transforms` and builds/codegens `funcs`
// through the SSA representation. A transform this pipeline cannot apply is
// reported, not skipped -- see the visit below.
// A schedule's `f.skip(Shape.Disc)` has to name something: an arm of a match
// written in `f`, wherever inlining has put it (see ir::ArmCursors), and a
// bare `f.skip()` a function the program has. Said here, before any transform
// runs, with what the program does have, since a cursor that finds nothing
// would otherwise be a directive the program was compiled without (compare
// the transforms below, none of which is allowed to be ignored either).
void check_branch_policies(const FuncMap &fmap,
                           const ir::BranchPolicyMap &policies) {
    // Every match arm in the program, with the function it is in now.
    std::vector<std::pair<std::string, ir::Provenance>> arms;
    for (const auto &[fname, f] : fmap) {
        for (const auto &block : f->blocks) {
            if (block->provenance.defined()) {
                arms.emplace_back(fname, block->provenance);
            }
        }
    }
    const auto written_in = [&](const std::string &fname) {
        std::set<std::string> names;
        for (const auto &[in, arm] : arms) {
            if (arm.func() == fname) {
                names.insert(arm.str());
            }
        }
        return names;
    };
    const auto spell = [](const std::set<std::string> &names) {
        std::string all;
        for (const std::string &n : names) {
            all += (all.empty() ? "" : ", ") + n;
        }
        return all;
    };
    for (const auto &[fname, policy] : policies) {
        if (policy.skip.all && !fmap.contains(fname) &&
            written_in(fname).empty()) {
            internal_error << "skip() on " << fname
                           << ": no such function, and no match arm was "
                              "written in one of that name.";
        }
        for (const ir::Location &cursor : policy.skip.arms) {
            std::string spelled;
            for (const std::string &n : cursor.names) {
                spelled += (spelled.empty() ? "" : ".") + n;
            }
            const bool found =
                std::any_of(arms.begin(), arms.end(), [&](const auto &arm) {
                    return arm.second.func() == fname &&
                           arm.second.matches(cursor.names);
                });
            if (found) {
                continue;
            }
            // What is there instead: the arms written in the function, and
            // the ones inlined into it, which are named by where they were
            // written.
            const std::set<std::string> own = written_in(fname);
            std::map<std::string, std::set<std::string>> inlined;
            for (const auto &[in, arm] : arms) {
                if (in == fname && arm.func() != fname) {
                    inlined[arm.func()].insert(arm.str());
                }
            }
            std::string instead =
                own.empty() ? "No match is written in " + fname + "."
                            : "The arms written in " + fname + " are: " +
                                  spell(own) + ".";
            for (const auto &[origin, names] : inlined) {
                instead += " The arms of " + origin + " inlined into it (" +
                           spell(names) + ") are named by " + origin + ".";
            }
            internal_error << "skip(" << spelled << ") on " << fname
                           << ": no arm of a match written in " << fname
                           << " takes " << spelled << ". " << instead;
        }
    }
}

ir::FuncMap convert(ir::FuncMap funcs, const ir::TransformMap &transforms,
                    const ir::TransformOrder &order,
                    const ir::BranchPolicyMap &policies,
                    const std::map<std::string, ir::Queue> &queues,
                    const CompilerOptions &options,
                    ir::Program *keep_ssa = nullptr) {
    FuncMap fmap;

    TypeMap func_type_map;

    for (const auto &[name, func] : funcs) {
        func_type_map[name] = func->call_type();
        auto f = build(func);
        fmap[name] = std::move(f);
    }

    check_branch_policies(fmap, policies);

    // Before any rewrite has touched it, so that a transform's golden can say
    // what it changed and not only what it ended at.
    if (options.dump_ssa_preschedule) {
        dump_ssa(std::cout, "preschedule", fmap, options.is_verbose);
    }

    // Sort before anything else, and before loopify in particular: loopify
    // replaces a run of recursive calls with pushes onto a stack, and a sort
    // applied after that has nothing left to permute.
    //
    // Every function, rather than the ones the schedule names, because the
    // recursion a `trace.sort(...)` is about lives in a traversal function that
    // lowering invented and the schedule cannot name. The keys are on the IR by
    // now, so a function no schedule sorted has none and this does nothing to
    // it.
    for (const auto &[name, f] : fmap) {
        sort_recursion(*f);
        // What the network compared was built by rule; this is where it is
        // looked at (see SSA/Simplify.h).
        simplify(*f);
    }

    // A division by a value that does not change while its loop runs becomes
    // a multiply, with the multiplier computed where the divisor is (see
    // SSA/InvariantDivision.h). It has to see a loop as a loop -- a
    // recursion's unchanging argument is a loop-invariant only once loopify()
    // has made the recursion one -- and see it before it is vectorized,
    // because what it leaves needs none of the guarding a vectorized division
    // does, and the vectorizer widens the multiply like any other arithmetic.
    // So it runs over every function just before the first vectorize, and
    // again after each loopify written later, over the functions that are not
    // a gang's: a loop such a loopify makes inside a gang's function has its
    // divisions guarded already.
    bool divided = false;
    const auto divide_all = [&]() {
        for (const auto &[fname, f] : fmap) {
            const bool gang =
                std::find(f->attributes.begin(), f->attributes.end(),
                          ir::Function::Attribute::vectorized) !=
                f->attributes.end();
            if (!gang) {
                divide_by_invariants(*f);
            }
        }
    };

    // The rest in the order the schedule wrote them, across functions as much
    // as within one, because the order decides what is built. A recursion
    // loopified and then vectorized is a loop each lane walks at its own
    // pace, over a stack of its own; vectorized and then loopified it is a
    // recursion the gang makes together, on one node with a mask of the lanes
    // that reached it, which loopify then puts on one stack of nodes and one
    // of masks -- a packet traversal. Only the schedule can say which.
    //
    // A schedule built without a source order -- none is today -- is applied
    // function by function in name order, each function's directives in the
    // order they were listed.
    ir::TransformOrder ordered = order;
    if (ordered.empty()) {
        for (const auto &[name, ts] : transforms) {
            for (size_t i = 0; i < ts.size(); i++) {
                ordered.emplace_back(name, i);
            }
        }
    }

    // See BONSAI_TIME_PASSES in Lower/Lower.cpp: the time each transform
    // takes, since this one pass is most of a vectorized compile.
    const bool timing = std::getenv("BONSAI_TIME_PASSES") != nullptr;
    for (const auto &[name, index] : ordered) {
        if (!fmap.contains(name)) {
            continue;
        }
        const ir::Transform &t = transforms.at(name).at(index);
        const auto started = std::chrono::steady_clock::now();
        const auto report = [&] {
            if (!timing) {
                return;
            }
            const std::chrono::duration<double> took =
                std::chrono::steady_clock::now() - started;
            const char *kind = std::visit(
                overloads{[](const ir::Bind &) { return "bind"; },
                          [](const ir::Collapse &) { return "collapse"; },
                          [](const ir::Defer &) { return "defer"; },
                          [](const ir::Loopify &) { return "loopify"; },
                          [](const ir::Split &) { return "split"; },
                          [](const ir::Sort &) { return "sort"; },
                          [](const ir::Vectorize &) { return "vectorize"; }},
                t);
            std::cerr << "[time]   " << name << "." << kind << ": "
                      << took.count() << " s\n";
        };
        const auto unimplemented = [&name](const std::string &what) {
            internal_error
                << what << "() is in the schedule for " << name
                << ", but the SSA pipeline does not apply it yet. Ignoring it "
                   "would compile a different program than the schedule asks "
                   "for, so it is an error instead. Compile without `-p ssa` "
                   "to use it today.";
        };

        {
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
                        if (!divided) {
                            divide_all();
                            divided = true;
                        }
                        const LoopSite at = resolve_loop(
                            fmap, name, v.i.names.back(), "vectorize");
                        vectorize(fmap, at.func, at.index, policies);
                    },
                    [&](const ir::Loopify &l) {
                        int size = 0;
                        if (l.queue_size.has_value()) {
                            const auto n =
                                get_constant_value<int64_t>(*l.queue_size);
                            internal_assert(n.has_value() && *n > 0)
                                << "loopify(" << *l.queue_size << ") on "
                                << name
                                << " needs a constant, positive stack depth";
                            size = int(*n);
                        }
                        loopify(fmap, name, size);
                        if (divided) {
                            divide_all();
                        }
                    },
                    [&](const ir::Defer &d) {
                        internal_assert(!d.callee.names.empty())
                            << "defer() requires a callee for: " << name;
                        const auto q = queues.find(d.queue);
                        internal_assert(q != queues.end())
                            << name << ".defer(" << d.callee.names.back()
                            << ", " << d.queue << ") names a queue no "
                            << "schedule block declares. Declare it: `"
                            << d.queue << " = <func>.queue(<loop>);`";
                        QueueSpec spec;
                        spec.name = d.queue;
                        spec.owner = q->second.owner;
                        internal_assert(!q->second.loop.names.empty())
                            << d.queue << " names no loop";
                        spec.loop = q->second.loop.names.back();
                        if (q->second.capacity.has_value()) {
                            const auto n = get_constant_value<int64_t>(
                                *q->second.capacity);
                            internal_assert(n.has_value() && *n > 0)
                                << d.queue << " = " << spec.owner << ".queue("
                                << spec.loop << ", " << *q->second.capacity
                                << ") needs a constant, positive capacity";
                            spec.capacity = uint64_t(*n);
                        }
                        // The entry, queue and flag types it made are the
                        // program's now, for the printer and the backends'
                        // declarations.
                        for (const Type &made :
                             defer(fmap, name, d.callee.names.back(), spec)) {
                            const auto *s = made.as<Struct_t>();
                            internal_assert(s) << made;
                            if (keep_ssa != nullptr) {
                                keep_ssa->types[s->name] = made;
                            }
                        }
                    },
                    // Applied earlier in lowering, by the pass named.
                    [&](const ir::Sort &) {}, // Lower/Sorts.cpp
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
                        if (b.lambda.defined()) {
                            // A function's bind to a hardware unit, applied
                            // by lower::LowerHardwareBinds before this; the
                            // function's body is the unit's already.
                            return;
                        }
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
        report();
    }
    if (!divided) {
        divide_all();
    }

    // Now that every bind has been applied, it is settled which loops run
    // iterations at the same time -- and so which atomics were asked for
    // against a parallelism the schedule did not take up. Those cost nothing
    // to remove and everything to keep, so they go here, after the schedule
    // and before the graph is turned back into statements.
    for (const auto &[name, f] : fmap) {
        DemoteAtomics::run(*f);
        // The atomics every thread of a GPU block makes to one word, once
        // per block instead (SSA/BlockAccumulates.h); before the allocas
        // are promoted, so the slots it makes become values.
        ReduceBlockAccumulates::run(*f);
    }

    // What the transforms built by rule is looked at once more (see
    // SSA/Simplify.h): a vectorized split loop's index is the outer index
    // broadcast plus the ramp of lanes, which is a ramp, and only as a ramp
    // is an access at it a dense vector load or store rather than a gather.
    for (const auto &[name, f] : fmap) {
        simplify(*f);
    }

    // Contraction after the schedule as well, and for a related reason: what a
    // transform produces is arithmetic too. A vectorized gang's widened
    // multiply and add are as fusible as the scalar pair they came from, and a
    // pass that ran before the schedule would have missed them.
    if (options.ffp_contract) {
        for (const auto &[name, f] : fmap) {
            contract_fp(*f);
        }
    }

    // A transform may have added functions -- vectorize() specializes the
    // callees of a gang -- whose types nothing has recorded yet, or changed
    // a function's shape -- defer() hands a chain of functions the queue and
    // has them return a flag. A signature is whatever the entry block takes
    // and the return says, so every function's is read back from the graph.
    for (const auto &[name, f] : fmap) {
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
    if (options.dump_ssa_postschedule) {
        dump_ssa(std::cout, "postschedule", fmap, options.is_verbose);
    }
    if (options.is_verbose && !options.dump_ssa_postschedule) {
        dump_ssa(std::cerr, "postschedule", fmap, /*include_imported=*/true);
    }

    // A queue's push stays one instruction while the schedule is applied,
    // for the vectorizer to recognize; from here on it is the fetch-and-add
    // and the store it stands for (see SSA/Defer.h).
    for (const auto &[name, f] : fmap) {
        lower_pushes(*f);
    }
    // A `mut` local the builder put in memory that nothing but loads and
    // stores ever touch is a value from here on (SSA/PromoteAllocas.h). After
    // the rewrites rather than before them, because a rewrite may be about
    // the memory: defer() saves a `mut` local of the producer's iteration in
    // the queue entry by its slot, and would not see a value to save. Before
    // the code generators, because a loop body handed a value can run
    // anywhere, where one handed the address of a slot on the function's
    // stack cannot -- the sample count a match on the sampler settled is
    // what the GPU thread loop's bound is, and it has to reach the launch.
    for (const auto &[name, f] : fmap) {
        promote_allocas(*f, f->blocks.front()->name);
    }
    // Every parfor body's arguments are its captures again, whatever the
    // rewrites above and the promotion left of that (see SSA/CloseBodies.h):
    // the code generators make a kernel of a bound loop's body from those
    // arguments.
    for (const auto &[name, f] : fmap) {
        close_parfor_bodies(*f);
    }

    ir::FuncMap new_funcs;

    for (const auto &[fname, f] : fmap) {
        // The relooper runs for every function regardless of what generates
        // code for it: reading a schedule's work as ordinary statements is
        // worth the pass on its own, and the C++ and CUDA backends print
        // from the statements.
        new_funcs[fname] = codegen_stmt(*f, func_type_map);

        // Every function's SSA is kept, and the LLVM backends generate from
        // it rather than from the statements: blocks with arguments are
        // blocks with phis, which is the form LLVM wants, and the relooper
        // re-derives structure that LLVM then discards -- a re-derivation
        // that cannot represent what partial linearization leaves of a
        // vectorized function at all. A program with a loop on the GPU
        // depends on this too: the loop's body is cut out of the block graph
        // as a kernel, and the device module compiles every function the
        // kernel reaches from the same SSA the host has (see CodeGen_PTX),
        // so that the two sides are one reading of the program.
        if (keep_ssa != nullptr) {
            keep_ssa->ssa_funcs[fname] = f;
        }
    }

    return new_funcs;
}

} // namespace

ir::Program ConvertToSSA::run(ir::Program program,
                              const CompilerOptions &options) const {
    ir::TransformMap transforms;
    ir::TransformOrder order;
    ir::BranchPolicyMap policies;
    std::map<std::string, ir::Queue> queues;
    if (const auto it = program.schedules.find(ir::Target::Host);
        it != program.schedules.end()) {
        transforms = it->second.func_transforms;
        order = it->second.transform_order;
        policies = it->second.branch_policies;
        queues = it->second.queues;
    }

    ir::Program new_program;
    new_program.types = program.types;
    new_program.externs = program.externs;
    new_program.schedules = program.schedules;
    new_program.funcs = convert(std::move(program.funcs), transforms, order,
                                policies, queues, options, &new_program);
    return new_program;
}

ir::FuncMap ConvertToSSA::run(ir::FuncMap funcs,
                              const CompilerOptions &options) const {
    return convert(std::move(funcs), {}, {}, {}, {}, options);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
