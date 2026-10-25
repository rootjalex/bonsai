#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/CloseBodies.h"
#include "SSA/CodeGen_Stmt.h"
#include "SSA/Contract.h"
#include "SSA/CloneFunction.h"
#include "SSA/Defer.h"
#include "SSA/InducedStorage.h"
#include "SSA/BlockAccumulates.h"
#include "SSA/Contention.h"
#include "SSA/GateDrains.h"
#include "SSA/HitContinuations.h"
#include "SSA/RoundGuard.h"
#include "SSA/DemoteAtomics.h"
#include "SSA/HeapArrays.h"
#include "SSA/DestinationPassing.h"
#include "SSA/HoistAllocations.h"
#include "SSA/QueueStorage.h"
#include "SSA/ReorderLoops.h"
#include "SSA/Specialize.h"
#include "SSA/Stage.h"
#include "SSA/InvariantDivision.h"
#include "SSA/MirrorLoads.h"
#include "SSA/PackBoolArrays.h"
#include "SSA/PromoteAllocas.h"
#include "SSA/Rewrite.h"
#include "SSA/ConstantIntervals.h"
#include "SSA/Simplify.h"
#include "SSA/SortRecursion.h"
#include "SSA/SSA.h"
#include "SSA/Unroll.h"

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
            Argument a = {arg.type, arg.name, arg.mutating, arg.unaliased,
                          arg.reducer};
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
    // Names for the results of `&&`, `||` and `ite` lowered as control flow
    // (see short_circuit): a `!` in the name, as a block's has, so that no
    // name the program wrote can be the same.
    uint64_t conditional_counter = 0;
    std::string conditional_name(const std::string &prefix) {
        return "!" + prefix + "_" + std::to_string(conditional_counter++);
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

        const bool then_needs_merge = !never_falls_through(node->then_body);
        const bool else_needs_merge =
            !node->else_body.defined() || !never_falls_through(node->else_body);
        const bool needs_merge = then_needs_merge || else_needs_merge;
        std::shared_ptr<Block> merge_block =
            needs_merge ? make_block("merge") : nullptr;

        // The arms carry where the `if` was written, for a schedule's
        // `skip` to find them (ir::Provenance::if_arm). An `if` with no
        // else whose arm always returns -- `if !(hit) { return {}; }` --
        // has the code after it as its other arm, reached by the false edge
        // alone: the merge block is that arm, and is named too.
        then_case->provenance = node->provenance;
        if (else_case) {
            else_case->provenance = node->provenance;
        } else if (!then_needs_merge && merge_block) {
            merge_block->provenance = node->provenance;
        }

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
            needs_merge =
                needs_merge || !arm.defined() || !never_falls_through(arm);
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
            if (!never_falls_through(node->arms[k])) {
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
        loop_exits.push_back(end);
        node->body.accept(this);
        loop_exits.pop_back();
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
        // A let of an effect that computes nothing -- `prefetch(...)`, bound
        // to a name nothing reads so that it has a place among statements --
        // is the effect alone here, an instruction with no name, as a store
        // is (see visit(const Intrinsic *)).
        if (const Intrinsic *in = node->value.as<Intrinsic>();
            in != nullptr && in->type.is<Void_t>()) {
            node->value.accept(this);
            return;
        }
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

    // The loops being built, innermost last: what a `break` leaves. A
    // `break` ends its block with a jump to the innermost loop's exit, as a
    // return ends its block; the statement after it, if any, is a merge the
    // if-conversion knows this arm does not reach (ir::always_returns).
    std::vector<std::shared_ptr<Block>> loop_exits;

    void visit(const Break *) override {
        internal_assert(!loop_exits.empty())
            << "break outside of any loop in " << block->name;
        internal_assert(loop_exits.back() != nullptr)
            << "break inside a parallel loop's body in " << block->name
            << ": a parallel body is a batch, and a break belongs after it";
        internal_assert(!block->terminator.defined());
        const std::shared_ptr<Block> &exit = loop_exits.back();
        block->terminator.data =
            Terminator::Jump{.name = exit->name, .args = {}};
        exit->preds.push_back(block);
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
        auto cont_block = make_block(call_name + (node->spawned ? "_spawn_cont"
                                                                : "_call_cont"));
        cont_block->preds.push_back(block);

        // End the current block with a Call -- or a Spawn, for `spawn
        // f(args);`: the same two edges, the call's and the continuation's,
        // under a terminator that says the two may run at once
        // (Terminator::Spawn). A spawned call has no value to receive.
        internal_assert(!block->terminator.defined());
        auto call_block = std::move(block);

        if (node->spawned) {
            call_block->terminator.data = Terminator::Spawn{
                .call = Terminator::Jump{.name = call_name,
                                         .args = std::move(args)},
                .cont = Terminator::Jump{.name = cont_block->name, .args = {}}};
        } else {
            call_block->terminator.data = Terminator::Call{
                .call = Terminator::Jump{.name = call_name,
                                         .args = std::move(args)},
                // 'args' is empty for now (implicit capture via CFG lookup)
                // Does not receive output of call, because value is dropped.
                .cont = Terminator::Jump{.name = cont_block->name, .args = {}},
                .drop = true};
        }

        block = cont_block;
    }

    // `join;` ends the block: everything spawned so far is waited for
    // before what follows runs (Terminator::Join). Its own block, so that
    // the point is visible to the deferral, which moves what follows a join
    // after the drain that runs the spawned calls.
    void visit(const Join *node) override {
        auto next_block = make_block("join");
        next_block->preds.push_back(block);
        internal_assert(!block->terminator.defined());
        block->terminator.data =
            Terminator::Join{.next = Terminator::Jump{.name = next_block->name}};
        block = next_block;
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
        std::vector<std::shared_ptr<Value>> conds;
        conds.reserve(node->conds.size());
        for (const auto &cond : node->conds) {
            conds.emplace_back(get_value(cond));
        }

        // Every value the run uses has to reach this block, including the ones
        // that only one of the calls passes: the run is a single terminator,
        // so there is nowhere later for them to be threaded in.
        rethread(args);
        for (auto &vs : varying) {
            rethread(vs);
        }
        rethread(keys);
        rethread(conds);

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
            .conds = std::move(conds),
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
        const bool on_stack = node->memory == Allocate::Stack;
        auto op = on_stack ? Instruction::Op::Alloca : Instruction::Op::Alloc;

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
                const bool readonly = ptr->get_type().is_readonly();
                ptr = block->make_instruction(
                    Ptr_t::make(current, readonly), Instruction::Op::FieldPtr,
                    {std::move(ptr), make_constant(u32, (uint64_t)idx)});
            } else {
                Expr idx = std::get<Expr>(value);
                auto i = get_value(idx);
                // The address of one element, typed as such -- the same way
                // the PtrTo visitor below types the GEPs it builds. The type
                // is what tells a later pass what a store through it writes,
                // and what a vectorized index turns it into: a per-lane index
                // makes it a vector of element addresses, a scatter. An
                // address formed inside read-only storage is read-only.
                Type element;
                if (current.defined()) {
                    current = current.element_of();
                    element = Ptr_t::make(current, ptr->get_type().is_readonly());
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
        case Accumulate::Argmin: {
            return Instruction::Op::AccArgmin;
        }
        case Accumulate::Argmax: {
            return Instruction::Op::AccArgmax;
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

        // Handle local variables: the place is the local's own storage, as
        // for a store -- a scalar's, or a pair's, which an argmin folds into
        // (Accumulate::Argmin on a running best and what holds it).
        if (node->loc.accesses.empty()) {
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
        // The places written alongside the pair, each with its value, after
        // the place and the value (Instruction::alongside); a mask the
        // vectorizer adds later goes between.
        for (const auto &[place, written] : node->alongside) {
            std::shared_ptr<Value> written_v = get_value(written);
            std::shared_ptr<Value> place_ptr =
                block->get_value(place.base, base_lookup_type(place));
            if (!place.accesses.empty()) {
                place_ptr = walk_accesses(std::move(place_ptr), place);
            }
            args.push_back(std::move(place_ptr));
            args.push_back(std::move(written_v));
        }
        block->make_side_effect(op, std::move(args), node->atomic);
        if (!node->alongside.empty()) {
            block->instrs.back()->alongside = uint32_t(node->alongside.size());
        }
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

        // A parallel body is no loop a `break` can leave: its iterations
        // are a batch, and a break among them belongs after the batch
        // (Lower/ForEachs.cpp puts an element-independent one there). The
        // null marks the boundary for visit(Break).
        loop_exits.push_back(nullptr);
        node->body.accept(this);
        loop_exits.pop_back();
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
        loop_exits.push_back(exit_block);
        node->body.accept(this);
        loop_exits.pop_back();
        internal_assert(!block->terminator.defined())
            << "The body of loop " << node->index << " ends in " << block->name
            << ", which already has a terminator: a loop body that leaves "
            << "early on every path has no latch";

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

    // Whether evaluating `e` does work that a condition may spare: a call of
    // any kind, or the unwrapping of an option, which is not defined when
    // the option is empty.
    static bool has_work(const Expr &e) {
        struct Finder : public Visitor {
            using Visitor::visit;
            bool found = false;
            void visit(const Call *) override { found = true; }
            void visit(const GeomOp *) override { found = true; }
            void visit(const SetOp *) override { found = true; }
            void visit(const AggOp *) override { found = true; }
            void visit(const MatchExpr *) override { found = true; }
            void visit(const Unwrap *) override { found = true; }
        };
        Finder finder;
        e.accept(&finder);
        return finder.found;
    }

    // `a && b` and `a || b` short-circuit: `b` is evaluated only when `a`
    // leaves the answer open, as C's do. Lowered as control flow -- `a`
    // decides between a block that evaluates `b` and the merge, where the
    // result is a block parameter -- only when `b` has work in it to spare
    // (has_work); `&&` of two values already computed stays one
    // instruction, which is what the branch would have been made into
    // anyway. What this is for: a traversal's filter written `intersects(q,
    // g) && alpha_accepts(r, g)` ran the alpha test on every candidate,
    // missed or not.
    void short_circuit(const BinOp *node) {
        const bool is_and = node->op == BinOp::LAnd;
        auto a = get_value(node->a);
        internal_assert(!block->terminator.defined());
        const std::string result = conditional_name(is_and ? "and" : "or");
        // The answer `a` settles on its own, bound to the result's name here
        // so that the merge finds it along the edge that skips `b`. Bound in
        // the lookups and not as an instruction named for the result: the
        // name is the merge's parameter's, and a value is told apart from
        // another by its name (value_key in SSA/AnalyzeDivergence.cpp), so
        // two definitions under it -- this constant and `b` -- would read as
        // one value threaded through, and the merge's parameter as nothing;
        // a vectorized merge then kept the constant alone.
        block->lookups[result] = make_constant(node->type, !is_and);
        std::shared_ptr<Block> rhs = make_block(is_and ? "and_rhs" : "or_rhs");
        std::shared_ptr<Block> merge = make_block(is_and ? "and_merge" : "or_merge");
        block->terminator.data = Terminator::Dispatch{
            .cond = std::move(a),
            // a == 0 first, a != 0 second, as an if's: `&&` wants `b` when
            // `a` holds, `||` when it does not.
            .targets = {Terminator::Jump{.name = is_and ? merge->name : rhs->name,
                                         .args = {}},
                        Terminator::Jump{.name = is_and ? rhs->name : merge->name,
                                         .args = {}}}};
        std::shared_ptr<Block> from = std::move(block);
        rhs->preds.push_back(from);
        merge->preds.push_back(from);
        block = rhs;
        // `b` under the result's name along the other edge, as itself: the
        // instruction it is keeps its own name.
        block->lookups[result] = get_value(node->b);
        merge->preds.push_back(block);
        set_block_jump(merge->name);
        block = merge;
        value = block->get_value(result, node->type);
    }

    void visit(const BinOp *node) override {
        if ((node->op == BinOp::LAnd || node->op == BinOp::LOr) &&
            has_work(node->b)) {
            short_circuit(node);
            return;
        }
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

        // One that computes nothing is a side effect: an instruction with no
        // name, kept in its place among the block's stores, which is how
        // the vectorizer and the backend treat it (a prefetch; see
        // ir::Intrinsic::prefetch).
        if (node->type.is<Void_t>()) {
            internal_assert(Intrinsic::has_effects(node->op))
                << "An intrinsic of no value and no effect: " << Expr(node);
            block->make_side_effect(Instruction::Op::Intrinsic,
                                    std::move(args));
            block->instrs.back()->intrinsic = node->op;
            value = nullptr;
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
            // A float is negated by subtracting it from *negative* zero,
            // which is `-x` for every `x` (`-0.0 - (+0.0)` is `-0.0`) and
            // what LLVM reads as `fneg`; from `+0.0` it would be `+0.0` at
            // `x = +0.0`, and a subtraction LLVM can only keep, which left
            // `0 - (a - b)` as `(b - a) + 0.0` in the FCPW leaf. An integer
            // is negated from zero.
            auto zero = splat(elem.is_float()
                                  ? Constant{elem, -0.0}
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
                const bool readonly = base->get_type().is_readonly();
                base = block->make_instruction(
                    Ptr_t::make(current, readonly), Instruction::Op::FieldPtr,
                    {std::move(base), make_constant(u32, (uint64_t)idx)});
            } else {
                if (!current.is_reference()) {
                    return nullptr; // a lane of a vector, which is a value
                }
                current = current.element_of();
                // An address formed inside read-only storage is read-only.
                const bool readonly = base->get_type().is_readonly();
                base = block->make_instruction(
                    Ptr_t::make(current, readonly), Instruction::Op::GEP,
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

    void visit(const StoredElement *node) override {
        internal_error << "An element of a tree's storage reached the SSA "
                       << "conversion unresolved: " << ir::Expr(node)
                       << ". Lower/Layouts.cpp spells these as reads of the "
                       << "layout's array.";
    }

    void visit(const TiledArray *node) override {
        internal_error << "A tiled array reached the SSA conversion as a "
                       << "whole: " << ir::Expr(node)
                       << ". Only an element of one can be read, and "
                       << "Lower/TiledArrays.cpp spells each read as the "
                       << "tile's.";
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

    // `select(c, a, b)` evaluates both arms and is one instruction, the
    // conditional move the program asked for by writing it. `ite(c, a, b)`
    // (Select::lazy) evaluates the arm it takes, as C's `c ? a : b` does:
    // control flow, each arm in a block of its own, the value a parameter
    // of the merge.
    void visit(const Select *node) override {
        if (node->lazy) {
            auto cond = get_value(node->cond);
            internal_assert(!block->terminator.defined());
            const std::string result = conditional_name("ite");
            std::shared_ptr<Block> then_case = make_block("ite_then");
            std::shared_ptr<Block> else_case = make_block("ite_else");
            std::shared_ptr<Block> merge = make_block("ite_merge");
            block->terminator.data = Terminator::Dispatch{
                .cond = std::move(cond),
                .targets = {Terminator::Jump{.name = else_case->name, .args = {}},
                            Terminator::Jump{.name = then_case->name, .args = {}}}};
            std::shared_ptr<Block> from = std::move(block);
            then_case->preds.push_back(from);
            else_case->preds.push_back(from);
            block = then_case;
            block->make_instruction(result, node->type, get_value(node->tvalue));
            merge->preds.push_back(block);
            set_block_jump(merge->name);
            block = else_case;
            block->make_instruction(result, node->type, get_value(node->fvalue));
            merge->preds.push_back(block);
            set_block_jump(merge->name);
            block = merge;
            value = block->get_value(result, node->type);
            return;
        }
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

// The type of the value of `f` named `name` -- a parameter's or an
// instruction's -- for a directive that names one (a split's chunk, a
// persistent loop's count), since the schedule's scope has no types.
std::optional<Type> type_of_value(const Function &f, const std::string &name) {
    std::optional<Type> type;
    for (const ir::ssa::Argument &a : f.blocks.front()->args) {
        if (a.name == name) {
            type = a.type;
        }
    }
    for (const auto &b : f.blocks) {
        for (const auto &in : b->instrs) {
            if (in->name == name) {
                type = in->type;
            }
        }
    }
    return type;
}

// Every parfor reachable from `start`, by function.
//
// A schedule names a function and a loop in it, but neither need be where the
// loop ends up: lowering moves a `map` into a `_traverse_arrayN` helper of its
// own, so the loop a schedule calls `process`'s is really in a function
// `process` calls. The Stmt-level pass follows the same calls, by name and
// with a "this is hacky" note; following the call graph is the general form of
// that.
//
// And in every copy a directive made of a function on the way
// (functions_standing_for, SSA/Rewrite.h): a vectorize's gang copies, which
// the gang's calls reach and the scalar function's never do, and a
// specialize's arms, whose loops `occluded[False].vectorize(tris.Interior.
// children)` names as `children!False`.
std::map<std::string, std::set<std::string>>
parfors_reachable_from(const FuncMap &fmap, const std::string &start) {
    std::map<std::string, std::set<std::string>> found;
    for (const std::string &name : functions_standing_for(fmap, start)) {
        for (const auto &block : fmap.at(name)->blocks) {
            if (const auto *p =
                    std::get_if<Terminator::ParFor>(&block->terminator.data)) {
                found[name].insert(p->index);
            }
        }
    }
    return found;
}

// Whether `loop` is the loop `candidate` names, or one of its variants: a
// specialized function's loops are the variants' (`p!VolPath`, SSA/
// Specialize.cpp), a split queue's drains the sub-queues' (`hits!Some!
// Diffuse`, SSA/Defer.cpp), and a schedule that names `p` or `hits` means
// all of them.
bool loop_matches(const std::string &loop, const std::string &candidate) {
    return loop == candidate ||
           (loop.size() > candidate.size() &&
            loop.compare(0, candidate.size(), candidate) == 0 &&
            loop[candidate.size()] == '!');
}

// The part of a loop's name past the name the schedule wrote: `!VolPath` of
// `p!VolPath` named as `p`. What a directive's products take too, so that a
// split of every variant's `s` makes each variant its own `s_gang!VolPath`.
std::string variant_suffix(const std::string &wanted, const std::string &index) {
    for (const std::string &candidate : {wanted, "_" + wanted}) {
        if (loop_matches(index, candidate)) {
            return index.substr(candidate.size());
        }
    }
    return "";
}

// The loops a schedule means by a name. A loop the program wrote is called
// what the program called it; one lowering generated is called `_` followed
// by that, because lowering labels what it invents, while a schedule names it
// without the underscore either way. A name matches a loop exactly or a
// variant of it (loop_matches): `render[VolPath].bind(p, ...)` is written
// by the parser as the loop `p!VolPath`, which is one loop; `render.bind(p,
// ...)` on a specialized `render` is every variant's `p`.
std::vector<LoopSite> resolve_loops(const FuncMap &fmap, const std::string &start,
                                    const std::string &wanted,
                                    const std::string &transform) {
    const auto found = parfors_reachable_from(fmap, start);
    for (const std::string &candidate : {wanted, "_" + wanted}) {
        std::vector<LoopSite> sites;
        // The function the schedule named wins over one it merely reaches.
        const auto here = found.find(start);
        if (here != found.end()) {
            for (const std::string &loop : here->second) {
                if (loop_matches(loop, candidate)) {
                    sites.push_back({start, loop});
                }
            }
        }
        if (!sites.empty()) {
            return sites;
        }
        for (const auto &[fname, loops] : found) {
            for (const std::string &loop : loops) {
                if (loop_matches(loop, candidate)) {
                    sites.push_back({fname, loop});
                }
            }
        }
        if (!sites.empty()) {
            return sites;
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
    return {};
}

// The one loop a schedule means, where a directive takes exactly one.
LoopSite resolve_loop(const FuncMap &fmap, const std::string &start,
                      const std::string &wanted, const std::string &transform) {
    const std::vector<LoopSite> sites = resolve_loops(fmap, start, wanted, transform);
    internal_assert(sites.size() == 1)
        << transform << "() on " << start << ": " << wanted << " names "
        << sites.size() << " loops, the variants of a specialized function or "
        << "the queues of a split; this directive takes one. Name it: `" << start
        << "[<Variant>]." << transform << "(" << wanted << ", ...)`.";
    return sites.front();
}

// The loop a directive on a nested pair means by its second name: the one
// nested with the loop its first name found, `anchor`. `collapse(w, i, k)`
// names the `i` inside `w`, and `reorder(i, w)` the `w` around `i` -- or
// inside it, once an earlier reorder has turned the nest round
// (`either_way`). The name is looked for in the anchor's function with the
// anchor's variant suffix first (a specialized function's `s!VolPath`
// beside its `p!VolPath`) and bare second, since a loop can have one copy
// where its partner has several: after `reorder(i, w)` a schedule's `i` is
// the nest's `i`, `i!prologue` and `i!epilogue` (SSA/ReorderLoops.h), three
// variants in the sense of loop_matches beside the one `w`, and only the
// nest's `i` is nested with it. Position settles what the name leaves
// open: of the loops the name matches, the one whose header lies in the
// anchor's body or, either way round, whose body holds the anchor's header.
// None -- the anchor is a variant with no such partner, as the distributed
// loops are -- answers nothing, and the directive passes that anchor over;
// more than one is refused.
std::optional<LoopSite> nested_loop(const FuncMap &fmap, const LoopSite &anchor,
                                    const std::string &wanted, const std::string &suffix,
                                    const std::string &transform, bool either_way) {
    const auto found = parfors_reachable_from(fmap, anchor.func);
    const auto here = found.find(anchor.func);
    if (here == found.end()) {
        return std::nullopt;
    }
    std::vector<std::string> candidates;
    for (const std::string &name :
         {wanted + suffix, "_" + wanted + suffix, wanted, "_" + wanted}) {
        for (const std::string &loop : here->second) {
            if (loop_matches(loop, name)) {
                candidates.push_back(loop);
            }
        }
        if (!candidates.empty()) {
            break;
        }
    }
    if (candidates.empty()) {
        return std::nullopt;
    }
    const Function &f = *fmap.at(anchor.func);
    // A parfor's header block and its body's name, by the index it goes by.
    const auto loop_of = [&](const std::string &index) {
        std::pair<const Block *, std::string> found_loop{nullptr, ""};
        for (const auto &block : f.blocks) {
            const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p != nullptr && p->index == index) {
                found_loop = {block.get(), p->body.name};
            }
        }
        return found_loop;
    };
    const auto [anchor_header, anchor_body] = loop_of(anchor.index);
    internal_assert(anchor_header != nullptr)
        << transform << "() on " << anchor.func << ": no parfor named " << anchor.index;
    std::vector<std::string> nested;
    for (const std::string &candidate : candidates) {
        if (candidate == anchor.index) {
            continue;
        }
        const auto [header, body] = loop_of(candidate);
        if (header == nullptr) {
            continue;
        }
        const bool inside = Cfg(f, anchor_body).contains(*header);
        const bool around = either_way && Cfg(f, body).contains(*anchor_header);
        if (inside || around) {
            nested.push_back(candidate);
        }
    }
    internal_assert(nested.size() <= 1)
        << transform << "() on " << anchor.func << ": " << wanted << " names "
        << nested.size() << " loops nested with " << anchor.index
        << "; this directive takes one. Name it: `" << anchor.func << "[<Variant>]."
        << transform << "(" << wanted << ", ...)`.";
    if (nested.empty()) {
        return std::nullopt;
    }
    return LoopSite{anchor.func, nested.front()};
}

// The name a schedule knows a function by: `vol_route` of `vol_route!True`
// and of `vol_route!after!hit`, the copies and continuations a directive made.
std::string base_of(const std::string &name) {
    return name.substr(0, name.find('!'));
}

// A queue declared on a loop that an earlier specialize copied is one queue
// per copy -- one per arm -- each owned by its arm's loop and named with
// the arm's variant (`rays!True`): the dispatch on the flag sits above the
// loop, and under each arm the loop, its queues, their drains and the rest
// pass are the arm's own, so that one arm's kernels alone ever launch, and
// what one arm never reads of its record is stored nowhere
// (SSA/InducedStorage.h). A queue on a loop with one copy is one queue, as
// before, and so is a queue on `root`. Every directive on the queue -- the
// deferrals, the cut, the splits -- is applied once per arm.
struct QueueArm {
    std::string variant;   // `!True`; empty for the one instance of an unspecialized loop
    std::string loop;      // what the queue is on: the arm's loop, or `root`
    std::string name;      // the queue's name for this arm
    std::string walk_loop; // the arm's loop, whose body the chain is reached from
    std::string entry;     // for `root`: the arm's first block (QueueSpec::region_entry)
    // For `root`: the blocks only this arm's selector reaches -- its loop,
    // and the drains and rest passes of the arm's earlier queues, which a
    // queue on `root` puts beside the loop rather than inside it -- and
    // so where the chain is reached from (arm_chain_copies). Empty for a
    // queue on a loop, whose drains lie in the loop's body.
    std::set<std::string> region;
};

// The labels the owner's specialize directives give the copies they make:
// `False` and `True` for a bool or a gang's count, a variant type's variant
// names. What tells a specialize's copy of a loop (`p!VolPath!True`) from
// another directive's product that carries a suffix too -- a queue split's
// drain (`hits!Big`), an interchange's prologue (`p!VolPath!prologue`).
std::set<std::string> specialize_labels(
    const FuncMap &fmap, const ir::TransformMap &transforms,
    const std::map<std::string, ir::Program::AdtStorage> &adt_storages,
    const std::string &owner) {
    std::set<std::string> labels;
    const auto ts = transforms.find(owner);
    const auto f = fmap.find(owner);
    if (ts == transforms.end() || f == fmap.end()) {
        return labels;
    }
    for (const ir::Transform &t : ts->second) {
        const auto *s = std::get_if<ir::Specialize>(&t);
        if (s == nullptr) {
            continue;
        }
        // A gang's count and an arm's bool field make `False` and `True`,
        // as a bool parameter does.
        if (!s->gang.empty() || !s->field.empty()) {
            labels.insert("False");
            labels.insert("True");
            continue;
        }
        for (const Argument &a : f->second->blocks.front()->args) {
            if (a.name != s->param) {
                continue;
            }
            Type type = a.type;
            if (const Ptr_t *p = type.as<Ptr_t>()) {
                type = p->etype;
            }
            if (type.is<Bool_t>()) {
                labels.insert("False");
                labels.insert("True");
            } else if (const Struct_t *st = type.as<Struct_t>()) {
                if (const auto it = adt_storages.find(st->name); it != adt_storages.end()) {
                    for (const auto &[vname, tag] : it->second.variants) {
                        labels.insert(vname);
                    }
                }
            }
        }
    }
    return labels;
}

// Whether `suffix` (`!VolPath!True`) is made of specialize labels alone.
bool arm_suffix(const std::string &suffix, const std::set<std::string> &labels) {
    if (suffix.size() < 2 || suffix[0] != '!') {
        return false;
    }
    for (size_t at = 1; at <= suffix.size();) {
        const size_t next = suffix.find('!', at);
        const std::string segment =
            suffix.substr(at, next == std::string::npos ? std::string::npos : next - at);
        if (!labels.contains(segment)) {
            return false;
        }
        if (next == std::string::npos) {
            break;
        }
        at = next + 1;
    }
    return true;
}

std::vector<QueueArm> queue_arms(const FuncMap &fmap, const std::string &qname,
                                 const ir::Queue &q, const std::set<std::string> &labels) {
    internal_assert(!q.loop.names.empty()) << qname << " names no loop";
    const std::string &loop = q.loop.names.back();
    std::vector<QueueArm> arms;
    if (loop == "root") {
        // The owner's outermost loops that a specialize made: one arm per
        // variant they carry, the arm's region beginning at the selector
        // the dispatch enters (its block's loop is the arm's copy;
        // SSA/Specialize.cpp). A drain of a queue made earlier -- an arm
        // queue's `rays!True`, its rest pass -- carries the variant too and
        // is not the arm's loop. An owner nobody specialized has one arm.
        const Function &O = *fmap.at(q.owner);
        std::vector<std::shared_ptr<Block>> loops;
        for (const auto &block : O.blocks) {
            if (std::holds_alternative<Terminator::ParFor>(block->terminator.data)) {
                loops.push_back(block);
            }
        }
        const auto is_drain = [&](const std::string &index) {
            for (const auto &[name, owned] : O.queue_sizes) {
                const size_t v = name.find('!');
                const std::string base = v == std::string::npos ? name : name.substr(0, v);
                const std::string variant = v == std::string::npos ? "" : name.substr(v);
                if (index == name || index.rfind(name + "!", 0) == 0 ||
                    index == base + "_rest" + variant) {
                    return true;
                }
            }
            return false;
        };
        std::map<std::string, std::vector<std::shared_ptr<Block>>> by_variant;
        for (const auto &block : loops) {
            bool inside = false;
            for (const auto &other : loops) {
                const auto &p = std::get<Terminator::ParFor>(other->terminator.data);
                if (other != block && Cfg(O, p.body.name).contains(*block)) {
                    inside = true;
                }
            }
            if (inside) {
                continue;
            }
            const std::string &index = std::get<Terminator::ParFor>(block->terminator.data).index;
            const size_t bang = index.find('!');
            if (bang != std::string::npos && arm_suffix(index.substr(bang), labels) &&
                !is_drain(index)) {
                by_variant[index.substr(bang)].push_back(block);
            }
        }
        if (by_variant.empty()) {
            return {QueueArm{"", loop, qname, "", ""}};
        }
        for (const auto &[variant, blocks] : by_variant) {
            internal_assert(blocks.size() == 1)
                << "[unimplemented] " << qname << " = " << q.owner << ".queue(root): "
                << q.owner << " has " << blocks.size() << " outermost loops of the arm "
                << variant << "; a queue on root takes one loop per arm";
            const auto &p = std::get<Terminator::ParFor>(blocks.front()->terminator.data);
            arms.push_back(QueueArm{variant, loop, qname + variant, p.index,
                                    blocks.front()->name});
        }
        // Each arm's region: what its selector reaches and no other arm's
        // does. The merge below the arms and everything after it is every
        // arm's and belongs to none; a drain of an arm's earlier queue,
        // made beside the arm's loop, is reached from that arm's selector
        // alone and is where the chain is reached through -- the calls of
        // a chain function from a split's drains (`shade!False!Matte`
        // calling `wetten`) are the arm's to retarget as much as the
        // loop's own.
        for (QueueArm &arm : arms) {
            const Cfg own(O, arm.entry);
            for (BlockId b : own.rpo) {
                arm.region.insert(own.block(b)->name);
            }
        }
        for (QueueArm &arm : arms) {
            for (const QueueArm &other : arms) {
                if (&other == &arm) {
                    continue;
                }
                const Cfg theirs(O, other.entry);
                for (BlockId b : theirs.rpo) {
                    arm.region.erase(theirs.block(b)->name);
                }
            }
        }
        return arms;
    }
    // The loop as declared, when it is still one loop; else its copies a
    // specialize made, by their labels. A copy of another kind -- an
    // interchange's prologue -- is neither.
    for (const LoopSite &site : resolve_loops(fmap, q.owner, loop, "queue")) {
        if (site.func != q.owner) {
            continue;
        }
        const std::string suffix = variant_suffix(loop, site.index);
        if (suffix.empty()) {
            return {QueueArm{"", site.index, qname, site.index, ""}};
        }
        if (arm_suffix(suffix, labels)) {
            arms.push_back(QueueArm{suffix, site.index, qname + suffix, site.index, ""});
        }
    }
    internal_assert(!arms.empty())
        << qname << " = " << q.owner << ".queue(" << loop << "): " << q.owner
        << " has no loop named " << loop << ", nor a specialize's copies of one";
    if (arms.size() == 1) {
        arms.front().variant.clear();
        arms.front().name = qname;
        return arms;
    }
    return arms;
}

// The functions of a queue's chain, by the names the schedule knows them by:
// the callee, the pushers, and everything that reaches one of them by calls
// -- the functions a deferral rewrites or hands the queue down through. The
// call graph is read over the base names, so copies and continuations count
// as the function they were made from.
std::set<std::string> chain_functions(const FuncMap &fmap, const std::string &callee,
                                      const std::vector<std::string> &pushers) {
    std::map<std::string, std::set<std::string>> callers;
    for (const auto &[f, F] : fmap) {
        for (const std::string &g : callees_of(*F)) {
            callers[base_of(g)].insert(base_of(f));
        }
    }
    std::set<std::string> on{callee};
    on.insert(pushers.begin(), pushers.end());
    std::vector<std::string> work(on.begin(), on.end());
    while (!work.empty()) {
        const std::string n = work.back();
        work.pop_back();
        for (const std::string &c : callers[n]) {
            if (on.insert(c).second) {
                work.push_back(c);
            }
        }
    }
    return on;
}

// The arm's copies of a queue's chain. Two arms' drains hand the chain two
// queues of two types, so no function that takes the queue, or hands it
// down, can serve both arms. Walking the calls down from the arm's region
// -- its loop's body for a queue on the loop, the loop and the drains
// beside it for a queue on `root` (QueueArm::region) -- a chain function
// is the arm's own already when its name carries the arm's
// variant -- the specialize copied it, handed the arm's key, as it copies
// `integrator_li` and everything the key is passed on to -- and is cloned
// under the arm's variant, the call retargeted, when it does not (a pusher
// the key never reaches). A function off the chain stays one function.
// Hands back, by base name, every function of the chain the arm calls: one
// name as a rule, and more where a stage split a continuation off it
// (`step!after` beside `step`) or where an earlier specialize copied it per
// variant of a key the chain hands it (`finish!Big` and `finish!Small`) --
// which one_copy tells apart.
std::map<std::string, std::set<std::string>>
arm_chain_copies(FuncMap &fmap, const std::string &owner, const QueueArm &arm,
                 const std::set<std::string> &on_chain) {
    std::map<std::string, std::set<std::string>> copies;
    const std::shared_ptr<Function> O = fmap.at(owner);
    std::vector<std::shared_ptr<Block>> start;
    if (!arm.region.empty()) {
        // An arm of a queue on `root`: its region (QueueArm::region), the
        // loop and the drains of its earlier queues beside it.
        for (const auto &block : O->blocks) {
            if (arm.region.contains(block->name)) {
                start.push_back(block);
            }
        }
    } else if (arm.walk_loop.empty()) {
        start = O->blocks; // the whole function: one arm, on root
    } else {
        // An arm of a queue on a loop: the loop's body, where a queue on
        // the loop drains too.
        for (const auto &block : O->blocks) {
            const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p != nullptr && p->index == arm.walk_loop) {
                const Cfg body(*O, p->body.name);
                for (BlockId b : body.rpo) {
                    start.push_back(body.block(b));
                }
            }
        }
    }
    std::set<std::string> visited;
    const std::function<void(const std::vector<std::shared_ptr<Block>> &)> walk =
        [&](const std::vector<std::shared_ptr<Block>> &blocks) {
            for (const auto &block : blocks) {
                Terminator::Jump *call = block->terminator.callee();
                if (call == nullptr) {
                    continue;
                }
                const std::string base = base_of(call->name);
                if (!on_chain.contains(base) || !fmap.contains(call->name)) {
                    continue;
                }
                std::string target = call->name;
                if (!carries_variant(target, arm.variant)) {
                    target = call->name + arm.variant;
                    if (!fmap.contains(target)) {
                        // A function's entry block carries its name, and a
                        // call of itself names it too (specialize_function
                        // does the same for its copies).
                        std::shared_ptr<Function> copy = clone_function(*fmap.at(call->name));
                        const std::string was = copy->blocks.front()->name;
                        copy->blocks.front()->name = target;
                        for (const auto &b : copy->blocks) {
                            for (Terminator::Jump *jump : jumps_of(*b)) {
                                if (jump->name == was) {
                                    jump->name = target;
                                }
                            }
                        }
                        fmap[target] = std::move(copy);
                    }
                    call->name = target;
                }
                copies[base].insert(target);
                if (visited.insert(target).second) {
                    walk(fmap.at(target)->blocks);
                }
            }
        };
    walk(start);
    return copies;
}

// The one function a chain function's copies stand for: the shortest, when
// every other is a continuation split off it (`step!after` of `step`,
// `vol_path_step!True!after!hit` of `vol_path_step!True`); none, when two
// are copies of it in their own right (`finish!Big` and `finish!Small`,
// which a queue's specialize made because the chain hands `finish` the key
// it specialized by), since a deferral's callee is one function.
std::string one_copy(const std::set<std::string> &copies) {
    std::string shortest;
    for (const std::string &c : copies) {
        if (shortest.empty() || c.size() < shortest.size()) {
            shortest = c;
        }
    }
    for (const std::string &c : copies) {
        if (c != shortest && c.rfind(shortest + "!", 0) != 0) {
            return "";
        }
    }
    return shortest;
}

// The arm's copy of the function a directive names: the function itself for
// the one arm of an unspecialized loop; else the shortest of its copies that
// carries the arm's variant (`vol_path_step!True`, not its continuation
// `vol_path_step!True!after!hit`).
std::string arm_copy_of(const FuncMap &fmap, const std::string &name, const QueueArm &arm) {
    if (arm.variant.empty()) {
        return name;
    }
    std::string best;
    for (const auto &[f, _] : fmap) {
        if ((f == name || f.rfind(name + "!", 0) == 0) && carries_variant(f, arm.variant) &&
            (best.empty() || f.size() < best.size())) {
            best = f;
        }
    }
    internal_assert(!best.empty())
        << name << " has no copy for the arm " << arm.variant << " of " << arm.loop
        << ": the arm's deferral has to come first, which copies the chain for it";
    return best;
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
        dump_induced(os, *f);
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
// Every function a schedule's name stands for when it says `skip`: the
// function itself and every function its calls reach -- the recursion that
// Lower/RecLoops.cpp extracted out of it into a function of its own, and the
// helpers a vectorized traversal calls under a mask, which the vectorizer
// specializes and linearizes under their own names (SSA/Vectorize.cpp).
// The linearizer looks a policy up under the name of the function it is
// running on; without this, `trace.skip(...)` sat under `trace` while the
// leaf it meant was in `_recloop_func1`, and did nothing, silently.
std::set<std::string> reach_of(const FuncMap &fmap, const std::string &start) {
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
            continue;
        }
        for (const auto &block : it->second->blocks) {
            if (const auto *call = block->terminator.callee()) {
                work.push_back(call->name);
            }
        }
    }
    return seen;
}

// The policies under every name they stand for (see reach_of): the bare
// `skip()` and the cursors that name a function, which are about the code
// as compiled wherever it is. A cursor naming a match (`Shape`, `Shape.Disc`)
// stays where it was written: it names a match *written in* the function
// the schedule said, and the linearizer looks it up by the arm's own
// function (ir::skips), so copying it along would widen it to every match
// on the type in everything the function calls.
ir::BranchPolicyMap expand_branch_policies(const FuncMap &fmap,
                                           const ir::BranchPolicyMap &policies) {
    // The names arms were written in, off the provenance each block carries.
    // A cursor names a function by the name the schedule's author knows,
    // which is not always a function of the module's: a geometric op's
    // implementation is renamed for its operand types by Lower/Geometrics.cpp
    // (`distmin` over a point and a triangle becomes
    // `distmin_Point_Triangle`), while its arms still say `distmin`, the name
    // `closest.skip(distmin)` uses and the one ir::skips matches them by.
    // Without this the cursor was dropped here and the leaf of the FCPW
    // closest point ran every Voronoi region on every packet, silently.
    std::set<std::string> written_in;
    for (const auto &[fname, f] : fmap) {
        for (const auto &block : f->blocks) {
            if (block->provenance.defined()) {
                written_in.insert(block->provenance.func());
            }
        }
    }
    ir::BranchPolicyMap expanded = policies;
    for (const auto &[name, policy] : policies) {
        std::vector<ir::Location> by_function;
        for (const ir::Location &cursor : policy.skip.arms) {
            if (!cursor.names.empty() &&
                (fmap.contains(cursor.names.front()) ||
                 written_in.contains(cursor.names.front()))) {
                by_function.push_back(cursor);
            }
        }
        if (!policy.skip.all && by_function.empty()) {
            continue;
        }
        for (const std::string &reached : reach_of(fmap, name)) {
            if (reached == name) {
                continue;
            }
            ir::ArmCursors &skip = expanded[reached].skip;
            skip.all = skip.all || policy.skip.all;
            skip.arms.insert(skip.arms.end(), by_function.begin(),
                             by_function.end());
        }
    }
    return expanded;
}

void check_branch_policies(const FuncMap &fmap,
                           const ir::BranchPolicyMap &policies) {
    // Every arm in the program, with the function it is in now.
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
            // An arm of a match written in the function, or an arm written
            // in the function the cursor names, anywhere the schedule's name
            // stands for (see reach_of and ir::ArmCursors).
            ir::ArmCursors one;
            one.arms.push_back(cursor);
            const std::set<std::string> reach = reach_of(fmap, fname);
            const bool found =
                std::any_of(arms.begin(), arms.end(), [&](const auto &arm) {
                    return reach.count(arm.first) != 0 &&
                           ((arm.second.func() == fname &&
                             arm.second.matches(cursor.names)) ||
                            one.names_function_of(arm.second));
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
            const std::string &first = cursor.names[0];
            const std::string as_function =
                fmap.contains(first)
                    ? ", and " + first + " has no arm in what " + fname +
                          " reaches"
                    : ", and nothing " + fname + " reaches is a function named " +
                          first;
            internal_error << "skip(" << spelled << ") on " << fname
                           << ": no arm of a match written in " << fname
                           << " takes " << spelled << as_function << ". "
                           << instead;
        }
    }
}

// The splits the schedule wrote on `spec`'s queue, as its tree: `q.specialize(
// k)` is the root, `q[A].specialize(k2)` the node under the root's variant
// `A` (ir::QueueSpecialize, QueueSpec::Split). A node split below is split
// itself.
void attach_splits(QueueSpec &spec,
                   const std::vector<ir::QueueSpecialize> &splits,
                   const std::map<std::string, ir::Program::AdtStorage>
                       &adt_storages,
                   const std::string &declared) {
    // `declared` is the queue as the schedule names it (`hits`); `spec.name`
    // is the instance being made, which carries an arm's variant when the
    // queue's loop was specialized (`hits!True`, QueueSpec::variant).
    spec.adt_storages = &adt_storages;
    for (const ir::QueueSpecialize &qs : splits) {
        if (qs.queue != declared) {
            continue;
        }
        if (!spec.split.has_value()) {
            spec.split = QueueSpec::Split{};
        }
        QueueSpec::Split *at = &*spec.split;
        std::string path = spec.name;
        for (const std::string &label : qs.under) {
            at = &at->under[label];
            path += "[" + label + "]";
        }
        internal_assert(at->key.empty())
            << path << ".specialize(" << qs.key << "): " << path
            << " is already split on " << at->key;
        at->key = qs.key;
    }
    if (!spec.split.has_value()) {
        return;
    }
    const std::function<void(const QueueSpec::Split &, const std::string &)>
        check = [&](const QueueSpec::Split &node, const std::string &path) {
            internal_assert(!node.key.empty())
                << path << " is split below (`" << path
                << "[...].specialize(...)`) but not itself: write `" << path
                << ".specialize(<value>)` too.";
            for (const auto &[label, under] : node.under) {
                check(under, path + "[" + label + "]");
            }
        };
    check(*spec.split, spec.name);
}

ir::FuncMap convert(ir::FuncMap funcs, const ir::TransformMap &transforms,
                    const ir::TransformOrder &order,
                    const ir::BranchPolicyMap &policies,
                    const std::map<std::string, ir::Queue> &queues,
                    const std::vector<ir::QueueSpecialize> &queue_splits,
                    const std::map<std::string, ir::Program::AdtStorage>
                        &adt_storages,
                    const CompilerOptions &options,
                    ir::Program *keep_ssa = nullptr) {
    FuncMap fmap;
    // How each queue a deferral makes stores its entries, for the pushes'
    // lowering at the end (SSA/Defer.h).
    QueueLayouts layouts;

    TypeMap func_type_map;

    // See BONSAI_TIME_PASSES in Lower/Lower.cpp: this one pass is most of a
    // scheduled compile, so its phases are timed one by one -- the graph's
    // construction, the passes over every function before and after the
    // directives, each directive (below), and the statements at the end.
    const bool timing = std::getenv("BONSAI_TIME_PASSES") != nullptr;
    auto phase_started = std::chrono::steady_clock::now();
    const auto phase = [&](const char *name) {
        if (!timing) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        std::cerr << "[time]   ssa " << name << ": "
                  << std::chrono::duration<double>(now - phase_started).count()
                  << " s\n";
        phase_started = now;
    };

    for (const auto &[name, func] : funcs) {
        func_type_map[name] = func->call_type();
        auto f = build(func);
        // The values the schedule reads by name -- a split's chunk -- are
        // uses the body does not show, carried over from the function the
        // parser stamped (ir::Function::schedule_uses) so that nothing here
        // removes them before the split reads them.
        f->schedule_uses = func->schedule_uses;
        // And which OptiX program the function is, if one, for the code
        // generator that compiles it as an entry point (CodeGen_OptiX).
        f->optix_program = func->optix_program;
        fmap[name] = std::move(f);
    }
    phase("build");

    check_branch_policies(fmap, policies);
    // Under every name they stand for, which is what the linearizer looks
    // them up by (see expand_branch_policies).
    ir::BranchPolicyMap expanded_policies =
        expand_branch_policies(fmap, policies);
    // A copy a directive makes of a function after this point -- a gang
    // specialize's arm -- stands under the function's name too, and a
    // vectorize written after it looks its policies up under the copy's
    // own: handed down from the function it was copied from
    // (Function::specialized_from) once the copies exist.
    const auto inherit_policies = [&]() {
        for (bool grew = true; grew;) {
            grew = false;
            for (const auto &[fname, f] : fmap) {
                if (f->specialized_from.empty() ||
                    expanded_policies.contains(fname)) {
                    continue;
                }
                const auto from = expanded_policies.find(f->specialized_from);
                if (from != expanded_policies.end()) {
                    expanded_policies[fname] = from->second;
                    grew = true;
                }
            }
        }
    };

    // Before any rewrite has touched it, so that a transform's golden can say
    // what it changed and not only what it ended at.
    if (options.dump_ssa_preschedule) {
        dump_ssa(std::cout, "preschedule", fmap, options.is_verbose);
    }

    // A sort is applied where the schedule wrote it, like every other
    // directive (the ordered loop below), over the functions its name stands
    // for: the recursion a `trace.sort(...)` is about lives in a traversal
    // function that lowering invented and the schedule cannot name, and in
    // whatever gang copies a vectorize written before it has made of that
    // function (Function::specialized_from). The keys are on the IR from
    // lowering, so a function no schedule sorted has none and the sort does
    // nothing to it. The order is what the schedule says: sorted and then
    // vectorized, a run is ordered by the executor the sort found, one lane
    // of a gang each; vectorized and then sorted, it is ordered once for
    // the gang, by the lanes' majority (SSA/SortRecursion.cpp). A sort
    // written after the loopify of its recursion finds no run left to
    // permute and is refused there (SSA/QueueRecursion.cpp).
    //
    // What every value is known to lie between, over the whole program as
    // it stands, is what lets a sort leave out the sign flip of a key that
    // cannot be negative (SSA/ConstantIntervals.h); the rewrites before and
    // after it make values the intervals never saw, which is fine, since an
    // unknown value is everything.
    const auto sort_reach = [&](const std::string &name) {
        const std::set<std::string> reach = functions_standing_for(fmap, name);
        const ConstantIntervals intervals(fmap);
        if (std::getenv("BONSAI_INTERVALS") != nullptr) {
            for (const auto &[fname, f] : fmap) {
                intervals.dump(std::cerr, *f);
            }
        }
        if (options.dump_intervals) {
            std::cout << "=== intervals before the sort of " << name << "\n";
            for (const auto &[fname, f] : fmap) {
                intervals.dump(std::cout, *f);
            }
        }
        for (const std::string &fname : reach) {
            const auto f = fmap.find(fname);
            if (f == fmap.end()) {
                continue;
            }
            sort_recursion(*f->second, fmap, options.target, intervals);
            // What the network compared was built by rule; this is where it
            // is looked at (see SSA/Simplify.h).
            simplify(*f->second);
        }
    };

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
    // A run-time divisor's multiplier is for a gang: the functions a
    // vectorize directive names, and every function those reach, since the
    // gang's calls are vectorized copies of them. A function none of them
    // reaches is scalar wherever it runs, and a scalar division by a
    // non-constant is not optimized (the user's rule, 2026-10-02): on the
    // GPU a thread's own multiplier was a 128-bit division expanded into a
    // loop, 78% of a material kernel (SSA/InvariantDivision.h). Constant
    // divisors are rewritten everywhere.
    std::set<std::string> vectorized_reach;
    {
        std::vector<std::string> work;
        for (const auto &[fname, ts] : transforms) {
            const bool vectorizes_one =
                std::any_of(ts.begin(), ts.end(), [](const ir::Transform &t) {
                    return std::holds_alternative<ir::Vectorize>(t);
                });
            if (vectorizes_one && fmap.contains(fname) &&
                vectorized_reach.insert(fname).second) {
                work.push_back(fname);
            }
        }
        while (!work.empty()) {
            const std::string fname = work.back();
            work.pop_back();
            for (const auto &block : fmap.at(fname)->blocks) {
                const auto *callee = block->terminator.callee();
                if (callee != nullptr && fmap.contains(callee->name) &&
                    vectorized_reach.insert(callee->name).second) {
                    work.push_back(callee->name);
                }
            }
        }
    }
    bool divided = false;
    const auto divide_all = [&]() {
        for (const auto &[fname, f] : fmap) {
            const bool gang =
                std::find(f->attributes.begin(), f->attributes.end(),
                          ir::Function::Attribute::vectorized) !=
                f->attributes.end();
            if (!gang) {
                divide_by_invariants(*f, vectorized_reach.contains(fname));
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
    // The schedule's directives are applied in the order they were written
    // (ir::TransformOrder, recorded by the parser): the order is part of what
    // a schedule says -- the drains that follow one producer loop run in the
    // order their queues were deferred (SSA/Defer.cpp, "Where the drain
    // sits"). A schedule built without a source order is applied function by
    // function in name order, each function's directives in the order they
    // were listed.
    ir::TransformOrder ordered = order;
    if (ordered.empty()) {
        for (const auto &[name, ts] : transforms) {
            for (size_t i = 0; i < ts.size(); i++) {
                ordered.emplace_back(name, i);
            }
        }
    }

    // Whether any directive of the schedule vectorizes a loop of one of
    // `fs`. A queue whose owner, drained callee and pushers are none of them
    // vectorized keeps its entries' short vectors whole
    // (QueueSpec::vector_leaves); one a gang will push onto or drain takes
    // them apart into a scalar per array. Read off the whole schedule here,
    // since the deferral makes the layout before the vectorize directive
    // that would need the scalars is applied.
    const auto vectorizes = [&](const std::vector<std::string> &fs) {
        for (const std::string &f : fs) {
            const auto ts = transforms.find(f);
            if (ts == transforms.end()) {
                continue;
            }
            for (const ir::Transform &t : ts->second) {
                if (std::holds_alternative<ir::Vectorize>(t)) {
                    return true;
                }
            }
        }
        return false;
    };

    // What every value lies between as the directives find the program
    // (--dump-intervals); each sort prints its own view again, above.
    if (options.dump_intervals) {
        const ConstantIntervals intervals(fmap);
        std::cout << "=== intervals before the directives\n";
        for (const auto &[name, f] : fmap) {
            intervals.dump(std::cout, *f);
        }
    }

    // Every function simplified once before the directives see it (see
    // SSA/Simplify.h), as it was when the sort of every function ran here.
    for (const auto &[fname, f] : fmap) {
        simplify(*f);
    }
    phase("simplify");

    // The time each transform takes (BONSAI_TIME_PASSES, above).
    phase("before the directives");
    // The functions a directive has been applied to so far: `specialize`
    // goes first among a function's, since its copies are what the
    // directives after it schedule, one variant's loops or all of them
    // (SSA/Specialize.h).
    std::set<std::string> scheduled;
    for (const auto &[name, index] : ordered) {
        const ir::Transform &t = transforms.at(name).at(index);
        if (!fmap.contains(name)) {
            // The function is gone. A queue's specialize copied it once per
            // variant and a later deferral erased the unreached original
            // (SSA/Defer.cpp: a dead caller of the chain goes), or nothing
            // ever reached it. A deferral written on it after the copies
            // were made applies to them (SSA/Stage.h, call_holders:
            // `vol_surface.defer(vol_subsurface, bssrdfs)` after
            // `hits.specialize(material)` turns the call into a push in each
            // material's copy); any other directive on a gone function is a
            // schedule for code that no longer exists, and is refused rather
            // than dropped.
            const std::string prefix = name + "!";
            const bool copies = std::any_of(
                fmap.begin(), fmap.end(), [&](const auto &kv) {
                    return kv.first.compare(0, prefix.size(), prefix) == 0;
                });
            if (!copies) {
                continue;
            }
            internal_assert(std::holds_alternative<ir::Defer>(t))
                << name << ": a directive on it comes after a queue's specialize "
                << "copied it per variant and the original went; only a "
                << "deferral applies to the copies (SSA/Stage.h, call_holders).";
        }
        if (std::holds_alternative<ir::Specialize>(t) &&
            std::get<ir::Specialize>(t).loop.empty() &&
            std::get<ir::Specialize>(t).gang.empty()) {
            // After another specialize is fine: `render.specialize(
            // integrator); render[VolPath].specialize(sampler)` copies a
            // copy, and the directives after both schedule the copies. The
            // loop form, `render.specialize(shadow, have_media)`, is the
            // other way round: it copies one loop as scheduled, so it comes
            // after the directives that made the loop (ir::Specialize); so
            // is the gang form, `occluded.specialize(count(i_lane) > 14)`,
            // which copies the gang copies as scheduled and comes after the
            // vectorize that made them.
            const ir::Specialize &sp = std::get<ir::Specialize>(t);
            internal_assert(!scheduled.contains(name))
                << name << ".specialize("
                << (sp.field.empty() ? sp.param : sp.arm + "::" + sp.field)
                << ") comes after another directive on " << name
                << ". specialize goes first: the copies it makes, one per "
                << "variant, are what the directives after it schedule -- "
                << "one variant's (`" << name << "[<Variant>].bind(...)`) or "
                << "every variant's (`" << name << ".bind(...)`).";
        }
        const bool counts = !std::holds_alternative<ir::Sort>(t) &&
                            !std::holds_alternative<ir::Specialize>(t) &&
                            !(std::holds_alternative<ir::Bind>(t) &&
                              std::get<ir::Bind>(t).lambda.defined());
        if (counts) {
            scheduled.insert(name);
        }
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
                          [](const ir::Persistent &) { return "persistent"; },
                          [](const ir::Prefetch &) { return "prefetch"; },
                          [](const ir::Reorder &) { return "reorder"; },
                          [](const ir::Split &) { return "split"; },
                          [](const ir::Sort &) { return "sort"; },
                          [](const ir::Specialize &) { return "specialize"; },
                          [](const ir::Stage &) { return "stage"; },
                          [](const ir::Unroll &) { return "unroll"; },
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
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, v.i.names.back(), "vectorize")) {
                            vectorize(fmap, at.func, at.index,
                                      expanded_policies);
                        }
                    },
                    [&](const ir::Unroll &u) {
                        internal_assert(!u.i.names.empty())
                            << "unroll() requires a loop name for: " << name;
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, u.i.names.back(), "unroll")) {
                            unroll_loop(*fmap.at(at.func), at.index);
                        }
                    },
                    [&](const ir::Specialize &s) {
                        if (!s.gang.empty()) {
                            // On a gang's state: the gang copies the
                            // vectorize of `s.gang` made, each a dispatcher
                            // over two arms (SSA/Specialize.h).
                            specialize_gang(fmap, name, s.gang, s.compare,
                                            s.threshold);
                            inherit_policies();
                            return;
                        }
                        specialize_loops(fmap, name, s.param, s.variant, s.loop,
                                         s.arm, s.field, adt_storages);
                    },
                    [&](const ir::Stage &s) {
                        internal_assert(!s.callee.names.empty())
                            << "stage() requires a callee for: " << name;
                        const auto q = queues.find(s.queue);
                        internal_assert(q != queues.end())
                            << name << ".stage(" << s.callee.names.back()
                            << ", " << s.queue << ") names a queue no "
                            << "schedule block declares. Declare it: `"
                            << s.queue << " = <func>.queue(<loop>);`";
                        // Once per arm of the queue's loop (queue_arms), on
                        // the arm's copy of the staged function.
                        for (const QueueArm &arm : queue_arms(
                                 fmap, s.queue, q->second,
                                 specialize_labels(fmap, transforms, adt_storages,
                                                   q->second.owner))) {
                        const std::string fname = arm_copy_of(fmap, name, arm);
                        QueueSpec spec;
                        spec.name = arm.name;
                        spec.variant = arm.variant;
                        spec.region_entry = arm.entry;
                        spec.owner = q->second.owner;
                        spec.layouts = &layouts;
                        attach_splits(spec, queue_splits, adt_storages, s.queue);
                        spec.vector_leaves =
                            !vectorizes({spec.owner, name, s.callee.names.back()});
                        spec.loop = arm.loop;
                        spec.storage = q->second.storage;
                        if (q->second.capacity.has_value()) {
                            const auto n = get_constant_value<int64_t>(
                                *q->second.capacity);
                            internal_assert(n.has_value() && *n > 0)
                                << s.queue << " = " << spec.owner << ".queue("
                                << spec.loop << ", " << *q->second.capacity
                                << ") needs a constant, positive capacity";
                            spec.capacity = uint64_t(*n);
                        }
                        if (!s.loop.empty()) {
                            // The cut on the same queue (SSA/Stage.h,
                            // stage_same_queue): no new queue, no new type.
                            // The first pass takes the arm's variant last,
                            // as the rest pass does (QueueSpec::variant).
                            stage_same_queue(fmap, fname, s.callee.names.back(),
                                             spec, s.loop + arm.variant);
                            continue;
                        }
                        for (const Type &made :
                             stage(fmap, fname, s.callee.names.back(), spec)) {
                            const auto *st = made.as<Struct_t>();
                            internal_assert(st) << made;
                            if (keep_ssa != nullptr) {
                                keep_ssa->types[st->name] = made;
                            }
                        }
                        } // per arm
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
                        loopify(fmap, name, size, l.variant);
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
                        const std::string &callee = d.callee.names.back();
                        if (!d.loop.empty()) {
                            // The cut on the same queue, before the call
                            // (SSA/Stage.h, defer_same_queue): a directive
                            // of its own, not a pusher of the queue.
                            for (const QueueArm &arm : queue_arms(
                                     fmap, d.queue, q->second,
                                     specialize_labels(fmap, transforms, adt_storages,
                                                       q->second.owner))) {
                                QueueSpec spec;
                                spec.name = arm.name;
                                spec.variant = arm.variant;
                                spec.region_entry = arm.entry;
                                spec.owner = q->second.owner;
                                spec.loop = arm.loop;
                                spec.layouts = &layouts;
                                defer_same_queue(fmap, arm_copy_of(fmap, name, arm), callee,
                                                 spec, d.loop + arm.variant);
                            }
                            return;
                        }
                        // Every `f.defer(callee, q)` on one queue and one
                        // callee is one deferral: a queue's entries are calls
                        // of one function, and the functions that push them
                        // are whichever the schedule named -- the trace
                        // kernel's call of itself, the camera ray's, the
                        // material kernel's call back. It is applied once, at
                        // the callee's own directive (`callee.defer(callee,
                        // q)`, a recursion's step) where there is one, else
                        // at the first written; the owner's directive is the
                        // initial push (`render.defer(step, rays)`,
                        // QueueSpec::initial_push) and the other functions'
                        // are the other pushers (QueueSpec::also_from). A
                        // lone directive on a function off the chain is a
                        // deferral of its own, queued where the call is made
                        // (test correctness/llvm/defer-off-chain).
                        const auto directive_on = [&](const std::string &f) {
                            const auto ts = transforms.find(f);
                            if (ts == transforms.end()) {
                                return false;
                            }
                            for (const ir::Transform &t : ts->second) {
                                const auto *other = std::get_if<ir::Defer>(&t);
                                if (other != nullptr && other->queue == d.queue &&
                                    other->loop.empty() &&
                                    !other->callee.names.empty() &&
                                    other->callee.names.back() == callee) {
                                    return true;
                                }
                            }
                            return false;
                        };
                        // The pushers other than the owner, in the order
                        // written.
                        std::vector<std::string> pushers;
                        for (const auto &[f, i] : ordered) {
                            if (f == q->second.owner || !fmap.contains(f) ||
                                std::find(pushers.begin(), pushers.end(), f) !=
                                    pushers.end()) {
                                continue;
                            }
                            const auto *other =
                                std::get_if<ir::Defer>(&transforms.at(f).at(i));
                            if (other != nullptr && other->queue == d.queue &&
                                other->loop.empty() &&
                                !other->callee.names.empty() &&
                                other->callee.names.back() == callee) {
                                pushers.push_back(f);
                            }
                        }
                        const std::string primary =
                            pushers.empty() ? name
                            : std::find(pushers.begin(), pushers.end(), callee) !=
                                    pushers.end()
                                ? callee
                                : pushers.front();
                        if (name != primary) {
                            return; // part of the primary's deferral
                        }
                        // Once per arm of the queue's loop (queue_arms): the
                        // arm's own copies of the chain, its own queue.
                        const std::set<std::string> chain_set =
                            chain_functions(fmap, callee, pushers);
                        for (const QueueArm &arm : queue_arms(
                                 fmap, d.queue, q->second,
                                 specialize_labels(fmap, transforms, adt_storages,
                                                   q->second.owner))) {
                        const std::map<std::string, std::set<std::string>> arm_copies =
                            arm_chain_copies(fmap, q->second.owner, arm, chain_set);
                        const auto callee_copies = arm_copies.find(callee);
                        if (!arm.variant.empty() && callee_copies == arm_copies.end()) {
                            // An arm that never calls into the chain -- the
                            // integrator that is not this one's -- has no
                            // queue of it.
                            continue;
                        }
                        // The callee as the arm calls it: itself, or the one
                        // copy a specialize made of it that the arm calls
                        // (`finish!Big`, when a queue's specialize upstream
                        // copied `finish` for the one drain that calls it).
                        // Two copies called side by side -- the chain hands
                        // the callee the key that specialize went by, so it
                        // was copied per variant -- are refused: a queue's
                        // drain calls one function (pbrt's BSSRDF kernel
                        // takes the basic texture evaluator for every entry,
                        // so vol_subsurface takes none and the material
                        // queues' split by evaluator leaves it one function).
                        std::string callee_arm = callee;
                        if (callee_copies != arm_copies.end()) {
                            callee_arm = one_copy(callee_copies->second);
                            std::string list;
                            for (const std::string &c : callee_copies->second) {
                                list += (list.empty() ? "" : ", ") + c;
                            }
                            internal_assert(!callee_arm.empty())
                                << d.queue << ": " << callee << " is called as " << list
                                << " where " << q->second.owner << " reaches it"
                                << (arm.variant.empty() ? "" : " in the arm " + arm.variant)
                                << ", copies a queue's specialize made of " << callee
                                << " because the chain hands it the key. A deferral's "
                                   "callee is one function: the argument has to be the "
                                   "same for every copy (as pbrt's BSSRDF kernel takes "
                                   "the basic evaluator for every entry).";
                        }
                        QueueSpec spec;
                        spec.name = arm.name;
                        spec.variant = arm.variant;
                        spec.region_entry = arm.entry;
                        spec.owner = q->second.owner;
                        spec.layouts = &layouts;
                        attach_splits(spec, queue_splits, adt_storages, d.queue);
                        // `f.sort(callee, |args| key)` on the deferred
                        // call: the queue is drained in key order. The
                        // Stmt-level lowering gave the callee the key as
                        // its last parameter (Lower/Sorts.cpp,
                        // apply_queue_sort); the deferral stores it as a
                        // leaf of the entry and sorts the pass by it
                        // (QueueSpec::sort_key).
                        for (const auto &[sorter, sts] : transforms) {
                            for (const ir::Transform &t : sts) {
                                const auto *sort = std::get_if<ir::Sort>(&t);
                                if (sort != nullptr && sort->loc.names.size() == 1 &&
                                    sort->loc.names[0] == callee) {
                                    spec.sort_key = "_sort_key";
                                    spec.sort_key_bits = sort->key_bits;
                                }
                            }
                        }
                        spec.initial_push = directive_on(q->second.owner);
                        {
                            std::vector<std::string> chain = {spec.owner, name, callee};
                            chain.insert(chain.end(), pushers.begin(), pushers.end());
                            spec.vector_leaves = !vectorizes(chain);
                        }
                        // Whether running an entry can push onto this queue,
                        // which is what asks for a second buffer (SSA/
                        // Defer.h): whether the callee reaches one of the
                        // queue's pushers along calls that stay calls once
                        // every directive is applied. A call another
                        // directive defers is a push onto another queue, run
                        // by that queue's drain after this one's pass; the
                        // calls after a staged one, or after one deferred
                        // with its continuation, move to that queue's drain
                        // with the rest of their function. So the ray queue
                        // is two buffers while the medium scattering runs
                        // inside the trace's drain and hands the turned ray
                        // straight back, and one once the scattering is a
                        // queue of its own and the material kernel another:
                        // every push onto it then comes after its pass.
                        {
                            const std::set<std::string> pusher_set(pushers.begin(),
                                                                   pushers.end());
                            std::set<std::string> seen;
                            std::vector<std::string> work{callee};
                            bool pushes_self = false;
                            while (!work.empty() && !pushes_self) {
                                const std::string g = work.back();
                                work.pop_back();
                                if (!seen.insert(g).second) {
                                    continue;
                                }
                                const auto gf = fmap.find(g);
                                if (gf == fmap.end()) {
                                    continue;
                                }
                                // The calls of g that other directives turn
                                // into pushes, and the boundaries after which
                                // g's calls move to another drain.
                                std::set<std::string> deferred_edges;
                                std::vector<std::string> boundaries;
                                if (const auto ts = transforms.find(g);
                                    ts != transforms.end()) {
                                    for (const ir::Transform &t : ts->second) {
                                        // A cut on the same queue moves no
                                        // call to another drain: its rest
                                        // runs in this drain's second pass.
                                        if (const auto *st = std::get_if<ir::Stage>(&t);
                                            st != nullptr && !st->callee.names.empty() &&
                                            st->loop.empty()) {
                                            boundaries.push_back(st->callee.names.back());
                                        } else if (const auto *other =
                                                       std::get_if<ir::Defer>(&t);
                                                   other != nullptr &&
                                                   other->loop.empty() &&
                                                   !other->callee.names.empty()) {
                                            const std::string &h = other->callee.names.back();
                                            deferred_edges.insert(h);
                                            if (has_nontail_call(fmap, g, h)) {
                                                boundaries.push_back(h);
                                            }
                                        }
                                    }
                                }
                                const auto moved = [&](const std::string &h) {
                                    return std::any_of(
                                        boundaries.begin(), boundaries.end(),
                                        [&](const std::string &b) {
                                            return calls_after(*gf->second, b, h);
                                        });
                                };
                                // A pusher reached pushes from this drain,
                                // unless its pushes lie after a boundary and
                                // so are made from the other drain -- the
                                // cycle rays to hits to rays.
                                if (pusher_set.contains(g) && !moved(callee)) {
                                    pushes_self = true;
                                    break;
                                }
                                for (const std::string &h : callees_of(*gf->second)) {
                                    if (!deferred_edges.contains(h) && !moved(h)) {
                                        work.push_back(h);
                                    }
                                }
                            }
                            spec.drain_pushes_self = pushes_self;
                        }
                        // The functions that make the call: for the primary
                        // and for every other pusher the schedule named,
                        // the function itself, the continuations split off
                        // from it and the copies a queue's specialize made of
                        // it, each where it is reached from the owner
                        // (SSA/Stage.h, call_holders) -- the medium-sample
                        // deferral the trace's function makes after its
                        // sample stage, the BSSRDF deferral the material
                        // kernel makes in each material's copy. The first is
                        // the deferral's own; the rest push alongside it
                        // (QueueSpec::also_from). A pusher none of whose
                        // functions is found is kept under its own name, for
                        // the deferral to refuse in its words.
                        std::vector<std::string> holders;
                        {
                            std::vector<std::string> named{primary};
                            for (const std::string &f : pushers) {
                                if (f != primary) {
                                    named.push_back(f);
                                }
                            }
                            for (const std::string &f : named) {
                                std::vector<std::string> of =
                                    call_holders(fmap, f, callee_arm, q->second.owner);
                                std::erase_if(of, [&](const std::string &h) {
                                    return !carries_variant(h, arm.variant);
                                });
                                if (std::getenv("BONSAI_EXPLAIN_DEFER") != nullptr) {
                                    std::cerr << "; " << d.queue << ": " << f
                                              << " stands for";
                                    for (const std::string &h : of) {
                                        std::cerr << " " << h;
                                    }
                                    std::cerr << (of.empty() ? " nothing reached from "
                                                                   + q->second.owner +
                                                                   " that calls " + callee
                                                             : "")
                                              << "\n";
                                }
                                if (of.empty()) {
                                    // The call may be there with a copy of
                                    // the callee as its callee: a queue's
                                    // specialize upstream copies the callee
                                    // along when the function hands it the
                                    // key, and a deferral's callee is one
                                    // function (pbrt's BSSRDF kernel takes
                                    // the basic texture evaluator for every
                                    // entry, so vol_subsurface takes none
                                    // and the material queues' split by
                                    // evaluator leaves it one function).
                                    const std::vector<std::string> copies =
                                        copies_called(fmap, f, callee_arm, q->second.owner);
                                    std::string list;
                                    for (const std::string &c : copies) {
                                        list += (list.empty() ? "" : ", ") + c;
                                    }
                                    internal_assert(copies.empty())
                                        << d.queue << ": " << f << " makes no call to "
                                        << callee << " where " << q->second.owner
                                        << " reaches it; its copies call " << list
                                        << ", copies a queue's specialize made of "
                                        << callee << " because " << f
                                        << " hands it the key. A deferral's callee is "
                                           "one function: the argument has to be the "
                                           "same for every copy (as pbrt's BSSRDF kernel "
                                           "takes the basic evaluator for every entry).";
                                    of.push_back(call_holder(fmap, f, callee_arm));
                                }
                                for (const std::string &h : of) {
                                    if (std::find(holders.begin(), holders.end(), h) ==
                                        holders.end()) {
                                        holders.push_back(h);
                                    }
                                }
                            }
                        }
                        for (size_t h = 1; h < holders.size(); h++) {
                            spec.also_from.push_back(holders[h]);
                        }
                        spec.loop = arm.loop;
                        spec.storage = q->second.storage;
                        if (q->second.capacity.has_value()) {
                            const auto n = get_constant_value<int64_t>(
                                *q->second.capacity);
                            internal_assert(n.has_value() && *n > 0)
                                << d.queue << " = " << spec.owner << ".queue("
                                << spec.loop << ", " << *q->second.capacity
                                << ") needs a constant, positive capacity";
                            spec.capacity = uint64_t(*n);
                        }
                        // A call that is not in tail position and not
                        // spawned is deferred with its continuation
                        // (SSA/Stage.h, defer_continuation); a tail call or
                        // a spawned one is deferred as it is. The entry,
                        // queue and flag types made are the program's now,
                        // for the printer and the backends' declarations.
                        const std::string &holder = holders.front();
                        const bool with_rest = has_nontail_call(fmap, holder, callee_arm);
                        for (const Type &made :
                             with_rest ? defer_continuation(fmap, holder, callee_arm, spec)
                                       : defer(fmap, holder, callee_arm, spec)) {
                            const auto *s = made.as<Struct_t>();
                            internal_assert(s) << made;
                            if (keep_ssa != nullptr) {
                                keep_ssa->types[s->name] = made;
                            }
                        }
                        } // per arm
                    },
                    // The keys were put on the recursion in lowering
                    // (Lower/Sorts.cpp); the run is ordered by them here,
                    // where the schedule wrote the sort (see sort_reach).
                    [&](const ir::Sort &) { sort_reach(name); },
                    // Applied earlier in lowering, by the pass named
                    // (Lower/Prefetches.cpp), to the function as a whole:
                    // written on an arm of a specialize, the fetches are
                    // taken back out of the copies that are not the arm's
                    // (ir::Prefetch::variant).
                    [&](const ir::Prefetch &p) {
                        if (p.variant.empty()) {
                            return;
                        }
                        for (const std::string &fname :
                             functions_standing_for(fmap, name)) {
                            if (carries_variant(fname, p.variant)) {
                                continue;
                            }
                            for (const auto &block : fmap.at(fname)->blocks) {
                                std::erase_if(block->instrs, [](const auto &instr) {
                                    return instr->op == Instruction::Op::Intrinsic &&
                                           instr->intrinsic == ir::Intrinsic::prefetch;
                                });
                            }
                        }
                    },
                    [&](const ir::Split &s) {
                        internal_assert(!s.i.names.empty() &&
                                        !s.io.names.empty() &&
                                        !s.ii.names.empty())
                            << "split() requires loop names for: " << name;
                        // The chunk: a constant, or an expression over the
                        // function's values that is one value for the whole
                        // loop (the two overloads of split, SSA/Rewrite.h).
                        // Every variant's loop, each split into pieces named
                        // for that variant (`s_gang!VolPath`), so that a
                        // later directive naming `s_gang` finds them all.
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, s.i.names.back(), "split")) {
                            const std::string suffix =
                                variant_suffix(s.i.names.back(), at.index);
                            Expr factor = s.factor;
                            if (!s.factor_name.empty()) {
                                // The named value's type, from the function
                                // the loop is in: a parameter's or an
                                // instruction's.
                                const std::optional<Type> type =
                                    type_of_value(*fmap.at(at.func), s.factor_name);
                                internal_assert(type.has_value())
                                    << "split(" << s.i.names.back() << ", "
                                    << s.factor_name << ") on " << name << ": "
                                    << at.func << " has no value named "
                                    << s.factor_name
                                    << ". The chunk names a parameter, or a value "
                                    << "computed before the loop that the program "
                                    << "keeps.";
                                factor = Var::make(*type, s.factor_name);
                            }
                            split(fmap, at.func, at.index, factor,
                                  s.io.names.back() + suffix,
                                  s.ii.names.back() + suffix, !s.generate_tail);
                        }
                    },
                    [&](const ir::Persistent &p) {
                        internal_assert(!p.i.names.empty() && !p.j.names.empty())
                            << "persistent() requires loop names for: " << name;
                        // The count: a constant expression, or a value of
                        // the function named as a split's chunk is. Every
                        // variant's loop, its workers named for the variant.
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, p.i.names.back(), "persistent")) {
                            const std::string suffix =
                                variant_suffix(p.i.names.back(), at.index);
                            Expr count = p.count;
                            if (!p.count_name.empty()) {
                                const std::optional<Type> type =
                                    type_of_value(*fmap.at(at.func), p.count_name);
                                internal_assert(type.has_value())
                                    << "persistent(" << p.i.names.back() << ", "
                                    << p.j.names.back() << ", " << p.count_name
                                    << ") on " << name << ": " << at.func
                                    << " has no value named " << p.count_name
                                    << ". The count names a parameter, or a value "
                                    << "computed before the loop that the program "
                                    << "keeps.";
                                count = Var::make(*type, p.count_name);
                            }
                            persistent(fmap, at.func, at.index, count,
                                       p.j.names.back() + suffix);
                        }
                    },
                    [&](const ir::Collapse &c) {
                        internal_assert(!c.io.names.empty() &&
                                        !c.ii.names.empty() &&
                                        !c.i.names.empty())
                            << "collapse() requires loop names for: " << name;
                        // Every variant's pair: the outer loop names the
                        // variants, the inner is the one inside each. An
                        // outer with none inside it -- the loops a reorder
                        // distributed beside its nest, which share the
                        // nest's name -- is no nest and is passed over; a
                        // directive that finds no pair at all is wrong.
                        size_t pairs = 0;
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, c.io.names.back(), "collapse")) {
                            const std::string suffix =
                                variant_suffix(c.io.names.back(), at.index);
                            const std::optional<LoopSite> in = nested_loop(
                                fmap, at, c.ii.names.back(), suffix, "collapse",
                                /*either_way=*/false);
                            if (!in.has_value()) {
                                continue;
                            }
                            collapse(fmap, at.func, at.index, in->index,
                                     c.i.names.back() + suffix);
                            pairs++;
                        }
                        internal_assert(pairs > 0)
                            << "collapse(" << c.io.names.back() << ", "
                            << c.ii.names.back() << ") on " << name
                            << ": no parfor named " << c.ii.names.back()
                            << " is nested in a parfor named " << c.io.names.back()
                            << ", so there is no nest to collapse.";
                    },
                    [&](const ir::Reorder &r) {
                        internal_assert(!r.inner.names.empty() &&
                                        !r.outer.names.empty())
                            << "reorder() requires loop names for: " << name;
                        // Every variant's pair, as collapse: the loop to be
                        // inner names the variants, the other is the loop
                        // nested with each, either way round (a second
                        // reorder of a nest names the distributed loops'
                        // name too, and those have no partner: passed over).
                        size_t pairs = 0;
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, r.inner.names.back(), "reorder")) {
                            const std::string suffix =
                                variant_suffix(r.inner.names.back(), at.index);
                            const std::optional<LoopSite> other = nested_loop(
                                fmap, at, r.outer.names.back(), suffix, "reorder",
                                /*either_way=*/true);
                            if (!other.has_value()) {
                                continue;
                            }
                            reorder(fmap, at.func, at.index, other->index,
                                    /*vector_leaves=*/!vectorizes({at.func}));
                            pairs++;
                        }
                        internal_assert(pairs > 0)
                            << "reorder(" << r.inner.names.back() << ", "
                            << r.outer.names.back() << ") on " << name
                            << ": no parfor named " << r.outer.names.back()
                            << " is nested with a parfor named " << r.inner.names.back()
                            << ", so there is no nest to interchange.";
                    },
                    [&](const ir::Bind &b) {
                        if (b.lambda.defined()) {
                            // A function's bind to a hardware unit, applied
                            // by lower::LowerHardwareBinds before this; the
                            // function's body is the unit's already.
                            return;
                        }
                        if (b.resource == ir::Resource::RTCore) {
                            // `f.bind(RTCore)`: the function's query on
                            // the ray tracing hardware, applied where the
                            // query is lowered (Lower/Trees.cpp).
                            internal_assert(b.i.names.empty())
                                << "bind(" << b.i.names.back()
                                << ", RTCore) on " << name << ": RTCore "
                                << "binds a function's query, not a loop; "
                                << "write `" << name << ".bind(RTCore)`.";
                            return;
                        }
                        internal_assert(!b.i.names.empty())
                            << "bind() requires a cursor for: " << name;
                        for (const LoopSite &at : resolve_loops(
                                 fmap, name, b.i.names.back(), "bind")) {
                            bind(fmap, at.func, at.index, b.resource);
                        }
                    },
                },
                t);
        }
        report();
    }
    if (!divided) {
        divide_all();
    }

    phase("directives");

    // Now that every bind has been applied, it is settled which loops run
    // iterations at the same time -- and so which atomics were asked for
    // against a parallelism the schedule did not take up. Those cost nothing
    // to remove and everything to keep, so they go here, after the schedule
    // and before the graph is turned back into statements. Which functions
    // the bound loops call is a fact of the whole program (a drained
    // function runs once per thread of its drain, with no loop of its own
    // to say so), settled before the functions are taken one at a time.
    const std::set<std::string> called_in_parallel =
        functions_called_in_parallel(fmap);
    // And of those functions' address parameters, which ones every call
    // running at once is handed storage of its own through -- the
    // sample's reducer handed down a path's calls, a drain's record slot --
    // which is a fact of every call site in the program, settled the same
    // way (SSA/Contention.h, parameter_contention).
    const std::map<std::string, std::vector<Contention>> params =
        parameter_contention(fmap, called_in_parallel);
    for (const auto &[name, f] : fmap) {
        const auto answers = params.find(name);
        DemoteAtomics::run(*f, called_in_parallel.contains(name),
                           answers == params.end() ? nullptr : &answers->second);
        // The atomics every thread of a GPU block makes to one word, once
        // per block instead (SSA/BlockAccumulates.h); before the allocas
        // are promoted, so the slots it makes become values.
        ReduceBlockAccumulates::run(*f);
    }

    phase("atomics after the binds");
    // Every array a transform induced -- a deferral's record, an interchange's
    // expansion of a value per iteration -- as a struct of arrays, now that
    // every directive has made its storage and every use of it, and before
    // the hit continuations copy a drain's body into a program whose
    // parameters mirror the values it reads (SSA/InducedStorage.h).
    split_induced_storage(fmap);
    phase("induced storage");

    // The launch gates, while every drain still carries its queue
    // (Terminator::ParFor::queue_base, cleared for host drains just below),
    // while a push is still one Push instruction naming its queue, and
    // before the signatures are read back so the mask parameters a split
    // adds land in the owner's exported type (SSA/GateDrains.h: pbrt's
    // haveBasicEvalMaterial / haveSubsurface / haveMedia as a rewrite of
    // the queues' owner).
    {
        std::vector<ir::Program::ArmMask> masks =
            gate_drains(fmap, queue_splits, adt_storages);
        if (keep_ssa != nullptr) {
            keep_ssa->arm_masks = std::move(masks);
        }
    }

    // A raygen's continuation after its nearest-hit trace becomes the
    // closest-hit program of that launch, pbrt's shape of the trace kernel
    // (SSA/HitContinuations.h). After the gates, with the drains final and
    // a push still one instruction; before the simplify below, which
    // specializes the two copies the split makes on hit and on miss.
    split_hit_continuations(fmap);
    phase("drain gates");

    // The round loops' guards and the inner queues' header fills
    // (SSA/RoundGuard.h): the loop becomes `do { launch } while
    // (bonsai_cuda_round_guard(..))` -- one synchronous four-byte read per
    // round in place of the generic host load -- and a header a queue made
    // inside the rounds restaged every round moves to the preheader. Here
    // because it needs the drains' queue facts and GPU bindings (like the
    // gates above), and before the simplify below so the dead thread of the
    // replaced read is swept with everything else.
    if (keep_ssa != nullptr) {
        rewrite_round_guards(fmap, *keep_ssa);
    }
    phase("round guards");

    // A queue's drain carries the queue's capacity and the address of its
    // count (Terminator::ParFor::capacity, set in Defer.cpp) for the launch
    // over the capacity a GPU bind makes of it (Bind.cpp). A drain that is
    // not GPU-bound -- run on the host, or not bound at all -- has no use
    // for them: it runs to the count as before. Cleared here so that the
    // instructions that computed them (the count's field pointer) are dead
    // and the simplify below removes them, leaving the host's drains exactly
    // as they were.
    for (const auto &[name, f] : fmap) {
        for (const auto &block : f->blocks) {
            auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p == nullptr || !p->capacity) {
                continue;
            }
            const bool gpu = p->binding == Resource::GPUThread ||
                             p->binding == Resource::GPUBlock ||
                             p->binding == Resource::OptixThread;
            if (!gpu) {
                p->capacity = nullptr;
                p->queue_base = nullptr;
                p->queue_slot = nullptr;
                p->count_address = nullptr;
            }
        }
    }

    // What the transforms built by rule is looked at once more (see
    // SSA/Simplify.h): a vectorized split loop's index is the outer index
    // broadcast plus the ramp of lanes, which is a ramp, and only as a ramp
    // is an access at it a dense vector load or store rather than a gather.
    // With what every value is known to lie between as the program now
    // stands, for the rules that are exact only under such a fact: the
    // traversal's prune against its running best, a float compare, is one
    // on the bits once the best is known non-negative, and joins the slab
    // test's compare -- which needs the best as the vectorized traversal
    // reads it, a load through the pointer its caller passed, before the
    // allocas are promoted and it is a value carried round the loop that one
    // pass cannot bound.
    {
        const ConstantIntervals intervals(fmap);
        if (options.dump_intervals) {
            std::cout << "=== intervals after the directives\n";
            for (const auto &[name, f] : fmap) {
                intervals.dump(std::cout, *f);
            }
        }
        // The directives have run, so the lanes of a vector built from its
        // lanes may be read back (SSA/Simplify.h, fold_built_lanes): the sort
        // read the shape it needed from such vectors as it ran.
        for (const auto &[name, f] : fmap) {
            simplify(*f, &intervals, /*fold_built_lanes=*/true);
        }
    }
    phase("simplify after the directives");

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
    // The foreign functions' signatures too: a call to one is turned back
    // into a Call naming it, typed from its declaration (CodeGen_Stmt.cpp).
    if (keep_ssa != nullptr) {
        for (const auto &[name, f] : keep_ssa->foreign_funcs) {
            func_type_map[name] = f->call_type();
        }
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
    phase("contraction, signatures and dumps");
    for (const auto &[name, f] : fmap) {
        lower_pushes(*f, layouts);
    }
    phase("pushes");
    // A pure value a loop computes from a cell in memory is computed where
    // the cell is written instead, into a mirror cell the loop reads
    // (SSA/MirrorLoads.h): the node test's limit from the argmin's best,
    // once per accepted hit rather than once per node. After the pushes,
    // whose entries are about the memory as it was; before the promotion
    // below, which makes the mirror a value carried round the loop.
    for (const auto &[name, f] : fmap) {
        mirror_derived_loads(*f);
    }
    phase("mirror derived loads");
    // A small local array of booleans -- a node's children's tests, which
    // the sort and the loopified descent read whole as a mask -- is the bits
    // of one word from here on (SSA/PackBoolArrays.h), assembled in a
    // register rather than stored a byte at a time and loaded back as a
    // word, which the store buffer cannot forward. After the directives,
    // whose vectorized children's loops have made their arrays vectors
    // already, and before the promotion below, which makes the word a value.
    for (const auto &[name, f] : fmap) {
        pack_bool_arrays(*f);
    }
    phase("pack bool arrays");
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
    phase("promote allocas");
    // And once more with the slots gone: a value that reached a match only
    // through a `mut` local -- the BxDF a material's arm builds and hands on
    // through the inliner's result slot -- is a struct with a constant tag
    // now, so the match folds and the arms nothing reaches go; each of the
    // wavefront's per-material copies is then one material's BSDF and not
    // every one's (SSA/Simplify.h).
    for (const auto &[name, f] : fmap) {
        simplify(*f, nullptr, /*fold_built_lanes=*/true);
    }
    phase("simplify after promotion");

    // Contraction after the schedule, and after the promotion: what a
    // transform produces is arithmetic too -- a vectorized gang's widened
    // multiply and add are as fusible as the scalar pair they came from --
    // and a product the program kept in a `mut` local is a value only now.
    // The inliner hands a callee's result back through such a local, so
    // `gaussian(x) - expX`, with `gaussian` ending in a product, is a load
    // minus a value until the slots are promoted; contracted earlier, the
    // subtraction stayed two roundings where gcc's (and pbrt's) is one.
    if (options.ffp_contract) {
        for (const auto &[name, f] : fmap) {
            contract_fp(*f);
        }
    }
    phase("contraction");
    // A slot that is filled and then copied into the program's answer is
    // the answer's own memory from here on, and the copy is gone
    // (SSA/DestinationPassing.h): an argmin's record is written where the
    // export puts it, as Embree's leaf writes the caller's RayHit. After the
    // promotion and the simplification, which turn the copy into a load and
    // a store with values between them; before the storage passes, which
    // would otherwise place a slot that is about to go.
    for (const auto &[name, f] : fmap) {
        pass_destinations(*f, fmap);
    }
    phase("pass destinations");
    // Storage made inside a loop that every iteration could share is made
    // once, before the loop (SSA/HoistAllocations.h): a pass's queues before
    // the loop over passes. After the binds, which decide which loops may be
    // crossed, and after the promotion, which decides what is storage at
    // all. Then what is a run-time-sized array at the top of a function no
    // loop calls goes to the heap, freed at the function's exits
    // (SSA/HeapArrays.h): once per call of the program, never per iteration.
    {
        const std::set<std::string> in_loops = called_inside_loops(fmap);
        for (const auto &[name, f] : fmap) {
            hoist_invariant_allocations(*f);
            // Where a queue's storage finally sits is what the schedule's
            // storage word is checked against, and an Extern queue's arrays
            // become parameters here, before the heap pass could take them
            // (SSA/QueueStorage.h).
            const bool exported =
                std::find(f->attributes.begin(), f->attributes.end(),
                          ir::Function::Attribute::exported) != f->attributes.end();
            place_queue_storage(*f, exported);
            heap_arrays(*f, in_loops.contains(name));
        }
    }
    phase("hoist allocations");
    // Every parfor body's arguments are its captures again, whatever the
    // rewrites above and the promotion left of that (see SSA/CloseBodies.h):
    // the code generators make a kernel of a bound loop's body from those
    // arguments.
    for (const auto &[name, f] : fmap) {
        close_parfor_bodies(*f);
    }
    phase("close parfor bodies");
    // What the code generators are handed: after the allocas are promoted
    // and the bodies closed, which is where a name a kernel cannot bind
    // shows up.
    if (options.dump_ssa_final) {
        dump_ssa(std::cout, "final", fmap, options.is_verbose);
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
    phase("statements");

    return new_funcs;
}

} // namespace

ir::Program ConvertToSSA::run(ir::Program program,
                              const CompilerOptions &options) const {
    const ir::TransformMap &transforms = program.schedule.func_transforms;
    const ir::TransformOrder &order = program.schedule.transform_order;
    const ir::BranchPolicyMap &policies = program.schedule.branch_policies;
    const std::map<std::string, ir::Queue> &queues = program.schedule.queues;
    const std::vector<ir::QueueSpecialize> &queue_splits =
        program.schedule.queue_specializations;

    ir::Program new_program;
    new_program.types = program.types;
    new_program.externs = program.externs;
    new_program.schedule = program.schedule;
    new_program.adt_storages = program.adt_storages;
    new_program.foreign_funcs = program.foreign_funcs;
    new_program.funcs =
        convert(std::move(program.funcs), transforms, order, policies, queues,
                queue_splits, new_program.adt_storages, options, &new_program);
    return new_program;
}

ir::FuncMap ConvertToSSA::run(ir::FuncMap funcs,
                              const CompilerOptions &options) const {
    return convert(std::move(funcs), {}, {}, {}, {}, {}, {}, options);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
