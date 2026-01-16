#include "SSA/Convert.h"

#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

namespace bonsai {
namespace ir {
namespace ssa {

struct FunctionBuilder : Visitor {
    std::shared_ptr<Value> value = nullptr;
    std::shared_ptr<Block> block = nullptr;

    std::shared_ptr<ssa::Function> function;

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
            Type type = arg.mutating ? Ptr_t::make(arg.type) : arg.type;
            // internal_assert(!arg.mutating)
            //     << "TODO: handle mutable arguments: " << func.name << " has "
            //     << arg.name << " as mutable";
            Argument a = {arg.type, arg.name};
            block->args.push_back(a);
            auto [_, inserted] = block->lookups.insert(
                {arg.name, std::make_shared<Value>(std::move(a))});
            internal_assert(inserted)
                << "Failed to insert argument: " << arg.name
                << " of function: " << func.name;
        }

        function->blocks.push_back(block);
        function->ret_type = func.ret_type;

        func.body.accept(this);
    }

    // TODO: cache for unmutable expressions!
    std::shared_ptr<Value> get_value(const Expr &expr) {
        value = nullptr;
        expr.accept(this);
        internal_assert(value) << expr << " failed to produce SSA value";
        return std::move(value);
    }

    void make_instruction(const std::string &name, Type type,
                          std::shared_ptr<Value> v) {
        internal_assert(block) << "Tried to append instruction to empty block";

        // If v already refers to an Instruction, just rename it (copy
        // propagation)
        if (auto instr_ptr =
                std::get_if<std::shared_ptr<Instruction>>(&v->data)) {
            internal_assert(*instr_ptr) << "Null instruction value";

            auto &instr = *instr_ptr;

            // Update name
            instr->name = name;

            // Update lookup table
            auto [it, inserted] = block->lookups.insert({name, v});
            if (!inserted) {
                it->second = v; // overwrite existing entry
            }

            return;
        }

        // Otherwise, create a new Set instruction
        std::vector<std::shared_ptr<Value>> vs = {std::move(v)};
        std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
            name, std::move(type), Instruction::Op::Set, std::move(vs), block);
        block->instrs.push_back(instr);

        auto [_, inserted] =
            block->lookups.insert({name, std::make_shared<Value>(instr)});
        internal_assert(inserted) << name << "already exists in block!\n";
    }

    uint64_t counter = 0;
    std::string get_unique_name() { return "@" + std::to_string(counter++); }

    std::shared_ptr<Value>
    make_instruction(Type type, Instruction::Op op,
                     std::vector<std::shared_ptr<Value>> vs) {
        internal_assert(block) << "Tried to append instruction to empty block";
        std::string name = get_unique_name();
        std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
            name, std::move(type), op, std::move(vs), block);
        block->instrs.push_back(instr);
        auto v = std::make_shared<Value>(std::move(instr));
        auto [_, inserted] = block->lookups.insert({name, v});
        internal_assert(inserted) << name << "already exists in block!\n";
        return v;
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
            {Terminator::Jump{.name = node->else_body.defined()
                                          ? else_case->name
                                          : merge_block->name},
             // v != 0
             Terminator::Jump{.name = then_case->name}}};

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

    void visit(const LetStmt *node) override {
        auto v = get_value(node->value);
        make_instruction(node->loc.base, node->loc.base_type, std::move(v));
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

    void visit(const CallStmt *node) override {
        auto func = get_value(node->func);
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());
        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }

        auto call_name = get_call_name(func);

        // Create continuation block.
        auto cont_block = make_block(call_name + "_call_cont");
        cont_block->preds.push_back(block);

        // End the current block with a Call
        internal_assert(!block->terminator.defined());
        auto call_block = std::move(block);

        call_block->terminator.data = Terminator::Call{
            .call = Terminator::Jump{.name = call_name, std::move(args)},
            // 'args' is empty for now (implicit capture via CFG lookup)
            // Does not receive output of call, because value is dropped.
            .cont = Terminator::Jump{.name = cont_block->name},
            .drop = true};

        block = cont_block;
    }

    void visit(const Append *node) override {
        auto v = get_value(node->value);
        auto loc = get_value(node->loc.to_expr());
        std::vector<std::shared_ptr<Value>> args = {std::move(loc),
                                                    std::move(v)};

        std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
            Instruction::Op::Append, std::move(args), block);
        block->instrs.push_back(instr);
    }

    void visit(const Allocate *node) override {
        if (node->memory == Allocate::Stack) {
            // TODO: require simple (non-struct) type?
            if (node->value.defined()) {
                // Handle like a LetStmt, this is a primitive type that will
                // just be updated.
                auto v = get_value(node->value);
                make_instruction(node->loc.base, node->loc.base_type,
                                 std::move(v));
            }
            return;
        }
        internal_assert(node->memory == Allocate::Heap)
            << "TODO: handle memory locations in SSA!";
        internal_assert(!node->value.defined())
            << "TODO: handle values in Allocate SSA! " << node->value;

        std::vector<std::shared_ptr<Value>> args;
        std::shared_ptr<Instruction> instr =
            std::make_shared<Instruction>(node->loc.base, node->loc.base_type,
                                          Instruction::Op::Alloc, args, block);
        block->instrs.push_back(instr);

        auto [_, inserted] = block->lookups.insert(
            {node->loc.base, std::make_shared<Value>(instr)});
        internal_assert(inserted)
            << node->loc.base << "already exists in block!\n";
    }

    void visit(const Store *node) override {
        if (node->loc.accesses.empty()) {
            internal_assert(node->loc.type.is_stack_allocatable())
                << "TODO: handle non-primitive (heap) stores in SSA: "
                << Stmt(node);
            auto v = get_value(node->value);

            // Overwrite the name with v (insert if missing).
            // All successors of the current block will receive v.
            block->lookups[node->loc.base] = v;
            return;
        }

        // Get the stored value *first*.
        // Before evaluating the lhs.
        auto v = get_value(node->value);

        // Create GEP
        // TODO: should this be a pointer to the type??
        auto var = block->get_value(node->loc.base, node->loc.base_type);

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
                var = make_instruction(Type(), Instruction::Op::GEP,
                                       {std::move(var), std::move(i)});
            }
        }

        std::vector<std::shared_ptr<Value>> args = {std::move(var),
                                                    std::move(v)};
        std::shared_ptr<Instruction> instr = std::make_shared<Instruction>(
            Instruction::Op::Store, std::move(args), block);
        block->instrs.push_back(instr);
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
        internal_assert(node->op == Extrema::eps) << "TODO: inf";
        value = make_instruction(node->type, Instruction::Op::Eps, {});
    }

    void visit(const Access *node) override {
        auto v = get_value(node->value);
        static const Type u32 = UInt_t::make(32);
        const Struct_t *struct_t = node->value.type().as<Struct_t>();
        internal_assert(struct_t)
            << node->value.type().as<Struct_t>() << " of " << Expr(node);
        auto idx = find_struct_index(node->field, struct_t->fields);
        auto vidx = make_constant(u32, (uint64_t)idx);
        value = make_instruction(node->type, Instruction::Op::LoadField,
                                 {std::move(v), std::move(vidx)});
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
        // case BinOp::Neq:
        //     return Instruction::Op::Neq;
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
        // case BinOp::Xor:
        //     return Instruction::Op::Xor;
        // case BinOp::BwAnd:
        //     return Instruction::Op::BwAnd;
        // case BinOp::BwOr:
        //     return Instruction::Op::BwOr;
        // case BinOp::Shl:
        //     return Instruction::Op::Shl;
        // case BinOp::Shr:
        //     return Instruction::Op::Shr;
        default: {
            internal_error << "TODO: handle: " << to_string(op);
        }
        }
    }

    void visit(const BinOp *node) override {
        auto a = get_value(node->a);
        auto b = get_value(node->b);
        auto op = get_binop(node->op);
        value = make_instruction(node->type, op, {std::move(a), std::move(b)});
    }

    void visit(const Broadcast *node) override {
        auto v = get_value(node->value);
        static const Type u32 = UInt_t::make(32);
        auto lanes = make_constant(u32, (uint64_t)node->lanes);
        value = make_instruction(node->type, Instruction::Op::Bc,
                                 {std::move(v), std::move(lanes)});
    }

    void visit(const Build *node) override {
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->values.size());
        for (const auto &arg : node->values) {
            args.emplace_back(get_value(arg));
        }
        value = make_instruction(node->type, Instruction::Op::MakeStruct,
                                 std::move(args));
    }

    void visit(const Cast *node) override {
        auto v = get_value(node->value);
        Instruction::Op op = (node->mode == Cast::Mode::Reinterpret)
                                 ? Instruction::Op::Reinterpret
                                 : Instruction::Op::Cast;
        value = make_instruction(node->type, op, {std::move(v)});
    }

    void visit(const Call *node) override {
        auto func = get_value(node->func);
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());
        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }

        auto call_name = get_call_name(func);

        // Create continuation block.
        auto cont_block = make_block(call_name + "_call_cont");
        cont_block->preds.push_back(block);

        // End the current block with a Call
        internal_assert(!block->terminator.defined());
        auto call_block = std::move(block);

        call_block->terminator.data = Terminator::Call{
            .call = Terminator::Jump{.name = call_name, std::move(args)},
            // TODO: `args` takes an argument that is the result of the call.
            .cont = Terminator::Jump{.name = cont_block->name},
            .drop = false};

        block = cont_block;

        std::string name = get_unique_name(); // name of returned item
        Argument arg{.type = node->type, .name = name};
        block->args.push_back(arg);
        // TODO value must now be the load of the first argument from the
        // cont_block!
        value = std::make_shared<Value>(std::move(arg));
    }

    void visit(const Extract *node) override {
        auto vec = get_value(node->vec);
        auto idx = get_value(node->idx);
        value = make_instruction(node->type, Instruction::Op::ExtractIdx,
                                 {std::move(vec), std::move(idx)});
    }

    Instruction::Op get_intrinsic(const Intrinsic::OpType &op) {
        switch (op) {
        case Intrinsic::abs:
            return Instruction::Op::Abs;
        // case Intrinsic::cos:
        //     return "cos";
        // case Intrinsic::cross:
        //     return "cross";
        // case Intrinsic::dot:
        //     return "dot";
        // case Intrinsic::fma:
        //     return "fma";
        case Intrinsic::max:
            return Instruction::Op::Max;
        case Intrinsic::min:
            return Instruction::Op::Min;
        // case Intrinsic::norm:
        //     return "norm";
        // case Intrinsic::pow:
        //     return "pow";
        // case Intrinsic::rand:
        //     return "rand";
        // case Intrinsic::round:
        //     return "round";
        // case Intrinsic::sin:
        //     return "sin";
        // case Intrinsic::sqr:
        //     return "sqr";
        // case Intrinsic::sqrt:
        //     return "sqrt";
        // case Intrinsic::tan:
        //     return "tan";
        default: {
            internal_error << "TODO: handle: " << to_string(op);
        }
        }
    }

    void visit(const Intrinsic *node) override {
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
        }
        std::vector<std::shared_ptr<Value>> args;
        args.reserve(node->args.size());
        for (const auto &arg : node->args) {
            args.emplace_back(get_value(arg));
        }
        auto op = get_intrinsic(node->op);
        value = make_instruction(node->type, op, std::move(args));
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

    Instruction::Op get_vreduce(const VectorReduce::OpType &op) {
        switch (op) {
        case VectorReduce::Add:
            return Instruction::Op::Add;
        // case VectorReduce::Idxmin:
        //     return "idxmin";
        // case VectorReduce::Idxmax:
        //     return "idxmax";
        case VectorReduce::Mul:
            return Instruction::Op::Mul;
        case VectorReduce::Min:
            return Instruction::Op::Min;
        case VectorReduce::Max:
            return Instruction::Op::Max;
        // case VectorReduce::Or:
        //     return "any";
        // case VectorReduce::And:
        //     return "all";
        default: {
            internal_error << "TODO: handle: " << to_string(op);
        }
        }
    }

    void visit(const VectorReduce *node) override {
        auto a = get_value(node->value);
        auto op = get_vreduce(node->op);
        value = make_instruction(node->type, op, {std::move(a)});
    }

    // RESTRICT_VISITOR(IntImm);
    // RESTRICT_VISITOR(UIntImm);
    // RESTRICT_VISITOR(FloatImm);
    // RESTRICT_VISITOR(BoolImm);
    RESTRICT_VISITOR(VecImm);
    RESTRICT_VISITOR(StringImm);
    // RESTRICT_VISITOR(Extrema);
    // RESTRICT_VISITOR(Var);
    // RESTRICT_VISITOR(BinOp);
    RESTRICT_VISITOR(UnOp);
    RESTRICT_VISITOR(Select);
    // RESTRICT_VISITOR(Cast);
    // RESTRICT_VISITOR(Broadcast);
    // RESTRICT_VISITOR(VectorReduce);
    RESTRICT_VISITOR(VectorShuffle);
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
    RESTRICT_VISITOR(PtrTo);
    RESTRICT_VISITOR(Deref);
    RESTRICT_VISITOR(AtomicAdd);

    // RESTRICT_VISITOR(CallStmt);
    RESTRICT_VISITOR(Print);
    // RESTRICT_VISITOR(Return);
    // RESTRICT_VISITOR(LetStmt);
    // RESTRICT_VISITOR(IfElse);
    RESTRICT_VISITOR(DoWhile);
    // RESTRICT_VISITOR(Sequence); // default behavior is fine.
    // RESTRICT_VISITOR(Allocate);
    RESTRICT_VISITOR(Free);
    // RESTRICT_VISITOR(Store);
    RESTRICT_VISITOR(Accumulate);
    RESTRICT_VISITOR(Label);
    RESTRICT_VISITOR(RecLoop);
    RESTRICT_VISITOR(Match);
    RESTRICT_VISITOR(Yield);
    RESTRICT_VISITOR(Iterate);
    RESTRICT_VISITOR(Scan);
    RESTRICT_VISITOR(YieldFrom);
    RESTRICT_VISITOR(ForAll);
    RESTRICT_VISITOR(ForEach);
    RESTRICT_VISITOR(Continue);
    RESTRICT_VISITOR(Launch);
    // RESTRICT_VISITOR(Append);
};

std::shared_ptr<ssa::Function>
build(const std::shared_ptr<ir::Function> &func) {
    std::cout << *func << std::endl;
    FunctionBuilder builder(*func);
    if (!builder.block->terminator.defined()) {
        builder.block->terminator.data = Terminator::Return{};
    }
    return builder.function;
}

ir::FuncMap ConvertToSSA::run(ir::FuncMap funcs,
                              const CompilerOptions &options) const {
    for (auto &[name, func] : funcs) {
        auto f = build(func);
        f->dump(std::cout);
    }
    return funcs;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
