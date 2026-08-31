#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IR/Function.h"
#include "IR/Type.h"

namespace bonsai {
namespace ir {
namespace ssa {

struct Argument {
    Type type;
    std::string name;
    // Only meaningful for a Function's entry-block arguments (see
    // FunctionBuilder in SSA/Convert.cpp and codegen_stmt in
    // SSA/CodeGen_Stmt.cpp); ignored elsewhere.
    bool mutating = false;
    // Likewise: carried through so that what lowering knew about the object
    // this parameter names survives the trip through this form (see
    // ir::Function::Argument::unaliased).
    bool unaliased = false;

    void dump(std::ostream &os) const;
};

struct Constant {
    Type type;
    // string -> function call!
    std::variant<bool, int64_t, uint64_t, double, std::string> data;

    void dump(std::ostream &os) const;
};

struct Instruction;

struct Value {
    std::variant<Argument, Constant, std::shared_ptr<Instruction>> data;

    Value(Argument argument) : data(std::move(argument)) {}
    Value(Constant constant) : data(std::move(constant)) {}
    Value(std::shared_ptr<Instruction> instr) : data(std::move(instr)) {};

    const Type &get_type() const;

    std::optional<Argument> get_argument() const;

    void dump(std::ostream &os) const;
};

struct Block;

struct Instruction {
    enum class Op {
        // Keep this sorted!
        Abs,
        // Accs are side-effect-y
        AccAdd,
        AccMul,
        AccSub,
        AccArgmin,
        AccArgmax,
        AccMin,
        AccMax,

        Add,
        // The address of a value, mirroring ir::PtrTo -- what a call site does
        // to an argument the callee takes by pointer (see
        // Lower/Mutability.cpp).
        //
        // This is deliberately *not* an Alloca and a Store. Most values that
        // get addressed already live somewhere -- a field of a struct that is
        // itself in memory, an element of an array -- and the backends know
        // how to name that place without copying anything. Deciding here that
        // the value needs a stack slot of its own throws that away, and it
        // cannot be taken back later: the pointer escapes into the call, so
        // SSA/PromoteAllocas.h will not touch it. Being pure rather than a
        // side-effecting pair also means the rewrites can move it to where it
        // is used instead of having to keep it in place.
        AddressOf,
        Alloc,  // on heap
        Alloca, // on stack
        // Is any lane of a mask set? A cross-lane reduction: its operand is
        // one value per lane but its result is a single uniform bool, which
        // is what lets a gang branch on it. This is what makes the latch of a
        // vectorized divergent loop uniform -- the gang goes round again as
        // long as any lane still wants to (see SSA/UniformizeLoops.h).
        Any,
        Append, // side-effect-y
        Bc,
        BwAnd,
        BwOr,
        Cast,
        Div,
        Eps,
        Eq,
        ExtractIdx,
        GEP,
        Inf,
        // Any of the intrinsics this IR has that are not spelled out above,
        // carried through verbatim: which one is in `intrinsic`.
        //
        // Abs, Max and Min have opcodes of their own because the passes here
        // reason about them directly -- the vectorizer has to know which of
        // their operands go per-lane, and a vector reduction lowers to them.
        // Nothing here has anything to say about `sqrt` or `pow` beyond
        // handing them back to the backend, so the whole rest of the set
        // rides on this one opcode rather than being enumerated twice.
        Intrinsic,
        LAnd,
        LOr,
        Leq,
        Load, // from ptr
        LoadField,
        Lt,
        MakeStruct,
        Max,
        Min,
        Mod,
        Mul,
        Ne,
        Print, // side-effect-y
        // The gang's lane indices: base + stride * <0, 1, ..., lanes-1>.
        // This is what a vectorized loop index becomes, and an index of
        // this shape is what makes a memory access dense rather than a
        // gather (see ir::Ramp).
        Ramp,
        // A reduction over the lanes of one value, as the source wrote it
        // (`sum(v)`, `max(v)`): which one is in `reduce`. Distinct from Any,
        // which reduces across the *gang* -- the lanes of a vectorized loop --
        // and is introduced by the vectorizer rather than by the program. The
        // two look alike and mean different things: this one is per-lane work
        // on a value a lane holds, and Any asks about the gang as a whole.
        Reduce,
        Reinterpret,
        Select,
        Set,
        Shl,
        Shr,
        // The storage size of `queried_type`, left for the backend to answer
        // (see ir::SizeOf).
        SizeOf,
        Store, // side-effect-y
        Sub,
        Xor,
    };

    // empty -> side-effect-y (store/append)
    std::string name;
    Type type;
    Op op;

    // The type an operation asks about rather than produces. Only SizeOf has
    // one: its answer is an integer, but what it is the size *of* is a type,
    // which no operand can carry.
    Type queried_type;

    // Which intrinsic this is. Only meaningful for Op::Intrinsic.
    ir::Intrinsic::OpType intrinsic = ir::Intrinsic::abs;

    // Which reduction this is. Only meaningful for Op::Reduce.
    ir::VectorReduce::OpType reduce = ir::VectorReduce::Add;

    std::vector<std::shared_ptr<Value>> operands;
    std::weak_ptr<Block> owner;

    Instruction(std::string name, Type type, Op op,
                std::vector<std::shared_ptr<Value>> operands,
                std::weak_ptr<Block> owner)
        : name(std::move(name)), type(std::move(type)), op(op),
          operands(std::move(operands)), owner(std::move(owner)) {}

    // Make a side-effect-y instruction
    Instruction(Op op, std::vector<std::shared_ptr<Value>> operands,
                std::weak_ptr<Block> owner)
        : op(op), operands(std::move(operands)), owner(std::move(owner)) {}

    void dump(std::ostream &os) const;
};

struct Terminator {
    struct Jump {
        // Unconditional
        std::string name;
        std::vector<std::shared_ptr<Value>> args;
    };
    struct Dispatch {
        // Conditional
        std::shared_ptr<Value> cond;
        std::vector<Jump> targets;
    };
    struct Return {
        std::shared_ptr<Value> value; // possibly empty
    };
    struct ParFor {
        std::string index;
        std::shared_ptr<Value> start, end, stride;

        // Body block varying index (first) and <n> uniform arguments.
        Jump body;
        Jump cont; // after the body. index out of scope.

        // The hardware a bind() put this loop on, if any. Carried rather than
        // acted on: at this level a bind changes nothing about the graph, it
        // only records what the loop is to run on, and code generation is
        // where that becomes a launch.
        std::optional<Resource> binding;
    };
    struct Yield {
        // Ends a ParFor block
    };
    struct Call {
        // Call + call's return continuation
        Jump call; // call to make
        Jump cont; // continuation to return to. if call returns a value, it is
                   // appended as the first argument to cont
        bool drop = true;
    };

    std::variant<std::monostate, Jump, Dispatch, Return, ParFor, Yield, Call>
        data;

    bool defined() const {
        return !std::holds_alternative<std::monostate>(data);
    }

    void dump(std::ostream &os) const;
};

struct Function;

struct Block : public std::enable_shared_from_this<Block> {
    std::string name;
    std::vector<Argument> args; // take the place of phis
    std::vector<std::shared_ptr<Instruction>> instrs;
    Terminator terminator;
    std::weak_ptr<Function> owner;

    // Duplicated data; for lookups *only*.
    std::map<std::string, std::shared_ptr<Value>> lookups;

    std::vector<std::weak_ptr<Block>> preds;

    // Give this block a parameter, and return the value standing for it.
    //
    // `args` and `lookups` are two halves of the same fact: the first is the
    // parameter list, the second is what the block hands back when something
    // asks for that name. Setting only the first leaves the name unresolvable,
    // and then the next instruction mentioning it threads in a *second*
    // parameter of the same name -- see forward_block_values, which is where
    // a name that is not in `lookups` gets one. Use this rather than pushing
    // onto `args` directly.
    std::shared_ptr<Value> add_argument(const Argument &arg);

    void make_instruction(const std::string &name, Type type,
                          std::shared_ptr<Value> v);
    std::shared_ptr<Value>
    make_instruction(Type type, Instruction::Op op,
                     std::vector<std::shared_ptr<Value>> vs,
                     bool allow_rename = false);
    void make_side_effect(Instruction::Op op,
                          std::vector<std::shared_ptr<Value>> vs);

    // If value is defined in this block, returns it, otherwise adds it as an
    // argument recursively until it finds the block it is defined in!
    std::shared_ptr<Value> get_value(const std::string &name, const Type &type);

    void dump(std::ostream &os) const;
    void dump() const; // defaults to std::cout
  private:
    // Mutates in place. Used for rethreading args in call instructions.
    void forward_block_values(std::vector<std::shared_ptr<Value>> &vs);
};

struct Function {
    // First block is entry block.
    std::vector<std::shared_ptr<Block>> blocks;
    ir::Type ret_type; // convenience.
    // Carried through from the originating ir::Function so that codegen_stmt
    // can reconstruct it faithfully (e.g. [[export]]).
    std::vector<ir::Function::Attribute> attributes;

    void dump(std::ostream &os) const;

    std::string get_unique_name() {
        return "@" + std::to_string(name_counter++);
    }

    // Makes sure `get_unique_name` will not hand out `name` again. A copy of
    // a function starts its counter at zero, so it has to be told about the
    // names its instructions already carry (see SSA/CloneFunction.h).
    void reserve_name(const std::string &name) {
        if (name.size() < 2 || name[0] != '@') {
            return;
        }
        const std::string digits = name.substr(1);
        if (digits.find_first_not_of("0123456789") != std::string::npos) {
            return;
        }
        name_counter = std::max<size_t>(name_counter, std::stoull(digits) + 1);
    }

  private:
    size_t name_counter = 0;
};

// What an opcode is called, for reporting one.
const char *op_name(Instruction::Op op);

// Useful helper for std::variant
template <class... Ts>
struct overloads : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
overloads(Ts...) -> overloads<Ts...>;

} // namespace ssa
} // namespace ir
} // namespace bonsai
