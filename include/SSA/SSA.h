#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "IR/Type.h"

namespace bonsai {
namespace ir {
namespace ssa {

struct Argument {
    Type type;
    std::string name;

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
        Add,
        Alloc,  // on heap
        Alloca, // on stack
        Append, // side-effect-y
        Bc,
        Cast,
        Div,
        Eps,
        Eq,
        ExtractIdx,
        GEP,
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
        Reinterpret,
        Set,
        Store, // side-effect-y
        Sub,
    };

    // empty -> side-effect-y (store/append)
    std::string name;
    Type type;
    Op op;

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

    std::variant<std::monostate, Jump, Dispatch, Return, ParFor, Yield, Call> data;

    bool defined() const {
        return !std::holds_alternative<std::monostate>(data);
    }

    void dump(std::ostream &os) const;
};

struct Function;

struct Block {
    std::string name;
    std::vector<Argument> args; // take the place of phis
    std::vector<std::shared_ptr<Instruction>> instrs;
    Terminator terminator;
    std::weak_ptr<Function> owner;

    // Duplicated data; for lookups *only*.
    std::map<std::string, std::shared_ptr<Value>> lookups;

    std::vector<std::weak_ptr<Block>> preds;

    // If value is defined in this block, returns it, otherwise adds it as an
    // argument recursively until it finds the block it is defined in!
    std::shared_ptr<Value> get_value(const std::string &name, const Type &type);

    void dump(std::ostream &os) const;
    void dump() const; // defaults to std::cout
};

struct Function {
    // First block is entry block.
    std::vector<std::shared_ptr<Block>> blocks;
    ir::Type ret_type; // convenience.

    void dump(std::ostream &os) const;
};

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
