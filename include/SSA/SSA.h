#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "IR/Function.h"
#include "IR/Provenance.h"
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
    // Likewise: a reduction variable, only accumulated into here (see
    // ir::Function::Argument::reducer). A deferral stores its address in the
    // entry rather than its contents, since every continuation of the
    // iteration adds into the one place (SSA/Defer.cpp).
    bool reducer = false;

    void dump(std::ostream &os) const;
};

// A constant with no value: any value of its type will do, because nothing
// reads it (see ir::Undef). Printed `undef`.
struct Undefined {
    bool operator==(const Undefined &) const { return true; }
};

std::ostream &operator<<(std::ostream &os, const Undefined &);

struct Constant {
    Type type;
    // string -> function call!
    std::variant<bool, int64_t, uint64_t, double, std::string, Undefined> data;

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
        // Fetch-and-add: add to what a pointer points at and produce the value
        // it held *before* the add. Mirrors ir::AtomicAdd.
        //
        // Not one of the Accs above, and the difference is the result. An
        // AccAdd is a side effect and nothing reads it, so a schedule that
        // proves two lanes never touch the same address may drop its atomicity
        // (SSA/DemoteAtomics.h) and leave the arithmetic alone. This one's
        // whole purpose is the value it returns -- which slot a thread claimed
        // -- so it stays atomic wherever it might be reached in parallel, and
        // it has to be an instruction with a name rather than a side effect
        // with none.
        AtomicAdd,
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
        // The address of a struct's field, by index into that struct. The
        // mirror of LoadField, which reads one: a write needs the place rather
        // than the value, and a GEP cannot stand in for it because the two are
        // rebuilt differently -- an index access indexes an array, a field
        // access names a member, and a WriteLoc keeps them apart.
        FieldPtr,
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
        // Complement. Which one it is follows the operand, as it does in
        // ir::UnOp: negating a truth value for a bool, and flipping the bits
        // for an integer. Both are one instruction on every backend, so this
        // is a real operation rather than something to synthesise -- lowering
        // it to a select works only for bools, and to an exclusive-or with
        // all ones only for integers.
        Not,
        // How many lanes of a mask are set. A cross-lane reduction like Any:
        // its operand is one bool per lane and its result one uniform count,
        // which is what lets a gang decide something by counting itself.
        // Vote is lowered to two of these and a compare (see lower_votes in
        // SSA/Vectorize.cpp). Before widening a gang holds one bool, and the
        // count of it is that bool as a number.
        Popcount,
        Print, // side-effect-y
        // Appends a value to a queue: operands are a pointer to the queue --
        // a struct of a count and an array of entries, made by defer() (see
        // SSA/Defer.h) -- and the entry; a third, when the push is made
        // under a mask, is that mask, as a predicated store carries one.
        // Named: its value is the slot the entry took. It is one instruction
        // rather than the fetch-and-add and the store it lowers to
        // (lower_pushes, SSA/Defer.cpp) because how a gang does it is not
        // lane by lane: the lanes that push compact into consecutive slots --
        // the count advances once by their number, and each takes the slot
        // at its rank among them -- and the vectorizer can only say so of an
        // instruction that still says "push". Atomic on the count unless
        // shown not to need to be, which is `atomic` below.
        Push,
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
        // Lanes picked out of the operands' concatenation by the constant
        // indices in `shuffle`, one per lane of the result (ir::Shuffle).
        Shuffle,
        // The storage size of `queried_type`, left for the backend to answer
        // (see ir::SizeOf).
        SizeOf,
        Store, // side-effect-y
        Sub,
        // The decision a gang makes once on a bool its lanes may hold
        // differently: the majority of the lanes that are on, ties going to
        // false. Scalar code is a gang of one, and there this is the bool.
        //
        // Emitted by sort_recursion() for every compare-and-swap of a run's
        // sorting network. A run of recursive calls is made once by whoever
        // executes it, in one order, so when a gang executes it the order
        // cannot follow each lane's own key -- one child is descended into
        // first for the whole gang, the way a packet tracer descends
        // (Wald, Slusallek, Benthin & Wagner, "Interactive Rendering with
        // Coherent Ray Tracing", Eurographics 2001, section 3), and the vote
        // is how the gang picks it. When the run is instead put on a stack by
        // loopify() the visits are each lane's own again, and
        // queue_recursion() strips the vote (see SSA/QueueRecursion.h).
        Vote,
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

    // The constant indices of a Shuffle, one per lane of the result, into
    // the concatenation of its operands (see ir::Shuffle). Only meaningful
    // for Op::Shuffle.
    std::vector<int> shuffle;

    // Whether an accumulate is indivisible. Only meaningful for the Acc ops
    // and for Push, whose count it guards. Carried rather than acted on, the
    // way a ParFor's binding is: what it costs is decided when code is
    // generated, and whether it is needed at all is decided by whether the
    // schedule made the loop around it parallel.
    bool atomic = false;

    // Whether a Store under a mask compacts: the lanes the mask has on write
    // their values into consecutive slots from the one address the store is
    // given, in lane order, rather than each into its own lane's slot (LLVM's
    // masked.compressstore; `vpcompressd` to memory). Only meaningful for a
    // masked Store whose address is one pointer to the first slot; the
    // compacting push of a gang writes each field of its entries this way
    // (see lower_pushes in SSA/Defer.cpp).
    bool compact = false;

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
        // A spawned call (ir::Accumulate::spawned): its value goes into a
        // reducer by the accumulate that is the continuation's first
        // instruction, and nothing else of the continuation depends on it.
        // A deferral may push the call and go on without waiting; the drain
        // makes the call and the accumulate (SSA/Defer.cpp).
        bool spawned = false;
    };
    struct MultiCall {
        // A run of calls to one callee, differing only in some arguments, and
        // then one continuation after all of them. This is what a tree query's
        // `from (a, b)` becomes -- see ir::MultiRecurse, whose shape it keeps.
        //
        // It is a terminator rather than N chained Calls because the order of
        // the run is a scheduling decision: sort() rewrites it by permuting
        // `varying`, which it can only do while the run is still one thing.
        // Expanding it into a chain of Calls is what code generation does, and
        // is the point past which the order can no longer be changed.
        Jump call;  // callee and the arguments every call in the run shares
        Jump cont;  // continuation, reached after the last call returns
        std::vector<size_t> varying_at; // indices into call.args
        // One entry per call, each holding `varying_at.size()` values.
        std::vector<std::vector<std::shared_ptr<Value>>> varying;
        // What a sort() orders the calls by, one key per call, or empty when
        // no schedule asked for an order. Consumed by sort_recursion(), which
        // permutes `varying` to match and then clears this -- so a MultiCall
        // reaching code generation should not have any.
        std::vector<std::shared_ptr<Value>> keys;
        bool drop = true;

        // `call.args` with entry `c` substituted in at `varying_at`.
        std::vector<std::shared_ptr<Value>> call_args(size_t c) const;
    };

    std::variant<std::monostate, Jump, Dispatch, Return, ParFor, Yield, Call,
                 MultiCall>
        data;

    bool defined() const {
        return !std::holds_alternative<std::monostate>(data);
    }

    // The callee, for a terminator that calls -- Call or MultiCall -- and null
    // for one that does not. A MultiCall's calls all go to the same place, so
    // there is one answer either way.
    //
    // Worth preferring over `get_if<Call>` at any site that is asking "does
    // this go somewhere else?": those sites are looking for the edge, not for
    // how many times it is taken, and one that checks only for Call silently
    // stops seeing branching recursions.
    const Jump *callee() const;
    Jump *callee();

    // The continuation of a terminator that has one -- Call, MultiCall or
    // ParFor -- and null otherwise.
    const Jump *continuation() const;

    void dump(std::ostream &os) const;
};

struct Function;

struct Block : public std::enable_shared_from_this<Block> {
    std::string name;
    std::vector<Argument> args; // take the place of phis
    std::vector<std::shared_ptr<Instruction>> instrs;
    Terminator terminator;
    std::weak_ptr<Function> owner;
    // What this block was in the program as written, for a schedule to point
    // at (see ir::Provenance): the arm of a match, on the block the arm
    // begins with. Undefined for most blocks. Copied wherever the block is.
    ir::Provenance provenance;

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
                          std::vector<std::shared_ptr<Value>> vs,
                          bool atomic = false);

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

    // The function this one is a copy of, specialized by vectorize() for the
    // shape a gang calls it in (see specialize in SSA/Vectorize.cpp); empty
    // for a function the program wrote. A schedule directive names the
    // program's function, and one written after the vectorize has to find
    // its work in the copies as well -- `trace.loopify(64)` after
    // `render.vectorize(s)` puts the gang's traversal on a stack.
    std::string specialized_from;

    // The queues this function owns (defer(), SSA/Defer.h), by name: the
    // value each one's storage was sized by -- the capacity every round of
    // its drain is bounded by -- and the block the storage was made in. A
    // later deferral whose producer loop is one of these drains sizes its
    // own queue by the first (each entry of the drained queue pushes at
    // most one, so the drain's queue holds no more than the drained one
    // does, whatever a round's count) and makes its storage in the second
    // (before the round loop, rather than once per round inside it).
    struct OwnedQueue {
        std::shared_ptr<Value> size;
        std::string point;
        // The drain's first and last blocks, `<queue>!drain` and
        // `<queue>!exit`. A later deferral whose producer loop is the same
        // as this one's -- another kernel of the same round -- puts its
        // drain after this one rather than before it, so that the drains
        // that follow one producer loop run in the order the schedule
        // deferred their queues, as pbrt's Render loop lists its kernels
        // (SSA/Defer.cpp, "Where the drain sits").
        std::string entry, exit;
        // The functions whose calls are this queue's pushes: the deferred
        // function and the other pushers the schedule named. A later
        // deferral placed after this drain in the same round may not reach
        // one of them: an entry pushed after the pass is over waits for a
        // round that the last round does not have.
        std::set<std::string> pushers;
    };
    std::map<std::string, OwnedQueue> queue_sizes;
    // The addresses of the record slots deferrals of this function have made
    // (SSA/Defer.cpp): a per-iteration slot in place of a local of the
    // producer's iteration -- a reducer's, or a mutable local the deferred
    // call is handed -- and the slot for the deferred call's value. The
    // record is pbrt's pixelSampleState: it outlives every frame, so its
    // addresses may be stored in a queue entry as the addresses they are.
    std::set<const Instruction *> record_slots;

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

// The zero of `type`: a constant for a scalar, and for an aggregate -- a short
// vector, a struct, a struct of them -- a value built from the zeros of its
// parts by instructions appended to `into`, since this form has no constant
// aggregates. For a slot that has to hold something before anything has been
// put in it.
std::shared_ptr<Value> zero_value(const Type &type, Function &func,
                                  const std::shared_ptr<Block> &into);

// A value of `type` that nothing reads: what a join is handed along an edge
// whose lanes are all inactive, what a skipped call's continuation is handed
// in place of the result, what a loop tracker holds before its lane has left.
// A constant of any type, aggregates included, since it builds nothing; the
// backend lowers it to `undef`, which costs no instruction and lets the
// selects and stores that carry it fold (see ir::Undef).
std::shared_ptr<Value> undef_value(const Type &type);

// Are `a` and `b` one definition? This form threads a definition onwards
// through block arguments under its own name, so two references to one name
// -- spelled as the instruction that defined it or as the argument it arrived
// as, in whatever blocks -- are two references to one value. Two equal
// constants of one type are one value too.
bool same_value(const Value &a, const Value &b);

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
