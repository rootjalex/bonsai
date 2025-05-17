#include "Lower/Defers.h"

#include "Lower/TopologicalOrder.h"

#include "Opt/Simplify.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Mutator.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

// Get a map from func to any func that calls it.
CallGraph invert_graph(const CallGraph &call_graph) {
    CallGraph calls_graph;
    for (const auto &[producer, consumers] : call_graph) {
        for (const auto &consumer : consumers) {
            calls_graph[consumer].insert(producer);
        }
    }
    return calls_graph;
}

bool write_types_match(const std::vector<TypedVar> &write0,
                       const std::vector<TypedVar> &write1) {
    if (write0.size() != write1.size()) {
        return false;
    }
    for (size_t i = 0; i < write0.size(); i++) {
        if (!equals(write0[i].type, write1[i].type)) {
            return false;
        }
    }
    return true;
}

struct FindUses : public Visitor {
    std::vector<TypedVar> store_types;
    bool is_return = false;
    const std::string &producer;
    const std::string &consumer;

    FindUses(const std::string &producer, const std::string &consumer)
        : producer(producer), consumer(consumer) {}

    void visit(const Var *node) override {
        internal_assert(node->name != producer)
            << producer << " not directly a return or store in " << consumer
            << ", Bonsai cannot defer it.";
    }

    // TODO: handle Froms
    RESTRICT_VISITOR(YieldFrom);

    void visit(const Return *node) override {
        const Call *call = node->value.as<Call>();
        if (!call) {
            return Visitor::visit(node);
        }
        const Var *func = call->func.as<Var>();
        internal_assert(func) << call->func;
        if (func->name == producer) {
            internal_assert(store_types.empty())
                << producer << " is both a stored-value and returned-value in "
                << consumer;
            is_return = true;
        }
    }

    void visit(const Store *node) override {
        const Call *call = node->value.as<Call>();
        if (!call) {
            return Visitor::visit(node);
        }
        const Var *func = call->func.as<Var>();
        internal_assert(func) << call->func;
        if (func->name == producer) {
            internal_assert(!is_return)
                << producer << " is both a stored-value and returned-value in "
                << consumer;
            // TODO: support multiple stores.
            internal_assert(store_types.empty())
                << producer << " is written twice in " << consumer;
            store_types = gather_write_vars(node->loc);
        }
    }
};

std::vector<TypedVar> find_write_type(const std::string &consumer,
                                      const std::string &producer,
                                      const std::string &responsible,
                                      const Program &program,
                                      const CallGraph &producer_to_consumer,
                                      const std::set<std::string> &visited) {
    // Doesn't matter past the func responsible for handling the queue.
    if (producer == responsible) {
        return {};
    }

    const auto &fiter = program.funcs.find(consumer);
    internal_assert(fiter != program.funcs.cend())
        << consumer << " not found in program";

    FindUses analysis(producer, consumer);
    fiter->second->body.accept(&analysis);
    internal_assert(!analysis.store_types.empty() || analysis.is_return)
        << "Cannot find write_type of " << consumer << " in " << producer;
    if (!analysis.is_return) {
        return analysis.store_types;
    }
    // Need to recursively analyse funcs that call consumer.
    std::vector<TypedVar> current;
    std::string prev;
    for (const auto &func : producer_to_consumer.at(consumer)) {
        if (visited.contains(func)) {
            continue;
        }
        std::set<std::string> next_visited = visited;
        next_visited.insert(func);
        auto rec = find_write_type(func, consumer, responsible, program,
                                   producer_to_consumer, next_visited);
        if (rec.empty()) {
            continue;
        }
        if (current.empty()) {
            current = rec;
            prev = func;
            continue;
        }
        internal_assert(write_types_match(current, rec))
            << "Write type of " << prev << " does not match write type of "
            << func << " for producer: " << consumer;
    }
    return current;
}

// Is there a call path from f0 to f1?
// e.g. true if f0 calls f1, or f0 calls g which calls f1, and so forth.
bool reachable(const std::string &f0, const std::string &f1,
               const CallGraph &consumer_to_producer,
               std::set<std::string> visited = {}) {
    for (const auto &producer : consumer_to_producer.at(f0)) {
        if (producer == f1) {
            return true;
        } else if (visited.contains(producer)) {
            continue;
        }
        std::set<std::string> next_visited = visited;
        next_visited.insert(producer);
        if (reachable(producer, f1, consumer_to_producer, next_visited)) {
            return true;
        }
    }
    return false;
}

std::string queued_func_name(const std::string &name) {
    return "_queued_" + name;
}

std::ostream &operator<<(std::ostream &os, const std::vector<TypedVar> &vars) {
    os << "{";
    for (size_t i = 0; i < vars.size(); i++) {
        if (i > 0) {
            os << ", ";
        }
        os << vars[i].name << " : " << vars[i].type;
    }
    os << "}";
    return os;
}

struct ReplaceUses : public Mutator {
    const std::string &producer;
    const std::string &consumer;
    const std::string &queue_name;
    const Type &queue_type;
    const CallGraph &consumer_to_producer;
    FuncMap &funcs;
    Expr queue;

    // TO avoid infinite recursion.
    std::set<std::string> mutated;
    // These are stacks.
    std::vector<std::vector<TypedVar>> write_types;
    std::vector<std::string> called_funcs;

    ReplaceUses(const std::string &producer, const std::string &consumer,
                const std::string &queue_name, const Type &queue_type,
                const CallGraph &consumer_to_producer, FuncMap &funcs)
        : producer(producer), consumer(consumer), queue_name(queue_name),
          queue_type(queue_type), consumer_to_producer(consumer_to_producer),
          funcs(funcs) {
        queue = Var::make(queue_type, queue_name);
    }

    Expr visit(const Var *node) override {
        if (node->type.is<Function_t>()) {
            internal_assert(
                !reachable(node->name, producer, consumer_to_producer))
                << node->name
                << " cannot be augmented with write information in deferal of "
                << producer;
        }
        return node;
    }

    // TODO: handle Froms
    RESTRICT_MUTATOR(Stmt, YieldFrom);

    void handle_func_build(const std::string &old_name,
                           const std::vector<TypedVar> &old_write_types,
                           bool accept_queue) {
        auto old_iter = funcs.find(old_name);
        internal_assert(old_iter != funcs.end()) << old_name;
        // Build a new function with the same initial args + write args
        // + the queue. The new function is a void return type.
        std::vector<Function::Argument> args = old_iter->second->args;
        std::vector<TypedVar> new_write_types;
        bool first = true;
        for (const auto &[name, type] : old_write_types) {
            std::string arg_name =
                name.starts_with("_write_") ? name : "_write_" + name;
            args.emplace_back(arg_name, type, /*default_value=*/Expr(),
                              first); // only array arg is mutating
            first = false;
            new_write_types.push_back({arg_name, type});
        }
        if (accept_queue) {
            args.emplace_back(queue_name, queue_type, /*default_value=*/Expr(),
                              true);
        }

        static const Type ret_type = Void_t::make();

        std::shared_ptr<Function> funcptr = nullptr;
        if (old_name == consumer) {
            // Directly modify this function, all calls to it must be modified.
            funcptr = old_iter->second;
            funcptr->args = std::move(args);
            funcptr->ret_type = ret_type;
        } else {
            std::string new_name = queued_func_name(old_name);
            funcptr = std::make_shared<Function>(
                new_name, std::move(args), ret_type, /*body=*/Stmt(),
                old_iter->second->interfaces, old_iter->second->attributes);
            // Insert without a body into the funcs list. this is to properly
            // handle recursion.
            auto [_, inserted] = funcs.try_emplace(new_name, funcptr);
            internal_assert(inserted) << new_name << " already in funcs map...";
        }

        // Now build the body.
        mutated.insert(old_name);
        write_types.push_back(std::move(new_write_types));
        called_funcs.push_back(old_name);
        Stmt body = mutate(old_iter->second->body);
        write_types.pop_back();
        called_funcs.pop_back();
        funcptr->body = std::move(body);
    }

    Stmt handle(const std::string &func,
                const std::vector<TypedVar> &curr_write_types,
                const std::vector<Expr> call_args) {
        internal_assert(!called_funcs.empty()) << func;

        const bool accept_queue =
            reachable(func, producer, consumer_to_producer);

        if (accept_queue || func == producer) {
            internal_assert(!called_funcs.empty())
                << "Function responsible for queueing: " << producer
                << " cannot directly return it -- TODO: implement heap "
                   "allocate lowering.";
            internal_assert(!curr_write_types.empty())
                << "In deferral of: " << producer << ", "
                << called_funcs.back();

            const bool do_enqueue =
                func == producer && called_funcs.back() == consumer;

            // Grab regular args.
            std::vector<Expr> args = call_args;
            // Insert write locations
            // TODO: THE FIRST OF THESE MUST BE MUTABLE!!
            {
                const auto &[name, type] = curr_write_types[0];
                // args.push_back(Var::make(Ptr_t::make(type), name));
                if (do_enqueue) {
                    args.push_back(PtrTo::make(Var::make(type, name)));
                } else {
                    args.push_back(Var::make(type, name));
                }
            }

            for (size_t i = 1; i < curr_write_types.size(); i++) {
                const auto &[name, type] = curr_write_types[i];
                args.push_back(Var::make(type, name));
            }
            if (accept_queue) {
                args.push_back(queue);
            }

            const std::string new_func_name =
                (func == consumer) ? func : queued_func_name(func);

            if (!mutated.contains(func)) {
                // Haven't mutated func, need to build new signature, insert
                // into funcs/visited, and then mutate body (handles recursion).
                handle_func_build(func, curr_write_types, accept_queue);
            }

            if (do_enqueue) {
                if (accept_queue) {
                    args.pop_back(); // don't store the queue.
                }
                Expr size = PtrTo::make(Extract::make(queue, 0));
                WriteLoc loc(queue_name, queue_type);
                loc.add_index_access(1); // get second element of tuple
                Expr one = make_one(size.type().element_of());
                Expr count = AtomicAdd::make(size, one);
                loc.add_index_access(count);
                Expr store = make_tuple(std::move(args));
                return Store::make(loc, store);
            } else {
                // This is a call.
                const auto &fiter = funcs.find(new_func_name);
                internal_assert(fiter != funcs.cend())
                    << func << " visited but " << new_func_name
                    << " not found in funcs.";

                // internal_assert(accept_queue) << func;

                Expr new_func =
                    Var::make(fiter->second->call_type(), new_func_name);

                return CallStmt::make(std::move(new_func), std::move(args));
            }
        }
        return Stmt();
    }

    Stmt build_write(Expr value) const {
        internal_assert(!write_types.empty())
            << "Cannot build write of " << value
            << " without location to write to.";
        auto curr_writes = write_types.back();
        internal_assert(!curr_writes.empty())
            << "Cannot build write of " << value
            << " without location to write to.";
        // TODO(ajr): this somewhat assumes a specific gather_write_vars ->
        // WriteLoc ordering, which may not be perfect...
        WriteLoc loc(curr_writes.front().name, curr_writes.front().type);
        for (size_t i = 1; i < curr_writes.size(); i++) {
            loc.add_index_access(
                Var::make(curr_writes[i].type, curr_writes[i].name));
        }
        return Sequence::make(
            {Store::make(std::move(loc), std::move(value)), Return::make()});
    }

    Stmt visit(const Return *node) override {
        // std::cout << "Visiting Return: " << Stmt(node);
        // std::cout << "Called func: " << called_funcs.back() << std::endl;
        // std::cout << "Producer: " << producer << std::endl;
        // std::cout << "Consumer: " << consumer << std::endl;
        if (called_funcs.back() == producer && (producer != consumer)) {
            // This is now a write!
            // std::cout << "now a write (1)\n\n";
            return build_write(node->value);
        }

        const Call *call = node->value.as<Call>();
        if (!call) {
            if (called_funcs.back() == producer ||
                called_funcs.back() == consumer) {
                // This is now a write!
                // std::cout << "now a write (2)\n\n";
                return build_write(node->value);
            }
            // std::cout << "NOT a write (1)\n\n";
            return Mutator::visit(node);
        }
        const Var *func = call->func.as<Var>();
        internal_assert(func) << call->func;

        internal_assert(!write_types.empty()) << Stmt(node);
        Stmt try_mutate = handle(func->name, write_types.back(), call->args);
        if (try_mutate.defined()) {
            // std::cout << "mutated\n\n";
            return Sequence::make({std::move(try_mutate), Return::make()});
        }

        if (called_funcs.back() == producer ||
            called_funcs.back() == consumer) {
            // This is now a write!
            // std::cout << "now a write (3)\n\n";
            return build_write(node->value);
        }
        // std::cout << "NOT a write (2)\n\n";
        return node;
    }

    Stmt visit(const Store *node) override {
        const Call *call = node->value.as<Call>();
        if (!call) {
            return Mutator::visit(node);
        }
        const Var *func = call->func.as<Var>();
        internal_assert(func) << call->func;

        auto store_types = gather_write_vars(node->loc);

        Stmt try_mutate = handle(func->name, store_types, call->args);
        if (try_mutate.defined()) {
            return try_mutate;
        }
        return node;
    }
};

std::string second_buffer(const std::string &queue_name) {
    return queue_name + "_dbl_buffer";
}

// TODO(ajr): make this findable via scheduling.
std::string idx_name(const std::string &queue_name) {
    return "_" + queue_name + "_task";
}

Stmt apply_queueing(const std::string &responsible, Stmt stmt,
                    const std::string &idx, const std::string &producer,
                    const std::string &consumer, const std::string &queue_name,
                    const Type &queue_type,
                    const CallGraph &consumer_to_producer, FuncMap &funcs) {

    // Mutate the body of responsible, below loop_idx.
    // responsible should become:
    // forall i in . . .
    //   foo()
    // =>
    // forall i in  . . .
    //   alloc queue
    //   foo()
    //   if work is recursive, double buffer and iterate
    //   else:
    //      forall task in queue:
    //        do task

    ReplaceUses mutator(producer, consumer, queue_name, queue_type,
                        consumer_to_producer, funcs);

    Expr queue = Var::make(queue_type, queue_name);
    Expr queue_size = Extract::make(queue, 0);
    Expr queue_data = Extract::make(queue, 1);

    // Expr initial = make_tuple(
    //     {make_zero(queue_size.type()), Build::make(queue_data.type())});

    auto process_producer_loop = [&](bool include_queue) {
        const std::string &name =
            (producer == consumer) ? producer : queued_func_name(producer);
        const auto &piter = funcs.find(name);
        internal_assert(piter != funcs.cend()) << name;
        const Expr func = Var::make(piter->second->call_type(), name);

        const size_t n_args = piter->second->args.size() - include_queue;

        std::string idx = idx_name(queue_name);
        Expr size = Extract::make(queue, 0);
        Expr data = Extract::make(queue, 1);

        Expr idx_var = Var::make(size.type(), idx);
        Expr datum = Extract::make(data, idx_var);
        std::vector<Expr> args(n_args + include_queue);

        for (size_t i = 0; i < n_args; i++) {
            args[i] = Extract::make(datum, i);
        }
        if (include_queue) {
            const std::string dbl_buffer = second_buffer(queue_name);
            Expr queue1 = Var::make(queue_type, dbl_buffer);
            args[n_args] = queue1;
        }

        Stmt call = CallStmt::make(std::move(func), std::move(args));

        ForAll::Slice slice{make_zero(size.type()), size,
                            make_one(size.type())};

        return ForAll::make(std::move(idx), std::move(slice), std::move(call));
    };

    auto make_buffer =
        [&](const std::string &buffer_name) -> std::pair<Stmt, Stmt> {
        const std::string data_name = buffer_name + "_data";
        Expr data = Var::make(queue_data.type(), data_name);
        std::vector<Expr> args = {make_zero(queue_size.type()), data};
        // TODO(ajr): make memory scheduable
        return {// First allocate the array
                Allocate::make(WriteLoc(data_name, queue_data.type()),
                               Allocate::Stack),
                // Then place the array and a count of zero into local memory
                Allocate::make(WriteLoc(buffer_name, queue_type),
                               Build::make(queue_type, args), Allocate::Stack)};
    };

    auto store_zero_size = [&](const std::string &buffer_name) -> Stmt {
        WriteLoc loc(buffer_name, queue_type);
        loc.add_index_access(0);
        return Store::make(loc, make_zero(queue_size.type()));
    };

    auto rewrite_loop_body = [&](const Stmt &body) {
        mutator.called_funcs = {responsible};
        mutator.write_types = {{}};
        // Remove return
        Stmt repl = mutator.mutate(body);
        Stmt ret;
        if (contains<Return>(repl)) {
            internal_assert(repl.as<Sequence>() &&
                            !repl.as<Sequence>()->stmts.empty())
                << repl;
            std::vector<Stmt> stmts = repl.as<Sequence>()->stmts;
            ret = stmts.back();
            internal_assert(ret.is<Return>()) << repl;
            stmts.pop_back();
            repl = Sequence::make(std::move(stmts));
            internal_assert(!contains<Return>(repl)) << repl;
        }

        Stmt handle_queue;
        WriteLoc queue_loc(queue_name, queue_type);

        if (reachable(producer, producer, consumer_to_producer)) {
            // Recursive func, must iteratively evaluate in a double-buffer
            // loop.
            // allocate second buffer
            // do {
            //   processs queue
            //   swap queue with second buffer
            // } while (!queue.empty())
            const std::string dbl_buffer = second_buffer(queue_name);
            Expr queue1 = Var::make(queue_type, dbl_buffer);
            // TODO: make sure this lowers correctly.
            Expr not_empty = queue_size > 0;
            WriteLoc dbl_buffer_loc(dbl_buffer, queue_type);
            const std::string tmp = "_tmp_" + dbl_buffer;
            WriteLoc tmp_loc(tmp, queue_type);
            Stmt process_queue = process_producer_loop(true);

            auto [make_data, make_queue] = make_buffer(dbl_buffer);

            handle_queue = Sequence::make(
                {make_data, make_queue,
                 DoWhile::make(
                     Sequence::make(
                         {process_queue,
                          // Do queue swap
                          LetStmt::make(tmp_loc, queue1),
                          Store::make(dbl_buffer_loc, queue),
                          store_zero_size(dbl_buffer),
                          Store::make(queue_loc, Var::make(queue_type, tmp))}),
                     not_empty)});
        } else {
            handle_queue = process_producer_loop(false);
        }

        Expr initial = make_tuple(
            {make_zero(queue_size.type()), Build::make(queue_data.type())});

        auto [make_data, make_queue] = make_buffer(queue_name);

        std::vector<Stmt> stmts = {make_data, make_queue,
                                   // perform original code
                                   repl, handle_queue};
        if (ret.defined()) {
            stmts.push_back(std::move(ret));
        }

        return Sequence::make(stmts);
    };

    // if idx == root, then wrap the whole body.
    if (idx == "root") {
        // Handle annoying case that should be inlined...
        if (const Return *ret = stmt.as<Return>()) {
            internal_assert(ret->value.defined()) << stmt;
            const Call *call = ret->value.as<Call>();
            internal_assert(call) << stmt;
            const Var *func = call->func.as<Var>();
            internal_assert(func && func->name.starts_with("_traverse_array"))
                << stmt;
            internal_assert(funcs.contains(func->name))
                << func->name << " " << stmt;
            Stmt body = std::move(funcs.at(func->name)->body);
            funcs.erase(func->name);
            return rewrite_loop_body(body);
        }
        return rewrite_loop_body(stmt);
    }

    // Otherwise, find the Loop body.
    // TODO(ajr): this could be useful for more succinct LoopTransforms as well.
    struct FindLoop : Mutator {
        const std::string &loop_idx;
        std::function<Stmt(const Stmt &)> mutator;
        FuncMap &funcs;

        FindLoop(const std::string &loop_idx,
                 std::function<Stmt(const Stmt &)> mutator, FuncMap &funcs)
            : loop_idx(loop_idx), mutator(std::move(mutator)), funcs(funcs) {}

        Stmt visit(const ForAll *node) override {
            if (node->index != loop_idx) {
                return Mutator::visit(node);
            }
            Stmt body = mutator(node->body);
            return ForAll::make(node->index, node->slice, std::move(body));
        }

        // TODO: this is hacky, need a better way.
        Expr visit(const Call *node) override {
            if (const Var *var = node->func.as<Var>()) {
                // TODO(ajr): hope to God it's impossible to have self-recursion
                // in these.
                if (var->name.starts_with("_traverse_array")) {
                    funcs[var->name]->body =
                        mutate(std::move(funcs[var->name]->body));
                    return node;
                }
            }
            return Mutator::visit(node);
        }
    };

    return FindLoop(idx, rewrite_loop_body, funcs).mutate(stmt);
}

void defer_call(const std::string &consumer, const std::string &producer,
                const std::string &responsible, const std::string &loop,
                const std::string &queue, Program &program,
                const std::map<std::string, Expr> &queue_sizes) {
    // Find map from consumers to producers.
    CallGraph consumer_to_producer = build_call_graph(program.funcs);
    internal_assert(consumer_to_producer[consumer].contains(producer))
        << producer << " is not called by " << consumer
        << "; it cannot be defered";
    CallGraph producer_to_consumer = invert_graph(consumer_to_producer);
    // Find the "write type" of producer. If consumer directly returns it,
    // then the "write type" is the "write type" of consumer. If consumer
    // stores it, then it is the storage type.
    // Note: to do this correctly, the storage must last. This should be
    // asserted to be on the Heap, I think.
    // At the very least, it must be allocated by `responsible`, I think.
    // TODO(ajr): also support accumulate operations.
    std::set<std::string> visited = {consumer};
    auto write_type = find_write_type(consumer, producer, responsible, program,
                                      producer_to_consumer, visited);
    internal_assert(!write_type.empty())
        << "Failed to find write type for " << consumer << ".defer(" << producer
        << ")";
    // Now, replace all calls to producer in consumer with a store
    // TODO(ajr): or accumulate.
    // All paths from responsible to consumer need to be augmented with
    // write parameters, including passing the queue through!

    Type queue_type;
    {
        std::vector<Type> etypes;
        const auto &piter = program.funcs.find(producer);
        internal_assert(piter != program.funcs.cend())
            << "Cannot find producer: " << producer << " in program functions.";
        etypes.reserve(piter->second->args.size() + write_type.size());
        for (const auto &arg : piter->second->args) {
            etypes.push_back(arg.type);
        }
        bool first = true;
        for (const auto &arg : write_type) {
            if (first) {
                // Make sure array arg is mutable
                etypes.push_back(Ptr_t::make(arg.type));
                first = false;
            } else {
                etypes.push_back(arg.type);
            }
        }

        const std::string location = responsible + "." + queue;

        const auto &siter = queue_sizes.find(location);
        internal_assert(siter != queue_sizes.cend())
            << queue << " at " << responsible << " was not given a size.";

        Type tuple_t = Tuple_t::make(std::move(etypes));
        Type array_t = Array_t::make(std::move(tuple_t), siter->second);
        static const Type count_t = UInt_t::make(64);
        queue_type = Tuple_t::make({count_t, array_t});
    }

    const auto &riter = program.funcs.find(responsible);
    internal_assert(riter != program.funcs.cend())
        << "Cannot find responsible: " << responsible
        << " in program functions.";

    riter->second->body = apply_queueing(
        responsible, std::move(riter->second->body), loop, producer, consumer,
        queue, queue_type, consumer_to_producer, program.funcs);
}

} // namespace

Program LowerDefers::run(Program program,
                         const CompilerOptions &options) const {
    if (program.schedules.empty()) {
        return program;
    }

    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    TransformMap &transforms = program.schedules[Target::Host].func_transforms;

    if (transforms.empty()) {
        return program;
    }

    // Need simplification to run to avoid unnecessary saved variables
    // TODO(ajr): might also want LICM/CSE here...
    program.funcs = opt::Simplify().run(std::move(program.funcs), options);

    std::map<std::string, Expr> queue_sizes;

    // TODO(ajr): should defers happen in any particular order...?
    for (const auto &[consumer, ts] : transforms) {
        for (const auto &t : ts) {
            if (std::holds_alternative<Defer>(t)) {
                // Steps:
                // (1) in `func` (== name), search for `producer`.
                //     assert all calls to producer are directly
                //     returns or stores. turn those returns and
                //     stores to queue writes, and add a queue
                //     parameter to the argument list.
                // (2) if `func` is a non-void return type, and
                //     the deferred call is in returns, then add
                //     a mutable write location to `func`'s arg
                //     list, based on the calls to `func`. rewrite
                //     calls to func to provide the mutable write
                //     location parameters
                // (3) assert that defer.loop has a queue available?
                const Defer &def = std::get<Defer>(t);
                internal_assert(def.producer.names.size() == 1)
                    << "TODO: Multi-location in defer() producer: "
                    << def.producer;
                const std::string &producer = def.producer.names.front();
                internal_assert(def.loop.names.size() == 1 ||
                                def.loop.names.size() == 2)
                    << "Failed to parse valid loop location in defer(): "
                    << def.loop;
                const std::string &responsible = (def.loop.names.size() == 1)
                                                     ? consumer
                                                     : def.loop.names.front();
                const std::string &loop_idx = (def.loop.names.size() == 1)
                                                  ? def.loop.names.front()
                                                  : def.loop.names.back();
                internal_assert(def.queue.names.size() == 1)
                    << "TODO: Multi-location in defer() queue: " << def.queue;
                const std::string &queue = def.queue.names.front();

                defer_call(consumer, producer, responsible, loop_idx, queue,
                           program, queue_sizes);
            } else if (std::holds_alternative<MakeQueue>(t)) {
                const MakeQueue &makeq = std::get<MakeQueue>(t);
                internal_assert(makeq.queue.names.size() == 1)
                    << "Multi-location in make_queue queue name: "
                    << makeq.queue;
                internal_assert(makeq.loop.names.size() == 1)
                    << "Multi-location in make_queue loop name: " << makeq.loop;
                internal_assert(makeq.queue_size.has_value())
                    << "TODO: dynamic queue sizes for: " << makeq.queue
                    << " at " << makeq.loop << " of " << consumer;
                const std::string location =
                    consumer + "." + makeq.queue.names.front();
                auto [_, inserted] =
                    queue_sizes.try_emplace(location, *makeq.queue_size);
                internal_assert(inserted)
                    << consumer << " already has a queue named " << makeq.queue
                    << ", can't build one at loop " << makeq.loop;
            }
        }
    }

    return program;
}

} // namespace lower
} // namespace bonsai
