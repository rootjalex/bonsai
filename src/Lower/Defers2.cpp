#include "Lower/Defers2.h"

#include "Lower/TopologicalOrder.h"

#include "Opt/Inline.h"
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
using namespace ir;

namespace {

// Defers the simple case where `location` = consumer = producer = responsible,
// and the loop_index = "root".
//
// func collisions() -> set<(Triangle, Triangle)> {
//     Q1: Queue<(Triangle, Triangle)> = (triangles1, triangles2);
//     Q2: Queue<(Triangle, Triangle)> = {};
//     out: dyn_array<(Triangle, Triangle)>;
//     do {
//         for all (t1, t2) in Q1 {
//           rec(t1, t2);
//         }
//         swap(Q1, Q2);
//         Q2.clear();
//     } while (!Q1.empty());
//     return out;
// }
void defer_simple(const std::string &location, const std::string &queue,
                  const std::map<std::string, Expr> &queue_sizes,
                  Program &program) {
    // Find the queue size.
    auto qit = queue_sizes.find(location + "." + queue);
    internal_assert(qit != queue_sizes.end()) << queue;
    // Find the function.
    lower::CallGraph call_graph = build_call_graph(program.funcs);
    std::string target;
    for (const std::string &call : call_graph[location]) {
        if (!call.starts_with("_traverse_tree")) {
            continue;
        }
        internal_assert(target.empty())
            << "two or more calls of `_traverse_tree` found!";
        target = call;
    }
    auto fit = program.funcs.find(target);
    internal_assert(fit != program.funcs.end()) << target;
    auto &function = fit->second;

    struct DeferImpl : ir::Mutator {
        DeferImpl(const std::string &queue_name, const Expr &queue_size,
                  const Type &queue_type,
                  const std::vector<Function::Argument> &arguments,
                  Program &program)
            : queue_name(queue_name), queue_size(queue_size),
              queue_type(queue_type), buffer_name(queue_name + "_buffer"),
              arguments(arguments), program(program) {}

        Stmt visit(const YieldFrom *node) override {
            std::vector<Stmt> stmts;
            // Update `from` to append to queues (and increment index).
            WriteLoc bq(buffer_name, queue_type);
            WriteLoc bq_stack = bq;
            bq_stack.add_struct_access("stack");

            WriteLoc bq_count = bq;
            bq_count.add_struct_access("count");
            bq_stack.add_index_access(bq_count.to_expr());
            Expr value = node->value;

            std::vector<Expr> values = {value};
            if (value.type().is<Tuple_t>()) {
                values = break_tuple(value);
            }

            for (int i = 0, e = values.size(); i < e; ++i) {
                stmts.push_back(Store::make(bq_stack, values[i]));
                stmts.push_back(Accumulate::make(bq_count,
                                                 Accumulate::OpType::Add,
                                                 make_one(bq_count.type)));
            }
            return Sequence::make(std::move(stmts));
        }

        Stmt visit(const RecLoop *node) override {
            // Update `rec` loop to respective do while.
            std::vector<Stmt> stmts;
            std::string idx = "q";
            Expr q = Var::make(queue_type, queue_name);
            Expr count_q = Access::make("count", q);
            Expr stack_q = Access::make("stack", q);
            Expr bq = Var::make(queue_type, buffer_name);
            Expr count_bq = Access::make("count", bq);
            Expr stack_bq = Access::make("stack", bq);
            Type i_type = count_q.type();
            Type q_type = stack_q.type();

            ForAll::Slice slice{
                .begin = make_zero(i_type),
                .end = count_q,
                .stride = make_one(i_type),
            };
            Stmt loop = ForAll::make(idx, slice, ir::Mutator::visit(node));
            Stmt swap = queue_swap(q, bq);
            Stmt body = Sequence::make({loop, swap});
            Expr cond = count_q != make_zero(i_type);
            return DoWhile::make(std::move(body), std::move(cond));
        }

        Stmt mutate(const Stmt &stmt) override {
            if (!entry) {
                return ir::Mutator::mutate(stmt);
            }
            entry = false;
            std::vector<Stmt> stmts;
            // Add queues to initial function body with counters.
            static const Type count_t = UInt_t::make(64);
            Type array_t = Array_t::make(queue_type, queue_size);
            queue_type = Struct_t::make("queue",
                                        {
                                            {"count", count_t},
                                            {"stack", array_t},
                                        },
                                        {{"count", Expr(0)}});
            const auto [_, inserted] =
                program.types.emplace("queue", queue_type);
            internal_assert(inserted);
            WriteLoc loc(queue_name, queue_type);
            stmts.push_back(Allocate::make(loc));
            stmts.push_back(Allocate::make(WriteLoc(buffer_name, queue_type)));

            Expr q = Var::make(queue_type, queue_name);
            Expr count_q = Access::make("count", q);
            Expr stack_q = Access::make("stack", q);
            std::vector<Expr> args;
            std::vector<Type> types;
            for (int i = 0, e = arguments.size(); i < e; ++i) {
                types.push_back(arguments[i].type);
                args.push_back(Var::make(arguments[i].type, arguments[i].name));
            }

            // Push the top of the tree.
            WriteLoc head = loc;
            head.add_struct_access("stack");
            head.add_index_access(make_zero(count_q.type()));
            stmts.push_back(
                Store::make(head, Build::make(Tuple_t::make(types), args)));
            // Increment the count.
            WriteLoc count = loc;
            count.add_struct_access("count");
            stmts.push_back(Accumulate::make(count, Accumulate::OpType::Add,
                                             make_one(count_t)));
            stmts.push_back(ir::Mutator::mutate(stmt));
            return Sequence::make(std::move(stmts));
        }

      private:
        bool entry = true;
        const std::string &queue_name;
        const Expr &queue_size;
        Type queue_type;
        std::string buffer_name;
        const std::vector<Function::Argument> &arguments;
        Program &program;

        // Analagous to:
        // swap(q1, q2); clear(q2);
        Stmt queue_swap(Expr q1, Expr q2) {
            std::vector<Stmt> stmts;
            internal_assert(ir::equals(q1.type(), q2.type()));
            WriteLoc q1w(queue_name, q1.type());
            WriteLoc q1w_count = q1w;
            q1w_count.add_struct_access("count");
            WriteLoc q1w_stack = q1w;
            q1w_stack.add_struct_access("stack");

            WriteLoc q2w(buffer_name, q2.type());
            WriteLoc q2w_count = q2w;
            q2w_count.add_struct_access("count");
            WriteLoc q2w_stack = q2w;
            q2w_stack.add_struct_access("stack");

            // q1 = q2;
            stmts.push_back(Store::make(q1w_count, q2w_count.to_expr()));
            stmts.push_back(Store::make(q1w_stack, q2w_stack.to_expr()));
            // clear q2
            stmts.push_back(Store::make(q2w_count, make_zero(q2w_count.type)));
            // For now we just assume the array will be written over.
            return Sequence::make(std::move(stmts));
        }
    };

    DeferImpl defer(/*queue_name=*/queue,
                    /*queue_size=*/qit->second,
                    /*queue_type=*/function->ret_type.element_of(),
                    function->args, program);
    function->body = defer.mutate(function->body);
}

} // namespace

Program LowerDefers2::run(Program program,
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

    // A map from queue name to queue size.
    std::map<std::string, Expr> queue_sizes;
    for (const auto &[consumer, ts] : transforms) {
        for (const auto &t : ts) {
            if (std::holds_alternative<MakeQueue>(t)) {
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
                continue;
            }
            if (std::holds_alternative<Defer>(t)) {
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
                const std::string &loop_index = (def.loop.names.size() == 1)
                                                    ? def.loop.names.front()
                                                    : def.loop.names.back();
                internal_assert(def.queue.names.size() == 1)
                    << "TODO: Multi-location in defer() queue: " << def.queue;
                const std::string &queue = def.queue.names.front();
                if (consumer == producer && consumer == responsible &&
                    loop_index == "root") {
                    defer_simple(consumer, queue, queue_sizes, program);
                    continue;
                }
                internal_error << "unsupported defer scheduling!";
            }
        }
    }

    return program;
}

} // namespace lower
} // namespace bonsai
