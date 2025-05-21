#include "Lower/Defers2.h"

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
//           rec(t1, t2, Q2, out);
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
    auto fit = program.funcs.find(location);
    internal_assert(fit != program.funcs.end()) << location;
    auto &function = fit->second;

    struct DeferImpl : ir::Mutator {
        DeferImpl(const std::string &queue_name, const Expr &queue_size)
            : queue_name(queue_name), queue_size(queue_size) {}

        Stmt visit(const YieldFrom *node) override {
            // 2. Update `from` to append to queues (and increment index).
            return node;
        }

        Stmt visit(const RecLoop *node) override {
            // 3. Update `rec` loop to respective do while.
            return node;
        }

        Stmt mutate(const Stmt &stmt) override {
            if (!entry) {
                return ir::Mutator::mutate(stmt);
            }
            entry = false;
            std::vector<Stmt> statements;
            // 1. Add queues to initial function body with counters.
            return stmt;
        }

      private:
        bool entry = true;
        const std::string &queue_name;
        const Expr &queue_size;
    };

    DeferImpl defer(queue, /*queue_size=*/qit->second);
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
