#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <map>
#include <set>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

bool may_nest(Resource outer, Resource inner) {
    switch (outer) {
    case Resource::GPUBlock:
        return inner == Resource::GPUThread;
    case Resource::GPUThread:
        // Nothing goes inside a thread but a gang of lanes, which vectorize()
        // makes and which is not a bind.
        return false;
    case Resource::CPUThread:
        return inner == Resource::CPUThread;
    case Resource::RTCore:
    case Resource::OptixThread:
    case Resource::TextureUnit:
        return false;
    }
    return false;
}

namespace {

// The blocks a parfor's body can reach, which is where anything nested inside
// it lives. A parfor body ends at a yield, so this stops at the loop.
set<string> body_of(const Function &func, const Terminator::ParFor &loop) {
    set<string> body;
    const Cfg region(func, loop.body.name);
    for (const auto &block : region.blocks()) {
        body.insert(block->name);
    }
    return body;
}

} // namespace

void bind(FuncMap &funcs, string func, string index, Resource resource) {
    internal_assert(funcs.contains(func))
        << "bind applied to unknown func: " << func;
    auto f = funcs[func];

    Terminator::ParFor *bound = nullptr;
    for (const auto &block : f->blocks) {
        auto *parfor = std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->index == index) {
            internal_assert(bound == nullptr)
                << "Two loops named " << index << " in " << func;
            bound = parfor;
        }
    }
    internal_assert(bound != nullptr)
        << "bind(" << index << ", " << to_string(resource) << ") on " << func
        << ": no parfor named " << index;

    internal_assert(!bound->binding.has_value())
        << "bind(" << index << ", " << to_string(resource) << ") on " << func
        << ": that loop is already bound to " << to_string(*bound->binding)
        << ", and a loop runs on one hardware resource";

    bound->binding = resource;

    // A loop over a queue put on the GPU is launched over the queue's
    // capacity, not its count: pbrt's `ForAllQueued`, a launch over
    // maxQueueSize whose kernel begins `if (index >= queue->Size()) return;`.
    // The count is then read on the device, by the guard, and the host
    // neither reads it back before the launch nor waits for the launch to
    // read the next -- the launches of a round go into the stream one after
    // another, which is where pbrt's render spends no time waiting and ours
    // did (apps/pbrt/PLAN.md, the launch shape). The guard is the split's
    // when the loop was split -- its body is then the inner loop, and the
    // split guarded the step by the count read through the loop's address
    // (Rewrite.cpp) -- and otherwise one made here, a block ahead of the
    // body that reads the count and lets an index short of it through.
    const bool on_gpu = resource == Resource::GPUThread ||
                        resource == Resource::GPUBlock ||
                        resource == Resource::OptixThread;
    if (on_gpu && bound->capacity != nullptr) {
        // The loop's body: whether this is a plain drain or a split one.
        shared_ptr<Block> body;
        for (const auto &block : f->blocks) {
            if (block->name == bound->body.name) {
                body = block;
            }
        }
        internal_assert(body != nullptr)
            << "bind(" << index << "): the loop's body " << bound->body.name
            << " is no block of " << func;
        // A split drain -- its body is the inner loop -- keeps the count
        // launch for now: the launch-over-capacity is built here only for a
        // drain bound whole (the raygens, `bind(rays, OptixThread)`), which
        // is where the launch gaps were largest. The split material kernels
        // (SSA/Rewrite.cpp's inner and outer loops) launch over the count as
        // before; extending the capacity launch to them is a later step
        // (apps/pbrt/PLAN.md). The carried values are dropped so nothing
        // downstream reads them.
        const bool split_drain =
            std::holds_alternative<Terminator::ParFor>(body->terminator.data);
        if (split_drain) {
            bound->capacity = nullptr;
            bound->queue_base = nullptr;
            bound->queue_slot = nullptr;
        }
        // The block the loop is the terminator of: where the extent and the
        // address of the count go.
        shared_ptr<Block> header;
        for (const auto &block : f->blocks) {
            if (std::get_if<Terminator::ParFor>(&block->terminator.data) == bound) {
                header = block;
            }
        }
        internal_assert(header != nullptr)
            << "bind(" << index << "): the loop is no terminator of " << func;
        const Type index_type = std::get<Terminator::ParFor>(header->terminator.data)
                                    .start->get_type();
        if (!split_drain) {
            // The extent: the queue's size as the loop's index type (its
            // count field is u32 and its index is u32, so this is a cast
            // only when the producer's push count came in another type).
            shared_ptr<Value> extent = bound->capacity;
            if (!extent->get_type().same_as(index_type)) {
                extent = header->make_instruction(index_type, Instruction::Op::Cast,
                                                  {extent});
            }
            // The count's address: the count field of the queue drained --
            // the header, or the slot of it for a double-buffered queue.
            shared_ptr<Value> queue = bound->queue_base;
            if (bound->queue_slot != nullptr) {
                queue = header->make_instruction(bound->queue_base->get_type(),
                                                 Instruction::Op::GEP,
                                                 {bound->queue_base, bound->queue_slot});
            }
            auto zero = std::make_shared<Value>(
                Constant{UInt_t::make(32), uint64_t(0)});
            shared_ptr<Value> count_address = header->make_instruction(
                Ptr_t::make(UInt_t::make(32)), Instruction::Op::FieldPtr,
                {queue, zero});
            bound->end = extent;
            bound->count_address = count_address; // read by the residency analysis
            // The guard, ahead of the body: it reads the count on the device
            // and lets an index short of it through (pbrt's `if (index >=
            // queue->Size()) return;`).
            internal_assert(!body->args.empty())
                << "bind(" << index << "): the body " << body->name
                << " takes no index";
            auto guard = std::make_shared<Block>();
            guard->name = body->name + "!inqueue";
            guard->owner = f;
            std::vector<shared_ptr<Value>> onward;
            for (const Argument &arg : body->args) {
                onward.push_back(guard->add_argument(Argument{arg.type, arg.name}));
            }
            auto address = guard->add_argument(
                Argument{count_address->get_type(), f->get_unique_name()});
            shared_ptr<Value> count = guard->make_instruction(
                UInt_t::make(32), Instruction::Op::Load, {address});
            if (!count->get_type().same_as(index_type)) {
                count = guard->make_instruction(index_type, Instruction::Op::Cast,
                                                {count});
            }
            auto in_queue = guard->make_instruction(
                Bool_t::make(), Instruction::Op::Lt, {onward[0], count});
            auto past = std::make_shared<Block>();
            past->name = body->name + "!past";
            past->owner = f;
            past->terminator.data = Terminator::Yield{};
            guard->terminator.data = Terminator::Dispatch{
                in_queue,
                {Terminator::Jump{past->name},
                 Terminator::Jump{body->name, std::move(onward)}}};
            std::vector<shared_ptr<Value>> to_guard = bound->body.args;
            to_guard.push_back(count_address);
            bound->body = Terminator::Jump{guard->name, std::move(to_guard)};
            f->blocks.push_back(guard);
            f->blocks.push_back(past);
            refresh_preds(*f);
        }
    }

    // Every other bound loop this one contains, or is contained by. Checked
    // after the fact rather than as the schedule is read, because the schedule
    // may bind the inner loop first and either order should mean the same
    // thing.
    const set<string> inside = body_of(*f, *bound);
    for (const auto &block : f->blocks) {
        const auto *other =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (other == nullptr || other == bound || !other->binding.has_value()) {
            continue;
        }
        if (inside.count(block->name)) {
            internal_assert(may_nest(resource, *other->binding))
                << "bind(" << index << ", " << to_string(resource) << ") on "
                << func << ": " << other->index << " is inside it and bound to "
                << to_string(*other->binding)
                << ", which does not run inside a " << to_string(resource);
        } else if (body_of(*f, *other)
                       .count(
                           // the block holding the loop being bound
                           [&] {
                               for (const auto &b : f->blocks) {
                                   if (std::get_if<Terminator::ParFor>(
                                           &b->terminator.data) == bound) {
                                       return b->name;
                                   }
                               }
                               return string();
                           }())) {
            internal_assert(may_nest(*other->binding, resource))
                << "bind(" << index << ", " << to_string(resource) << ") on "
                << func << ": it is inside " << other->index << ", which is "
                << "bound to " << to_string(*other->binding) << ", and a "
                << to_string(resource) << " does not run inside that";
        }
    }
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
