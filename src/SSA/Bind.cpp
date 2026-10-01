#include "SSA/Analysis.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "Error.h"
#include "Utils.h"

#include <algorithm>
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
    // did (apps/pbrt/PLAN.md, the launch shape). The guard is a block made
    // here ahead of the body, which reads the count and lets an index short
    // of it through; for a loop that was split -- its body is then the inner
    // loop -- it is the split's own guard on the step, the block where the
    // global index is formed and tested against the end the split threaded
    // through both new loops (Rewrite.cpp), retargeted at the count read on
    // the device.
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
        // A split drain: this is the outer loop (the blocks, `bind(hits_blk,
        // GPUBlock)`), its body the inner loop's block, and that loop's body
        // the split's step. The launch values reached the outer loop through
        // the split (Rewrite.cpp); the inner loop has none and is left be.
        const bool split_drain =
            std::holds_alternative<Terminator::ParFor>(body->terminator.data);
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
        // The extent: the queue's size as the loop's index type (its count
        // field is u32 and its index is u32, so this is a cast only when the
        // producer's push count came in another type).
        shared_ptr<Value> extent = bound->capacity;
        if (!extent->get_type().same_as(index_type)) {
            extent = header->make_instruction(index_type, Instruction::Op::Cast,
                                              {extent});
        }
        // The count's address: the count field of the queue drained -- the
        // header, or the slot of it for a double-buffered queue.
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
        const shared_ptr<Value> old_end = bound->end;
        bound->end = extent;
        bound->count_address = count_address; // read by the residency analysis
        if (split_drain) {
            auto *inner = std::get_if<Terminator::ParFor>(&body->terminator.data);
            shared_ptr<Block> step;
            for (const auto &block : f->blocks) {
                if (block->name == inner->body.name) {
                    step = block;
                }
            }
            internal_assert(step != nullptr)
                << "bind(" << index << "): the inner loop's body " << inner->body.name
                << " is no block of " << func;
            // The end the split threaded through the inner loop's block and
            // the step (an argument of each, under one name) was this loop's
            // end, the count as read on the host; the test is the only thing
            // that read it, and the test is about to read the count on the
            // device instead. Out of both blocks and both jumps, so that the
            // kernel does not capture it and the host's read goes dead with
            // the launch wait it was.
            for (size_t k = 0; k < bound->body.args.size(); k++) {
                if (bound->body.args[k] != old_end) {
                    continue;
                }
                // A jump's arguments follow the loop's index, which the loop
                // passes itself: the k-th value fills the (k+1)-th argument.
                const string threaded = body->args[k + 1].name;
                bound->body.args.erase(bound->body.args.begin() + k);
                body->args.erase(body->args.begin() + k + 1);
                for (size_t j = 0; j < inner->body.args.size(); j++) {
                    const auto passed = inner->body.args[j]->get_argument();
                    if (passed && passed->name == threaded) {
                        inner->body.args.erase(inner->body.args.begin() + j);
                        step->args.erase(step->args.begin() + j + 1);
                        break;
                    }
                }
                break;
            }
            // The count's address, threaded as the end was: an argument of
            // the inner loop's block passed by this loop, and an argument of
            // the step passed by the inner loop.
            const Type address_type = count_address->get_type();
            auto address_in = body->add_argument(
                Argument{address_type, f->get_unique_name()});
            bound->body.args.push_back(count_address);
            auto address_at_step = step->add_argument(
                Argument{address_type, f->get_unique_name()});
            inner->body.args.push_back(address_in);
            // The split's guard: the step ends in a dispatch on `index <
            // end` (Rewrite.cpp). Its end becomes the count, loaded on the
            // device ahead of the test -- pbrt's `if (index >= queue->Size())
            // return;` -- so a step past the count ends at the tail as one
            // past the end did.
            auto *dispatch = std::get_if<Terminator::Dispatch>(&step->terminator.data);
            internal_assert(dispatch != nullptr)
                << "bind(" << index << "): the split's step " << step->name
                << " ends in no dispatch; a drain is split with a guard";
            auto *test = std::get_if<shared_ptr<Instruction>>(&dispatch->cond->data);
            internal_assert(test != nullptr && (*test)->op == Instruction::Op::Lt &&
                            (*test)->operands.size() == 2)
                << "bind(" << index << "): the split's step " << step->name
                << " does not dispatch on `index < end`";
            const size_t before = step->instrs.size();
            shared_ptr<Value> count = step->make_instruction(
                UInt_t::make(32), Instruction::Op::Load, {address_at_step});
            if (!count->get_type().same_as(index_type)) {
                count = step->make_instruction(index_type, Instruction::Op::Cast,
                                               {count});
            }
            (*test)->operands[1] = count;
            // make_instruction appends, and the test needs the count ahead
            // of it: the new instructions move to just before the test.
            auto at = std::find(step->instrs.begin(), step->instrs.end(), *test);
            internal_assert(at != step->instrs.end())
                << "bind(" << index << "): the split's test is not in its step";
            std::rotate(at, step->instrs.begin() + before, step->instrs.end());
        } else {
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
