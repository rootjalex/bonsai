#include "SSA/RoundGuard.h"

#include "IR/Program.h"
#include "SSA/Analysis.h"

#include "Error.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::string;
using std::vector;

const char *const guard_name = "bonsai_cuda_round_guard";

const Instruction *def_of(const shared_ptr<Value> &v) {
    if (v == nullptr) {
        return nullptr;
    }
    const auto *i = std::get_if<shared_ptr<Instruction>>(&v->data);
    return i == nullptr ? nullptr : i->get();
}

bool is_zero(const shared_ptr<Value> &v) {
    if (v == nullptr) {
        return false;
    }
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return false;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u == 0;
    }
    if (const auto *s = std::get_if<int64_t>(&c->data)) {
        return *s == 0;
    }
    return false;
}

bool is_one(const shared_ptr<Value> &v) {
    if (v == nullptr) {
        return false;
    }
    const auto *c = std::get_if<Constant>(&v->data);
    if (c == nullptr) {
        return false;
    }
    if (const auto *u = std::get_if<uint64_t>(&c->data)) {
        return *u == 1;
    }
    if (const auto *s = std::get_if<int64_t>(&c->data)) {
        return *s == 1;
    }
    return false;
}

// The name a value's storage goes by: the allocation's, or the argument
// threading it into this block -- threading keeps the name (Block::
// get_value), so the name follows a queue across blocks and function copies.
std::optional<string> root_name(const shared_ptr<Value> &v) {
    if (v == nullptr) {
        return std::nullopt;
    }
    if (const auto arg = v->get_argument()) {
        return arg->name;
    }
    const Instruction *d = def_of(v);
    if (d == nullptr) {
        return std::nullopt;
    }
    switch (d->op) {
    case Instruction::Op::Alloca:
    case Instruction::Op::Alloc:
        return d->name;
    case Instruction::Op::GEP:
    case Instruction::Op::ExtractIdx:
    case Instruction::Op::Load:
    case Instruction::Op::LoadField:
    case Instruction::Op::FieldPtr:
    case Instruction::Op::Cast:
        return d->operands.empty() ? std::nullopt : root_name(d->operands[0]);
    default:
        return std::nullopt;
    }
}

bool on_gpu(const Terminator::ParFor &loop) {
    return loop.binding.has_value() &&
           (*loop.binding == Resource::GPUThread ||
            *loop.binding == Resource::GPUBlock ||
            *loop.binding == Resource::OptixThread);
}

// Every value a block refers to, instructions and terminator both, handed to
// `each`. For deciding whether an instruction the rewrite orphaned is dead.
template <typename F> void each_value(const Block &block, F each) {
    for (const auto &instr : block.instrs) {
        for (const auto &v : instr->operands) {
            each(v);
        }
    }
    const auto jump = [&](const Terminator::Jump &j) {
        for (const auto &v : j.args) {
            each(v);
        }
    };
    std::visit(
        [&](const auto &t) {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, Terminator::Jump>) {
                jump(t);
            } else if constexpr (std::is_same_v<T, Terminator::Dispatch>) {
                each(t.cond);
                for (const auto &j : t.targets) {
                    jump(j);
                }
            } else if constexpr (std::is_same_v<T, Terminator::Return>) {
                each(t.value);
            } else if constexpr (std::is_same_v<T, Terminator::ParFor>) {
                each(t.start);
                each(t.end);
                each(t.stride);
                each(t.capacity);
                each(t.queue_base);
                each(t.queue_slot);
                each(t.count_address);
                jump(t.body);
                jump(t.cont);
            } else if constexpr (std::is_same_v<T, Terminator::Call>) {
                jump(t.call);
                jump(t.cont);
            } else if constexpr (std::is_same_v<T, Terminator::MultiCall>) {
                jump(t.call);
                jump(t.cont);
                for (const auto &vs : t.varying) {
                    for (const auto &v : vs) {
                        each(v);
                    }
                }
                for (const auto &v : t.keys) {
                    each(v);
                }
                for (const auto &v : t.conds) {
                    each(v);
                }
            }
        },
        block.terminator.data);
}

bool used_anywhere(const Function &f, const shared_ptr<Value> &value) {
    bool used = false;
    for (const auto &block : f.blocks) {
        each_value(*block, [&](const shared_ptr<Value> &v) {
            used = used || v == value;
        });
    }
    return used;
}

// A header whose test the rewrite replaces.
struct Candidate {
    shared_ptr<Block> header;
    shared_ptr<Value> cond;       // the Eq, to delete if orphaned
    shared_ptr<Value> count_load; // the LoadField, the read being removed
    shared_ptr<Value> queue_view; // the ExtractIdx/Load, to delete if orphaned
    shared_ptr<Value> queues;     // the header allocation the count is read of
    shared_ptr<Value> slot;       // which queue of a double buffer, or null
    string round;                 // the header argument the latch steps
    const Loop *loop = nullptr;   // the round loop itself (the Cfg snapshot's)
};

} // namespace

void pipeline_round_guards(FuncMap &funcs, ir::Program &program) {
    bool rewrote_any = false;
    for (auto &[fname, f] : funcs) {
        if (f->blocks.empty()) {
            continue;
        }
        // The queue drains, by the facts a GPU bind reads (Terminator::
        // ParFor::queue_base, set by Defer, carried outward by a split).
        struct Drain {
            string family;
            string block;
            bool gpu;
        };
        vector<Drain> drains;
        for (const auto &block : f->blocks) {
            const auto *p =
                std::get_if<Terminator::ParFor>(&block->terminator.data);
            if (p == nullptr || p->queue_base == nullptr) {
                continue;
            }
            const auto family = root_name(p->queue_base);
            if (family.has_value()) {
                drains.push_back(Drain{*family, block->name, on_gpu(*p)});
            }
        }
        if (drains.empty()) {
            continue;
        }

        const Cfg cfg(*f);
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest forest = compute_loop_forest(cfg, dom);

        vector<Candidate> candidates;
        for (const auto &header : f->blocks) {
            const auto *d =
                std::get_if<Terminator::Dispatch>(&header->terminator.data);
            if (d == nullptr || d->targets.size() != 2) {
                continue;
            }
            // The test: `count == 0`, the count a host load through the
            // queue's header (LoadField of field 0 of the queue read this
            // round).
            const Instruction *eq = def_of(d->cond);
            if (eq == nullptr || eq->op != Instruction::Op::Eq ||
                eq->operands.size() != 2) {
                continue;
            }
            shared_ptr<Value> load_v;
            if (is_zero(eq->operands[1])) {
                load_v = eq->operands[0];
            } else if (is_zero(eq->operands[0])) {
                load_v = eq->operands[1];
            } else {
                continue;
            }
            const Instruction *load = def_of(load_v);
            if (load == nullptr || load->op != Instruction::Op::LoadField ||
                load->operands.size() != 2 || !is_zero(load->operands[1])) {
                continue;
            }
            const Instruction *view = def_of(load->operands[0]);
            if (view == nullptr) {
                continue;
            }
            shared_ptr<Value> queues, slot;
            if (view->op == Instruction::Op::ExtractIdx &&
                view->operands.size() == 2) {
                queues = view->operands[0];
                slot = view->operands[1];
            } else if (view->op == Instruction::Op::Load &&
                       view->operands.size() == 1) {
                queues = view->operands[0];
            } else {
                continue;
            }
            const auto family = root_name(queues);
            if (!family.has_value()) {
                continue;
            }
            // The loop this header heads must drain that queue on the GPU,
            // and nothing in it on the host: a host drain reads its own trip
            // count back each round, so the loop would wait regardless and
            // the host's view of this count must stay exact.
            const BlockId hid = cfg.find(*header);
            const Loop *loop = hid == NO_BLOCK ? nullptr : forest.find(hid);
            if (loop == nullptr) {
                continue;
            }
            bool gpu_drain = false, host_drain = false;
            for (const Drain &drain : drains) {
                const BlockId b = cfg.find(drain.block);
                if (b == NO_BLOCK || !loop->blocks.contains(b)) {
                    continue;
                }
                if (!drain.gpu) {
                    host_drain = true;
                } else if (drain.family == *family) {
                    gpu_drain = true;
                }
            }
            if (!gpu_drain || host_drain) {
                continue;
            }
            // The round: the header argument one predecessor starts at zero
            // and another steps by one -- the loop's own counter, which the
            // ring is indexed by.
            string round;
            for (size_t k = 0; k < header->args.size() && round.empty(); k++) {
                bool stepped = false, started = false;
                for (const auto &w : header->preds) {
                    const auto pred = w.lock();
                    if (pred == nullptr) {
                        continue;
                    }
                    const auto look = [&](const Terminator::Jump &j) {
                        if (j.name != header->name || k >= j.args.size()) {
                            return;
                        }
                        const Instruction *step = def_of(j.args[k]);
                        if (step != nullptr &&
                            step->op == Instruction::Op::Add &&
                            step->operands.size() == 2 &&
                            (is_one(step->operands[0]) ||
                             is_one(step->operands[1]))) {
                            stepped = true;
                        }
                        if (is_zero(j.args[k])) {
                            started = true;
                        }
                    };
                    std::visit(
                        [&](const auto &t) {
                            using T = std::decay_t<decltype(t)>;
                            if constexpr (std::is_same_v<T, Terminator::Jump>) {
                                look(t);
                            } else if constexpr (std::is_same_v<
                                                     T, Terminator::Dispatch>) {
                                for (const auto &j : t.targets) {
                                    look(j);
                                }
                            } else if constexpr (std::is_same_v<
                                                     T, Terminator::ParFor>) {
                                look(t.body);
                                look(t.cont);
                            } else if constexpr (std::is_same_v<
                                                     T, Terminator::Call>) {
                                look(t.cont);
                            }
                        },
                        pred->terminator.data);
                }
                if (stepped && started) {
                    round = header->args[k].name;
                }
            }
            if (round.empty()) {
                continue;
            }
            candidates.push_back(Candidate{header, d->cond, load_v,
                                           load->operands[0], queues, slot,
                                           round, loop});
        }

        for (Candidate &c : candidates) {
            Block &header = *c.header;
            auto old = std::get<Terminator::Dispatch>(header.terminator.data);

            // The continuation: the dispatch as it was, on the guard's
            // answer in place of the count -- zero keeps its meaning, so the
            // targets stand.
            auto cont = std::make_shared<Block>();
            cont->name = header.name + "!go";
            cont->owner = header.owner;
            auto go = cont->add_argument(
                Argument{UInt_t::make(32), f->get_unique_name()});
            auto zero = std::make_shared<Value>(
                Constant{UInt_t::make(32), uint64_t(0)});
            auto stop = cont->make_instruction(Bool_t::make(),
                                               Instruction::Op::Eq, {go, zero});
            cont->terminator.data =
                Terminator::Dispatch{stop, std::move(old.targets)};

            // The count's address, where the test loaded its value: the
            // runtime snapshots it on the stream instead.
            shared_ptr<Value> queue_at = c.queues;
            if (c.slot != nullptr) {
                queue_at = header.make_instruction(
                    c.queues->get_type(), Instruction::Op::GEP,
                    {c.queues, c.slot});
            }
            auto field_zero = std::make_shared<Value>(
                Constant{UInt_t::make(32), uint64_t(0)});
            auto count_address = header.make_instruction(
                Ptr_t::make(UInt_t::make(32)), Instruction::Op::FieldPtr,
                {queue_at, field_zero});
            auto round = header.get_value(c.round, UInt_t::make(32));
            header.terminator.data = Terminator::Call{
                Terminator::Jump{guard_name, {round, count_address}},
                Terminator::Jump{cont->name, {}},
                /*drop=*/false};

            // The graph: the continuation takes the header's place as the
            // targets' predecessor, and follows it in the block list.
            const auto &targets =
                std::get<Terminator::Dispatch>(cont->terminator.data).targets;
            for (const auto &target : targets) {
                for (const auto &block : f->blocks) {
                    if (block->name != target.name) {
                        continue;
                    }
                    for (auto &w : block->preds) {
                        if (w.lock() == c.header) {
                            w = cont;
                        }
                    }
                }
            }
            cont->preds = {c.header};
            for (auto it = f->blocks.begin(); it != f->blocks.end(); ++it) {
                if (*it == c.header) {
                    f->blocks.insert(it + 1, cont);
                    break;
                }
            }

            // The replaced test may still ride the dispatch's jumps into
            // block arguments nothing reads: the launch's end, threaded by
            // the defer and left behind when the GPU bind replaced it with
            // the capacity (Bind.cpp scrubs the split case and leaves the
            // plain one). Dropped here -- the argument and every jump's
            // value for it -- where the receiver is provably unread, so the
            // load this pass exists to remove goes dead.
            auto &moved =
                std::get<Terminator::Dispatch>(cont->terminator.data);
            for (Terminator::Jump &jump : moved.targets) {
                shared_ptr<Block> target;
                for (const auto &block : f->blocks) {
                    if (block->name == jump.name) {
                        target = block;
                    }
                }
                if (target == nullptr) {
                    continue;
                }
                for (size_t k = jump.args.size(); k-- > 0;) {
                    if (jump.args[k] != c.cond &&
                        jump.args[k] != c.count_load &&
                        jump.args[k] != c.queue_view) {
                        continue;
                    }
                    if (k >= target->args.size()) {
                        continue;
                    }
                    const Argument &arg = target->args[k];
                    const auto held = target->get_value(arg.name, arg.type);
                    if (used_anywhere(*f, held)) {
                        continue;
                    }
                    for (const auto &w : target->preds) {
                        const auto pred = w.lock();
                        if (pred == nullptr) {
                            continue;
                        }
                        const auto drop = [&](Terminator::Jump &j) {
                            if (j.name == target->name && k < j.args.size()) {
                                j.args.erase(j.args.begin() + k);
                            }
                        };
                        std::visit(
                            [&](auto &t) {
                                using T = std::decay_t<decltype(t)>;
                                if constexpr (std::is_same_v<T,
                                                             Terminator::Jump>) {
                                    drop(t);
                                } else if constexpr (
                                    std::is_same_v<T, Terminator::Dispatch>) {
                                    for (auto &j : t.targets) {
                                        drop(j);
                                    }
                                } else if constexpr (
                                    std::is_same_v<T, Terminator::ParFor>) {
                                    drop(t.body);
                                    drop(t.cont);
                                } else if constexpr (
                                    std::is_same_v<T, Terminator::Call>) {
                                    drop(t.cont);
                                }
                            },
                            pred->terminator.data);
                    }
                    target->args.erase(target->args.begin() + k);
                }
            }

            // The orphaned test itself: each erased only once nothing is
            // left that reads it.
            const auto erase_if_dead = [&](const shared_ptr<Value> &v) {
                if (v == nullptr || used_anywhere(*f, v)) {
                    return;
                }
                const Instruction *d = def_of(v);
                std::erase_if(header.instrs, [&](const auto &i) {
                    return i.get() == d;
                });
            };
            erase_if_dead(c.cond);
            erase_if_dead(c.count_load);
            erase_if_dead(c.queue_view);
            rewrote_any = true;
        }

        // The headers rebuilt every round: a queue made inside this loop (a
        // material or shadow queue, made inside the ray queue's rounds) is
        // emptied before its producers run by storing its whole header --
        // count zero and the arrays -- once a round, and the arrays never
        // change: on the renderer's host that was a dozen staged copies and
        // their ring events every round, a third of the launch gap (the
        // nsys numbers, apps/pbrt/PLAN.md). The fill moves to the loop's
        // preheader -- everything it stores but the count is defined
        // outside the loop -- and in its place only the count is zeroed,
        // one word on the stream. Defer cannot place it there itself: when
        // a queue is built the graph is not yet walkable past its own
        // unterminated blocks, where this pass sees the finished loop.
        std::map<const Instruction *, BlockId> def_block;
        std::map<string, BlockId> named_def; // an instruction's name -> block
        std::map<string, const Instruction *> named_instr;
        for (const auto &block : cfg.blocks()) {
            const BlockId b = cfg.find(block->name);
            for (const auto &instr : block->instrs) {
                def_block[instr.get()] = b;
                if (!instr->name.empty()) {
                    named_def[instr->name] = b;
                    named_instr[instr->name] = instr.get();
                }
            }
        }
        // root_name, through threads: a value that arrives as a block
        // argument carries the name of the instruction it threads
        // (Block::get_value), and names are unique in a function, so the
        // walk continues at that instruction.
        const std::function<std::optional<string>(const shared_ptr<Value> &)>
            rooted = [&](const shared_ptr<Value> &v) -> std::optional<string> {
            const std::function<std::optional<string>(const Instruction *)>
                of_instr = [&](const Instruction *d) -> std::optional<string> {
                if (d == nullptr) {
                    return std::nullopt;
                }
                switch (d->op) {
                case Instruction::Op::Alloca:
                case Instruction::Op::Alloc:
                    return d->name;
                case Instruction::Op::GEP:
                case Instruction::Op::ExtractIdx:
                case Instruction::Op::Load:
                case Instruction::Op::LoadField:
                case Instruction::Op::FieldPtr:
                case Instruction::Op::Cast:
                    return d->operands.empty() ? std::nullopt
                                               : rooted(d->operands[0]);
                default:
                    return std::nullopt;
                }
            };
            if (v == nullptr) {
                return std::nullopt;
            }
            if (const auto arg = v->get_argument()) {
                const auto it = named_instr.find(arg->name);
                if (it == named_instr.end()) {
                    return arg->name;
                }
                return of_instr(it->second);
            }
            return of_instr(def_of(v));
        };
        std::set<string> parameters;
        for (const Argument &arg : f->blocks.front()->args) {
            parameters.insert(arg.name);
        }
        for (Candidate &c : candidates) {
            const Loop &loop = *c.loop;
            // Whether `v` holds the same value on every round. A value
            // threaded into the loop is an argument of an in-loop block
            // under the name it was defined with (Block::get_value keeps
            // the name), so the test is of the name: it is a parameter of
            // the function, or an instruction defined outside the loop, and
            // nothing inside the loop defines that name -- then the thread
            // only carries it, the back edge included.
            const auto movable = [&](const shared_ptr<Value> &v) {
                if (v == nullptr) {
                    return false;
                }
                if (std::holds_alternative<Constant>(v->data)) {
                    return true;
                }
                string name;
                if (const auto arg = v->get_argument()) {
                    name = arg->name;
                } else if (const Instruction *d = def_of(v)) {
                    name = d->name;
                }
                if (name.empty()) {
                    return false;
                }
                const auto defined = named_def.find(name);
                if (defined != named_def.end() &&
                    loop.blocks.contains(defined->second)) {
                    return false; // made anew each round
                }
                return parameters.contains(name) ||
                       defined != named_def.end();
            };
            // The value in the preheader: a constant as it is, an
            // instruction defined outside the loop as it is when its block
            // dominates the preheader, anything else threaded in by name
            // (the graph is complete here, so the walk is safe).
            const auto in_preheader =
                [&](const shared_ptr<Block> &preheader,
                    const shared_ptr<Value> &v) -> shared_ptr<Value> {
                if (std::holds_alternative<Constant>(v->data)) {
                    return v;
                }
                if (const Instruction *d = def_of(v)) {
                    const auto it = def_block.find(d);
                    const BlockId at = cfg.find(preheader->name);
                    if (it != def_block.end() && at != NO_BLOCK &&
                        dom.dominates(it->second, at)) {
                        return v;
                    }
                }
                string name;
                if (const auto arg = v->get_argument()) {
                    name = arg->name;
                } else if (const Instruction *d = def_of(v)) {
                    name = d->name;
                }
                return preheader->get_value(name, v->get_type());
            };
            // The preheader: the one way into the loop that is not a back
            // edge.
            shared_ptr<Block> preheader;
            bool one_way_in = true;
            for (const auto &w : c.header->preds) {
                const auto pred = w.lock();
                if (pred == nullptr) {
                    continue;
                }
                const BlockId b = cfg.find(pred->name);
                if (b != NO_BLOCK && loop.blocks.contains(b)) {
                    continue;
                }
                one_way_in = one_way_in && preheader == nullptr;
                preheader = pred;
            }
            if (preheader == nullptr || !one_way_in) {
                continue;
            }
            for (const auto &block : cfg.blocks()) {
                const BlockId b = cfg.find(block->name);
                if (b == NO_BLOCK || !loop.blocks.contains(b)) {
                    continue;
                }
                vector<shared_ptr<Instruction>> fills;
                for (const auto &instr : block->instrs) {
                    if (instr->op != Instruction::Op::Store ||
                        instr->operands.size() != 2) {
                        continue;
                    }
                    const Instruction *made = def_of(instr->operands[1]);
                    if (made == nullptr ||
                        made->op != Instruction::Op::MakeStruct ||
                        made->operands.empty() ||
                        !is_zero(made->operands[0])) {
                        continue;
                    }
                    // The pointer: the queue header's slot, rooted outside
                    // the loop in a header allocation of Defer's.
                    shared_ptr<Value> base = instr->operands[0], index;
                    if (const Instruction *gep = def_of(base);
                        gep != nullptr && gep->op == Instruction::Op::GEP &&
                        gep->operands.size() == 2) {
                        base = gep->operands[0];
                        index = gep->operands[1];
                    }
                    const auto family = root_name(base);
                    if (!family.has_value() || !family->ends_with("_queue") ||
                        !movable(base) ||
                        (index != nullptr && !movable(index))) {
                        continue;
                    }
                    bool all_movable = true;
                    for (const auto &part : made->operands) {
                        all_movable = all_movable && movable(part);
                    }
                    if (all_movable) {
                        fills.push_back(instr);
                    }
                }
                for (const auto &fill : fills) {
                    const Instruction *made = def_of(fill->operands[1]);
                    shared_ptr<Value> base = fill->operands[0], index;
                    if (const Instruction *gep = def_of(base);
                        gep != nullptr && gep->op == Instruction::Op::GEP) {
                        base = gep->operands[0];
                        index = gep->operands[1];
                    }
                    shared_ptr<Value> at = in_preheader(preheader, base);
                    if (index != nullptr) {
                        at = preheader->make_instruction(
                            base->get_type(), Instruction::Op::GEP,
                            {at, in_preheader(preheader, index)});
                    }
                    vector<shared_ptr<Value>> parts;
                    for (const auto &part : made->operands) {
                        parts.push_back(in_preheader(preheader, part));
                    }
                    auto filled = preheader->make_instruction(
                        made->type, Instruction::Op::MakeStruct,
                        std::move(parts));
                    preheader->make_side_effect(Instruction::Op::Store,
                                                {at, filled});
                    // The count: zeroed in the fill's place only when the
                    // loop does not already reset it -- a queue the rounds
                    // drain once a pass is reset after its drain (Defer),
                    // which covers every round after the preheader's zero,
                    // so a second word a round would be waste.
                    const auto queue_name = root_name(fill->operands[0]);
                    const bool debug_rounds =
                        std::getenv("BONSAI_DEBUG_ROUNDS") != nullptr;
                    bool reset_in_loop = false;
                    for (const auto &other : cfg.blocks()) {
                        const BlockId ob = cfg.find(other->name);
                        if (ob == NO_BLOCK || !loop.blocks.contains(ob)) {
                            continue;
                        }
                        for (const auto &instr : other->instrs) {
                            // A word of zero stored through the queue's
                            // header: its count, the only header word
                            // anything writes (the pointer may arrive as a
                            // threaded argument, so the test is of the
                            // root, not the field pointer's shape).
                            if (instr == fill ||
                                instr->op != Instruction::Op::Store ||
                                instr->operands.size() != 2 ||
                                !is_zero(instr->operands[1]) ||
                                !instr->operands[1]->get_type().is<UInt_t>()) {
                                continue;
                            }
                            if (debug_rounds) {
                                std::cerr << "rounds: zero store in "
                                          << other->name << " roots at "
                                          << rooted(instr->operands[0])
                                                 .value_or("<none>")
                                          << " against "
                                          << queue_name.value_or("<none>")
                                          << "\n";
                            }
                            reset_in_loop =
                                reset_in_loop ||
                                rooted(instr->operands[0]) == queue_name;
                        }
                    }
                    if (!reset_in_loop) {
                        auto field_zero = std::make_shared<Value>(
                            Constant{UInt_t::make(32), uint64_t(0)});
                        auto count_ptr = block->make_instruction(
                            Ptr_t::make(UInt_t::make(32)),
                            Instruction::Op::FieldPtr,
                            {fill->operands[0], field_zero});
                        auto zero = std::make_shared<Value>(
                            Constant{UInt_t::make(32), uint64_t(0)});
                        block->make_side_effect(Instruction::Op::Store,
                                                {count_ptr, zero});
                    }
                    const shared_ptr<Value> stored = fill->operands[1];
                    const shared_ptr<Value> pointer = fill->operands[0];
                    std::erase_if(block->instrs, [&](const auto &i) {
                        return i == fill;
                    });
                    // The fill's MakeStruct (and the slot it stored through)
                    // are the store's alone: gone with it.
                    for (const auto &orphan : {stored, pointer}) {
                        if (def_of(orphan) == nullptr ||
                            used_anywhere(*f, orphan)) {
                            continue;
                        }
                        const Instruction *d = def_of(orphan);
                        std::erase_if(block->instrs, [&](const auto &i) {
                            return i.get() == d;
                        });
                    }
                }
            }
        }
    }

    if (!rewrote_any || program.foreign_funcs.contains(guard_name)) {
        return;
    }
    // The runtime's declaration, built as the parser builds one (Parser.cpp,
    // parse_foreign_function) except that the count's parameter is the
    // pointer it is: the parser admits arrays because a program's author
    // means a buffer, where this address is one word of a queue header, and
    // both reach C as a pointer (CodeGen_LLVM::declare_foreign_function).
    // runtime/bonsai_cuda.h implements it; the JIT pins it by address
    // (CodeGen/JIT.cpp).
    std::vector<ir::Function::Argument> args;
    args.push_back(ir::Function::Argument("round", UInt_t::make(32)));
    args.push_back(
        ir::Function::Argument("count", Ptr_t::make(UInt_t::make(32))));
    program.foreign_funcs[guard_name] = std::make_shared<ir::Function>(
        guard_name, std::move(args), UInt_t::make(32), ir::Stmt(),
        ir::Function::InterfaceList{},
        std::vector<ir::Function::Attribute>{ir::Function::Attribute::foreign});
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
