#include "SSA/QueueStorage.h"

#include "Error.h"
#include "SSA/Analysis.h"
#include "Utils.h"

#include <algorithm>

#include <map>
#include <set>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

namespace {

using std::shared_ptr;
using std::string;
using std::vector;

// Which processor a loop's binding runs its body on.
bool gpu_bound(Resource r) {
    return r == Resource::GPUBlock || r == Resource::GPUThread ||
           r == Resource::OptixThread;
}

// The allocation an address was computed from, through the instructions
// that derive one address from another -- an element, a field, a slot of
// an array of headers, the address of a value -- and through a load of an
// array handle out of a header, so that a read of an entry through the
// handle a header holds roots at the header. Null when the address is not
// an allocation's: a parameter, a value computed from one.
const Instruction *storage_of(const shared_ptr<Value> &v, bool *through_load) {
    const shared_ptr<Value> *at = &v;
    while (true) {
        const auto *instr = std::get_if<shared_ptr<Instruction>>(&(*at)->data);
        if (instr == nullptr) {
            return nullptr;
        }
        const Instruction::Op op = (*instr)->op;
        if (op == Instruction::Op::Alloca || op == Instruction::Op::Alloc) {
            return instr->get();
        }
        if ((op == Instruction::Op::GEP || op == Instruction::Op::FieldPtr ||
             op == Instruction::Op::AddressOf || op == Instruction::Op::ExtractIdx ||
             op == Instruction::Op::Load) &&
            !(*instr)->operands.empty()) {
            if (op == Instruction::Op::Load) {
                *through_load = true;
            }
            at = &(*instr)->operands[0];
            continue;
        }
        return nullptr;
    }
}

struct Loops {
    Cfg cfg;
    // Per block: the bound parfors whose bodies hold it, outermost first,
    // as (the block that launches, the binding).
    vector<vector<std::pair<BlockId, Resource>>> bound_around;
    // Per block: whether any loop (bound or not, a parfor's body or a
    // sequential loop) holds it.
    vector<bool> in_loop;

    explicit Loops(const Function &func) : cfg(func) {
        const DomTree dom = compute_dominator_tree(cfg);
        const LoopForest forest = compute_loop_forest(cfg, dom);
        bound_around.resize(cfg.size());
        in_loop.assign(cfg.size(), false);
        for (BlockId b : cfg.rpo) {
            if (forest.innermost(b) != nullptr) {
                in_loop[b] = true;
            }
        }
        for (BlockId b : cfg.rpo) {
            const auto *p = std::get_if<Terminator::ParFor>(&cfg[b].terminator.data);
            if (p == nullptr) {
                continue;
            }
            const BlockId body = cfg.id(p->body.name);
            if (body == NO_BLOCK) {
                continue;
            }
            for (BlockId inside : reachable_from(cfg, body)) {
                in_loop[inside] = true;
                if (p->binding.has_value()) {
                    bound_around[inside].emplace_back(b, *p->binding);
                }
            }
        }
        // Outermost first: a body launched from a block that is itself in a
        // body is inner to it. Sort by the depth of the launching block.
        for (auto &around : bound_around) {
            std::stable_sort(around.begin(), around.end(),
                             [&](const auto &x, const auto &y) {
                                 return bound_around[x.first].size() <
                                        bound_around[y.first].size();
                             });
        }
    }

    BlockId block_of(const Instruction &instr) const {
        const shared_ptr<Block> owner = instr.owner.lock();
        return owner ? cfg.id(owner->name) : NO_BLOCK;
    }
};

string describe(const Instruction &alloc) {
    return alloc.name + " (in " + to_string(*alloc.storage) + " memory)";
}

} // namespace

size_t place_queue_storage(Function &func, bool exported) {
    vector<shared_ptr<Instruction>> placed;
    for (const auto &block : func.blocks) {
        for (const auto &instr : block->instrs) {
            if ((instr->op == Instruction::Op::Alloca ||
                 instr->op == Instruction::Op::Alloc) &&
                instr->storage.has_value()) {
                placed.push_back(instr);
            }
        }
    }
    if (placed.empty()) {
        return 0;
    }
    const Loops loops(func);

    // Every access of each placed allocation: the block it is in, whether
    // it is of the entries (through an array handle, or of an array
    // directly) rather than of the header's own words, and the captures a
    // bound loop takes of it.
    struct Access {
        BlockId block;
        bool entries;
        const Instruction *by;  // null for a loop's capture
        const Terminator::ParFor *capture = nullptr;
    };
    std::map<const Instruction *, vector<Access>> accesses;
    for (const auto &block : func.blocks) {
        const BlockId b = loops.cfg.id(block->name);
        if (b == NO_BLOCK) {
            continue;
        }
        for (const auto &instr : block->instrs) {
            if (instr->op != Instruction::Op::Load &&
                instr->op != Instruction::Op::Store &&
                instr->op != Instruction::Op::AtomicAdd) {
                continue;
            }
            if (instr->operands.empty()) {
                continue;
            }
            bool through_load = false;
            const Instruction *root = storage_of(instr->operands[0], &through_load);
            if (root == nullptr || !root->storage.has_value()) {
                continue;
            }
            // Of the entries: an array's storage, or the header's reached
            // through a handle loaded out of it. The header is a struct
            // (`Queue_<q>`), or an array of them for a double-buffered or
            // split queue; an entry array's elements are the entry's leaves,
            // never a struct (SSA/Defer.cpp).
            const Type stored = root->type.is_reference()
                                    ? root->type
                                    : root->type.as<Ptr_t>()->etype;
            const Array_t *as_array = stored.as<Array_t>();
            const bool array = as_array != nullptr && !as_array->etype.is<Struct_t>();
            accesses[root].push_back({b, array || through_load, instr.get()});
        }
        // A loop whose body takes the storage, or an address into it, as a
        // capture: the body reads and writes it through that argument -- a
        // drain reads the entries through the header it is handed -- and
        // runs where the loop is bound, or where the loop sits if it is not.
        if (const auto *p = std::get_if<Terminator::ParFor>(&block->terminator.data)) {
            for (const shared_ptr<Value> &arg : p->body.args) {
                bool through_load = false;
                const Instruction *root = storage_of(arg, &through_load);
                if (root != nullptr && root->storage.has_value()) {
                    accesses[root].push_back({b, /*entries=*/true, nullptr, p});
                }
            }
        }
    }

    size_t made = 0;
    for (const shared_ptr<Instruction> &alloc : placed) {
        const Storage storage = *alloc->storage;
        const BlockId home = loops.block_of(*alloc);
        internal_assert(home != NO_BLOCK) << alloc->name << " is in no block";
        const auto &uses = accesses[alloc.get()];

        internal_assert(storage != Storage::DeviceShared)
            << "[unimplemented] " << describe(*alloc)
            << ": a queue in shared memory lives in one block's memory and "
               "must be pushed to and drained inside one iteration of a loop "
               "bound to GPUBlock; the PTX backend allocates no shared memory "
               "for a queue yet (apps/pbrt/PLAN.md)";

        const StorageSide side = side_of(storage);
        for (const Access &use : uses) {
            const auto &around = loops.bound_around[use.block];
            // Where the access runs: a captured use runs where its loop is
            // bound, if it is; anything else where the enclosing bound loop
            // runs, if any.
            const bool on_gpu =
                (use.capture != nullptr && use.capture->binding.has_value())
                    ? gpu_bound(*use.capture->binding)
                    : !around.empty() && gpu_bound(around.back().second);
            const auto where = [&]() -> string {
                if (use.capture != nullptr) {
                    return "the loop over " + use.capture->index +
                           (use.capture->binding.has_value()
                                ? " bound to " + string(to_string(*use.capture->binding))
                                : string(", which runs on the host,")) +
                           " takes it as a capture";
                }
                if (around.empty()) {
                    return "it is accessed by host code (block " +
                           loops.cfg.name(use.block) + ")";
                }
                return "it is accessed inside the loop over " +
                       std::get_if<Terminator::ParFor>(
                           &loops.cfg[around.back().first].terminator.data)
                           ->index +
                       " bound to " + string(to_string(around.back().second));
            };
            if (side == StorageSide::Host) {
                internal_assert(!on_gpu)
                    << describe(*alloc) << " is host memory, and " << where()
                    << ": a kernel cannot address the host's memory. Put the "
                       "queue in DeviceGlobal, Managed or an Extern memory of "
                       "the device, or drain it on the CPU.";
            } else if (side == StorageSide::Device) {
                internal_assert(!use.entries || on_gpu)
                    << describe(*alloc) << " is device memory, and " << where()
                    << ": only the device addresses its entries, the host only "
                       "the header's count. Drain the queue on the GPU, or put "
                       "it in Managed memory, which both address.";
            }
            // One thread's storage, written from another thread's loop: the
            // allocation sits inside a loop bound to CPU threads, and the
            // access is inside a further thread-bound loop nested in it.
            if (side == StorageSide::Host) {
                const auto &at_home = loops.bound_around[home];
                if (!at_home.empty() && at_home.back().second == Resource::CPUThread &&
                    around.size() > at_home.size()) {
                    const auto &inner = around[at_home.size()];
                    internal_assert(inner.second != Resource::CPUThread)
                        << describe(*alloc) << " is one thread's storage -- it is "
                        << "placed inside the loop over "
                        << std::get_if<Terminator::ParFor>(
                               &loops.cfg[at_home.back().first].terminator.data)
                               ->index
                        << " bound to CPUThread -- and the loop over "
                        << std::get_if<Terminator::ParFor>(
                               &loops.cfg[inner.first].terminator.data)
                               ->index
                        << ", also bound to CPUThread and nested inside it, "
                           "accesses it: more than one thread would write one "
                           "thread's queue.";
                }
            }
        }

        if (!is_extern(storage)) {
            continue;
        }
        // The caller's storage: one per call.
        internal_assert(!loops.in_loop[home])
            << describe(*alloc) << " is the caller's storage, of which there "
            << "is one per call, but it is made once per iteration of a loop "
            << "(block " << loops.cfg.name(home)
            << "). Place the queue at a loop the storage can be hoisted out "
               "of, or at root.";
        internal_assert(exported)
            << describe(*alloc) << " is the caller's storage, but "
            << func.blocks.front()->name
            << " is not exported: its callers are functions of the program, "
               "and none of them has queue storage to pass. Export the "
               "function, or use the function's own memory (DeviceGlobal, "
               "Managed, Heap).";
        // The header stays the function's (it was given the memory's own
        // word by the deferral); the arrays become parameters.
        if (!alloc->type.is_reference()) {
            continue;
        }
        const Array_t *array = alloc->type.as<Array_t>();
        internal_assert(array != nullptr && array->size.defined())
            << describe(*alloc) << " is not a sized array: " << alloc->type;
        // A constant capacity stays in the parameter's type, and the header
        // the C++ backend prints says the array's bytes
        // (`BONSAI_<f>_<array>_BYTES`); a capacity only known at run time --
        // a render's pass of pixels, pbrt's maxQueueSize -- makes the
        // parameter an unsized array, and the header says the element's
        // bytes (`_ELEMENT_BYTES`) for the caller to multiply by the
        // capacity it computes as the program does. The drain's launch
        // still runs over the capacity the function computes.
        const bool constant = get_constant_value<int64_t>(array->size).has_value();
        const Type parameter_type =
            constant ? alloc->type : Array_t::make(array->etype, Expr());
        Argument arg{parameter_type, alloc->name, /*mutating=*/true};
        arg.storage = storage;
        const shared_ptr<Value> parameter = func.blocks.front()->add_argument(arg);
        replace_uses(func, alloc.get(), parameter);
        made++;
    }
    return made;
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
