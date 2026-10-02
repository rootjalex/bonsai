#pragma once

#include <optional>
#include <string>

namespace bonsai {
namespace ir {

// The memory a queue's storage lives in, and who owns it: the optional last
// argument of `queue()` (IR/Schedule.h).
//
//     hits = render.queue(p_band, DeviceGlobal);
//     hits = render.queue(p_band, 4096, ExternDevice);
//
// Two things are said at once, which Halide says with two directives:
// `store_in(MemoryType)` names the memory, and the storage of a pipeline's
// inputs and outputs is the caller's by being a buffer parameter. Here the
// memory types are the host's heap and stack, the device's global and
// shared memory, and managed memory (cudaMallocManaged) the host and the
// device both address; and the Extern kinds are the same memories owned by
// the caller of the queue's function, which takes the storage as buffer
// parameters, sized by the capacity the header exports, and allocates and
// frees nothing for the queue -- pbrt allocates its queues once, before its
// timer, and so can a driver here. Where the storage is placed -- one per
// iteration of which loop -- is `queue()`'s loop argument, as it was: a
// queue at a loop bound to CPU threads is per thread, at a loop bound to
// GPU blocks per block; none of that is a storage word.
//
// Without the argument the memory is what the drains' binding implies:
// device memory when they run on the GPU, the heap otherwise (SSA/HeapArrays,
// CodeGen_LLVM_SSA::device_resident_allocations), which is what every
// schedule written before the argument existed means.
//
// Whether a storage is valid for a queue is checked against the hardware
// hierarchy once the schedule is applied (SSA/QueueStorage.h): every push
// and the drain must run inside one iteration of a loop bound at the
// memory's level -- one block for DeviceShared, one thread for Stack --
// and an Extern queue's storage must reach the function's top.
enum class Storage {
    Heap,
    Stack,
    DeviceGlobal,
    DeviceShared,
    Managed,
    ExternHost,
    ExternDevice,
    ExternManaged,
};

inline const char *to_string(Storage storage) {
    switch (storage) {
    case Storage::Heap:
        return "Heap";
    case Storage::Stack:
        return "Stack";
    case Storage::DeviceGlobal:
        return "DeviceGlobal";
    case Storage::DeviceShared:
        return "DeviceShared";
    case Storage::Managed:
        return "Managed";
    case Storage::ExternHost:
        return "ExternHost";
    case Storage::ExternDevice:
        return "ExternDevice";
    case Storage::ExternManaged:
        return "ExternManaged";
    }
    return "<unknown storage>";
}

// The storage a name spells, or none: how the parser tells the word from a
// capacity variable in `queue()`'s second slot.
inline std::optional<Storage> storage_named(const std::string &name) {
    for (const Storage s : {Storage::Heap, Storage::Stack, Storage::DeviceGlobal,
                            Storage::DeviceShared, Storage::Managed,
                            Storage::ExternHost, Storage::ExternDevice,
                            Storage::ExternManaged}) {
        if (name == to_string(s)) {
            return s;
        }
    }
    return std::nullopt;
}

// The caller's memory rather than the function's.
inline bool is_extern(Storage storage) {
    return storage == Storage::ExternHost || storage == Storage::ExternDevice ||
           storage == Storage::ExternManaged;
}

// Which processor addresses the memory directly: the host alone, the device
// alone, or both.
enum class StorageSide { Host, Device, Both };

inline StorageSide side_of(Storage storage) {
    switch (storage) {
    case Storage::Heap:
    case Storage::Stack:
    case Storage::ExternHost:
        return StorageSide::Host;
    case Storage::DeviceGlobal:
    case Storage::DeviceShared:
    case Storage::ExternDevice:
        return StorageSide::Device;
    case Storage::Managed:
    case Storage::ExternManaged:
        return StorageSide::Both;
    }
    return StorageSide::Host;
}

} // namespace ir
} // namespace bonsai
