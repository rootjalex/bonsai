#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace lower {

// Replaces a return `call` or store `call` with an atomic write to the
// specified `queue`. Then, finds all code that calls this function, and
// forces it to pass a write location and the queue through. Queue allocations
// happen at the specified loop index, which looks like `qfunc.index`. If
// `qfunc` is not provided, it is assumed that `qfunc` is this func.
// func dumb_sum(i : i32, n : i32, a : array[f32, n], acc : f32) -> f32 {
//     if (i == n) {
//         return acc;
//     } else {
//         return dumb_sum(i + 1, n, a, acc + a[i]);
//     }
// }
// dumb_sum.defer(dumb_sum, <loop id>, <queue id>)
// func dumb_sum_defer(i : i32, n : i32, a : array[f32, n, acc : f32], out : mut
// f32, queue : Queue) -> void {
//     if (i == n) {
//         out = acc;
//         return;
//     } else {
//         queue.insert(i + 1, n, a, acc + a[i], out);
//         return;
//     }
// }
// The queue allocation is placed at the designated loop.
// Note: if the queue size is not scheduled, this becomes a dynamic
// queue (by default). Queue sizes should be scheduled via the `make_queue`
// or `make_queues` scheduling commands.
// Note: if `queue` is `local` and `loop` is innermost, this is just
// iteration aligning.
class LowerDefers : public Pass {
  public:
    const std::string name() const override { return "lower-defers"; }

    // Requires full-program (needs access to schedule).
    ir::Program run(ir::Program program,
                    const CompilerOptions &options) const override;
};

} // namespace lower
} // namespace bonsai
