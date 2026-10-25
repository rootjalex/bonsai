#include "SSA/Unroll.h"

#include "Error.h"
#include "SSA/Analysis.h"
#include "SSA/CloneFunction.h"
#include "SSA/InsertPreheader.h"
#include "Utils.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {
namespace ssa {

using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// A constant's value as a count, or nothing for a value that is not one.
std::optional<int64_t> constant_count(const Value &v) {
    const auto *c = std::get_if<Constant>(&v.data);
    if (c == nullptr) {
        return std::nullopt;
    }
    return std::visit(
        overloads{
            [](uint64_t u) -> std::optional<int64_t> { return int64_t(u); },
            [](int64_t i) -> std::optional<int64_t> { return i; },
            [](auto) -> std::optional<int64_t> { return std::nullopt; },
        },
        c->data);
}

// The index's value for one copy, as a constant of the index's type.
shared_ptr<Value> index_constant(const Type &t, int64_t k) {
    if (t.is<UInt_t>()) {
        return std::make_shared<Value>(Constant{t, uint64_t(k)});
    }
    return std::make_shared<Value>(Constant{t, k});
}

} // namespace

size_t unroll_loop(Function &func, const string &index) {
    // The function's name, for the messages: its entry block's.
    const string fname = func.blocks.empty() ? string("?") : func.blocks.front()->name;
    shared_ptr<Block> loop;
    for (const auto &block : func.blocks) {
        const auto *parfor =
            std::get_if<Terminator::ParFor>(&block->terminator.data);
        if (parfor != nullptr && parfor->index == index) {
            internal_assert(!loop)
                << "Two loops named " << index << " in " << fname;
            loop = block;
        }
    }
    internal_assert(loop) << "unroll(" << index << "): no loop of that name in "
                          << fname;
    const Terminator::ParFor parfor =
        std::get<Terminator::ParFor>(loop->terminator.data);
    internal_assert(!parfor.binding.has_value())
        << "unroll(" << index << ") in " << fname
        << ": the loop is bound to hardware (bind), so its iterations are what "
           "the hardware runs, not a body to write out once per iteration.";

    const std::optional<int64_t> start = constant_count(*parfor.start);
    const std::optional<int64_t> end = constant_count(*parfor.end);
    const std::optional<int64_t> stride = constant_count(*parfor.stride);
    internal_assert(start.has_value() && end.has_value() &&
                    stride.has_value() && *stride > 0)
        << "unroll(" << index << ") in " << fname
        << ": the loop's extent is not a constant. unroll() writes the body "
           "out once per iteration, so it has to know how many there are: "
           "split the loop by a constant chunk and unroll the inner loop, as "
           "`split(i, io, ii, 4, true).unroll(ii)`.";
    const int64_t count =
        *end <= *start ? 0 : (*end - *start + *stride - 1) / *stride;
    internal_assert(count >= 1)
        << "unroll(" << index << ") in " << fname
        << ": the loop runs no iterations";

    // The body hands on to the next copy at its one yield.
    const string exit = unify_yields(func, parfor.body.name);
    internal_assert(!exit.empty())
        << "unroll(" << index << ") in " << fname
        << ": the loop's body never yields";
    const Cfg region(func, parfor.body.name);
    vector<shared_ptr<Block>> blocks;
    for (const BlockId b : region.rpo) {
        blocks.push_back(region.block(b));
    }
    shared_ptr<Block> body;
    for (const auto &block : func.blocks) {
        if (block->name == parfor.body.name) {
            body = block;
        }
    }
    internal_assert(body && !body->args.empty())
        << "unroll(" << index << ") in " << fname
        << ": the body takes no index";
    const Type index_t = body->args.front().type;

    // Copy k runs iteration `start + k * stride`: the index is that constant
    // throughout it and no block of it takes the index (substitute_argument,
    // SSA/InsertPreheader.h), so a copy is entered with the arguments the
    // loop handed its body and nothing else -- from the loop's block for the
    // first, from the yield of the copy before it for the rest; the last
    // copy's yield goes where the loop's continuation went.
    const string index_name = body->args.front().name;
    shared_ptr<Block> previous;
    for (int64_t k = 0; k < count; k++) {
        const string suffix = "!unroll" + std::to_string(k);
        const std::map<string, shared_ptr<Block>> copies =
            clone_region(func, blocks, suffix);
        const shared_ptr<Block> entry = copies.at(parfor.body.name);
        const shared_ptr<Block> yield = copies.at(exit);
        for (const auto &block : blocks) {
            func.blocks.push_back(copies.at(block->name));
        }
        // The copy's yield still ends it here, so the region below its
        // entry is the copy alone.
        substitute_argument(func, entry->name, index_name,
                            index_constant(index_t, *start + k * *stride));
        const Terminator::Jump into{entry->name, parfor.body.args};
        if (k == 0) {
            loop->terminator.data = into;
        } else {
            previous->terminator.data = into;
        }
        previous = yield;
    }
    previous->terminator.data = parfor.cont;

    // The loop's own body is reached by nothing now.
    remove_unreachable_blocks(func);
    refresh_preds(func);
    return size_t(count);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
