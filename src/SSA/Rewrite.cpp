#include "SSA/Convert.h"

#include "SSA/Analysis.h"
#include "SSA/InsertPreheader.h"
#include "SSA/QueueRecursion.h"
#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <functional>
#include <iostream>
#include <optional>

namespace bonsai {
namespace ir {
namespace ssa {

using std::map;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::tuple;
using std::vector;

namespace {

template <typename T>
struct OrderedSet {
  private:
    vector<T> order;
    set<T> seen;

  public:
    void insert(T t) {
        if (seen.insert(t).second) {
            // insertion took place.
            order.emplace_back(std::move(t));
        }
    }

    vector<T> get() { return std::move(order); }
};

struct Continuation {
    // TODO: figure this out with scoping and such...
    // vector<Block> blocks;
    vector<pair<string, string>> fb_pairs; // function, block pairs
};

bool is_trivial_terminator(const BlockMap &bmap, const std::string &bname) {
    const auto &iter = bmap.find(bname);
    internal_assert(iter != bmap.cend());
    const auto &block = iter->second;
    return block->args.empty() && block->instrs.empty() &&
           (std::holds_alternative<Terminator::Yield>(block->terminator.data) ||
            std::holds_alternative<Terminator::Return>(block->terminator.data));
}

tuple<Type, Continuation> FindPath(const FuncMap &funcs,
                                   const std::string &func, const Cursor &owner,
                                   const Cursor &cursor) {
    using ContStack = vector<pair<string, string>>;
    ContStack cont_stack;
    // Each stack frame of the cursor logs the types needed for the
    // continuation, and the set of values saved. The values are for
    // deduplication. Ideally, we would track unique values through dataflow
    // analysis: this is an important TODO.
    vector<pair<vector<Type>, set<shared_ptr<Value>>>> state_stack;
    set<string> visited; // func_name + block_name

    // Current set of uniforms in DFS traversal.
    // TODO: using strings seems somewhat error-prone,
    // but can't get unique shared_ptr<Value> from Argument type
    vector<set<string>> uniforms;

    auto get_bmap = [&](const std::string &fname) {
        const auto &f = funcs.find(fname);
        internal_assert(f != funcs.cend()) << fname;
        return make_block_map(f->second);
    };

    auto flatten_state_stack = [&state_stack]() {
        vector<Type> types;
        for (const auto &ts : state_stack) {
            for (const auto &t : ts.first) {
                types.push_back(t);
            }
        }
        return types;
    };

    std::function<bool(const shared_ptr<Value> &)> is_uniform =
        [&](const shared_ptr<Value> &value) -> bool {
        return std::visit(overloads{
                              [&](const std::shared_ptr<Instruction> &i) {
                                  return uniforms.back().contains(i->name);
                              },
                              [&](const Constant &c) { return true; },
                              [&](const Argument &a) {
                                  return uniforms.back().contains(a.name);
                              },
                          },
                          value->data);
    };

    auto save_type_state =
        [&](const std::vector<std::shared_ptr<Value>> &args) {
            for (const auto &arg : args) {
                // TODO: deduplicate through all frames?
                // That requires dataflow through all blocks,
                // Similar to uniform analysis.
                if (!is_uniform(arg) &&
                    !state_stack.back().second.contains(arg)) {
                    state_stack.back().first.push_back(arg->get_type());
                    state_stack.back().second.insert(arg);
                }
            }
        };

    auto propagate_uniformity = [&](const vector<bool> &umask,
                                    const shared_ptr<Block> &block) {
        // umask says which arguments are uniform.
        internal_assert(umask.size() == block->args.size());
        uniforms.push_back({});
        for (size_t i = 0; i < block->args.size(); i++) {
            if (umask[i]) {
                uniforms.back().insert(block->args[i].name);
            }
        }

        for (const auto &instr : block->instrs) {
            if (instr->name.empty())
                continue;

            // TODO: are any instructions not uniform? e.g., an rng?
            const bool args_are_uniform = std::all_of(
                instr->operands.cbegin(), instr->operands.cend(), is_uniform);
            if (args_are_uniform) {
                uniforms.back().insert(instr->name);
            }
        }
    };

    // auto push_parfor_args_as_uniform = [&](const shared_ptr<Block>
    // &block) {
    //     // All arguments to this block are uniform *except* the first!
    //     std::vector<bool> mask(block->args.size(), true);
    //     mask[0] = false;
    //     propagate_uniformity(mask, block);
    // };

    auto make_mask = [&](const vector<shared_ptr<Value>> &values,
                         bool add_front_false) {
        vector<bool> mask;
        size_t size = values.size() + (add_front_false ? 1 : 0);
        mask.reserve(size);

        if (add_front_false) {
            mask.push_back(false);
        }
        std::transform(values.cbegin(), values.cend(), std::back_inserter(mask),
                       [&](auto v) { return is_uniform(v); });
        return mask;
    };

    using VisitorRetT = std::optional<tuple<vector<Type>, ContStack>>;

    std::function<VisitorRetT(const BlockMap &, const shared_ptr<Block> &,
                              const string &, Cursor, const vector<bool> &)>
        visit_producer = [&](const BlockMap &bmap,
                             const shared_ptr<Block> &block,
                             const std::string &fname, Cursor c,
                             const vector<bool> &mask) -> VisitorRetT {
        internal_assert(block);
        if (visited.contains(fname + block->name)) {
            return VisitorRetT{};
        }
        visited.insert(fname + block->name);
        propagate_uniformity(mask, block);

        auto visit_target = [&](const Terminator::Jump &target) {
            auto arg_mask = make_mask(target.args, false);
            return visit_producer(bmap, bmap.at(target.name), fname, c,
                                  arg_mask);
        };

        auto ret = std::visit(
            overloads{
                [&](const std::monostate &m) -> VisitorRetT {
                    funcs.at(func)->dump(std::cout);
                    internal_error << "FindPath called on unfinished block: "
                                   << block->name << " of function " << fname;
                },
                [&](const Terminator::Jump &j) -> VisitorRetT {
                    return visit_target(j);
                },
                [&](const Terminator::Dispatch &d) -> VisitorRetT {
                    internal_assert(!d.targets.empty())
                        << block->name << " of function " << fname;

                    // TODO: handle multiple paths, need union of branch
                    // returns.
                    for (size_t i = 0; i < d.targets.size(); i++) {
                        auto rec = visit_target(d.targets[i]);
                        if (rec) {
                            // TODO: handle possible merge!
                            return rec;
                        }
                    }
                    return {};
                },
                [&](const Terminator::Return &r) -> VisitorRetT { return {}; },
                [&](const Terminator::ParFor &p) -> VisitorRetT {
                    if (c.ids.front() == p.index) {
                        // Part of the cursor!
                        auto producer_block = bmap.at(p.body.name);

                        std::vector<bool> mask;

                        if (c.ids.size() == cursor.ids.size()) {
                            // This is the first producer loop! Start
                            // tracking uniforms.
                            mask =
                                std::vector<bool>(p.body.args.size() + 1, true);
                            mask[0] = false;
                        } else {
                            // Track as usual, this is a nested loop.
                            mask = make_mask(p.body.args, true);
                        }

                        c.ids.pop_front();

                        return visit_producer(bmap, producer_block, fname, c,
                                              mask);
                    } else {
                        // Not part of cursor, skip to continuation.
                        return visit_target(p.cont);
                    }
                },
                [&](const Terminator::Yield &y) -> VisitorRetT { return {}; },
                [&](const Terminator::Call &call) -> VisitorRetT {
                    if (c.ids.front() == call.call.name) {
                        // Part of the path!
                        c.ids.pop_front();

                        state_stack.push_back({});
                        save_type_state(call.cont.args);
                        cont_stack.push_back({fname, call.cont.name});

                        VisitorRetT rec;
                        if (c.ids.empty()) {
                            // Found it! Save the arguments too.
                            save_type_state(call.call.args);
                            rec = {flatten_state_stack(), cont_stack};
                        } else {
                            internal_assert(call.cont.args.empty())
                                << "TODO: support stack continuations!";
                            // TODO: also assert trivial? I guess empty arg
                            // blocks are trivial? they can return constants
                            // I guess.
                            BlockMap new_bmap = get_bmap(call.call.name);
                            const auto &fiter = funcs.find(call.call.name);
                            internal_assert(fiter != funcs.cend())
                                << call.call.name;
                            auto new_block =
                                fiter->second->blocks.front()->name;
                            auto new_mask = make_mask(call.call.args, false);
                            rec =
                                visit_producer(new_bmap, new_bmap.at(new_block),
                                               call.call.name, c, new_mask);
                        }

                        state_stack.pop_back();
                        cont_stack.pop_back();
                        return rec;
                    } else {
                        // Not part of path.
                        // TODO: is a called value uniform if all args are?
                        // can we have side-effects?? yes if rng is supported!!
                        // const bool ret_is_uniform =
                        //     std::all_of(call.cont.args.cbegin(),
                        //                 call.cont.args.cend(), is_uniform);

                        auto cont_mask = make_mask(call.cont.args, !call.drop);
                        return visit_producer(bmap, bmap.at(call.cont.name),
                                              fname, c, cont_mask);
                    }
                },
            },
            block->terminator.data);
        // Pop frame added by propogate_uniformity()
        uniforms.pop_back();
        return ret;
    };

    std::function<VisitorRetT(const BlockMap &, const shared_ptr<Block> &,
                              const string &, Cursor)>
        visit_owner = [&](const BlockMap &bmap, const shared_ptr<Block> &block,
                          const std::string &fname, Cursor o) -> VisitorRetT {
        internal_assert(block);
        if (visited.contains(fname + block->name)) {
            return VisitorRetT{};
        }
        visited.insert(fname + block->name);

        auto visit_target = [&](const Terminator::Jump &target) {
            auto arg_mask = make_mask(target.args, false);
            return visit_owner(bmap, bmap.at(target.name), fname, o);
        };

        auto ret = std::visit(
            overloads{
                [&](const std::monostate &m) -> VisitorRetT {
                    funcs.at(func)->dump(std::cout);
                    internal_error << "FindPath called on unfinished block: "
                                   << block->name << " of function " << fname;
                },
                [&](const Terminator::Jump &j) -> VisitorRetT {
                    return visit_target(j);
                },
                [&](const Terminator::Dispatch &d) -> VisitorRetT {
                    internal_assert(!d.targets.empty())
                        << block->name << " of function " << fname;

                    // TODO: handle multiple paths, need union of branch
                    // returns.
                    for (size_t i = 0; i < d.targets.size(); i++) {
                        auto rec = visit_target(d.targets[i]);
                        if (rec) {
                            // TODO: handle possible merge!
                            return rec;
                        }
                    }
                    return {};
                },
                [&](const Terminator::Return &r) -> VisitorRetT { return {}; },
                [&](const Terminator::ParFor &p) -> VisitorRetT {
                    internal_assert(!o.ids.empty());
                    internal_error << "TODO: parfor owning loop";
                },
                [&](const Terminator::Yield &y) -> VisitorRetT { return {}; },
                [&](const Terminator::Call &call) -> VisitorRetT {
                    internal_error << "TODO: call owner";
                },
            },
            block->terminator.data);
        return ret; // TODO: memoize?
    };
    auto bmap = get_bmap(func);
    const auto &fiter = funcs.find(func);
    internal_assert(fiter != funcs.cend()) << func;
    auto entry = fiter->second->blocks.front()->name;
    internal_assert(!owner.ids.empty()) << owner.to_string();
    const bool owner_is_root =
        owner.ids.size() == 1 && owner.ids.front() == "root";
    internal_assert(owner_is_root) << "TODO: handle non-root owner of defer";
    const vector<bool> true_mask(fiter->second->blocks.front()->args.size(),
                                 true);
    auto ret = visit_producer(bmap, bmap.at(entry), func, cursor, true_mask);
    internal_assert(ret) << "Failed to find cursor: " << cursor.to_string()
                         << " in func: " << func;
    // TODO: pack continuation order into "good" order?
    auto [types, pairs] = *ret;
    return {Tuple_t::make(types), Continuation{pairs}};
}

tuple<Type, vector<Continuation> /*, vector<???>*/>
FindPaths(const FuncMap &funcs, const std::string &func, const Cursor &owner,
          const vector<Cursor> &cursors) {

    vector<Type> state_ts;
    vector<Continuation> conts;
    // TODO: OrderedSet<???>

    for (const auto &c : cursors) {
        auto [state_t, cont] = FindPath(funcs, func, owner, c /*, ???*/);
        state_ts.push_back(state_t);
        conts.push_back(cont);
    }
    internal_assert(state_ts.size() == 1) << "TODO: make TaggedUnion";
    Type state_t = state_ts[0]; // MakeTaggedUnion(state_ts)
    return {state_t, conts /*, ???*/};
}

// Finds first Yield or Return block via DFS.
// Does not recurse into parfors (skips to continuation)
shared_ptr<Block> GetTerminatorBlock(const shared_ptr<Function> &func,
                                     const string &origin) {
    BlockMap bmap = make_block_map(func);

    // Memoize results.
    map<string, shared_ptr<Block>> memo;
    set<std::string> visiting;

    std::function<shared_ptr<Block>(const string &)> dfs =
        [&](const string &bname) -> shared_ptr<Block> {
        if (const auto &m = memo.find(bname); m != memo.end()) {
            return m->second;
        }
        internal_assert(bmap.contains(bname)) << bname;

        if (visiting.contains(bname)) {
            // Cycle.
            return nullptr;
        }

        const auto block = bmap.at(bname);

        visiting.insert(bname);

        auto ret = std::visit(
            overloads{
                [&](const std::monostate &m) -> shared_ptr<Block> {
                    func->dump(std::cout);
                    internal_error
                        << "GetTerminatorBlock called on unfinished block: "
                        << bname;
                },
                [&](const Terminator::Jump &j) -> shared_ptr<Block> {
                    return dfs(j.name);
                },
                [&](const Terminator::Dispatch &d) -> shared_ptr<Block> {
                    internal_assert(!d.targets.empty()) << bname;
                    shared_ptr<Block> first = dfs(d.targets[0].name);
                    for (size_t i = 1; i < d.targets.size(); i++) {
                        shared_ptr<Block> next = dfs(d.targets[i].name);
                        if (first == nullptr && next != nullptr) {
                            // Track only valid blocks.
                            first = next;
                        }
                        internal_assert(first == next || next == nullptr)
                            << "Diverging CF in terminator of: " << bname;
                    }
                    return first;
                },
                [&](const Terminator::Return &r) -> shared_ptr<Block> {
                    return block;
                },
                [&](const Terminator::ParFor &p) -> shared_ptr<Block> {
                    return dfs(p.cont.name);
                },
                [&](const Terminator::Yield &y) -> shared_ptr<Block> {
                    return block;
                },
                [&](const Terminator::Call &c) -> shared_ptr<Block> {
                    return dfs(c.cont.name);
                },
            },
            block->terminator.data);
        visiting.erase(bname);
        memo[bname] = ret;
        return ret;
    };

    return dfs(origin);
}

} // namespace

std::string Cursor::to_string() const {
    std::string s;
    for (const auto &id : ids) {
        if (!s.empty()) {
            s += ".";
        }
        s += id;
    }
    return s;
}

void split(FuncMap &funcs, string func, string idx, int factor, string outer,
           string inner, bool exact) {
    internal_assert(funcs.contains(func))
        << "split applied to unknown func:" << func;
    auto f = funcs[func];

    vector<shared_ptr<Block>> blocks;

    for (auto &block : f->blocks) {
        blocks.push_back(block);
        if (!std::holds_alternative<Terminator::ParFor>(
                block->terminator.data)) {

            continue;
        }
        Terminator::ParFor parfor =
            std::get<Terminator::ParFor>(block->terminator.data);
        if (parfor.index != idx) {
            continue;
        }

        internal_assert(exact) << "TODO: handle guards inside split()";

        // Whether an index this loop would have visited is one the split
        // still visits. Without a tail the two loops together cover
        // start, start+factor, start+2*factor, ... and each chunk walks
        // `factor` wide by the original stride, so the cover is exact only if
        // the chunk is a whole number of strides and the range is a whole
        // number of chunks. Get either wrong and the split runs off the end of
        // what it was asked to iterate, which is a wrong answer rather than a
        // slow one -- it writes past the range the program reasoned about.
        const auto as_int =
            [](const std::shared_ptr<Value> &v) -> std::optional<int64_t> {
            const auto *c = std::get_if<Constant>(&v->data);
            if (c == nullptr) {
                return std::nullopt;
            }
            if (const auto *i = std::get_if<int64_t>(&c->data)) {
                return *i;
            }
            return std::nullopt;
        };

        const auto stride_n = as_int(parfor.stride);
        internal_assert(stride_n.has_value())
            << "split(" << idx << ") on " << func
            << " needs a constant stride to know that its chunks line up with "
               "the steps the loop takes";
        internal_assert(*stride_n > 0 && factor % *stride_n == 0)
            << "split(" << idx << ", " << factor << ") on " << func
            << " does not divide the loop's stride of " << *stride_n
            << ", so a chunk would start part way through a step";

        // A range only known at run time is the caller's assertion to make:
        // that is what asking for no tail means. One known here is checked.
        const auto start_n = as_int(parfor.start);
        const auto end_n = as_int(parfor.end);
        if (start_n.has_value() && end_n.has_value()) {
            internal_assert((*end_n - *start_n) % factor == 0)
                << "split(" << idx << ", " << factor << ") on " << func
                << " does not divide the loop's range of [" << *start_n << ":"
                << *end_n << "), so without a tail it would run "
                << (factor - (*end_n - *start_n) % factor)
                << " iterations past the end";
        }

        // The body's index becomes the inner loop's, and is renamed to say so.
        // parfor i in start:end:stride body(i) cont()
        // ->
        // parfor outer in start:end:factor inner_loop(outer) cont()
        // block inner_loop(o):
        //   parfor inner in 0:factor:stride step(inner, o) inner_cont()
        // block step(n, o): body(o + n)
        // block inner_cont(): yield
        //
        // The inner loop counts from zero and the body is handed the sum,
        // rather than the inner loop counting from `outer` and the body being
        // handed its index directly. The second is one instruction cheaper,
        // but it makes the inner range depend on the outer index, and a range
        // that does has no single trip count -- so nothing could collapse the
        // two loops afterwards, or reason about the inner one on its own. The
        // Stmt-level split makes the same choice.
        Type itype = parfor.start->get_type();
        auto split_factor = std::make_shared<Value>(Constant{itype, factor});
        auto zero = std::make_shared<Value>(Constant{itype, int64_t(0)});

        // TODO: truly unique name generation?
        shared_ptr<Block> outer_yield = std::make_shared<Block>();
        outer_yield->name = parfor.body.name + "_yield_" + outer;
        outer_yield->terminator.data = Terminator::Yield{};
        outer_yield->owner = f;

        Argument outer_arg{itype, outer};
        auto v_outer_arg = std::make_shared<Value>(outer_arg);

        // The arguments the loop was already threading into its body, which
        // both new blocks have to carry so that they still arrive. Taken from
        // what the body block declares rather than from what the jump passes:
        // a jump may pass a constant or an instruction's result, which has no
        // Argument to copy, and dropping those left the blocks below with
        // fewer parameters than their callers supply.
        const BlockMap bmap = make_block_map(f);
        internal_assert(bmap.contains(parfor.body.name))
            << func << " has no block " << parfor.body.name;
        const auto &loop_body = bmap.at(parfor.body.name);
        internal_assert(!loop_body->args.empty())
            << parfor.body.name << " has no index argument";
        const std::vector<Argument> carried(loop_body->args.begin() + 1,
                                            loop_body->args.end());
        internal_assert(carried.size() == parfor.body.args.size())
            << parfor.body.name << " takes " << loop_body->args.size()
            << " arguments but the loop passes it "
            << (parfor.body.args.size() + 1);

        internal_assert(std::holds_alternative<Constant>(parfor.stride->data))
            << "TODO: handle non-Constant strides in split mining";

        // Where the two indices become the one the body expects.
        shared_ptr<Block> step = std::make_shared<Block>();
        step->name = parfor.body.name + "_step_" + inner;
        step->owner = f; // This *MUST* exist before make_instruction
        const Argument inner_arg{itype, inner};
        auto v_inner = step->add_argument(inner_arg);
        auto v_outer = step->add_argument(outer_arg);
        for (const Argument &arg : carried) {
            step->add_argument(arg);
        }
        auto absolute = step->make_instruction(itype, Instruction::Op::Add,
                                               {v_outer, v_inner});
        std::vector<shared_ptr<Value>> to_body = {absolute};
        for (const Argument &arg : carried) {
            to_body.push_back(std::make_shared<Value>(arg));
        }
        step->terminator.data =
            Terminator::Jump{parfor.body.name, std::move(to_body)};

        shared_ptr<Block> inner_loop = std::make_shared<Block>();
        inner_loop->name = parfor.body.name + "_split_" + outer;
        inner_loop->owner = f;
        inner_loop->add_argument(outer_arg);
        for (const Argument &arg : carried) {
            inner_loop->add_argument(arg);
        }

        std::vector<shared_ptr<Value>> to_step = {v_outer_arg};
        for (const Argument &arg : carried) {
            to_step.push_back(std::make_shared<Value>(arg));
        }
        inner_loop->terminator.data =
            Terminator::ParFor{inner,
                               zero,
                               split_factor,
                               parfor.stride,
                               Terminator::Jump{step->name, std::move(to_step)},
                               Terminator::Jump{outer_yield->name}};

        // TODO: lookups? or are those only necessary in construction?
        inner_loop->preds = {block};
        outer_yield->preds = {inner_loop};

        blocks.push_back(inner_loop);
        blocks.push_back(step);
        blocks.push_back(outer_yield);

        // TODO: fix loop body predecessors?

        block->terminator.data = Terminator::ParFor{
            outer,
            parfor.start,
            parfor.end,
            split_factor,
            Terminator::Jump{inner_loop->name, parfor.body.args},
            parfor.cont};
    }
    if (blocks.size() == f->blocks.size()) {
        internal_error << "Did not find loop: " << idx
                       << " in function: " << func;
    }
    f->blocks = std::move(blocks);
}

namespace {

// Does `func` call itself?
bool is_recursive(const Function &func) {
    for (const auto &block : func.blocks) {
        const auto *call =
            std::get_if<Terminator::Call>(&block->terminator.data);
        if (call != nullptr && call->call.name == func.blocks.front()->name) {
            return true;
        }
    }
    return false;
}

// The functions that call themselves, among those `start` can reach.
//
// The schedule names the function the programmer wrote, but the recursion is
// not always in it. A tree query is lowered into a traversal function of its
// own before any of this runs, so `trace.loopify(64)` names a function whose
// body is a call to the traversal, and it is the traversal that recurses.
std::set<std::string> recursive_functions_from(const FuncMap &funcs,
                                               const std::string &start) {
    std::set<std::string> found;
    std::set<std::string> seen;
    std::vector<std::string> work{start};
    while (!work.empty()) {
        const std::string name = work.back();
        work.pop_back();
        if (!seen.insert(name).second) {
            continue;
        }
        const auto it = funcs.find(name);
        if (it == funcs.end()) {
            continue; // an extern, or something not compiled here
        }
        if (is_recursive(*it->second)) {
            found.insert(name);
        }
        for (const auto &block : it->second->blocks) {
            if (const auto *call =
                    std::get_if<Terminator::Call>(&block->terminator.data)) {
                work.push_back(call->call.name);
            }
        }
    }
    return found;
}

} // namespace

void loopify(FuncMap &funcs, std::string func, int size) {
    internal_assert(funcs.contains(func))
        << "loopify applied to unknown func:" << func;
    auto f = funcs[func];

    if (size > 0) {
        // A recursion that branches cannot become a loop by itself, since the
        // second call has to happen after the first one comes back. It is put
        // on an explicit stack instead (see SSA/QueueRecursion.h).
        const std::set<std::string> targets =
            recursive_functions_from(funcs, func);
        internal_assert(!targets.empty())
            << "loopify(" << size << ") on " << func << ": neither it nor "
            << "anything it calls is recursive, so there is nothing to put on "
            << "a stack";
        for (const std::string &target : targets) {
            queue_recursion(*funcs[target], size_t(size));

            // The target is a loop now, and it is only a function at all
            // because Lower/RecLoops.cpp had to extract the recursion into
            // one. Left standing it costs more than the call: what the
            // traversal folds into is the caller's, so every update to it
            // becomes a store through a pointer and reading the answer back
            // becomes a load per field. Putting it back where it came from is
            // what lets that stay in registers, and the recursion that stopped
            // it from happening is exactly what was just removed.
            auto &attrs = funcs[target]->attributes;
            if (std::find(attrs.begin(), attrs.end(),
                          ir::Function::Attribute::always_inlined) ==
                attrs.end()) {
                attrs.push_back(ir::Function::Attribute::always_inlined);
            }
        }
        return;
    }

    // Find any tail calls with empty return continuations and convert them into
    // jumps.

    BlockMap bmap = make_block_map(f);

    // The blocks whose tail call becomes a back edge, i.e. the latches of the
    // loop this leaves behind.
    std::set<std::string> latches;

    for (auto &block : f->blocks) {
        if (!std::holds_alternative<Terminator::Call>(block->terminator.data)) {
            continue;
        }

        Terminator::Call call =
            std::get<Terminator::Call>(block->terminator.data);
        if (call.call.name != func) {
            // Not a tail call.
            continue;
        }

        internal_assert(call.cont.args.empty())
            << "Cannot loopify tail-call in: " << func
            << ", has continuation arguments to: " << call.cont.name;

        internal_assert(bmap.contains(call.cont.name))
            << "BlockMap for " << func
            << " does not contain continutation target: " << call.cont.name;

        const auto cont = bmap.at(call.cont.name);

        // The call has to be the last thing the function does, so that going
        // round the loop again is the same as making it. That means a
        // continuation which does nothing but return -- either returning
        // nothing, or returning exactly what the call produced, which is
        // `return f(...)` and is just as much a tail call.
        const auto *returns =
            std::get_if<Terminator::Return>(&cont->terminator.data);
        bool is_tail = cont->instrs.empty() && returns != nullptr;
        if (is_tail && !call.drop) {
            // A call whose result is kept hands it to the continuation as a
            // leading argument; returning that argument, and nothing else, is
            // what makes this a tail call rather than a use of the result.
            is_tail = cont->args.size() == 1 && returns->value != nullptr &&
                      std::holds_alternative<Argument>(returns->value->data) &&
                      std::get<Argument>(returns->value->data).name ==
                          cont->args[0].name;
        }
        internal_assert(is_tail)
            << "Cannot loopify the call to " << func << " in " << block->name
            << ": its continuation " << call.cont.name << " does more than "
            << "return, so the call is not in tail position";

        // Replace call terminator with direct jump.
        block->terminator.data = call.call;
        latches.insert(block->name);

        // Remove block as predecessor to continuation block.
        std::erase_if(cont->preds, [&](const auto &p) {
            const auto ptr = p.lock();
            internal_assert(ptr);
            return ptr->name == block->name;
        });

        // Add block as predecessor to entry block.
        f->blocks[0]->preds.push_back(block);
    }

    if (latches.empty()) {
        return;
    }

    // The back edges close on the entry block, which makes the entry the loop
    // header -- and then the values carried around the loop are the function's
    // own parameters, reassigned on every iteration. A parameter is not
    // storage, so nothing can be assigned to it; the loop needs a header of
    // its own, entered from a preheader (see SSA/InsertPreheader.h).
    insert_preheader(*f, f->blocks[0]->name, latches);
}

// TODO: make this accept non-constant sizes!
void defer(FuncMap &funcs, const string &func, const Queue_t &queue_t,
           const vector<Cursor> &cursors) {
    internal_assert(funcs.contains(func))
        << "defer applied to unknown func:" << func;
    auto [state_type, conts] = FindPaths(funcs, func, queue_t.owner, cursors);

    std::cout << state_type << std::endl;
    for (const auto &c : conts) {
        std::cout << "Continuation: {";
        bool first = true;
        for (const auto &p : c.fb_pairs) {
            if (!first) {
                std::cout << ", ";
            }
            first = false;
            std::cout << "(" << p.first << ", " << p.second << ")";
        }
        std::cout << "}\n";
    }

    std::cerr << "here\n";
    exit(-1);
}

} // namespace ssa
} // namespace ir
} // namespace bonsai
