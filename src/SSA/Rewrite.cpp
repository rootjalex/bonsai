#include "SSA/Convert.h"

#include "SSA/Rewrite.h"
#include "SSA/SSA.h"

#include "IR/Analysis.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Lower/Intrinsics.h"

#include "Utils.h"

#include <iostream>

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

using BlockMap = map<string, shared_ptr<Block>>;

BlockMap make_block_map(const shared_ptr<Function> &func) {
    BlockMap bmap;
    for (const auto &block : func->blocks) {
        bmap[block->name] = block;
    }
    return bmap;
}

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

tuple<Type, Continuation>
FindPath(const FuncMap &funcs, const std::string &func, const Cursor &cursor) {
    using ContStack = vector<pair<string, string>>;
    ContStack cont_stack;
    vector<vector<Type>> state_stack;
    set<string> visited; // func_name + block_name

    auto get_bmap = [&](const std::string &fname) {
        const auto &f = funcs.find(fname);
        internal_assert(f != funcs.cend()) << fname;
        return make_block_map(f->second);
    };

    auto flatten_state_stack = [&state_stack]() {
        vector<Type> types;
        for (const auto &ts : state_stack) {
            for (const auto &t : ts) {
                types.push_back(t);
            }
        }
        return types;
    };

    auto save_type_state =
        [&](const std::vector<std::shared_ptr<Value>> &args) {
            for (const auto &arg : args) {
                // TODO: don't save uniform vars! But need to track
                // them somewhere else...
                // TODO: deduplicate!! this tracks all live state...
                if (!std::holds_alternative<Constant>(arg->data)) {
                    state_stack.back().push_back(arg->get_type());
                }
            }
        };

    using VisitorRetT = std::optional<tuple<vector<Type>, ContStack>>;

    std::function<VisitorRetT(const BlockMap &, const shared_ptr<Block> &,
                              const string &fname, Cursor c)>
        visitor = [&](const BlockMap &bmap, const shared_ptr<Block> &block,
                      const std::string &fname, Cursor c) -> VisitorRetT {
        internal_assert(block);
        if (visited.contains(fname + block->name)) {
            return VisitorRetT{};
        }
        visited.insert(fname + block->name);

        auto ret = std::visit(
            overloads{
                [&](const std::monostate &m) -> VisitorRetT {
                    funcs.at(func)->dump(std::cout);
                    internal_error << "FindPath called on unfinished block: "
                                   << block->name << " of function " << fname;
                },
                [&](const Terminator::Jump &j) -> VisitorRetT {
                    return visitor(bmap, bmap.at(j.name), fname, c);
                },
                [&](const Terminator::Dispatch &d) -> VisitorRetT {
                    internal_assert(!d.targets.empty())
                        << block->name << " of function " << fname;

                    // TODO: handle multiple paths, need union of branch
                    // returns.
                    for (size_t i = 0; i < d.targets.size(); i++) {
                        auto rec =
                            visitor(bmap, bmap.at(d.targets[i].name), fname, c);
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
                        c.ids.pop_front();
                        return visitor(bmap, bmap.at(p.body.name), fname, c);
                    }
                    return visitor(bmap, bmap.at(p.cont.name), fname, c);
                },
                [&](const Terminator::Yield &y) -> VisitorRetT { return {}; },
                [&](const Terminator::Call &call) -> VisitorRetT {
                    if (c.ids.front() == call.call.name) {
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
                            // blocks are trivial? they can return constants I
                            // guess.
                            BlockMap new_bmap = get_bmap(call.call.name);
                            const auto &fiter = funcs.find(call.call.name);
                            internal_assert(fiter != funcs.cend())
                                << call.call.name;
                            auto new_block =
                                fiter->second->blocks.front()->name;
                            rec = visitor(new_bmap, new_bmap.at(new_block),
                                          call.call.name, c);
                        }

                        state_stack.pop_back();
                        cont_stack.pop_back();
                        return rec;
                    }
                    return visitor(bmap, bmap.at(call.cont.name), fname, c);
                },
            },
            block->terminator.data);
        return ret; // TODO: memoize?
    };
    auto bmap = get_bmap(func);
    const auto &fiter = funcs.find(func);
    internal_assert(fiter != funcs.cend()) << func;
    auto entry = fiter->second->blocks.front()->name;
    auto ret = visitor(bmap, bmap.at(entry), func, cursor);
    internal_assert(ret) << "Failed to find cursor: " << cursor.to_string()
                         << " in func: " << func;
    // TODO: pack continuation order into "good" order?
    auto [types, pairs] = *ret;
    return {Tuple_t::make(types), Continuation{pairs}};
}

tuple<Type, vector<Continuation> /*, vector<???>*/>
FindPaths(const FuncMap &funcs, const std::string &func,
          const vector<Cursor> &cursors) {

    vector<Type> state_ts;
    vector<Continuation> conts;
    // TODO: OrderedSet<???>

    for (const auto &c : cursors) {
        auto [state_t, cont] = FindPath(funcs, func, c /*, ???*/);
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
    internal_assert(funcs.contains(func)) << func;
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

        // parfor i in start:end:stride body(i) cont()
        // ->
        // parfor outer in start:end:factor inner_loop(outer) cont()
        // block inner_loop(o):
        //  parfor inner in o:o+factor:stride body(i) inner_cont()
        // block inner_cont(): yield

        Type itype = parfor.start->get_type();
        auto split_factor = std::make_shared<Value>(Constant{itype, factor});

        // TODO: truly unique name generation?
        shared_ptr<Block> outer_yield = std::make_shared<Block>();
        outer_yield->name = parfor.body.name + "_yield_" + outer;
        outer_yield->terminator.data = Terminator::Yield{};
        outer_yield->owner = f;

        shared_ptr<Block> inner_loop = std::make_shared<Block>();
        inner_loop->name = parfor.body.name + "_split_" + outer;
        inner_loop->owner = f; // This *MUST* exist before make_instruction
        Argument outer_arg{itype, outer};
        auto v_outer_arg = std::make_shared<Value>(outer_arg);
        auto inner_end = inner_loop->make_instruction(
            itype, Instruction::Op::Add, {v_outer_arg, split_factor});

        internal_assert(std::holds_alternative<Constant>(parfor.stride->data))
            << "TODO: handle non-Constant strides in split mining";

        inner_loop->terminator.data = Terminator::ParFor{
            inner,         v_outer_arg, inner_end,
            parfor.stride, parfor.body, Terminator::Jump{outer_yield->name}};
        inner_loop->args.push_back(outer_arg);
        for (const auto &arg : parfor.body.args) {
            auto as_arg = arg->get_argument();
            if (as_arg.has_value()) {
                inner_loop->args.push_back(*as_arg);
            }
        }

        // TODO: lookups? or are those only necessary in construction?
        inner_loop->preds = {block};
        outer_yield->preds = {inner_loop};

        blocks.push_back(inner_loop);
        blocks.push_back(outer_yield);

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

// TODO: make this accept non-constant sizes!
void defer(FuncMap &funcs, const string &func, const Queue_t &queue_t,
           const vector<Cursor> &cursors) {
    auto [state_type, conts] = FindPaths(funcs, func, cursors);

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
