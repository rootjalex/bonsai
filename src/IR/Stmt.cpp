#include "IR/Stmt.h"

#include "IR/Equality.h"
#include "IR/Printer.h"
#include "Utils.h"

#include <algorithm>

namespace bonsai {
namespace ir {

Stmt CallStmt::make(Expr func, std::vector<Expr> args) {
    internal_assert(func.defined()) << "CallStmt::make received undefined func";
    internal_assert(std::all_of(args.cbegin(), args.cend(),
                                [](const Expr &e) { return e.defined(); }))
        << "CallStmt::make received undefined arg to func: " << func;

    CallStmt *node = new CallStmt;
    const Function_t *f = func.type().as<Function_t>();
    internal_assert(f) << "CallStmt::make received bad function type: "
                       << func.type();

    node->func = std::move(func);
    node->args = std::move(args);
    return node;
}

Stmt MultiRecurse::make(Expr func, std::vector<Expr> args,
                        std::vector<size_t> varying_at,
                        std::vector<std::vector<Expr>> varying,
                        std::vector<Expr> keys) {
    internal_assert(func.defined())
        << "MultiRecurse::make received undefined func";
    const Function_t *f = func.type().as<Function_t>();
    internal_assert(f) << "MultiRecurse::make received bad function type: "
                       << func.type();
    internal_assert(std::all_of(args.cbegin(), args.cend(),
                                [](const Expr &e) { return e.defined(); }))
        << "MultiRecurse::make received undefined arg to func: " << func;
    internal_assert(!varying.empty())
        << "MultiRecurse::make received no calls to make";

    for (size_t at : varying_at) {
        internal_assert(at < args.size())
            << "MultiRecurse::make varying position " << at
            << " is past the end of a " << args.size() << "-argument call";
    }
    for (const auto &vs : varying) {
        internal_assert(vs.size() == varying_at.size())
            << "MultiRecurse::make received " << vs.size() << " varying values "
            << "for " << varying_at.size() << " varying positions";
        internal_assert(std::all_of(vs.cbegin(), vs.cend(),
                                    [](const Expr &e) { return e.defined(); }))
            << "MultiRecurse::make received an undefined varying value";
    }

    internal_assert(keys.empty() || keys.size() == varying.size())
        << "MultiRecurse::make received " << keys.size() << " sort keys for "
        << varying.size() << " calls";
    internal_assert(std::all_of(keys.cbegin(), keys.cend(),
                                [](const Expr &e) { return e.defined(); }))
        << "MultiRecurse::make received an undefined sort key";

    MultiRecurse *node = new MultiRecurse;
    node->func = std::move(func);
    node->args = std::move(args);
    node->varying_at = std::move(varying_at);
    node->varying = std::move(varying);
    node->keys = std::move(keys);
    return node;
}

std::vector<Expr> MultiRecurse::call_args(size_t c) const {
    internal_assert(c < varying.size())
        << "MultiRecurse::call_args asked for call " << c << " of "
        << varying.size();
    std::vector<Expr> result = args;
    for (size_t k = 0; k < varying_at.size(); k++) {
        result[varying_at[k]] = varying[c][k];
    }
    return result;
}

Stmt Print::make(std::vector<Expr> args) {
    internal_assert(std::all_of(args.cbegin(), args.cend(), [](const Expr &e) {
        return e.defined();
    })) << "Print::make received undefined arg.";
    Print *node = new Print;
    node->args = std::move(args);
    return node;
}

Stmt Return::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Return::make";
    Return *node = new Return;
    node->value = std::move(value);
    return node;
}

Stmt Return::make() {
    Return *node = new Return;
    node->value = ir::Expr();
    return node;
}

// Stmt LetStmt::make(std::string name, Expr value, Stmt body) {
Stmt LetStmt::make(WriteLoc loc, Expr value) {
    internal_assert(loc.defined())
        << "Undefined write location in LetStmt::make";
    internal_assert(value.defined()) << "Undefined value in LetStmt::make";
    // internal_assert(body.defined()) << "Undefined body in LetStmt::make";
    LetStmt *node = new LetStmt;
    node->loc = std::move(loc);
    node->value = std::move(value);
    // node->body = std::move(body);
    return node;
}

Stmt IfElse::make(Expr cond, Stmt then_body, Stmt else_body) {
    internal_assert(cond.defined()) << "Undefined condition in IfElse::make";
    internal_assert(cond.type().defined() &&
                    (cond.type().is_bool() || cond.type().is<Option_t>()))
        << "Non-boolean condition in IfElse::make: " << cond << " of type "
        << cond.type();
    if (cond.type().is<Option_t>()) {
        cond = Cast::make(Bool_t::make(), cond);
    }
    internal_assert(then_body.defined())
        << "Undefined then_body in IfElse::make";
    IfElse *node = new IfElse;
    node->cond = std::move(cond);
    node->then_body = std::move(then_body);
    node->else_body = std::move(else_body);
    return node;
}

Stmt SwitchStmt::make(Expr value, std::vector<Stmt> arms,
                      std::vector<Provenance> provenance) {
    internal_assert(value.defined()) << "Undefined value in SwitchStmt::make";
    internal_assert(value.type().defined() && value.type().is_int_or_uint())
        << "Non-integer value in SwitchStmt::make: " << value << " of type "
        << value.type();
    internal_assert(arms.size() >= 2)
        << "A switch on " << value << " with " << arms.size()
        << " arm(s); one arm is just that arm";
    internal_assert(provenance.empty() || provenance.size() == arms.size())
        << "A switch on " << value << " with " << arms.size()
        << " arms and the provenance of " << provenance.size();
    SwitchStmt *node = new SwitchStmt;
    node->value = std::move(value);
    node->arms = std::move(arms);
    node->provenance = std::move(provenance);
    return node;
}

Stmt DoWhile::make(Stmt body, Expr cond) {
    internal_assert(body.defined()) << "Undefined body in DoWhile::make";
    internal_assert(cond.defined()) << "Undefined condition in DoWhile::make";
    internal_assert(cond.type().defined() && cond.type().is_bool())
        << "Non-boolean condition in DoWhile::make: " << cond << " of type "
        << cond.type();

    DoWhile *node = new DoWhile;
    node->body = std::move(body);
    node->cond = std::move(cond);
    return node;
}

Stmt While::make(Expr cond, Stmt body) {
    internal_assert(cond.defined()) << "Undefined condition in While::make";
    internal_assert(cond.type().defined() && cond.type().is_bool())
        << "Non-boolean condition in While::make: " << cond << " of type "
        << cond.type();
    internal_assert(body.defined()) << "Undefined body in While::make";

    While *node = new While;
    node->cond = std::move(cond);
    node->body = std::move(body);
    return node;
}

Stmt Sequence::make(std::vector<Stmt> stmts) {
    internal_assert(!stmts.empty()) << "Empty stmts in Sequence::make";
    if (stmts.size() == 1) {
        return stmts[0];
    }
    Sequence *node = new Sequence;
    node->stmts = std::move(stmts);
    return node;
}

Stmt Allocate::make(WriteLoc loc, Memory memory) {
    internal_assert(loc.defined())
        << "Undefined write location in Allocate::make";
    internal_assert(loc.accesses.empty())
        << "Allocate::make must be a base write location: " << loc;

    if (memory == Memory::Heap) {
        // Try to put small things on the stack.
        memory = (loc.type.defined() && loc.type.is_stack_allocatable())
                     ? Memory::Stack
                     : Memory::Heap;
    }

    Allocate *node = new Allocate;
    node->loc = std::move(loc);
    node->value = Expr();
    node->memory = memory;
    return node;
}

Stmt Allocate::make(WriteLoc loc, Expr value, Memory memory, bool unaliased) {
    internal_assert(loc.defined())
        << "Undefined write location in Allocate::make";
    internal_assert(loc.accesses.empty())
        << "Allocate::make must be a base write location: " << loc;

    if (memory == Memory::Heap) {
        // Try to put small things on the stack.
        memory = (loc.type.defined() && loc.type.is_stack_allocatable())
                     ? Memory::Stack
                     : Memory::Heap;
    }

    Allocate *node = new Allocate;
    node->loc = std::move(loc);
    node->value = std::move(value);
    node->memory = memory;
    node->unaliased = unaliased;
    return node;
}

Stmt Free::make(Expr var) {
    internal_assert(var.defined()) << "Undefined var in Free::make";
    Free *node = new Free;
    ir::Type type = var.type();
    internal_assert((type.is<Array_t>() ||
                     (type.is<Ptr_t>() && type.element_of().is<Struct_t>())))
        << "unexpected type in Free::make, " << type;
    node->value = std::move(var);
    return node;
}

namespace {

// Does a value of this type hold one element per lane of a `lanes`-wide mask:
// a vector that wide, or an aggregate whose every leaf is one? The latter is
// how a gang carries a struct -- as a struct of gang-wide fields -- and a
// masked store of one writes each field under the mask (see
// CodeGen_LLVM::create_masked_store_at).
bool has_lanes(const Type &type, uint32_t lanes) {
    if (type.is_vector()) {
        return type.lanes() == lanes;
    }
    if (const Struct_t *s = type.as<Struct_t>()) {
        return std::all_of(s->fields.begin(), s->fields.end(),
                           [&](const TypedVar &f) {
                               return has_lanes(f.type, lanes);
                           });
    }
    if (const Array_t *a = type.as<Array_t>()) {
        return has_lanes(a->etype, lanes);
    }
    return false;
}

// Does any part of `type` hold one value per lane of a gang of `lanes`? A
// value none of whose parts does is uniform -- the same for every lane -- and
// a masked store of it writes it once if any lane is on. What is neither is
// an aggregate only partly widened, which is a mistake.
bool mentions_lanes(const Type &type, uint32_t lanes) {
    if (type.is_vector()) {
        return type.lanes() == lanes;
    }
    if (const Struct_t *s = type.as<Struct_t>()) {
        return std::any_of(s->fields.begin(), s->fields.end(),
                           [&](const TypedVar &f) {
                               return mentions_lanes(f.type, lanes);
                           });
    }
    if (const Array_t *a = type.as<Array_t>()) {
        return mentions_lanes(a->etype, lanes);
    }
    return false;
}

} // namespace

Stmt Store::make(WriteLoc loc, Expr value, Expr mask, bool compact) {
    internal_assert(loc.defined()) << "Undefined write location in Store::make";
    internal_assert(value.defined()) << "Undefined value in Store::make";
    internal_assert(!compact || mask.defined())
        << "A compacting store has to say which lanes it compacts: " << loc;
    if (mask.defined() && mask.type().defined() && value.type().defined()) {
        internal_assert(mask.type().is_bool() && mask.type().is_vector())
            << "Store mask must be a boolean vector, got: " << mask.type();
        // Per lane, or uniform (stored once if any lane is on); never half
        // of each.
        internal_assert(has_lanes(value.type(), mask.type().lanes()) ||
                        !mentions_lanes(value.type(), mask.type().lanes()))
            << "Store mask has " << mask.type().lanes()
            << " lanes but stores a value of type: " << value.type();
    }
    Store *node = new Store;
    node->loc = std::move(loc);
    node->value = std::move(value);
    node->mask = std::move(mask);
    node->compact = compact;
    return node;
}

Stmt Accumulate::make(WriteLoc loc, OpType op, Expr value, bool atomic) {
    internal_assert(loc.defined())
        << "Undefined write location in Accumulate::make";
    internal_assert(value.defined()) << "Undefined value in Accumulate::make";
    Accumulate *node = new Accumulate;
    node->loc = std::move(loc);
    node->op = op;
    node->value = std::move(value);
    node->atomic = atomic;
    return node;
}

Stmt Label::make(std::string name, Stmt body) {
    internal_assert(!name.empty()) << "Label::make received empty name";

    Label *node = new Label;
    node->name = std::move(name);
    node->body = std::move(body);
    return node;
}

Stmt RecLoop::make(std::vector<Arg> args, Stmt body) {
    internal_assert(body.defined()) << "RecLoop::make received undefined body";

    for (const auto &arg : args) {
        internal_assert(!arg.var.name.empty())
            << "RecLoop::make received empty arg name";
        internal_assert(arg.var.type.defined())
            << "RecLoop::make received undefined arg type: " << arg.var.name;
        internal_assert(arg.init.defined())
            << "RecLoop::make received no starting value for: " << arg.var.name;
        internal_assert(equals(arg.init.type(), arg.var.type))
            << "RecLoop::make starts " << arg.var.name << " : " << arg.var.type
            << " at " << arg.init << ", which is a " << arg.init.type();
    }

    RecLoop *node = new RecLoop;
    node->args = std::move(args);
    node->body = std::move(body);
    return node;
}

Stmt RecLoop::make(const std::vector<TypedVar> &vars, Stmt body) {
    std::vector<Arg> args;
    args.reserve(vars.size());
    for (const TypedVar &var : vars) {
        args.push_back(Arg{var, Expr(var)});
    }
    return make(std::move(args), std::move(body));
}

Stmt MatchVariant::make(Expr value, std::vector<Arm> arms) {
    internal_assert(value.defined())
        << "MatchVariant::make received no value to match on";
    const ADT_t *adt = value.type().as<ADT_t>();
    internal_assert(adt) << "MatchVariant::make received a non-ADT value: "
                         << value << " of type " << value.type();
    internal_assert(!arms.empty())
        << "MatchVariant::make received no arms for " << adt->name;

    // Every variant once, and nothing else: a match that leaves one out has
    // no answer for a value that is it, and one that names a variant twice
    // has two.
    std::set<std::string> seen;
    for (const Arm &arm : arms) {
        const auto index = adt->index_of(arm.variant);
        internal_assert(index.has_value())
            << adt->name << " has no variant called " << arm.variant;
        internal_assert(seen.insert(arm.variant).second)
            << "Variant " << arm.variant << " of " << adt->name
            << " is matched twice";
        // An arm names every field of its variant, or none of them and reads
        // what it needs through `Unwrap` -- which is how a match built by the
        // compiler rather than written in the source reaches them.
        internal_assert(arm.bindings.empty() ||
                        arm.bindings.size() == adt->fields(*index).size())
            << arm.variant << " has " << adt->fields(*index).size()
            << " fields but the arm names " << arm.bindings.size();
        internal_assert(arm.body.defined())
            << "Arm for " << arm.variant << " has no body";
    }
    for (size_t i = 0; i < adt->variants.size(); i++) {
        internal_assert(seen.count(adt->variant_name(i)))
            << "Match on " << adt->name << " does not say what to do when it "
            << "is " << adt->variant_name(i);
    }

    MatchVariant *node = new MatchVariant;
    node->value = std::move(value);
    node->arms = std::move(arms);
    return node;
}

Stmt Match::make(Expr loc, Match::Arms arms) {
    return Match::make(std::move(loc), std::move(arms), Expr());
}

Stmt Match::make(Expr loc, Match::Arms arms, Expr volume_map) {
    internal_assert(loc.defined()) << "Undefined match location in Match::make";
    internal_assert(!arms.empty()) << "Received no match arms in Match::make";
    const BVH_t *bvh = loc.type().as<BVH_t>();
    internal_assert(bvh) << "Match is only implemented for BVH_t, received: "
                         << loc;
    internal_assert(bvh->nodes.size() == arms.size())
        << "Incorrect number of match arms for BVH type: " << loc.type()
        << " with " << arms.size() << " arms.";
    // Make sure all match arms exist.
    const size_t n = bvh->nodes.size();
    for (size_t i = 0; i < n; i++) {
        std::string_view name = bvh->nodes[i].name();
        const bool found =
            arms.cend() !=
            std::find_if(arms.cbegin(), arms.cend(), [&name](const auto &arm) {
                return arm.first.name() == name;
            });
        internal_assert(found) << "Match does not contain match arm: " << name;
    }
    Match *node = new Match;
    node->loc = std::move(loc);
    node->arms = std::move(arms);
    node->volume_map = std::move(volume_map);
    return node;
}

Stmt Yield::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Yield::make";
    Yield *node = new Yield;
    node->value = std::move(value);
    return node;
}

Stmt Iterate::make(Expr value) {
    internal_assert(value.defined()) << "Undefined value in Iterate::make";
    internal_assert(value.type().is_iterable())
        << "Non-iterable type in Iterate::make" << value;
    Iterate *node = new Iterate;
    node->value = std::move(value);
    return node;
}

Stmt Scan::make(std::optional<AggOp::OpType> op, WriteLoc loc, Expr func,
                Expr value) {
    internal_assert(value.defined()) << "Undefined value in Scan::make";
    internal_assert(op.has_value() == loc.defined())
        << "A reducing scan needs an accumulator, and only a reducing scan "
           "has one";
    Scan *node = new Scan;
    node->op = std::move(op);
    node->loc = std::move(loc);
    node->func = std::move(func);
    node->value = std::move(value);
    return node;
}

Stmt YieldFrom::make(Expr value) {
    return YieldFrom::make(std::move(value), {});
}

Stmt YieldFrom::make(Expr value, std::vector<Expr> keys) {
    internal_assert(value.defined()) << "Undefined value in YieldFrom::make";
    internal_assert(std::all_of(keys.cbegin(), keys.cend(),
                                [](const Expr &e) { return e.defined(); }))
        << "Undefined sort key in YieldFrom::make";
    YieldFrom *node = new YieldFrom;
    node->value = std::move(value);
    node->keys = std::move(keys);
    return node;
}

Stmt ForEach::make(std::string name, Expr iter, Stmt body) {
    internal_assert(!name.empty()) << "Undefined name in ForEach::make";
    internal_assert(iter.defined()) << "Undefined iterator in ForEach::make";
    internal_assert(iter.type().is_iterable())
        << "ForEach requires iterable: " << iter;
    internal_assert(body.defined()) << "Undefined body in ForEach::make";

    ForEach *node = new ForEach;
    node->name = std::move(name);
    node->iter = std::move(iter);
    node->body = std::move(body);
    return node;
}

Stmt ForAll::make(std::string index, Slice slice, Stmt body) {
    ForAll *node = new ForAll;
    internal_assert(!index.empty()) << "Empty index name in ForAll::make";
    // Header is optionally defined.
    internal_assert(slice.begin.defined())
        << "Undefined Slice.begin in ForAll::make";
    internal_assert(slice.end.defined())
        << "Undefined Slice.end in ForAll::make";
    internal_assert(slice.stride.defined())
        << "Undefined Slice.stride in ForAll::make";
    internal_assert(body.defined()) << "Undefined body in ForAll::make";
    internal_assert(equals(slice.begin.type(), slice.end.type()));
    internal_assert(equals(slice.begin.type(), slice.stride.type()));
    node->index = std::move(index);
    node->slice = std::move(slice);
    node->body = std::move(body);
    return node;
}

Type ForAll::index_type() const { return slice.begin.type(); }

Expr ForAll::count() const {
    Type idx_t = index_type();
    Expr b = slice.begin, e = slice.end, s = slice.stride;
    // ((e - b) + (s - 1)) / s
    Expr x = BinOp::make(BinOp::OpType::Sub, e, b);
    Expr y = BinOp::make(BinOp::OpType::Sub, s, make_one(idx_t));
    Expr z = BinOp::make(BinOp::OpType::Add, x, y);
    return BinOp::make(BinOp::OpType::Div, z, s);
}

Stmt ParFor::make(std::string index, Slice slice, Stmt body,
                  std::optional<Resource> binding) {
    ParFor *node = new ParFor;
    internal_assert(!index.empty()) << "Empty index name in ParFor::make";
    // Header is optionally defined.
    internal_assert(slice.begin.defined())
        << "Undefined Slice.begin in ParFor::make";
    internal_assert(slice.end.defined())
        << "Undefined Slice.end in ParFor::make";
    internal_assert(slice.stride.defined())
        << "Undefined Slice.stride in ParFor::make";
    internal_assert(body.defined()) << "Undefined body in ParFor::make";
    internal_assert(equals(slice.begin.type(), slice.end.type()));
    internal_assert(equals(slice.begin.type(), slice.stride.type()));
    node->index = std::move(index);
    node->slice = std::move(slice);
    node->body = std::move(body);
    node->binding = binding;
    return node;
}

Type ParFor::index_type() const { return slice.begin.type(); }

Expr ParFor::count() const {
    Type idx_t = index_type();
    Expr b = slice.begin, e = slice.end, s = slice.stride;
    // ((e - b) + (s - 1)) / s
    Expr x = BinOp::make(BinOp::OpType::Sub, e, b);
    Expr y = BinOp::make(BinOp::OpType::Sub, s, make_one(idx_t));
    Expr z = BinOp::make(BinOp::OpType::Add, x, y);
    return BinOp::make(BinOp::OpType::Div, z, s);
}

Stmt Continue::make() {
    static Stmt global_break = new Continue;
    return global_break;
}

Stmt Launch::make(std::string func, Expr n, std::vector<Expr> args) {
    internal_assert(!func.empty()) << "Launch::make received undefined func";
    internal_assert(n.defined() && n.type().is_int_or_uint())
        << "Launch::make received undefined count: " << n;
    internal_assert(std::all_of(args.cbegin(), args.cend(),
                                [](const Expr &e) { return e.defined(); }))
        << "Launch::make received undefined arg to func: " << func;

    Launch *node = new Launch;
    node->func = std::move(func);
    node->n = std::move(n);
    node->args = std::move(args);
    return node;
}

Stmt Append::make(WriteLoc loc, Expr value) {
    internal_assert(!loc.base.empty())
        << "Append::make received empty allocation name";
    internal_assert(loc.type.defined())
        << "Append::make received untyped allocation";
    internal_assert(value.defined()) << "Append::make received undefined value";
    Append *node = new Append;
    node->loc = std::move(loc);
    node->value = std::move(value);
    return node;
}

} // namespace ir
} // namespace bonsai
