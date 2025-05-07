#include "IR/Printer.h"

#include "IR/Expr.h"
#include "IR/Stmt.h"
#include "IR/Type.h"
#include "Utils.h"

#include <sstream>
#include <vector>

namespace bonsai {
namespace ir {
namespace {

template <typename IRNode>
void print_annotation(const IRNode &node, std::ostream &os) {
    if (std::string ann = node->get_annotation(); !ann.empty()) {
        os << ' ' << '\"' << ann << '\"';
    }
}
} // namespace

std::ostream &operator<<(std::ostream &os, const Program &program) {
    Printer printer(os);
    printer.print(program);
    return os;
}

std::string to_string(const Expr &expr) {
    std::ostringstream oss;
    oss << expr;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Expr &expr) {
    if (expr.defined()) {
        Printer printer(os);
        printer.print(expr);
    } else {
        os << "(undef-expr)";
    }
    return os;
}

std::string to_string(const Interface &interface) {
    std::ostringstream oss;
    oss << interface;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Interface &interface) {
    if (interface.defined()) {
        Printer printer(os);
        printer.print(interface);
    } else {
        os << "(undef-interface)";
    }
    return os;
}

std::string to_string(const Type &type) {
    std::ostringstream oss;
    oss << type;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Type &type) {
    if (type.defined()) {
        Printer printer(os);
        printer.print(type);
    } else {
        os << "(undef-type)";
    }
    return os;
}

std::string to_string(const Stmt &stmt) {
    std::ostringstream oss;
    oss << stmt;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Stmt &stmt) {
    if (stmt.defined()) {
        Printer printer(os);
        printer.print(stmt);
    } else {
        os << "(undef-stmt)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &stream, const Indentation &indentation) {
    for (int i = 0; i < indentation.indent * 2; i++) {
        stream << " ";
    }
    return stream;
}

std::ostream &operator<<(std::ostream &os, const WriteLoc &loc) {
    if (loc.defined()) {
        Printer printer(os);
        printer.print(loc);
    } else {
        os << "(undef-loc)";
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const Function &func) {
    Printer printer(os);
    printer.print(func);
    return os;
}

std::ostream &operator<<(std::ostream &os, const Target &target) {
    // TODO: flesh this out.
    switch (target) {
    case Target::Host: {
        os << "host";
        break;
    }
    default: {
        internal_error << "Support for non-host target? add printing support!";
    }
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const Schedule &schedule) {
    Printer printer(os, /*verbose=*/true);
    printer.print(schedule);
    return os;
}

std::string to_string(const Layout &layout) {
    std::ostringstream oss;
    oss << layout;
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const Layout &layout) {
    if (layout.defined()) {
        Printer printer(os);
        printer.print(layout);
    } else {
        os << "(undef-layout)";
    }
    return os;
}

void Printer::print(const Program &program) {
    {
        // We always verbosely print program types and externs.
        ScopedValue<bool> _(verbose, true);
        for (const auto &[name, type] : program.types) {
            os << "type " << name << " = ";
            print(type);
            os << "\n";
        }
        if (!program.types.empty()) {
            os << std::endl;
        }
        for (const auto &[name, type] : program.externs) {
            os << "extern " << name << " : ";
            print(type);
            os << "\n";
        }
        if (!program.externs.empty()) {
            os << std::endl;
        }
    }

    for (const auto &[name, func] : program.funcs) {
        if (!verbose && func->is_imported()) {
            continue;
        }
        print(*func);
        os << '\n' << '\n';
    }
    if (!program.funcs.empty()) {
        os << std::endl;
    }

    {
        // Similarly, we always verbosely print the schedule.
        ScopedValue<bool> _(verbose, true);
        for (const auto &[target, schedule] : program.schedules) {
            os << "schedule " << target << "{\n";
            print(schedule);
            os << '}';
        }
        if (!program.schedules.empty()) {
            os << std::endl;
        }
    }
}

void Printer::print(const Type &type) {
    if (type.defined()) {
        type->accept(this);
    } else {
        // Due to parsing pre-type inference, sometimes
        // have a ton of undefined types.
        os << "unknown";
    }
}

void Printer::print(const Function &function) {
    os << "func ";
    if (function.is_exported()) {
        os << "[[export]] ";
    }
    os << function.name;

    if (!function.interfaces.empty()) {
        os << "<";
        bool first = true;
        for (const auto &[name, interface] : function.interfaces) {
            if (!first) {
                os << ", ";
            }
            first = false;

            os << name;
            if (!interface.is<IEmpty>()) {
                os << " : ";
                print(interface);
            }
        }
        os << ">";
    }
    os << "(";
    bool first = true;
    for (const auto &arg : function.args) {
        if (!first) {
            os << ", ";
        }
        first = false;

        os << arg.name;
        if (arg.type.defined()) {
            os << " : ";
            if (arg.mutating) {
                os << "mut ";
            }
            os << arg.type;
        }
        if (arg.default_value.defined()) {
            os << " = " << arg.default_value;
        }
    }

    os << ") -> " << function.ret_type << " {\n";
    set_indent(1);
    function.body.accept(this);
    os << "}";
}

void Printer::print(const Schedule &schedule) {
    for (const auto &[name, type] : schedule.tree_types) {
        os << name << " : ";
        print(type);
        os << '\n';
    }

    for (const auto &[name, layout] : schedule.tree_layouts) {
        os << name << " : ";
        print(layout);
        os << '\n';
    }
    // TODO: the rest of the schedule.
}

void Printer::print(const Interface &interface) {
    internal_assert(interface.defined());
    interface->accept(this);
}

void Printer::print_type_list(const std::vector<Type> &types) {
    for (size_t i = 0; i < types.size(); i++) {
        print(types[i]);
        if (i < types.size() - 1) {
            os << ", ";
        }
    }
}

void Printer::print(const Expr &expr) {
    ScopedValue<bool> old(implicit_parens, false);
    expr.accept(this);
    print_annotation(expr, os);
}

void Printer::print_no_parens(const Expr &expr) {
    ScopedValue<bool> old(implicit_parens, true);
    expr.accept(this);
}

void Printer::print_expr_list(const std::vector<Expr> &exprs) {
    for (size_t i = 0; i < exprs.size(); i++) {
        print_no_parens(exprs[i]);
        if (i < exprs.size() - 1) {
            os << ", ";
        }
    }
}

void Printer::print(const Stmt &stmt) {
    stmt->accept(this);
    print_annotation(stmt, os);
}

void Printer::print(const WriteLoc &loc) {
    if (verbose) {
        os << "(";
        print(loc.type);
        os << ")";
    }
    os << loc.base;
    for (const auto &value : loc.accesses) {
        if (std::holds_alternative<std::string>(value)) {
            os << "." << std::get<std::string>(value);
        } else {
            os << "[";
            print_no_parens(std::get<Expr>(value));
            os << "]";
        }
    }
}

void Printer::print(const Layout &layout) { layout->accept(this); }

void Printer::visit(const Void_t *node) { os << "void"; }

void Printer::visit(const Int_t *node) { os << "i" << node->bits; }

void Printer::visit(const UInt_t *node) { os << "u" << node->bits; }

void Printer::visit(const Index_t *node) { os << "idx"; }

void Printer::visit(const Float_t *node) {
    if (node->is_ieee754()) {
        os << "f" << node->bits();
    } else if (node->is_bfloat16()) {
        os << "bf" << node->bits();
    } else {
        os << "f" << node->exponent << "_" << node->mantissa;
    }
}

void Printer::visit(const Bool_t *node) { os << "bool"; }

void Printer::visit(const Ptr_t *node) {
    os << "(";
    print(node->etype);
    os << "*)";
}

void Printer::visit(const Ref_t *node) {
    os << "(const " << node->name << "&)";
}

void Printer::visit(const Vector_t *node) {
    print(node->etype);
    os << "x" << node->lanes;
}

void Printer::visit(const Struct_t *node) {
    if (verbose) {
        os << "struct ";
    }
    os << node->name;
    if (verbose) {
        os << "{ ";
        bool first = true;
        for (const auto &[key, value] : node->fields) {
            if (!first) {
                os << "; ";
            }
            first = false;
            // TODO: flip? if easier to read.
            os << key << " : ";
            print(value);
        }
        os << " }";
    }
}

void Printer::visit(const Tuple_t *node) {
    os << "(";
    print_type_list(node->etypes);
    os << ")";
}

void Printer::visit(const Array_t *node) {
    print(node->etype);
    os << "[";
    ir::Expr size = node->size;
    if (size.defined()) {
        if (std::optional<uint64_t> constant_size = get_constant_value(size)) {
            os << std::to_string(*constant_size);
        } else {
            print_no_parens(node->size);
        }
    }
    os << "]";
}

void Printer::visit(const Option_t *node) {
    os << "option<";
    print(node->etype);
    os << ">";
}

void Printer::visit(const Set_t *node) {
    os << "set<";
    print(node->etype);
    os << ">";
}

void Printer::visit(const Function_t *node) {
    os << "Fn(";
    for (size_t i = 0; i < node->arg_types.size(); i++) {
        if (node->arg_types[i].is_mutable) {
            os << "mut ";
        }
        print(node->arg_types[i].type);
        if (i < node->arg_types.size() - 1) {
            os << ", ";
        }
    }
    os << ") -> ";
    print(node->ret_type);
}

void Printer::visit(const Generic_t *node) {
    if (node->interface.is<IEmpty>()) {
        os << node->name;
    } else {
        os << "(" << node->name << " : ";
        print(node->interface);
        os << ")";
    }
}

void Printer::print(const BVH_t::Node &node) {
    const auto print_volume = [&](const BVH_t::Volume &volume) {
        internal_assert(volume.struct_type.is<Struct_t>());
        os << volume.struct_type.as<Struct_t>()->name;
        internal_assert(!volume.initializers.empty());
        os << "(";
        for (size_t i = 0; i < volume.initializers.size(); i++) {
            if (i != 0) {
                os << ", ";
            }
            os << volume.initializers[i];
        }
        os << ")";
    };

    const Struct_t *as_struct = node.struct_type.as<Struct_t>();
    internal_assert(as_struct);

    os << as_struct->name;
    internal_assert(!as_struct->fields.empty());
    os << "(";
    for (size_t i = 0; i < as_struct->fields.size(); i++) {
        if (i != 0) {
            os << ", ";
        }
        os << as_struct->fields[i].name << " : " << as_struct->fields[i].type;
    }
    os << ")";

    if (node.volume.has_value()) {
        os << " with ";
        print_volume(*node.volume);
    }
}

void Printer::visit(const BVH_t *node) {
    os << "tree[[";
    print(node->primitive);
    os << "]] " << node->name;
    ;
    if (!verbose) {
        return;
    }

    internal_assert(!node->nodes.empty());
    for (size_t i = 0; i < node->nodes.size(); i++) {
        os << "\n  | ";
        print(node->nodes[i]);
    }
}

void Printer::visit(const IEmpty *node) { os << "IEmpty"; }

void Printer::visit(const IFloat *node) { os << "IFloat"; }

void Printer::visit(const IVector *node) {
    os << "IVector";
    if (node->etype.defined()) {
        os << "[[";
        print(node->etype);
        os << "]]";
    }
}

void Printer::visit(const IntImm *node) {
    if (verbose) {
        os << "(";
        print(node->type);
        os << ")";
    }
    os << node->value;
}

void Printer::visit(const UIntImm *node) {
    if (verbose) {
        os << "(";
        print(node->type);
        os << ")" << node->value;
        return;
    }
    os << node->value << 'u';
}

void Printer::visit(const FloatImm *node) {
    if (verbose) {
        os << "(";
        print(node->type);
        os << ")" << node->value;
        return;
    }
    switch (node->type.bits()) {
    case 64:
        os << node->value;
        break;
    case 32:
        os << node->value << "f";
        break;
    case 16:
        os << node->value << "h";
        break;
    default:
        internal_error << "Bad bit-width for float" << node->type;
    }
}

void Printer::visit(const BoolImm *node) {
    const auto *str = node->value ? "true" : "false";
    os << str;
}

void Printer::visit(const VecImm *node) {
    os << "(";
    print(node->type);
    os << ")[";
    for (int i = 0, e = node->type.lanes(); i < e; ++i) {
        print(node->values[i]);
        if (i + 1 == e) {
            continue;
        }
        os << ",";
    }
    os << "]";
}

void Printer::visit(const Infinity *node) {
    os << "(";
    print(node->type);
    os << ")";
    os << "inf";
}

void Printer::visit(const Var *node) { os << node->name; }

void Printer::open() {
    if (!implicit_parens) {
        os << "(";
    }
}

void Printer::close() {
    if (!implicit_parens) {
        os << ")";
    }
}

std::string to_string(const BinOp::OpType &op) {
    switch (op) {
    case BinOp::Add:
        return "+";
    case BinOp::Mul:
        return "*";
    case BinOp::Div:
        return "/";
    case BinOp::Sub:
        return "-";
    case BinOp::Mod:
        return "%";
    case BinOp::Neq:
        return "!=";
    case BinOp::Eq:
        return "==";
    case BinOp::Le:
        return "<=";
    case BinOp::Lt:
        return "<";
    case BinOp::LAnd:
        return "&&";
    case BinOp::LOr:
        return "||";
    case BinOp::Xor:
        return "^";
    case BinOp::BwAnd:
        return "&";
    case BinOp::BwOr:
        return "|";
    case BinOp::Shl:
        return "<<";
    case BinOp::Shr:
        return ">>";
    default:
        internal_error << "unsupported op: " << op;
    }
}

void Printer::visit(const BinOp *node) {
    open();
    print(node->a);
    os << " ";
    // TODO: handle min/max/etc.
    os << to_string(node->op);
    os << " ";
    print(node->b);
    close();
}

std::string to_string(const UnOp::OpType &op) {
    switch (op) {
    case UnOp::Neg:
        return "-";
    case UnOp::Not:
        return "!";
    }
}

void Printer::visit(const UnOp *node) {
    os << to_string(node->op);
    print(node->a);
}

void Printer::visit(const Select *node) {
    os << "select(";
    print_no_parens(node->cond);
    os << ", ";
    print_no_parens(node->tvalue);
    os << ", ";
    print_no_parens(node->fvalue);
    os << ")";
}

void Printer::visit(const Cast *node) {
    os << "cast<";
    print(node->type);
    os << ">(";
    print_no_parens(node->value);
    os << ")";
}

void Printer::visit(const Broadcast *node) {
    const ir::Expr &value = node->value;
    print(value.type());
    os << "x" << node->lanes << "(";
    print_no_parens(value);
    os << ")";
}

std::string to_string(const VectorReduce::OpType &op) {
    switch (op) {
    case VectorReduce::Add:
        return "+";
    case VectorReduce::Idxmin:
        return "idxmin";
    case VectorReduce::Idxmax:
        return "idxmax";
    case VectorReduce::Mul:
        return "*";
    case VectorReduce::Min:
        return "min";
    case VectorReduce::Max:
        return "max";
    case VectorReduce::Or:
        return "any";
    case VectorReduce::And:
        return "all";
    }
}

void Printer::visit(const VectorReduce *node) {
    // TODO: print type?
    os << "reduce<" << to_string(node->op) << ">(";
    print_no_parens(node->value);
    os << ")";
}

void Printer::visit(const VectorShuffle *node) {
    // TODO: print type?
    os << "shuffle(";
    print_no_parens(node->value);
    os << ", {";
    print_expr_list(node->idxs);
    os << "})";
}

void Printer::visit(const Ramp *node) {
    // TODO: print type?
    os << "ramp(";
    print_no_parens(node->base);
    os << ", ";
    print_no_parens(node->stride);
    os << ", " << node->lanes << ")";
}

void Printer::visit(const Extract *node) {
    // TODO: parens?
    print(node->vec);
    os << "[";
    print_no_parens(node->idx);
    os << "]";
}

void Printer::visit(const Build *node) {
    os << "build<";
    print(node->type);
    os << ">(";
    print_expr_list(node->values);
    os << ")";
}

void Printer::visit(const Access *node) {
    // TODO: parens?
    print(node->value);
    os << "." << node->field;
}

void Printer::visit(const Unwrap *node) {
    // TODO: parens?
    os << "(";
    print_no_parens(node->value);
    os << " as " << node->type.as<Struct_t>()->name << ")";
}

std::string to_string(const Intrinsic::OpType &op) {
    switch (op) {
    case Intrinsic::abs:
        return "abs";
    case Intrinsic::cos:
        return "cos";
    case Intrinsic::cross:
        return "cross";
    case Intrinsic::dot:
        return "dot";
    case Intrinsic::fma:
        return "fma";
    case Intrinsic::max:
        return "max";
    case Intrinsic::min:
        return "min";
    case Intrinsic::norm:
        return "norm";
    case Intrinsic::pow:
        return "pow";
    case Intrinsic::rand:
        return "rand";
    case Intrinsic::sin:
        return "sin";
    case Intrinsic::sqrt:
        return "sqrt";
    case Intrinsic::tan:
        return "tan";
    }
}

void Printer::visit(const Intrinsic *node) {
    // TODO: print type?
    os << to_string(node->op) << "(";
    print_expr_list(node->args);
    os << ")";
}

std::string to_string(const Generator::OpType &op) {
    switch (op) {
    case Generator::iter:
        return "iter";
    case Generator::range:
        return "range";
    }
}

void Printer::visit(const Generator *node) {
    os << to_string(node->op) << "(";
    print_expr_list(node->args);
    os << ")";
}

// TODO: work on syntax?
void Printer::visit(const Lambda *node) {
    os << "|";
    const size_t n = node->args.size();
    for (size_t i = 0; i < n; i++) {
        os << node->args[i].name;
        if (node->args[i].type.defined()) {
            os << " : ";
            print(node->args[i].type);
        }
        if (i < n - 1) {
            os << ", ";
        }
    }
    os << "| ";
    print_no_parens(node->value);
}

std::string to_string(const GeomOp::OpType &op) {
    return GeomOp::intrinsic_name(op);
}

void Printer::visit(const GeomOp *node) {
    // TODO: print type?
    os << to_string(node->op) << "(";
    print_no_parens(node->a);
    os << ", ";
    print_no_parens(node->b);
    os << ")";
}

std::string to_string(const SetOp::OpType &op) {
    switch (op) {
    case SetOp::argmin:
        return "argmin";
    case SetOp::filter:
        return "filter";
    case SetOp::map:
        return "map";
    case SetOp::product:
        return "product";
    }
}

void Printer::visit(const SetOp *node) {
    // TODO: print type?
    os << to_string(node->op) << "(";
    print_no_parens(node->a);
    os << ", ";
    print_no_parens(node->b);
    os << ")";
}

void Printer::visit(const Call *node) {
    // TODO: print type?
    print_no_parens(node->func);
    os << "(";
    print_expr_list(node->args);
    os << ")";
}

void Printer::visit(const Instantiate *node) {
    print_no_parens(node->expr);
    os << "[[";
    bool first = true;
    for (const auto &[key, value] : node->types) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << key << " -> ";
        print(value);
    }
    os << "]]";
}

void Printer::visit(const PtrTo *node) {
    os << "(&";
    print_no_parens(node->expr);
    os << ")";
}

void Printer::visit(const Deref *node) {
    os << "(*";
    print_no_parens(node->expr);
    os << ")";
}

void Printer::visit(const CallStmt *node) {
    os << get_indent();
    print_no_parens(node->func);
    os << '(';
    print_expr_list(node->args);
    os << ')' << '\n';
}

void Printer::visit(const Print *node) {
    os << get_indent();
    os << "print(";
    print_no_parens(node->value);
    os << ")\n";
}

void Printer::visit(const Return *node) {
    os << get_indent();
    os << "return";
    if (node->value.defined()) {
        os << ' ';
        print_no_parens(node->value);
    }
    os << "\n";
}

void Printer::visit(const LetStmt *node) {
    // ScopedBinding<> bind(known_type, node->name);
    os << get_indent() << "let " << node->loc << " = ";
    print_no_parens(node->value);
    os << " in\n";
    // TODO: fix this!! bring back SSA
    // print(node->body);
}

void Printer::visit(const IfElse *node) {
    os << get_indent();
    while (true) {
        os << "if (";
        print_no_parens(node->cond);
        os << ") {\n";
        indent++;
        print(node->then_body);
        indent--;

        if (!node->else_body.defined()) {
            break;
        }

        if (const IfElse *nested_if = node->else_body.as<IfElse>()) {
            os << get_indent() << "} else ";
            node = nested_if;
        } else {
            os << get_indent() << "} else {\n";
            indent++;
            print(node->else_body);
            indent--;
            break;
        }
    }

    os << get_indent() << "}\n";
}

void Printer::visit(const DoWhile *node) {
    os << get_indent();
    os << "do {\n";
    indent++;
    print(node->body);
    indent--;
    os << get_indent() << "} while (";
    print_no_parens(node->cond);
    os << ")\n";
}

void Printer::visit(const Sequence *node) {
    for (const auto &stmt : node->stmts) {
        stmt.accept(this);
    }
}

void Printer::visit(const Allocate *node) {
    os << get_indent();
    if (node->memory != Allocate::Memory::Stack) {
        os << "alloc ";
    }
    print(node->loc);
    os << " : mut ";
    print(node->loc.base_type);
    if (node->value.defined()) {
        os << " := ";
        print_no_parens(node->value);
    }
    os << "\n";
}

void Printer::visit(const Store *node) {
    os << get_indent();
    print(node->loc);
    os << " = ";
    print_no_parens(node->value);
    os << "\n";
}

void Printer::visit(const Accumulate *node) {
    os << get_indent();
    print(node->loc);
    switch (node->op) {
    case Accumulate::OpType::Add: {
        os << " += ";
        break;
    }
    case Accumulate::OpType::Mul: {
        os << " *= ";
        break;
    }
    case Accumulate::OpType::Sub: {
        os << " -= ";
        break;
    }
    case Accumulate::OpType::Argmin: {
        os << " argmin= ";
        break;
    }
    case Accumulate::OpType::Argmax: {
        os << " argmax= ";
        break;
    }
    default: {
        internal_error
            << "TODO: implement printing for all Accumulate op types!";
    }
    }
    print_no_parens(node->value);
    os << "\n";
    // TODO: fix this!! bring back SSA
    // print(node->body);
}

void Printer::visit(const Label *node) {
    os << get_indent();
    os << "#" << node->name << "{";
    if (node->body.defined()) {
        os << "\n";
        print(node->body);
    }
    os << "}\n";
}

void Printer::visit(const RecLoop *node) {
    os << get_indent() << "rec(";

    const size_t n = node->args.size();
    for (size_t i = 0; i < n; i++) {
        os << node->args[i].name;
        os << " : ";
        print(node->args[i].type);
        if (i < n - 1) {
            os << ", ";
        }
    }
    os << ") {\n";
    indent++;
    print(node->body);
    indent--;
    os << get_indent() << "}\n";
}

void Printer::visit(const Match *node) {
    os << get_indent();
    os << "match ";
    print(node->loc);
    os << " {\n";
    for (const auto &arm : node->arms) {
        os << get_indent() << "| ";
        print(arm.first);
        os << " ->\n";
        indent++;
        print(arm.second);
        indent--;
    }
    os << get_indent() << "}\n";
}

void Printer::visit(const Yield *node) {
    os << get_indent();
    os << "yield ";
    print_no_parens(node->value);
    os << "\n";
}

void Printer::visit(const Scan *node) {
    os << get_indent();
    os << "scan ";
    print_no_parens(node->value);
    os << "\n";
}

void Printer::visit(const YieldFrom *node) {
    os << get_indent();
    os << "from ";
    print_no_parens(node->value);
    os << "\n";
}

void Printer::visit(const ForEach *node) {
    os << get_indent();
    os << "foreach " << node->name << " in ";
    print_no_parens(node->iter);
    os << " {\n";
    indent++;
    print(node->body);
    indent--;
    os << get_indent() << "}\n";
}

void Printer::visit(const ForAll *node) {
    os << get_indent();
    os << "forall " << node->index << " in [";

    const ForAll::Slice &s = node->slice;
    print_no_parens(s.begin);
    os << ":";
    print_no_parens(s.end);
    os << ":";
    print_no_parens(s.stride);
    os << "] {\n";
    ++indent;
    print(node->body);
    --indent;
    os << get_indent() << "}\n";
}

void Printer::visit(const Continue *node) {
    os << get_indent();
    os << "continue\n";
}

void Printer::visit(const Launch *node) {
    os << get_indent() << "launch ";
    print_no_parens(node->n);
    os << " " << node->func << "(";
    print_expr_list(node->args);
    os << ")\n";
}

void Printer::visit(const Name *node) {
    os << get_indent();
    os << node->name;
    os << " : ";
    print(node->type);
    os << ";\n";
}

void Printer::visit(const Pad *node) {
    os << get_indent();
    os << node->bits;
    os << ";\n";
}

void Printer::visit(const Switch *node) {
    os << get_indent();
    os << "switch " << node->field << " {\n";
    for (const auto &[value, name, layout] : node->arms) {
        os << get_indent();
        if (value.has_value()) {
            os << *value;
        } else {
            os << "_";
        }
        os << " => ";
        if (name.has_value()) {
            os << *name;
        }
        os << "\n";
        indent++;
        layout.accept(this);
        indent--;
    }
    os << get_indent() << "};\n";
}

void Printer::visit(const Chain *node) {
    os << get_indent() << "{\n";
    indent++;
    for (const auto &layout : node->layouts) {
        print(layout);
    }
    indent--;
    os << get_indent() << "};\n";
}

void Printer::visit(const Group *node) {
    os << get_indent();
    os << "group[";
    print_no_parens(node->size);
    os << "]";
    if (!node->name.empty()) {
        os << " " << node->name;
        os << " : ";
        print(node->index_t);
    }
    os << "\n";
    print(node->inner);
}

void Printer::visit(const Materialize *node) {
    os << get_indent();
    os << node->name;
    os << " = ";
    print_no_parens(node->value);
    os << ";\n";
}

} // namespace ir
} // namespace bonsai
