#include "IR/Printer.h"

#include <sstream>
#include <vector>

#include "IR/Expr.h"
#include "IR/Stmt.h"
#include "IR/Type.h"

namespace bonsai {
namespace ir {

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
    os << "func " << func.name;
    if (!func.interfaces.empty()) {
        os << "<";
        bool first = true;
        for (const auto &[name, interface] : func.interfaces) {
            if (!first) {
                os << ", ";
            }
            first = false;

            os << name;
            if (!interface.is<IEmpty>()) {
                os << " : ";
                printer.print(interface);
            }
        }
        os << ">";
    }
    os << "(";
    bool first = true;
    for (const auto &arg : func.args) {
        if (!first) {
            os << ", ";
        }
        first = false;

        os << arg.name;
        if (arg.type.defined()) {
            os << " : " << arg.type;
        }
        if (arg.default_value.defined()) {
            os << " = " << arg.default_value;
        }
    }

    os << ") -> " << func.ret_type << " {\n";
    printer.set_indent(1);
    func.body.accept(&printer);
    os << "}";
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

    for (const auto &[name, type] : schedule.tree_types) {
        os << name << " : ";
        printer.print(type);
        os << "\n";
    }
    // TODO: the rest of the schedule.

    return os;
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
    // ScopedValue<bool> old(implicit_parens, false);
    bool temp = implicit_parens;
    implicit_parens = false;
    expr.accept(this);
    implicit_parens = temp;
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

void Printer::print(const Stmt &stmt) { stmt->accept(this); }

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

void Printer::visit(const Void_t *node) { os << "void"; }

void Printer::visit(const Int_t *node) { os << "i" << node->bits; }

void Printer::visit(const UInt_t *node) { os << "u" << node->bits; }

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
    print_type_list(node->arg_types);
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
    const auto print_param = [&](const BVH_t::Param &param) {
        os << param.name << " : ";
        print(param.type);
    };

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

    os << node.name;
    if (node.params.size()) {
        os << "(";
        for (size_t i = 0; i < node.params.size(); i++) {
            if (i != 0) {
                os << ", ";
            }
            print_param(node.params[i]);
        }
        os << ")";
    }
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
    os << "(";
    print(node->type);
    os << ")";
    os << node->value;
}

void Printer::visit(const UIntImm *node) {
    os << "(";
    print(node->type);
    os << ")";
    os << node->value;
}

void Printer::visit(const FloatImm *node) {
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
    case BinOp::And:
        return "&&";
    case BinOp::Or:
        return "||";
    case BinOp::Xor:
        return "^";
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
    os << "x" << node->lanes << "(";
    print_no_parens(node->value);
    os << ")";
}

std::string to_string(const VectorReduce::OpType &op) {
    switch (op) {
    case VectorReduce::Add:
        return "+";
    case VectorReduce::Idxmin:
        return "argmin";
    case VectorReduce::Idxmax:
        return "argmax";
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

std::string to_string(const Intrinsic::OpType &op) {
    switch (op) {
    case Intrinsic::abs:
        return "abs";
    case Intrinsic::cos:
        return "cos";
    case Intrinsic::cross:
        return "cross";
    case Intrinsic::fma:
        return "fma";
    case Intrinsic::max:
        return "max";
    case Intrinsic::min:
        return "min";
    case Intrinsic::sin:
        return "sin";
    case Intrinsic::sqrt:
        return "sqrt";
    }
}

void Printer::visit(const Intrinsic *node) {
    // TODO: print type?
    os << to_string(node->op) << "(";
    print_expr_list(node->args);
    os << ")";
}

// TODO: work on syntax?
void Printer::visit(const Lambda *node) {
    os << "|";
    const size_t n = node->args.size();
    // TODO: might need Lambdas to store arg types as well...
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
    switch (op) {
    case GeomOp::distance:
        return "distance";
    case GeomOp::intersects:
        return "intersects";
    case GeomOp::contains:
        return "contains";
    }
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

void Printer::visit(const Print *node) {
    os << get_indent();
    os << "print(";
    print_no_parens(node->value);
    os << ")\n";
}

void Printer::visit(const Return *node) {
    os << get_indent();
    os << "return ";
    print_no_parens(node->value);
    os << "\n";
}

void Printer::visit(const Store *node) {
    os << get_indent();
    os << node->name << "[";
    if (node->index.defined()) {
        print_no_parens(node->index);
    }
    os << "] = ";
    print_no_parens(node->value);
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

void Printer::visit(const Sequence *node) {
    for (const auto &stmt : node->stmts) {
        stmt.accept(this);
    }
}

void Printer::visit(const Assign *node) {
    os << get_indent();
    print(node->loc);
    if (node->mutating) {
        os << " = ";
    } else {
        os << " := ";
    }
    print_no_parens(node->value);
    os << "\n";
    // TODO: fix this!! bring back SSA
    // print(node->body);
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

void Printer::visit(const Match *node) {
    os << get_indent();
    os << "match ";
    print(node->loc);
    os << "{\n";
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

} // namespace ir
} // namespace bonsai
