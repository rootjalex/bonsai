#include "Parser/Parser.h"

#include "Parser/Lexer.h"
#include "Parser/Token.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Layout.h"
#include "IR/Operators.h"
#include "IR/Printer.h"
#include "IR/Type.h"
#include "IR/TypeEnforcement.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <tuple>

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace parser {

namespace {

// This provides the ability to report errors using a nicer `<<` interface
// instead of string concatenation. It should only be used in the Parser class.
class ParseErrorReport {
  public:
    ParseErrorReport(std::string back_trace, std::string file_name,
                     Token current)
        : back_trace(back_trace), file_name(file_name),
          current(std::move(current)) {}

    [[noreturn]] ~ParseErrorReport() noexcept(false) {
        const uint64_t begin_line = current.line_begin(),
                       begin_column = current.column_begin();
        std::ifstream file(file_name);
        std::string line;
        for (int i = 1; i <= begin_line; ++i) {
            internal_assert(static_cast<bool>(std::getline(file, line)));
        }

        std::stringstream ss;
        ss << back_trace << ":" << begin_line << ":" << begin_column
           << ": [parse error] " << os.str() << "\n"
           << "\n";
        ss << line << "\n";
        const std::string pointer = std::string(
            std::max<int64_t>(current.column_end() - begin_column, 1), '^');
        const std::string offset =
            std::string(std::max<int64_t>(begin_column - 1, 0), ' ');

        ss << offset << pointer;
        internal_error << ss.str();
    }

    template <typename T>
    ParseErrorReport &operator<<(const T &value) {
        os << value;
        return *this;
    }

  private:
    std::string back_trace;
    std::string file_name;
    Token current;
    std::ostringstream os;
};

/*
program := program_element*
program_element := import_stmt | element_decl | interface_decl | extern_decl |
func_decl
// TODO: support methods in elements? and interfaces / inheritance?
import_stmt := import name(/name)*;
element_decl := element IDENTIFIER ('{' (IDENTIFIER : type)+ '}') | '=' type ';'
interface_decl := interface IDENTIFIER {' func_decl+ '}'
extern_decl := extern IDENTIFIER : type;
func_decl := func IDENTIFIER '( (IDENTIFIER : type)+ ')'
             (-> type)?
             ('{' stmt '}' | '=' expr ';')

*/

struct Parser {
  public:
    Parser(TokenStream tokens) {
        context.emplace_back(std::move(tokens));
        // Add Intrinsic types!
    }

    ir::Program parse_program() {
        internal_assert(frames.empty());
        new_frame();
        parse_program_stream(/*allow_externs=*/true);
        end_frame();
        internal_assert(frames.empty());
        return std::move(program);
    }

  private:
    // Stores a stack of all program streams.
    std::vector<TokenStream> context;
    // Filenames of everything visited so far, to avoid double-imports.
    std::set<std::string> visited_files;

    const TokenStream &tokens() const { return context.back(); }

    TokenStream &tokens() { return context.back(); }

    ir::Program program;
    // Function variable frames. Maps name to type and mutability.
    std::list<std::map<std::string, std::pair<ir::Type, bool>>> frames;
    // TODO: if we allow nested functions or any other way to allow nested
    // generics, we need this to be a stack!
    ir::TypeMap current_generics;
    const ir::Type u32 = ir::UInt_t::make(32), i32 = ir::Int_t::make(32),
                   f32 = ir::Float_t::make_f32();

    const std::string &file_name() const { return tokens().file_name(); }

    std::string back_trace() const {
        std::ostringstream out;
        for (size_t i = 0; i < context.size(); ++i) {
            out << std::string(2 * i, ' ') << context[i].file_name();
            if (i + 1 < context.size()) {
                out << "\n" << std::string(2 * i, ' ') << "-> ";
            }
        }
        return out.str();
    }

    // Reports the error with relevant token location information. This will
    // always point at the last consumed token.
    inline ParseErrorReport report_error() const {
        return ParseErrorReport{back_trace(), file_name(),
                                tokens().current_token()};
    }

    ir::Type get_type_from_frame(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                internal_assert(!program.funcs.contains(name))
                    << "found a value in the current frame with the same name "
                       "as a previously defined function: "
                    << name;
                return found->second.first;
            }
        }

        if (auto it = program.funcs.find(name); it != program.funcs.end()) {
            return it->second->call_type();
        }
        report_error() << "Cannot check type of unknown var: " << name;
    }

    bool name_in_scope(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                return true;
            }
        }
        return false;
    }

    bool is_mutable(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                return found->second.second;
            }
        }
        report_error() << "Cannot check mutability of unknown var: " << name;
    }

    void add_type_to_frame(const std::string &name, ir::Type type, bool mut) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                report_error() << name << " shadows another variable";
            }
        }
        frames.back()[name] = {std::move(type), mut};
    }

    void modify_type_in_frame(const std::string &name, ir::Type type) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            auto &frame = *it;
            auto found = frame.find(name);
            if (found != frame.end()) {
                internal_assert(!found->second.first.defined())
                    << "Attempt to modify defined type for name: " << name;
                found->second.first = std::move(type);
                return;
            }
        }
        report_error() << "Cannot modify_type of unknown var: " << name
                       << " to type " << type;
    }

    void new_frame() { frames.emplace_back(); }

    void end_frame() { frames.pop_back(); }

    Token peek(uint32_t k = 0) const { return tokens().peek(k); }

    std::optional<Token> consume(Token::Type type) {
        const Token token = peek();

        if (!tokens().consume(type)) {
            return {};
        }

        return token;
    }

    // Automatically consumes the token.
    Token consume() {
        Token token = peek();
        tokens().consume(token.type);
        return token;
    }

    // Consumes the next token in the stream if the expected type
    //  is met, and reports an error otherwise.
    Token expect(Token::Type type) {
        if (std::optional<Token> token = consume(type)) {
            return *token;
        }
        Token current = consume();
        report_error() << "expected: " << Token::token_type_string(type)
                       << ", received: "
                       << Token::token_type_string(current.type);
    }

    std::string get_id() {
        const Token token = expect(Token::Type::IDENTIFIER);
        std::string id = std::get<std::string>(token.value);
        if (id.empty()) {
            report_error() << "unexpected: empty identifier name.";
        }
        // We special-case the string "_", which is used as a wild card in
        // layout switch statements.
        if (id.size() > 1 && id.starts_with('_')) {
            report_error() << "identifier: " << id
                           << " cannot begin with `_`. This is reserved for "
                              "compiler-generated variables.";
        }
        return id;
    }

    void parse_program_stream(const bool allow_externs) {
        while (!tokens().empty()) {
            parse_program_element(allow_externs);
        }
    }

    void parse_program_element(const bool allow_externs) {
        switch (peek().type) {
        case Token::Type::IMPORT:
            return parse_import();
        case Token::Type::ELEMENT:
            return parse_element();
        case Token::Type::INTERFACE:
            return parse_interface_def();
        case Token::Type::EXTERN:
            if (!allow_externs) {
                report_error() << "Parsed extern during import.";
            }
            return parse_extern();
        case Token::Type::FUNC:
            return parse_function();
        case Token::Type::SCHEDULE:
            return parse_schedule();
        default:
            report_error() << "failure in parse_program_element";
        }
    }

    void parse_import() {
        // Imports are essentially code inlining. Just load and parse the file.
        // TODO: support source/header separations?
        expect(Token::Type::IMPORT);
        std::string name = get_id();

        // Handle (/name)*
        while (!consume(Token::Type::SEMICOL)) {
            expect(Token::Type::SLASH);
            name += "/" + get_id();
        }

        name += ".bonsai";

        auto [_, inserted] = visited_files.insert(name);
        if (!inserted) {
            return; // We've already imported this file.
        }

        for (const auto &tks : context) {
            if (tks.file_name() == name) {
                report_error() << "Import cycle found.";
            }
        }

        try {
            TokenStream tokens = lex(name);
            context.emplace_back(std::move(tokens));
            parse_program_stream(/*allow_externs=*/false);
            context.pop_back();
        } catch (const Error &e) {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception &e) {
            report_error() << "Failure to parse imported file: " << name
                           << " with message: " << e.what();
        }
    }

    void parse_element() {
        expect(Token::Type::ELEMENT);

        // TODO: support methods as well.
        // TODO: figure out overloading policy for that.
        // TODO: for error handling, should we have beginLoc/endLoc like Simit?
        const std::string name = get_id();

        if (program.types.contains(name)) {
            report_error() << "Redefinition of type: " << name;
        }

        // Support inline aliasing.
        if (consume(Token::Type::ASSIGN)) {
            ir::Type alias = parse_type();
            expect(Token::Type::SEMICOL);
            program.types[name] = alias;
            return;
        }

        // Regular type declaration.
        ir::Struct_t::Map fields;
        ir::Struct_t::DefMap defaults;
        expect(Token::Type::LSQUIGGLE);
        do {
            // Handle multiple names with a single type.
            std::vector<std::string> names;
            do {
                const std::string field_name = get_id();
                // Verify this is not a duplicate.
                if (std::find_if(fields.cbegin(), fields.cend(),
                                 [&](const auto &p) {
                                     return p.name == field_name;
                                 }) != fields.cend()) {
                    report_error() << "Duplicate field name: " << field_name
                                   << " in element definition: " << name;
                }
                names.push_back(field_name);

                if (consume(Token::Type::ASSIGN)) {
                    ir::Expr _default = parse_expr();
                    internal_assert(ir::is_constant_expr(_default))
                        << "Field default values must be constants, received: "
                        << _default << " for field " << field_name
                        << " of element " << name;
                    defaults[field_name] = std::move(_default);
                }
            } while (consume(Token::Type::COMMA));

            expect(Token::Type::COL);
            ir::Type type = parse_type();
            for (const auto &field_name : names) {
                fields.emplace_back(field_name, type);
                if (defaults.contains(field_name)) {
                    defaults[field_name] =
                        constant_cast(type, defaults[field_name]);
                }
            }

            if (consume(Token::Type::ASSIGN)) {
                ir::Expr _default = parse_expr();
                internal_assert(ir::is_constant_expr(_default))
                    << "Field default values must be constants, received: "
                    << _default << " for typed fields " << " of element "
                    << name;
                _default = constant_cast(type, _default);
                internal_assert(_default.defined());
                for (const auto &field_name : names) {
                    internal_assert(!defaults.contains(field_name))
                        << "Duplicate default value for field " << field_name
                        << " of element " << name
                        << " individually marked: " << defaults[field_name]
                        << " and group marked: " << _default;
                    defaults[field_name] = _default;
                }
            }
            expect(Token::Type::SEMICOL);
        } while (!consume(Token::Type::RSQUIGGLE));

        program.types[name] = defaults.empty()
                                  ? ir::Struct_t::make(name, std::move(fields))
                                  : ir::Struct_t::make(name, std::move(fields),
                                                       std::move(defaults));
    }

    void parse_interface_def() {
        internal_error << "TODO: implement parse_interface_def";
    }

    void parse_extern() {
        // TODO: what if an extern name-conflicts with a type or something?
        // probably should check conflicts for all symbols?
        expect(Token::Type::EXTERN);
        const std::string name = get_id();
        if (std::find_if(program.externs.cbegin(), program.externs.cend(),
                         [&](const auto &p) { return p.name == name; }) !=
            program.externs.cend()) {
            report_error() << "Redefinition of extern: " << name;
        }
        // TODO: should we support defaults? that makes passing in args harder.
        expect(Token::Type::COL);
        ir::Type type = parse_type();
        expect(Token::Type::SEMICOL);
        add_type_to_frame(name, type, /* mutable */ false);
        program.externs.emplace_back(name, std::move(type));
    }

    std::vector<ir::Function::Argument> parse_func_args() {
        expect(Token::Type::LPAREN);
        std::vector<ir::Function::Argument> args;
        if (peek().type != Token::Type::RPAREN) {
            // parse arg list
            do {
                // TODO: can we accept multiple args with one type here, as in
                // element definitions?
                bool mutating = false;
                std::string arg_name = get_id();
                expect(Token::Type::COL);
                if (consume(Token::Type::MUT)) {
                    mutating = true;
                }
                ir::Type type = parse_type();

                ir::Expr default_value;

                if (consume(Token::Type::ASSIGN)) {
                    // Optional default value.
                    // TODO: default value may need to be cast to type.
                    // For now, assume that's done in type inference.
                    // TODO: this should not perform computation!
                    // Can we easily prevent that? Enforce is_constant?
                    default_value = parse_expr();
                    if (!ir::is_constant_expr(default_value)) {
                        report_error()
                            << "Function default values must be constants";
                    }
                }

                add_type_to_frame(arg_name, type, mutating);
                args.push_back(ir::Function::Argument{
                    std::move(arg_name),
                    std::move(type),
                    std::move(default_value),
                    mutating,
                });
            } while (consume(Token::Type::COMMA));
        }
        expect(Token::Type::RPAREN);
        return args;
    }

    ir::Function::InterfaceList parse_func_interfaces() {
        ir::Function::InterfaceList interfaces;
        if (consume(Token::Type::LT)) {
            // Parse generics.
            if (!current_generics.empty()) {
                report_error() << "Nested generics.";
            }

            do {
                const std::string iname = get_id();
                if (current_generics.contains(iname)) {
                    report_error() << "Duplicate interface name.";
                }
                ir::Interface interface = consume(Token::Type::COL).has_value()
                                              ? parse_interface()
                                              : ir::IEmpty::make();
                interfaces.emplace_back(iname, interface);
                current_generics[iname] =
                    ir::Generic_t::make(iname, std::move(interface));

            } while (consume(Token::Type::COMMA));
            expect(Token::Type::GT);
        }
        return interfaces;
    }

    void parse_geometric_intrinsic(const std::string &name) {
        // TODO: support generics for geometric intrinsics.
        new_frame();
        std::vector<ir::Function::Argument> args = parse_func_args();

        // Build a unique identifier, because all function names are unique.
        std::string typed_name = name;

        for (const auto &arg : args) {
            if (!arg.type.is<ir::Struct_t>()) {
                report_error() << "Geometric primitives only operator on "
                                  "elements, instead received: "
                               << arg.name << " : " << arg.type;
            }
            typed_name += "_" + arg.type.as<ir::Struct_t>()->name;
        }

        ir::Type ret_type;
        if (consume(Token::Type::RARROW)) {
            ret_type = parse_type();
        }

        // Do *NOT* support recursive geometric intrinsics.
        // These are hard (if not impossible) to optimize.

        ir::Stmt body;

        // TODO: the syntax of -> type = expr is ugly, maybe disallow, and just
        // do inference?

        if (consume(Token::Type::ASSIGN)) {
            ir::Expr expr = parse_expr();
            if (!ret_type.defined() && expr.type().defined()) {
                ret_type = expr.type();
            } else {
                if (!(ir::equals(ret_type, expr.type()) ||
                      (ret_type.is<ir::Option_t>() &&
                       ir::equals(ret_type.as<ir::Option_t>()->etype,
                                  expr.type())))) {
                    report_error() << "Mismatching types: " << ret_type
                                   << " versus " << expr.type();
                }
            }
            expect(Token::Type::SEMICOL);
            body = ir::Return::make(expr);
        } else {
            body = parse_sequence();
            ir::Type body_type = ir::get_return_type(body);
            if (!ret_type.defined() && body_type.defined()) {
                ret_type = std::move(body_type);
            } else if (ret_type.defined() && body_type.defined()) {
                if (!(ir::equals(ret_type, body_type) ||
                      (ret_type.is<ir::Option_t>() &&
                       ir::equals(ret_type.as<ir::Option_t>()->etype,
                                  body_type)))) {
                    report_error() << "Mismatching types: " << ret_type
                                   << " versus " << body_type;
                }
            }
        }

        if (!ret_type.defined()) {
            report_error()
                << "Unknown return type for geometric predicate is not allowed";
        }

        if (is_geometric_predicate(name) &&
            !ret_type.is<ir::Bool_t, ir::Option_t>()) {
            report_error() << "Geometric predicates must return a truth-y "
                              "value, instead returns type: "
                           << ret_type;
        } else if (is_geometric_metric(name) && !ret_type.is_numeric()) {
            report_error() << "Geometric metrics must return a Real-y value, "
                              "instead returns type: "
                           << ret_type;
        }

        end_frame();

        auto func = std::make_shared<ir::Function>(
            typed_name, std::move(args), std::move(ret_type), std::move(body),
            ir::Function::InterfaceList{}, std::vector<ir::Function::Attribute>{});

        auto [_, inserted] =
            program.funcs.try_emplace(std::move(typed_name), std::move(func));
        if (!inserted) {
            report_error() << "Duplicate geometric func detected, of name: "
                           << typed_name;
        }
    }

    void parse_function() {
        expect(Token::Type::FUNC);
        
        std::vector<ir::Function::Attribute> attributes;
        if (context.size() > 1) { // in an imported file.
            attributes.push_back(ir::Function::Attribute::imported);
        }

        if (consume(Token::Type::LBRACKET) && consume(Token::Type::LBRACKET)) {
            std::string attribute = get_id();
            if (attribute == "export") {
                attributes.push_back(ir::Function::Attribute::exported);
            } else {
                report_error() << "unexpected attribute: " << attribute;
            }
            expect(Token::Type::RBRACKET);
            expect(Token::Type::RBRACKET);
        }
        const std::string name = get_id();
        if (is_geometric_intrinsic(name)) {
            return parse_geometric_intrinsic(name); // special case.
        }

        if (program.funcs.contains(name)) {
            report_error() << "Redefinition of func: " << name;
        }

        ir::Function::InterfaceList interfaces = parse_func_interfaces();
        new_frame();
        std::vector<ir::Function::Argument> args = parse_func_args();

        // Optional RARROW with return_type: otherwise, requires type inference!
        ir::Type ret_type;
        if (consume(Token::Type::RARROW)) {
            ret_type = parse_type();
        }
        const bool ret_type_set = ret_type.defined();

        // Functions can either be inlined: '=' Expr;
        // or with '{' body '}'

        // To support recursive functions, we need to insert at least
        // the type of the function into the program now. Note that this
        // can break if the return type is not defined! We should assert
        // that the return type is defined for recursive calls, or implement
        // really good type unification or something.
        program.funcs[name] = std::make_shared<ir::Function>(
            name, std::move(args), std::move(ret_type), ir::Stmt(),
            std::move(interfaces), std::move(attributes));

        ir::Stmt body;

        // TODO: the syntax of -> type = expr is ugly, maybe disallow, and just
        // do inference?

        if (consume(Token::Type::ASSIGN)) {
            ir::Expr expr = parse_expr();
            if (!ret_type_set && expr.type().defined()) {
                ret_type = expr.type();
            }
            expect(Token::Type::SEMICOL);
            body = ir::Return::make(expr);
        } else {
            body = parse_sequence();
            // TODO: do type checking if they don't match?
            if (!ret_type_set) {
                ret_type = ir::get_return_type(body);
            }
        }

        end_frame();
        current_generics.clear();

        internal_assert(!program.funcs[name]->body.defined())
            << "Woah, how did " << name
            << " get a function body before being parsed?";

        program.funcs[name]->body = std::move(body);

        if (!ret_type_set && ret_type.defined()) {
            // we were able to statically infer the return type
            program.funcs[name]->ret_type = std::move(ret_type);
        }
    }

    ir::Stmt parse_sequence() {
        std::vector<ir::Stmt> stmts;
        expect(Token::Type::LSQUIGGLE);
        // brackets enclose a new frame!
        new_frame();

        while (!consume(Token::Type::RSQUIGGLE)) {
            stmts.push_back(parse_statement());
        }
        // close the frame.
        end_frame();
        internal_assert(!stmts.empty())
            << "Failed to parse a Sequence at line: "
            << peek().line_begin(); // TODO: this fails when the stream is empty
        if (stmts.size() == 1) {
            return std::move(stmts[0]);
        } else {
            return ir::Sequence::make(std::move(stmts));
        }
    }

    // Parses a call statement, i.e., a call statement with no return value.
    // This assumes the next token will have an id found in this program's
    // functions.
    // TODO(cgyurgyik): Need to eventually extend this to support struct methods
    // as well, e.g., `a.foo(1)`.
    std::optional<ir::Stmt> parse_call_statement(std::string id) {
        auto it = program.funcs.find(id);
        if (it == program.funcs.end()) {
            return {};
        }
        const ir::Function &function = *it->second;
        expect(Token::Type::LPAREN);

        std::vector<ir::Expr> args = parse_expr_list_until(Token::Type::RPAREN);
        ir::Expr v = ir::Var::make(function.call_type(), std::move(id));
        expect(Token::Type::SEMICOL);
        return ir::CallStmt::make(std::move(v), std::move(args));
    }

    // if expr stmt [elif expr stmt]* [else stmt]?
    // return expr;
    // TODO: allow labels? or inline funcs? of some kind.
    // { stmt* }
    // TODO: allow mut modifier!
    // name [: type]? =  expr;
    // name += expr; ?
    // call? no bc no side effects...
    // for? not while.
    ir::Stmt parse_statement() {
        if (peek().type == Token::Type::LSQUIGGLE) {
            return parse_sequence();
        } else if (consume(Token::Type::IF)) {
            ir::Expr cond = parse_expr(); // no required parens
            ir::Stmt then_case = parse_statement();
            internal_assert(!consume(Token::Type::ELIF))
                << "TODO: implement elif parsing for line: "
                << peek().line_begin();
            if (consume(Token::Type::ELSE)) {
                ir::Stmt else_case = parse_statement();
                return ir::IfElse::make(std::move(cond), std::move(then_case),
                                        std::move(else_case));
            } else {
                return ir::IfElse::make(std::move(cond), std::move(then_case));
            }
        } else if (consume(Token::Type::RETURN)) {
            if (consume(Token::Type::SEMICOL)) {
                return ir::Return::make();
            }
            ir::Expr ret = parse_expr();
            expect(Token::Type::SEMICOL);
            return ir::Return::make(std::move(ret));
        } else if (consume(Token::Type::PRINT)) {
            expect(Token::Type::LPAREN);
            ir::Expr value = parse_expr();
            expect(Token::Type::RPAREN);
            expect(Token::Type::SEMICOL);
            return ir::Print::make(value);
        } else if (peek().type == Token::Type::IDENTIFIER) {
            std::string id = get_id();
            // TODO(cgyurgyik): This assumes that functions are declared before
            // they're called. This isn't the only place this constraint holds,
            // we should eventual support mutual recursion.
            if (std::optional<ir::Stmt> call = parse_call_statement(id)) {
                return *call;
            }
            // TODO: allow tuple declaration/assignment?
            // TODO: how to do SSA in parsing?
            ir::WriteLoc loc = parse_write_loc(std::move(id));
            if (loc.accesses.empty() && !name_in_scope(loc.base)) {
                // Just a regular variable write
                // might be an Assign (if labelled `mut`)
                return parse_name_decl(std::move(loc));
            } else if (consume(Token::Type::ASSIGN)) {
                return parse_assign(std::move(loc));
            } else {
                // Must be an accumulate.
                return parse_accumulate(std::move(loc));
            }
        }
        internal_error << "[unimplemented] parse_statement for "
                       << peek().to_string();
    }

    ir::Stmt parse_name_decl(ir::WriteLoc loc) {
        // This is a never-seen-before write to a variable.
        internal_assert(!loc.type.defined());
        ir::Type type_label;
        bool _mutable = false;

        if (consume(Token::Type::COL)) {
            if (consume(Token::Type::MUT)) {
                _mutable = true;
                if (peek().type == Token::Type::IDENTIFIER) {
                    type_label = parse_type();
                } // otherwise just a `mut` label.
                // TODO: should we ever allow "just" a mut label?
            } else {
                type_label = parse_type();
            }
        }

        expect(Token::Type::ASSIGN);
        ir::Expr value = parse_expr();
        ir::Type type = value.type();
        expect(Token::Type::SEMICOL);

        if (const auto *build = value.as<ir::Build>();
            build && !type.defined()) {
            // When assigning a vector with an initializer list, this may occur,
            // e.g., `v: vector[i32, 2] = {1, 2};`, which is parsed as:
            //       `v: vector[i32, 2] = build<unknown>((i32)1, (i32)2)`
            if (const auto *vector_type = type_label.as<ir::Vector_t>();
                vector_type &&
                std::all_of(build->values.begin(), build->values.end(),
                            [](const ir::Expr &e) { return is_const(e); })) {
                value = ir::VecImm::make(build->values);
            } else {
                value = ir::Build::make(type_label, build->values);
            }
        }

        // TODO: do type-forcing here!
        if (type_label.defined() && type.defined()) {
            if (!ir::equals(type_label, value.type()) && is_const(value)) {
                value = constant_cast(type_label, value);
            }
            internal_assert(ir::equals(type_label, value.type()))
                << "Mismatching assignment: " << loc
                << " is labelled with type: " << type_label << " but " << value
                << " has type " << type;
        }
        ir::Type write_type = type_label.defined() ? type_label : type;
        add_type_to_frame(loc.base, write_type, _mutable);

        loc = ir::WriteLoc(loc.base, std::move(write_type));
        if (!_mutable) {
            return ir::LetStmt::make(std::move(loc), std::move(value));
        } else {
            return ir::Assign::make(std::move(loc), std::move(value),
                                    /*mutating*/ false);
        }
    }

    ir::Stmt parse_assign(ir::WriteLoc loc) {
        ir::Expr value = parse_expr();
        expect(Token::Type::SEMICOL);

        const bool mutating = name_in_scope(loc.base);

        if (mutating && !is_mutable(loc.base)) {
            report_error() << "Cannot assign to non-mutable variable: " << loc.base;
        }

        // TODO: do type forcing here!
        if (loc.type.defined() && value.type().defined()) {
            internal_assert(ir::equals(loc.type, value.type()))
                << "Mismatching assignment: " << loc
                << " has type: " << loc.type << " but " << value << " has type "
                << value.type();
        }

        if (!loc.type.defined() && value.type().defined()) {
            // TODO: do type forcing here!
            if (loc.accesses.empty()) {
                modify_type_in_frame(loc.base, value.type());
                loc = ir::WriteLoc(loc.base, value.type());
            }
            // e.g. if accesses wasn't empty, we could still infer a partial
            // type.
        }

        return ir::Assign::make(std::move(loc), std::move(value), mutating);
    }

    ir::Stmt parse_accumulate(ir::WriteLoc loc) {
        ir::Accumulate::OpType op = ir::Accumulate::OpType::Add;
        // Try to parse an accumulate
        if (consume(Token::Type::PLUS)) {
            op = ir::Accumulate::OpType::Add;
        } else if (consume(Token::Type::STAR)) {
            op = ir::Accumulate::OpType::Mul;
        } else {
            report_error() << "Unknown token when parsing Accumulate at line:";
        }
        expect(Token::Type::ASSIGN);
        ir::Expr value = parse_expr();
        expect(Token::Type::SEMICOL);
        return ir::Accumulate::make(std::move(loc), op, std::move(value));
    }

    // We follow C++'s operator precedence.
    // https://en.cppreference.com/w/cpp/language/operator_precedence

    // TODO: precedence 2: member access...
    // precedence 5: mul, div, mod
    // precedence 6: addition/subtraction
    // precedence 7: bit shifts
    // precedence 9: relational operators
    // precedence 10: equality operators
    // precedence 11: bitwise and
    // precedence 12: bitwise xor
    // precedence 13: bitwise or
    // precedence 14: logical and
    // precedence 15: logical or

    // TODO: parse member access first somehow...?
    // expr := muldivmod_expr
    ir::Expr parse_expr() { return parse_or(); }

    struct BinOperator {
        const ir::BinOp::OpType op;
        const Token::Type token;
        const bool flip = false;
    };

    template <size_t N, typename F>
    ir::Expr
    parse_bin_op_with_precedence(F parse_sub_expr,
                                 const std::array<BinOperator, N> &ops) {
        ir::Expr expr = parse_sub_expr();
        while (true) {
            // Find the first matching operation
            auto it = std::find_if(ops.begin(), ops.end(), [&](const auto &op) {
                return consume(op.token);
            });

            // If no operation is found, we're done here.
            if (it == ops.end()) {
                return expr;
            }

            // Parse the right-hand side and create the binary operation
            ir::Expr rhs = parse_sub_expr();
            if (it->flip) {
                expr = ir::BinOp::make(it->op, std::move(rhs), std::move(expr));
            } else {
                expr = ir::BinOp::make(it->op, std::move(expr), std::move(rhs));
            }
        }
    }

    // muldivmod_expr := addsub_expr (('*' | '/' | '%') addsub_expr)*
    ir::Expr parse_mul_div_mod() {
        return parse_bin_op_with_precedence<3>(
            [this]() { return parse_base_expr(); },
            {{
                {ir::BinOp::Mul, Token::Type::STAR},
                {ir::BinOp::Div, Token::Type::SLASH},
                {ir::BinOp::Mod, Token::Type::MOD},
            }});
    }

    // addsub_expr := shift_expr (('+' | '-') shift_expr)*
    ir::Expr parse_add_sub() {
        return parse_bin_op_with_precedence<2>(
            [this]() { return parse_mul_div_mod(); },
            {{
                {ir::BinOp::Add, Token::Type::PLUS},
                {ir::BinOp::Sub, Token::Type::MINUS},
            }});
    }

    // shift_expr := rel_expr (('<<' | '>>') rel_expr)*
    ir::Expr parse_shift() {
        return parse_bin_op_with_precedence<2>(
            [this]() { return parse_add_sub(); },
            {{
                {ir::BinOp::Shl, Token::Type::SHIFT_LEFT},
                {ir::BinOp::Shr, Token::Type::SHIFT_RIGHT},
            }});
    }

    // rel_expr := eq_expr (('<=' | '<') eq_expr)*
    ir::Expr parse_rels() {
        return parse_bin_op_with_precedence<4>(
            [this]() { return parse_shift(); },
            {{
                {ir::BinOp::Lt, Token::Type::LT},
                {ir::BinOp::Le, Token::Type::LEQ},
                {ir::BinOp::Lt, Token::Type::GT, /* flip */ true},
                {ir::BinOp::Le, Token::Type::GEQ, /* flip */ true},
            }});
    }

    // eq_expr := bwand_expr (('==' | '!=') bwand_expr)*
    ir::Expr parse_eqs() {
        return parse_bin_op_with_precedence<2>(
            [this]() { return parse_rels(); },
            {{
                {ir::BinOp::Eq, Token::Type::EQ},
                {ir::BinOp::Neq, Token::Type::NEQ},
            }});
    }

    // bwand_expr := xor_expr ('&' xor_expr)*
    ir::Expr parse_bwand() {
        return parse_bin_op_with_precedence<1>(
            [this]() { return parse_eqs(); },
            {{{ir::BinOp::BwAnd, Token::Type::BITWISE_AND}}});
    }

    // xor_expr := bwor_expr ('^' bwor_expr)*
    ir::Expr parse_xor() {
        return parse_bin_op_with_precedence<1>(
            [this]() { return parse_bwand(); },
            {{{ir::BinOp::Xor, Token::Type::XOR}}});
    }

    // bwor_expr := and_expr ('|' and_expr)*
    ir::Expr parse_bwor() {
        return parse_bin_op_with_precedence<1>(
            [this]() { return parse_xor(); },
            {{{ir::BinOp::BwOr, Token::Type::BAR}}});
    }

    // and_expr := or_expr ('&&' or_expr)*
    ir::Expr parse_and() {
        return parse_bin_op_with_precedence<1>(
            [this]() { return parse_bwor(); },
            {{{ir::BinOp::LAnd, Token::Type::LOGICAL_AND}}});
    }

    // or_expr := base_expr ('||' base_expr)*
    ir::Expr parse_or() {
        return parse_bin_op_with_precedence<1>(
            [this]() { return parse_and(); },
            {{{ir::BinOp::LOr, Token::Type::LOGICAL_OR}}});
    }

    // base_expr := '(' expr ')' | name (('.' field) | ('[' index (, index)* ']'
    // |
    // () ) | lambda args? : expr
    ir::Expr parse_base_expr() {
        if (consume(Token::Type::LPAREN)) {
            ir::Expr inner = parse_expr();
            if (consume(Token::Type::COMMA)) {
                // Tuple constructor.
                std::vector<ir::Expr> values = {std::move(inner)};
                std::vector<ir::Type> etypes = {values[0].type()};
                if (!etypes.back().defined()) {
                    report_error() << "[unimplemented] tuple construction "
                                      "with value of unknown type: "
                                   << values.back();
                }
                do {
                    ir::Expr next = parse_expr();
                    values.emplace_back(std::move(next));
                    etypes.push_back(values.back().type());
                    // TODO: improve type inference to handle this?
                    if (!etypes.back().defined()) {
                        report_error() << "[unimplemented] tuple construction "
                                          "with value of unknown type: "
                                       << values.back();
                    }
                } while (consume(Token::Type::COMMA));
                // TODO: gracefully
                ir::Type tuple_t = ir::Tuple_t::make(std::move(etypes));
                inner = ir::Build::make(std::move(tuple_t), std::move(values));
            }
            expect(Token::Type::RPAREN);
            return inner;
            // TODO: do these have the correct precedence?
        } else if (consume(Token::Type::MINUS)) {
            ir::Expr inner = parse_expr();
            return ir::UnOp::make(ir::UnOp::Neg, std::move(inner));
        } else if (consume(Token::Type::NOT)) {
            ir::Expr inner = parse_expr();
            return ir::UnOp::make(ir::UnOp::Not, std::move(inner));
        } else if (peek().type == Token::Type::IDENTIFIER) {
            return parse_identifier();
            // Parse literals.
        } else if (consume(Token::Type::TRUE)) {
            return ir::BoolImm::make(true);
        } else if (consume(Token::Type::FALSE)) {
            return ir::BoolImm::make(false);
        } else if (peek().type == Token::Type::INT_LITERAL) {
            // can't know concrete type yet, let type inference figure it out.
            const int64_t value = parse_int_literal();
            // default (pre-type casting) is i32
            return ir::IntImm::make(i32, value);
        } else if (peek().type == Token::Type::UINT_LITERAL) {
            // can't know concrete type yet, let type inference figure it out.
            const Token token = expect(Token::Type::UINT_LITERAL);
            const uint64_t value = std::get<uint64_t>(token.value);
            // default (pre-type casting) is u32
            return ir::UIntImm::make(u32, value);
        } else if (peek().type == Token::Type::FLOAT_LITERAL) {
            // can't know concrete type yet, let type inference figure it out.
            const Token token = expect(Token::Type::FLOAT_LITERAL);
            const double value = std::get<double>(token.value);
            // default (pre-type casting) is f32
            return ir::FloatImm::make(f32, value);
        } else if (consume(Token::Type::BAR)) {
            std::vector<ir::TypedVar> args = parse_lambda_args();
            new_frame();
            for (const auto &arg : args) {
                add_type_to_frame(arg.name, arg.type, /* mutable */ false);
            }
            // Optionally allow squiggles for lambda expression body.
            const bool has_squiggles =
                consume(Token::Type::LSQUIGGLE).has_value();
            ir::Expr expr = parse_expr();
            if (has_squiggles) {
                expect(Token::Type::RSQUIGGLE);
            }
            end_frame();
            return ir::Lambda::make(std::move(args), std::move(expr));
        } else if (consume(Token::Type::LSQUIGGLE)) {
            // Could be a build or an empty
            // TODO: is there an easy way to always know the type?
            if (consume(Token::Type::RSQUIGGLE)) {
                // Empty!
                static const std::vector<ir::Expr> empty = {};
                return ir::Build::make(ir::Type(), empty);
            } else {
                std::vector<ir::Expr> args =
                    parse_expr_list_until(Token::Type::RSQUIGGLE);
                // TODO: can we know the type?
                return ir::Build::make(ir::Type(), std::move(args));
            }
            // TODO: should also support e.g. Type{} notation.
        } else if (consume(Token::Type::STAR)) {
            // Option dereference.
            ir::Expr arg = parse_identifier();
            ir::Type atype = arg.type();
            internal_assert(atype.defined() && atype.is<ir::Option_t>())
                << "Parsed dereference of non-option: " << arg;
            ir::Type etype = atype.as<ir::Option_t>()->etype;
            // TODO(ajr): do we want an explicit Deref IR node?
            return ir::Cast::make(std::move(etype), std::move(arg));
        } else if (peek().type == Token::Type::STRING_LITERAL) {
            internal_error << "[unimplemented] string literals: "
                           << peek().to_string();
        }
        Token token = consume();
        report_error() << "unexpected token: "
                       << Token::token_type_string(token);
    }

    template <typename OpType, typename Container>
    std::optional<OpType>
    try_match_pattern(const std::string &name, const size_t arg_count,
                      const Container &patterns, size_t n_args) {
        for (const auto &p : patterns) {
            if (name == p.name) {
                if constexpr (requires { p.skippable; }) {
                    if (p.skippable && arg_count != p.n_args) {
                        return {};
                    }
                }
                if constexpr (requires { p.n_args; }) {
                    if (arg_count != p.n_args) {
                        report_error()
                            << p.name << " takes " << p.n_args
                            << " argument(s), received " << arg_count;
                    }
                } else {
                    if (arg_count != n_args) {
                        report_error()
                            << p.name << " takes " << n_args
                            << " argument(s), received " << arg_count;
                    }
                }

                return p.op;
            }
        }
        return {};
    }

    ir::Expr try_match_intrinsics(const std::string &name,
                                  std::vector<ir::Expr> args) {
        // Numerical intrinsics
        struct IntrinsicPattern {
            const std::string_view name;
            size_t n_args;
            ir::Intrinsic::OpType op;
            bool skippable = false;
        };

        static constexpr auto IPATTERNS = std::to_array<IntrinsicPattern>({
            {"abs", 1, ir::Intrinsic::abs},
            {"cos", 1, ir::Intrinsic::cos},
            {"cross", 2, ir::Intrinsic::cross},
            {"dot", 2, ir::Intrinsic::dot},
            {"fma", 3, ir::Intrinsic::fma},
            // These two are skippable because they might be parsed as
            // single-argument reductions below.
            {"max", 2, ir::Intrinsic::max, .skippable = true},
            {"min", 2, ir::Intrinsic::min, .skippable = true},
            {"norm", 1, ir::Intrinsic::norm},
            {"sin", 1, ir::Intrinsic::sin},
            {"sqrt", 1, ir::Intrinsic::sqrt},
        });

        if (auto op = try_match_pattern<ir::Intrinsic::OpType>(
                name, args.size(), IPATTERNS, 0)) {
            return ir::Intrinsic::make(*op, std::move(args));
        }

        // Set operations
        struct SetPattern {
            const std::string_view name;
            ir::SetOp::OpType op;
        };

        static constexpr auto SPATTERNS = std::to_array<SetPattern>({
            {"argmin", ir::SetOp::argmin},
            {"filter", ir::SetOp::filter},
            {"map", ir::SetOp::map},
            {"product", ir::SetOp::product},
        });

        if (auto op = try_match_pattern<ir::SetOp::OpType>(name, args.size(),
                                                           SPATTERNS, 2)) {
            return ir::SetOp::make(*op, std::move(args[0]), std::move(args[1]));
        }

        // Geometry operations
        struct GeomPattern {
            const std::string_view name;
            ir::GeomOp::OpType op;
        };

        static const auto GPATTERNS = std::to_array<GeomPattern>({
            {"distmax", ir::GeomOp::distmax},
            {"distmin", ir::GeomOp::distmin},
            {"intersects", ir::GeomOp::intersects},
            {"contains", ir::GeomOp::contains},
        });

        if (auto op = try_match_pattern<ir::GeomOp::OpType>(name, args.size(),
                                                            GPATTERNS, 2)) {
            return ir::GeomOp::make(*op, std::move(args[0]),
                                    std::move(args[1]));
        }

        // Vector reductions
        struct ReducePattern {
            const std::string_view name;
            ir::VectorReduce::OpType op;
        };

        static const auto RPATTERNS = std::to_array<ReducePattern>({
            {"sum", ir::VectorReduce::Add},
            {"all", ir::VectorReduce::And},
            {"idxmax", ir::VectorReduce::Idxmax},
            {"idxmin", ir::VectorReduce::Idxmin},
            {"max", ir::VectorReduce::Max},
            {"min", ir::VectorReduce::Min},
            {"prod", ir::VectorReduce::Mul},
            {"any", ir::VectorReduce::Or},
        });

        if (auto op = try_match_pattern<ir::VectorReduce::OpType>(
                name, args.size(), RPATTERNS, 1)) {
            return ir::VectorReduce::make(*op, std::move(args[0]));
        }

        return ir::Expr();
    }

    ir::Expr parse_identifier() {
        const std::string name = get_id();
        std::vector<std::string> fields; // possibly nested
        while (consume(Token::Type::PERIOD)) {
            // Member access.
            const std::string field_name = get_id();
            fields.push_back(field_name);
        }

        // Only use if not an intrinsic!
        auto make_expr = [&]() {
            // Just a variable, possibly with field accesses.
            // the type might have been provided already, or
            // might need to be inferred later.
            ir::Type var_type = get_type_from_frame(name); // never undefined.
            ir::Expr expr = ir::Var::make(var_type, name);

            for (const auto &field : fields) {
                expr = ir::Access::make(field, expr);
            }
            return expr;
        };

        std::vector<ir::Type> template_types;
        if (consume(Token::Type::LBRACKET)) {
            if (consume(Token::Type::LBRACKET)) {
                // Parse template type list
                template_types = parse_type_list_until(Token::Type::RBRACKET);
                expect(Token::Type::RBRACKET);
                if (template_types.empty()) {
                    report_error() << "Template syntax expects type arguments, "
                                      "but did not receive any for name: "
                                   << name;
                }
                if (peek().type != Token::Type::LPAREN) {
                    report_error() << "Template syntax supported only for "
                                      "function calls, found on name: "
                                   << name;
                }
            } else {
                std::vector<ir::Expr> idxs =
                    parse_expr_list_until(Token::Type::RBRACKET);
                if (idxs.empty()) {
                    report_error() << "Indexing into array/vector expects at "
                                      "least one index for name: "
                                   << name;
                }
                ir::Expr expr = make_expr();
                for (auto &idx : idxs) {
                    expr = ir::Extract::make(std::move(expr), std::move(idx));
                }
                return expr;
            }
        }

        if (consume(Token::Type::LPAREN)) {
            std::vector<ir::Expr> args =
                parse_expr_list_until(Token::Type::RPAREN);

            if (fields.empty()) {

                // Check for built-in intrinsics first. These are not
                // over-ridable!
                ir::Expr intrinsic = try_match_intrinsics(name, args);
                if (intrinsic.defined()) {
                    if (!template_types.empty()) {
                        report_error()
                            << "Intrinsics do not accept template parameters: "
                            << intrinsic;
                    }
                    return intrinsic;
                }

                // Checking program.funcs first means that users can override
                // built-in functions. That could be dangerous.
                if (program.funcs.contains(name)) {
                    const auto &func = program.funcs[name];
                    // TODO: handle default params!
                    if (args.size() != func->args.size()) {
                        report_error()
                            << "Call to: " << name
                            << " has incorrect number of arguments.\n"
                            << "Expected: " << func->args.size()
                            << " but parsed " << args.size();
                    }

                    if (func->interfaces.size() != template_types.size()) {
                        report_error()
                            << "Call to: " << name
                            << " has incorrect number of template paramters.\n"
                            << "Expected: " << func->interfaces.size()
                            << " but parsed " << template_types.size();
                    }

                    const size_t n_generics = func->interfaces.size();

                    ir::TypeMap instantiations;
                    for (size_t i = 0; i < n_generics; i++) {
                        instantiations[func->interfaces[i].name] =
                            template_types[i];

                        internal_assert(ir::satisfies(
                            template_types[i], func->interfaces[i].interface))
                            << "Template type: " << template_types[i]
                            << " in call to " << name
                            << " does not satisfy interface: "
                            << func->interfaces[i].interface;
                    }

                    // TODO(ajr): may want to lift this into an analysis
                    // function, for reuse in type inference.
                    const size_t n_args = func->args.size();
                    std::vector<ir::Type> arg_types(n_args);

                    for (size_t i = 0; i < func->args.size(); i++) {
                        arg_types[i] = func->args[i].type;
                        // TODO: we could push types down here, because
                        // we know the arg types. That mixes type
                        // inference with parsing though, not sure we
                        // want that (more than we already do...).
                        ir::Type expected_type =
                            func->interfaces.empty()
                                ? func->args[i].type
                                : replace(instantiations, func->args[i].type);

                        if (args[i].type().defined() &&
                            !ir::equals(expected_type, args[i].type())) {
                            report_error()
                                << "Argument " << i << " of call to function "
                                << name << " has incorrect type. Expected "
                                << expected_type
                                << " but parsed: " << args[i].type();
                        }
                    }

                    ir::Type ftype;

                    // TODO: if we allowed partially-defined types, we could
                    // make: Fn(arg_types) -> (undef) that would make type
                    // inference easier, but we'd have to change a lot of
                    // the error handling in IR/Type.cpp
                    // For now, leave ftype undefined in the else case
                    if (func->ret_type.defined()) {
                        ftype = ir::Function_t::make(func->ret_type,
                                                     std::move(arg_types));
                    }

                    ir::Expr f = ir::Var::make(ftype, name);

                    if (!func->interfaces.empty()) {
                        f = ir::Instantiate::make(std::move(f),
                                                  std::move(instantiations));
                    }
                    return ir::Call::make(std::move(f), std::move(args));
                } else if (name_in_scope(name)) {
                    if (!template_types.empty()) {
                        report_error() << "Error: cannot pass template "
                                          "parameters to lambda "
                                       << name;
                    }
                    ir::Type var_type =
                        get_type_from_frame(name); // never undefined.
                    ir::Expr expr = ir::Var::make(var_type, name);
                    return ir::Call::make(std::move(expr), std::move(args));
                }

                // Special built-ins with template parameters.
                if (name == "cast") {
                    if (template_types.size() != 1) {
                        report_error() << "cast() expects a single template "
                                          "parameter, instead received: "
                                       << template_types.size();
                    }
                    if (args.size() != 1) {
                        report_error() << "cast() expects a single argument, "
                                          "instead received: "
                                       << args.size();
                    }
                    return ir::Cast::make(std::move(template_types[0]),
                                          std::move(args[0]));
                } else if (name == "eps") {
                    if (template_types.size() != 1) {
                        report_error() << "eps() expects a single template "
                                          "parameter, instead received: "
                                       << template_types.size();
                    }
                    if (args.size() != 0) {
                        report_error()
                            << "eps() expects no arguments, instead received: "
                            << args.size();
                    }

                    // TODO: or template?
                    ir::Type type = std::move(template_types[0]);
                    internal_assert(type.is_float())
                        << "eps takes only floating point template types, "
                           "instead "
                           "received: "
                        << type;
                    // TODO: this should be handled in codegen...
                    return ir::FloatImm::make(type, machine_epsilon(type));
                }

                if (!template_types.empty()) {
                    report_error()
                        << name
                        << " does not take template parameters, but received: "
                        << template_types.size();
                }

                // Special built-ins without template parameters
                if (name == "permute") {
                    internal_assert(args.size() == 2)
                        << "permute takes two arguments, received: "
                        << args.size();
                    internal_assert(args[1].is<ir::Build>())
                        << "permute expects the second argument to be a list "
                           "of indexes, "
                           "instead "
                           "received: "
                        << args[1];
                    return ir::VectorShuffle::make(
                        std::move(args[0]), args[1].as<ir::Build>()->values);
                } else if (name == "select") {
                    internal_assert(args.size() == 3)
                        << "select takes 3 arguments, received: "
                        << args.size();
                    return ir::Select::make(std::move(args[0]),
                                            std::move(args[1]),
                                            std::move(args[2]));
                } else if (name == "range") {
                    internal_assert(args.size() == 3)
                        << "range takes 3 arguments, received: " << args.size();
                    internal_assert(args[0].type().defined() &&
                                    args[0].type().is<ir::Array_t>())
                        << "range() expects first argument to be an array "
                           "type: "
                        << args[0] << " is " << args[0].type();
                    internal_assert(args[1].type().defined() &&
                                    args[1].type().is_int_or_uint())
                        << "range() expects second argument to be an integer "
                           "type: "
                        << args[1] << " is " << args[1].type();
                    internal_assert(args[2].type().defined() &&
                                    args[2].type().is_int_or_uint())
                        << "range() expects third argument to be an integer "
                           "type: "
                        << args[2] << " is " << args[2].type();
                    internal_assert(ir::equals(args[1].type(), args[2].type()))
                        << "range() expects second and third arguments to be "
                           "same type "
                        << "arg1: " << args[1] << " is " << args[1].type()
                        << " arg2: " << args[2] << " is " << args[2].type();
                    // TODO: make this an intrinsic?
                    ir::Type ret_type =
                        ir::Array_t::make(args[0].type().element_of(), args[2]);
                    ir::Type call_type = ir::Function_t::make(
                        std::move(ret_type),
                        {args[0].type(), args[1].type(), args[2].type()});
                    ir::Expr func =
                        ir::Var::make(std::move(call_type), "range");
                    return ir::Call::make(std::move(func), std::move(args));
                }

                // Not intrinsic or set op, not sure what this is.
                // TODO: could be a ctor of a type?
                report_error() << "Unknown function call " << name;
            } else {
                // method access!
                // TODO: type inference via interface?
                ir::Expr expr = make_expr();
                if (!template_types.empty()) {
                    report_error()
                        << "TODO: support passing template types to a method "
                           "access: "
                        << expr << " received " << template_types.size();
                }
                return ir::Call::make(std::move(expr), std::move(args));
            }
        }

        if (!template_types.empty()) {
            report_error()
                << "TODO: support template arguments in constructors for: "
                << name;
        }

        if (peek().type == Token::Type::LSQUIGGLE) {
            if (fields.empty() && program.types.contains(name)) {
                expect(Token::Type::LSQUIGGLE);
                ir::Type type = program.types.at(name);
                internal_assert(type.defined());
                // Look for named struct build
                if (consume(Token::Type::PERIOD)) {
                    // TODO: this could use better error handling/messaging.
                    std::map<std::string, ir::Expr> args;
                    do {
                        std::string field = get_id();
                        expect(Token::Type::ASSIGN);
                        ir::Expr value = parse_expr();
                        internal_assert(value.defined());
                        args[std::move(field)] = std::move(value);
                    } while (consume(Token::Type::COMMA) &&
                             consume(Token::Type::PERIOD));
                    expect(Token::Type::RSQUIGGLE);
                    return ir::Build::make(std::move(type), std::move(args));
                }
                // Otherwise just a regular list-like struct build.
                auto args = parse_expr_list_until(Token::Type::RSQUIGGLE);
                return ir::Build::make(std::move(type), std::move(args));
            }
            // otherwise ignore, not a struct build, e.g. maybe `if` expr { body
            // }
        }

        if (name == "inf") {
            if (!fields.empty()) {
                report_error() << "`inf` is a reserved keyword for infinity";
            }
            return ir::Infinity::make(f32);
        }

        return make_expr();
    }

    std::vector<ir::Expr> parse_expr_list_until(const Token::Type &token) {
        std::vector<ir::Expr> exprs;
        if (consume(token)) {
            return exprs;
        }
        do {
            ir::Expr expr = parse_expr();
            exprs.emplace_back(std::move(expr));
        } while (consume(Token::Type::COMMA));
        expect(token);
        return exprs;
    }

    std::vector<ir::Type> parse_type_list_until(const Token::Type &token) {
        std::vector<ir::Type> types;
        if (consume(token)) {
            return types;
        }
        do {
            ir::Type type = parse_type();
            types.emplace_back(std::move(type));
        } while (consume(Token::Type::COMMA));
        expect(token);
        return types;
    }

    std::vector<ir::TypedVar> parse_lambda_args() {
        // arg := name (':' type)?
        // args := arg (',' arg)*
        // TODO: should we allow no arg lambdas?
        // not sure I want that for now. doesn't
        // that imply some sort of side effects?
        // but maybe we need that for rng?
        std::vector<ir::TypedVar> args;
        do {
            auto def = parse_name_def<false, true>(false);
            internal_assert(!def.value.defined())
                << "Lambdas cannot have default function values! " << def.name
                << " assigned default: " << def.value;
            internal_assert(!def.mut)
                << "TODO: support mutable arguments in lambdas. Argument: "
                << def.name << " marked as mutable.";
            args.push_back({std::move(def.name), std::move(def.type)});
        } while (!consume(Token::Type::BAR));
        return args;
    }

    struct NameDef {
        std::string name;
        ir::Type type;  // optional
        ir::Expr value; // optional
        bool mut = false;
    };

    // TODO: allow `mut` flag!
    template <bool T_REQUIRED, bool E_CONST>
    NameDef parse_name_def(const bool expr_allowed) {
        NameDef def;

        def.name = get_id();

        if constexpr (T_REQUIRED) {
            expect(Token::Type::COL);
            if (consume(Token::Type::MUT)) {
                def.mut = true;
            }
            def.type = parse_type();
        } else if (consume(Token::Type::COL)) {
            if (consume(Token::Type::MUT)) {
                def.mut = true;
                // might have no type label, just mut
                if (peek().type != Token::Type::ASSIGN) {
                    def.type = parse_type();
                }
            } else {
                // no mut label, must be type.
                def.type = parse_type();
            }
        }

        if (expr_allowed && consume(Token::Type::ASSIGN)) {
            def.value = parse_expr();
            if constexpr (E_CONST) {
                internal_assert(ir::is_constant_expr(def.value))
                    << "Expected constant value for name: " << def.name
                    << " but instead received: " << def.value;
            }
        }
        return def;
    }

    ir::WriteLoc parse_write_loc(std::string base) {
        ir::Type base_type;
        if (name_in_scope(base)) {
            base_type = get_type_from_frame(base);
        }

        ir::WriteLoc loc(std::move(base), base_type);

        while ((peek().type == Token::Type::PERIOD) ||
               (peek().type == Token::Type::LBRACKET)) {
            if (consume(Token::Type::PERIOD)) {
                const std::string field_name = get_id();
                loc.add_struct_access(field_name);
            } else {
                expect(Token::Type::LBRACKET);
                ir::Expr index = parse_expr();
                loc.add_index_access(std::move(index));
            }
        }
        return loc;
    }

    // type := i[N] | u[N]
    //         | f[N] | bf[N] | f[N]_[N]
    //         | bool | vector[type, int]
    //         | option[type] | declared_type
    ir::Type parse_type() {
        if (consume(Token::Type::LPAREN)) {
            // Tuple type.
            auto etypes = parse_type_list_until(Token::Type::RPAREN);
            if (etypes.empty()) {
                report_error() << "Cannot construct empty tuple type.";
            }
            return ir::Tuple_t::make(std::move(etypes));
        }
        // TODO: support tuples of types! AKA unnamed structs.
        const std::string name = get_id();
        // Signed integer types.
        std::regex int_pattern("^i(\\d+)$");
        // Unsigned integer types.
        std::regex uint_pattern("^u(\\d+)$");
        // Floating point types.
        std::regex float_pattern("^b?f(\\d+)$");
        // Explicit declaration of exponent and mantissa bits.
        std::regex float_pattern_explicit("^f(\\d+)_(\\d+)$");
        // TODO(cgyurgyik): Support explicit declaration for fixed point.

        std::smatch match;
        if (std::regex_match(name, match, int_pattern)) {
            const uint32_t bits = std::stoul(match[1].str());
            return ir::Int_t::make(bits);
        } else if (std::regex_match(name, match, uint_pattern)) {
            const uint32_t bits = std::stoul(match[1].str());
            return ir::UInt_t::make(bits);
        } else if (std::regex_match(name, match, float_pattern_explicit)) {
            const uint32_t exponent = std::stoul(match[1].str());
            const uint32_t mantissa = std::stoul(match[2].str());
            return ir::Float_t::make(exponent, mantissa);
        } else if (std::regex_match(name, match, float_pattern)) {
            const uint32_t bits = std::stoul(match[1].str());
            if (!name.empty() && name.front() == 'b') {
                if (bits == 16) {
                    return ir::Float_t::make_bf16();
                }
                report_error() << "brain float (bfloat) expects width 16";
            }

            // Assume default types for floating precision is IEEE-754 standard.
            switch (bits) {
            case 64:
                return ir::Float_t::make_f64();
            case 32:
                return ir::Float_t::make_f32();
            case 16:
                return ir::Float_t::make_f16();
            default:
                report_error() << "unsupported floating point type: f" << bits;
            }
        } else if (name == "bool") {
            return ir::Bool_t::make();
        } else if (name == "void") {
            return ir::Void_t::make();
        }
        // Now look for built-ins
        else if (name == "vector") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parse_type();
            expect(Token::Type::COMMA);
            const int64_t lanes = parse_int_literal();
            // TODO: this upper bound is arbitrary. Can't imagine needing a
            // larger one though?
            internal_assert(lanes > 0 && lanes < 1025)
                << "Vector lane count is invalid: " << lanes;
            expect(Token::Type::RBRACKET);
            return ir::Vector_t::make(std::move(etype),
                                      static_cast<uint32_t>(lanes));
        } else if (name == "option") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parse_type();
            if (etype.is_bool()) {
                report_error()
                    << "Bonsai does not support option[bool] "
                       "because the semantics are easily misconstrued.";
            }
            expect(Token::Type::RBRACKET);
            return ir::Option_t::make(std::move(etype));
        } else if (name == "set") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parse_type();
            expect(Token::Type::RBRACKET);
            // TODO: assert etype is a struct_t? or volume_t?
            return ir::Set_t::make(std::move(etype));
        } else if (name == "array") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parse_type();
            ir::Expr size;
            if (consume(Token::Type::COMMA)) {
                size = parse_expr();
            }
            expect(Token::Type::RBRACKET);
            return ir::Array_t::make(std::move(etype), std::move(size));
        } else if (current_generics.contains(name)) {
            return current_generics[name];
        }
        // Must be a user-defined type, or an error.
        auto it = program.types.find(name);
        if (it == program.types.end()) {
            report_error() << "unknown type: " << name;
        }
        return it->second;
    }

    // interface = iPrimitive | iVector[[interface]] | interface (`|`
    // interface)*
    ir::Interface parse_interface() {
        // TODO: Support interface unions and UDIs!
        return parse_primitive_interface();
    }

    ir::Interface parse_primitive_interface() {
        const std::string name = get_id();
        if (name == "IFloat") {
            return ir::IFloat::make();
        } else if (name == "IVector") {
            ir::Interface inner;
            if (consume(Token::Type::LBRACKET)) {
                expect(Token::Type::LBRACKET);
                inner = parse_interface();
                expect(Token::Type::RBRACKET);
                expect(Token::Type::RBRACKET);
            } else {
                inner = ir::IEmpty::make();
            }
            return ir::IVector::make(std::move(inner));
        } else if (current_generics.contains(name)) {
            return current_generics[name].as<ir::Generic_t>()->interface;
        }
        report_error() << "Unrecognized primitive interface: " << name;
    }

    int64_t parse_int_literal() {
        // TODO: might need a "tryparse_int_literal"...
        const Token val = expect(Token::Type::INT_LITERAL);
        return std::get<int64_t>(val.value);
    }

    void parse_schedule() {
        expect(Token::Type::SCHEDULE);
        // TODO: parse target name / info?
        expect(Token::Type::LSQUIGGLE);

        ir::Schedule schedule;

        ir::TypeMap trees;
        do {
            switch (peek().type) {
            case Token::Type::TREE: {
                ir::Type tree = parse_tree();
                internal_assert(tree.is<ir::BVH_t>());
                internal_assert(!trees.contains(tree.as<ir::BVH_t>()->name));
                // Would std::move() but I think rhs is evaluated before lhs.
                trees[tree.as<ir::BVH_t>()->name] = tree;
                break;
            }
            case Token::Type::IDENTIFIER: {
                const std::string name = get_id();
                // TODO: if func, is schedule.
                // TODO: if type, is ????
                // if extern, is tree-assignment
                const auto extern_iter = std::find_if(
                    program.externs.cbegin(), program.externs.cend(),
                    [&name](const auto &p) { return p.name == name; });
                if (extern_iter != program.externs.cend()) {
                    // name : tree_type;
                    if (!extern_iter->type.is<ir::Set_t>()) {
                        report_error() << "Extern: " << name
                                       << " is not a set, cannot be "
                                          "type-reassigned in a schedule.";
                    }

                    expect(Token::Type::COL);
                    const std::string tree_name = get_id();
                    expect(Token::Type::SEMICOL);

                    const auto tree_iter = trees.find(tree_name);

                    if (tree_iter == trees.cend()) {
                        report_error() << "Assigning extern: " << name
                                       << " to non-tree type: " << tree_name;
                    }

                    // TODO: assert Set[Primitive] matches Tree type!
                    schedule.tree_types[name] = tree_iter->second;
                } else {
                    // TODO: support func schedule parsing.
                    report_error()
                        << "Schedule name: " << name << " is not an extern.";
                }
                break;
            }
            case Token::Type::LAYOUT: {
                expect(Token::Type::LAYOUT);
                std::string name = get_id();
                const auto extern_iter = std::find_if(
                    program.externs.cbegin(), program.externs.cend(),
                    [&name](const auto &p) { return p.name == name; });

                if (extern_iter == program.externs.cend()) {
                    report_error()
                        << "Layout name: " << name << " is not an extern.";
                }

                ir::Layout layout = parse_top_level_layout();
                const auto [_, inserted] =
                    schedule.tree_layouts.emplace(name, std::move(layout));
                if (!inserted) {
                    report_error()
                        << "Layout for " << name << " already exists.";
                }
                break;
            }
            default: {
                consume();
                report_error() << "Unknown schedule statement.";
            }
            }
        } while (!consume(Token::Type::RSQUIGGLE));

        // TODO: figure out schedule merging.
        internal_assert(program.schedules.empty());

        program.schedules[ir::Target::Host] = schedule;
    }

    // tree := `tree` name = (`|` adt_node (`with` volume)?)+
    ir::Type parse_tree() {
        expect(Token::Type::TREE);
        expect(Token::Type::LBRACKET);
        expect(Token::Type::LBRACKET);
        ir::Type primitive = parse_type();
        expect(Token::Type::RBRACKET);
        expect(Token::Type::RBRACKET);
        auto [name, params, volume] = parse_node();

        if (program.types.contains(name)) {
            report_error() << "Tree named: " << name
                           << " conflicts with existing type: "
                           << program.types[name];
        }

        if (volume.has_value() != !params.empty()) {
            report_error() << "Parsing of tree " << name
                           << " has incompatible volume and params";
        }

        expect(Token::Type::ASSIGN);
        expect(Token::Type::BAR);

        // Empty reference for now.
        program.types[name] = ir::Ref_t::make(name);

        std::vector<ir::BVH_t::Node> nodes;

        do {
            auto [nname, nparams, nvolume] = parse_node();
            ir::Type struct_type =
                ir::Struct_t::make(std::move(nname), std::move(nparams));
            ir::BVH_t::Node node{std::move(struct_type), std::move(nvolume)};
            nodes.emplace_back(std::move(node));
        } while (consume(Token::Type::BAR));

        expect(Token::Type::SEMICOL);

        // TODO: assert that all volumes only have initializers from
        // parent.params or node.params BVH_t::make asserts this. we should
        // catch that failure, and report a backtrace.
        ir::Type type;
        if (volume.has_value()) {
            type = ir::BVH_t::make(std::move(primitive), std::move(name),
                                   std::move(params), std::move(nodes),
                                   std::move(*volume));
        } else {
            type = ir::BVH_t::make(std::move(primitive), std::move(name),
                                   std::move(nodes));
        }

        return type;
    }

    ir::BVH_t::Volume parse_volume() {
        std::string name = get_id();

        internal_assert(program.types.contains(name))
            << "Unknown volume type: " << name;
        ir::Type type = program.types[std::move(name)];

        expect(Token::Type::LPAREN);

        std::vector<std::string> initializers;
        do {
            std::string iname = get_id();
            initializers.emplace_back(std::move(iname));
        } while (consume(Token::Type::COMMA));

        expect(Token::Type::RPAREN);

        return ir::BVH_t::Volume{std::move(type), std::move(initializers)};
    }

    std::tuple<std::string, ir::Struct_t::Map, std::optional<ir::BVH_t::Volume>>
    parse_node() {
        std::string name = get_id();

        std::vector<ir::TypedVar> params;
        std::optional<ir::BVH_t::Volume> volume;

        if (consume(Token::Type::LPAREN)) {
            params = parse_tree_params();
            expect(Token::Type::RPAREN);
        }

        if (consume(Token::Type::WITH)) {
            volume = parse_volume();
        }

        return {name, params, volume};
    }

    std::vector<ir::TypedVar> parse_tree_params() {
        // arg := name (':' type)?
        // args := arg (',' arg)*
        // no mutability allowed.
        std::vector<ir::TypedVar> args;
        do {
            std::vector<std::string> names;
            do {
                names.push_back(get_id());
            } while (!consume(Token::Type::COL));

            ir::Type type = parse_type();

            for (auto &name : names) {
                args.push_back({std::move(name), type});
            }
        } while (consume(Token::Type::COMMA));
        return args;
    }

    // Wrapper that adds built-ins into scope and then calls parse_layout()
    ir::Layout parse_top_level_layout() {
        new_frame();

        // TODO: support other non-u32 indexing.
        // add_type_to_frame("count", u32, /* mutable=*/false);
        // add_type_to_frame("index", u32, /* mutable=*/false);
        // TODO: add range() and other built-ins!
        // ir::Interface interface = ir::IEmpty::make(); // TODO: IArray
        // ir::Type range_type; // TODO: T[], int_t, int_t -> T[]
        // add_type_to_frame("range", range_type, /*mutable=*/false);

        ir::Layout layout = parse_layout();
        expect(Token::Type::SEMICOL);

        end_frame();
        return layout;
    }

    ir::Layout parse_layout() {
        // Default.
        static ir::Expr count = ir::Var::make(u32, "count");
        switch (peek().type) {
        case Token::Type::LSQUIGGLE: {
            consume();
            std::vector<ir::Layout> layouts;
            new_frame();
            do {
                layouts.emplace_back(parse_layout());
                expect(Token::Type::SEMICOL);
            } while (!consume(Token::Type::RSQUIGGLE));
            end_frame();
            return ir::Chain::make(std::move(layouts));
        }
        case Token::Type::GROUP: {
            consume();
            ir::Expr size;
            if (consume(Token::Type::LBRACKET)) {
                size = parse_expr();
                expect(Token::Type::RBRACKET);
            }
            std::string name;
            ir::Type index_t;
            if (peek().type == Token::Type::IDENTIFIER) {
                name = get_id();
                expect(Token::Type::COL);
                index_t = parse_type();
                if (!index_t.is_int_or_uint()) {
                    report_error()
                        << "Group index type must be integer: " << name
                        << " is labelled " << index_t;
                }
                add_type_to_frame(name, index_t, /* mutable=*/false);
            }
            ir::Layout inner = parse_layout();
            if (!size.defined()) {
                ir::Expr isize = inner.count();
                if (!isize.defined() || !is_const(isize)) {
                    report_error() << "Cannot infer size of doubly-nested "
                                      "dynamic-sized layout groups";
                }
                size = (count + (isize - make_one(isize.type()))) / isize;
            }
            return ir::Group::make(std::move(size), std::move(name),
                                   std::move(index_t), std::move(inner));
        }
        case Token::Type::IDENTIFIER: {
            std::string name = get_id();
            if (consume(Token::Type::COL)) {
                ir::Type type = parse_type();
                if (!type.is_primitive()) {
                    report_error() << "Layout received name: " << name
                                   << " with non-primitive type: " << type;
                }
                add_type_to_frame(name, type, /* mutable=*/false);
                return ir::Name::make(std::move(name), std::move(type));
            } else {
                expect(Token::Type::ASSIGN);
                // TODO: insert built-ins to frame, here or somewhere?
                ir::Expr expr = parse_expr();
                if (!expr.defined() || !expr.type().defined() ||
                    !expr.type().is_primitive()) {
                    report_error()
                        << "Layout received materialization of name: " << name
                        << " with non-primitive type: " << expr;
                }
                add_type_to_frame(name, expr.type(), /* mutable=*/false);
                return ir::Materialize::make(std::move(name), std::move(expr));
            }
        }
        case Token::Type::INT_LITERAL: {
            const int64_t value = parse_int_literal();
            return ir::Pad::make(value);
        }
        case Token::Type::SWITCH: {
            consume();
            std::string name = get_id();
            expect(Token::Type::LSQUIGGLE);

            std::vector<ir::Split::Arm> arms;
            do {
                std::optional<int64_t> value;
                switch (peek().type) {
                case Token::Type::INT_LITERAL: {
                    value = parse_int_literal();
                    break;
                }
                case Token::Type::IDENTIFIER: {
                    const std::string id = get_id();
                    if (id != "_") {
                        report_error() << "Unknown switch parameter id: " << id;
                    }
                    break;
                }
                default: {
                    // internal_error << "Unknown switch parameter." <<
                    // tokens();
                    report_error() << "Unknown switch parameter." << peek();
                }
                }
                if (std::any_of(arms.begin(), arms.end(), [&](const auto &arm) {
                        return arm.value == value;
                    })) {
                    report_error()
                        << "Duplicate switch parameter: "
                        << (value.has_value() ? std::to_string(*value) : "_");
                }
                expect(Token::Type::ASSIGN);
                expect(Token::Type::GT);
                std::optional<std::string> node_name;
                if (peek().type == Token::Type::IDENTIFIER &&
                    peek(1).type == Token::Type::LSQUIGGLE) {
                    // named split.
                    node_name = get_id();
                }
                ir::Layout inner = parse_layout();
                expect(Token::Type::SEMICOL);
                arms.push_back(
                    {std::move(value), std::move(node_name), std::move(inner)});
            } while (!consume(Token::Type::RSQUIGGLE));
            return ir::Split::make(std::move(name), std::move(arms));
        }
        default: {
            internal_error << "TODO: " << tokens();
        }
        }
    }
};

} // namespace

ir::Program parse(const std::string &filename) {
    TokenStream tokens = lex(filename);
    // Don't enforce types when building ASTs, might need to do some inference.
    ir::global_disable_type_enforcement();
    return Parser(tokens).parse_program();
}

} // namespace parser
} // namespace bonsai
