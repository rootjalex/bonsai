#include "Parser/Parser.h"

#include "Parser/Lexer.h"
#include "Parser/Token.h"

#include "IR/Analysis.h"
#include "IR/Equality.h"
#include "IR/Printer.h"
#include "IR/Type.h"
#include "IR/TypeEnforcement.h"

#include <iostream>
#include <regex>
#include <span>
#include <sstream>

#include "Error.h"
#include "Utils.h"

namespace bonsai {
namespace parser {

namespace {

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
  private:
    TokenStream tokens;
    ir::Program program;
    std::list<std::map<std::string, std::pair<ir::Type, bool>>> frames;
    // TODO: if we allow nested functions or any other way to allow nested
    // generics, we need this to be a stack!
    std::map<std::string, ir::Type> current_generics;
    const ir::Type u32 = ir::UInt_t::make(32), i32 = ir::Int_t::make(32),
                   f32 = ir::Float_t::make_f32();

    ir::Type get_type_from_frame(const std::string &name) const {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            if (found != frame.cend()) {
                return found->second.first;
            }
        }
        internal_error << "Cannot check type of unknown var: " << name
                       << " required at line: " << peek().lineBegin;
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
        internal_error << "Cannot check mutability of unknown var: " << name;
        return false;
    }

    void add_type_to_frame(const std::string &name, ir::Type type, bool mut) {
        for (auto it = frames.rbegin(); it != frames.rend(); it++) {
            const auto &frame = *it;
            const auto &found = frame.find(name);
            internal_assert(found == frame.cend())
                << name << " shadows another variable (of the same name)";
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
        internal_error << "Cannot modify_type of unknown var: " << name
                       << " to type " << type;
    }

    void new_frame() { frames.emplace_back(); }

    void end_frame() { frames.pop_back(); }

  public:
    Parser(TokenStream _tokens) : tokens(std::move(_tokens)) {
        // Add Intrinsic types!
    }

    ir::Program parseProgram() {
        internal_assert(frames.empty());
        new_frame();
        while (!tokens.empty()) {
            parseProgramElement();
        }
        end_frame();
        internal_assert(frames.empty());
        return std::move(program);
    }

  private:
    Token peek(uint32_t k = 0) const { return tokens.peek(k); }

    std::optional<Token> consume(Token::Type type) {
        const Token token = peek();

        if (!tokens.consume(type)) {
            return {};
        }

        return token;
    }

    Token expect(Token::Type type) {
        if (std::optional<Token> token = consume(type)) {
            return *token;
        }
        internal_error << "Expected " << Token::token_type_string(type)
                       << ", instead received: " + peek().to_string()
                       << " at line: " << peek().lineBegin << ":"
                       << peek().colBegin;
    }

    void parseProgramElement() {
        switch (peek().type) {
        case Token::Type::IMPORT:
            return parseImport();
        case Token::Type::ELEMENT:
            return parseElement();
        case Token::Type::INTERFACE:
            return parseInterfaceDef();
        case Token::Type::EXTERN:
            return parseExtern();
        case Token::Type::FUNC:
            return parseFunction();
        default: {
            internal_error << "Failure in parseProgramElement: " +
                                  peek().to_string();
        }
        }
    }

    void parseImport() {
        // Imports are essentially code inlining. Just load and parse the file.
        // TODO: support source/header separations?
        expect(Token::Type::IMPORT);
        const Token id = expect(Token::Type::IDENTIFIER);
        std::string name = std::get<std::string>(id.value);

        // Handle (/name)*
        while (!consume(Token::Type::SEMICOL)) {
            expect(Token::Type::SLASH);
            const Token path = expect(Token::Type::IDENTIFIER);
            name += "/" + std::get<std::string>(path.value);
        }

        name += ".bonsai";

        ir::Program imported;
        try {
            // TODO: recursive imports break everything. Fix this somehow?
            imported = parse(name);
        } catch (const std::exception &e) {
            internal_error << "Failure to parse imported file: " << name
                           << " with message: " << e.what();
        }

        // Can't import externs, those are function arguments to the generated
        // stub.
        internal_assert(imported.externs.empty())
            << "Imported file: " << name << " contains externs.";

        for (auto &[fname, func] : imported.funcs) {
            internal_assert(!program.funcs.contains(fname))
                << "Redefinition of function: " << fname
                << " from imported file: " << name;
            program.funcs[fname] = std::move(func);
        }

        for (auto &[tname, type] : imported.types) {
            internal_assert(!program.types.contains(tname))
                << "Redefinition of type: " << tname
                << " from imported file: " << name;
            program.types[tname] = std::move(type);
        }
    }

    void parseElement() {
        expect(Token::Type::ELEMENT);

        // TODO: support methods as well.
        // TODO: figure out overloading policy for that.
        // TODO: for error handling, should we have beginLoc/endLoc like Simit?
        const Token id = expect(Token::Type::IDENTIFIER);
        const std::string name = std::get<std::string>(id.value);

        internal_assert(!program.types.contains(name))
            << "Redefinition of type: " << name << " on line " << id.lineBegin;

        // Support inline aliasing.
        if (consume(Token::Type::ASSIGN)) {
            ir::Type alias = parseType();
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
                const Token field = expect(Token::Type::IDENTIFIER);
                const std::string field_name =
                    std::get<std::string>(field.value);
                names.push_back(field_name);
                if (consume(Token::Type::ASSIGN)) {
                    ir::Expr _default = parseExpr();
                    internal_assert(ir::is_constant_expr(_default))
                        << "Field default values must be constants, received: "
                        << _default << " for field " << field_name
                        << " of element " << name;
                    defaults[field_name] = std::move(_default);
                }
            } while (consume(Token::Type::COMMA));

            expect(Token::Type::COL);
            ir::Type type = parseType();
            for (const auto &field_name : names) {
                internal_assert(fields.cend() ==
                                std::find_if(fields.cbegin(), fields.cend(),
                                             [&field_name](const auto &p) {
                                                 return p.first == field_name;
                                             }))
                    << "Duplicate field name: " << field_name
                    << " in element definition: " << name;
                fields.emplace_back(field_name, type);
                if (defaults.contains(field_name)) {
                    defaults[field_name] =
                        constant_cast(type, defaults[field_name]);
                }
            }

            if (consume(Token::Type::ASSIGN)) {
                ir::Expr _default = parseExpr();
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

    void parseInterfaceDef() {
        internal_error << "TODO: implement parseInterfaceDef";
    }

    void parseExtern() {
        // TODO: what if an extern name-conflicts with a type or something?
        // probably should check conflicts for all symbols?
        expect(Token::Type::EXTERN);
        const Token id = expect(Token::Type::IDENTIFIER);
        const std::string name = std::get<std::string>(id.value);
        internal_assert(
            program.externs.cend() ==
            std::find_if(program.externs.cbegin(), program.externs.cend(),
                         [&name](const auto &p) { return p.first == name; }))
            << "Redefinition of extern: " << name << " on line "
            << id.lineBegin;
        // TODO: should we support defaults? that makes passing in args harder.
        expect(Token::Type::COL);
        ir::Type type = parseType();
        expect(Token::Type::SEMICOL);
        add_type_to_frame(name, type, /* mutable */ false);
        program.externs.emplace_back(name, std::move(type));
    }

    void parseFunction() {
        expect(Token::Type::FUNC);
        const Token id = expect(Token::Type::IDENTIFIER);
        const std::string name = std::get<std::string>(id.value);
        internal_assert(!program.funcs.contains(name))
            << "Redefinition of func: " << name << " on line " << id.lineBegin;

        ir::Function::InterfaceList interfaces;
        if (consume(Token::Type::LT)) {
            // Parse generics.
            internal_assert(current_generics.empty())
                << "Nested generics in definition of: " << name;

            do {
                const Token itoken = expect(Token::Type::IDENTIFIER);
                const std::string iname = std::get<std::string>(itoken.value);
                internal_assert(!current_generics.contains(iname))
                    << "Duplicate interface name: " << iname
                    << " in definition of func: " << name;
                ir::Interface interface = consume(Token::Type::COL).has_value()
                                              ? parseInterface()
                                              : ir::IEmpty::make();
                interfaces.emplace_back(iname, interface);
                current_generics[iname] =
                    ir::Generic_t::make(iname, std::move(interface));

            } while (consume(Token::Type::COMMA));
            expect(Token::Type::GT);
        }

        expect(Token::Type::LPAREN);

        new_frame();
        std::vector<ir::Function::Argument> args;
        if (peek().type != Token::Type::RPAREN) {
            // parse arg list
            do {
                // TODO: can we accept multiple args with one type here, as in
                // element definitions?
                const Token arg_id = expect(Token::Type::IDENTIFIER);
                const std::string arg_name =
                    std::get<std::string>(arg_id.value);
                expect(Token::Type::COL);
                // TODO: handle `mut`! Use ParseNameDef?
                ir::Type type = parseType();

                ir::Expr default_value;

                if (consume(Token::Type::ASSIGN)) {
                    // Optional default value.
                    // TODO: default value may need to be cast to type.
                    // For now, assume that's done in type inference.
                    // TODO: this should not perform computation!
                    // Can we easily prevent that? Enforce is_constant?
                    default_value = parseExpr();
                    internal_assert(ir::is_constant_expr(default_value))
                        << "Function default values must be constants, "
                           "received: "
                        << default_value << " for argument " << arg_name
                        << " of func " << name;
                }

                add_type_to_frame(arg_name, type,
                                  /* mutable */ false); // TODO: handle mutable
                                                        // args in functions.
                args.push_back(ir::Function::Argument{
                    arg_name, std::move(type), std::move(default_value)});
            } while (consume(Token::Type::COMMA));
        }
        expect(Token::Type::RPAREN);

        // Optional RARROW with return_type: otherwise, requires type inference!
        ir::Type ret_type;
        if (consume(Token::Type::RARROW)) {
            ret_type = parseType();
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
            std::move(interfaces));

        ir::Stmt body;

        // TODO: the syntax of -> type = expr is ugly, maybe disallow, and just
        // do inference?

        if (consume(Token::Type::ASSIGN)) {
            ir::Expr expr = parseExpr();
            if (!ret_type_set && expr.type().defined()) {
                ret_type = expr.type();
            }
            expect(Token::Type::SEMICOL);
            body = ir::Return::make(expr);
        } else {
            body = parseSequence();
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

    ir::Stmt parseSequence() {
        std::vector<ir::Stmt> stmts;
        expect(Token::Type::LSQUIGGLE);
        // brackets enclose a new frame!
        new_frame();

        while (!consume(Token::Type::RSQUIGGLE)) {
            stmts.push_back(parseStmt());
        }
        // close the frame.
        end_frame();
        internal_assert(!stmts.empty())
            << "Failed to parse a Sequence at line: "
            << peek().lineBegin; // TODO: this fails when the stream is empty
        if (stmts.size() == 1) {
            return std::move(stmts[0]);
        } else {
            return ir::Sequence::make(std::move(stmts));
        }
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
    ir::Stmt parseStmt() {
        if (peek().type == Token::Type::LSQUIGGLE) {
            return parseSequence();
        } else if (consume(Token::Type::IF)) {
            ir::Expr cond = parseExpr(); // no required parens
            ir::Stmt then_case = parseStmt();
            internal_assert(!consume(Token::Type::ELIF))
                << "TODO: implement elif parsing for line: "
                << peek().lineBegin;
            if (consume(Token::Type::ELSE)) {
                ir::Stmt else_case = parseStmt();
                return ir::IfElse::make(std::move(cond), std::move(then_case),
                                        std::move(else_case));
            } else {
                return ir::IfElse::make(std::move(cond), std::move(then_case));
            }
        } else if (consume(Token::Type::RETURN)) {
            ir::Expr ret = parseExpr();
            expect(Token::Type::SEMICOL);
            return ir::Return::make(std::move(ret));
        } else if (consume(Token::Type::PRINT)) {
            expect(Token::Type::LPAREN);
            ir::Expr value = parseExpr();
            expect(Token::Type::RPAREN);
            expect(Token::Type::SEMICOL);
            return ir::Print::make(value);
        } else if (peek().type == Token::Type::IDENTIFIER) {
            // TODO: allow tuple declaration/assignment?
            // TODO: how to do SSA in parsing?
            ir::WriteLoc loc = parseWriteLoc();
            if (loc.accesses.empty() && !name_in_scope(loc.base)) {
                // Just a regular variable write
                // might be an Assign (if labelled `mut`)
                return parseNameDecl(std::move(loc));
            } else if (consume(Token::Type::ASSIGN)) {
                return parseAssign(std::move(loc));
            } else {
                // Must be an accumulate.
                return parseAccumulate(std::move(loc));
            }
        }
        internal_error << "TODO: implement parseStmt for "
                       << peek().to_string();
    }

    ir::Stmt parseNameDecl(ir::WriteLoc loc) {
        // This is a never-seen-before write to a variable.
        internal_assert(!loc.type.defined());
        ir::Type type_label;
        bool _mutable = false;

        if (consume(Token::Type::COL)) {
            if (consume(Token::Type::MUT)) {
                _mutable = true;
                if (peek().type == Token::Type::IDENTIFIER) {
                    type_label = parseType();
                } // otherwise just a `mut` label.
                // TODO: should we ever allow "just" a mut label?
            } else {
                type_label = parseType();
            }
        }

        expect(Token::Type::ASSIGN);
        ir::Expr value = parseExpr();
        ir::Type type = value.type();
        expect(Token::Type::SEMICOL);

        if (const auto *build = value.as<ir::Build>();
            build && !type.defined()) {
            // When assigning a vector with an initializer list, this may occur,
            // e.g., `v: vector[i32, 2] = {1, 2};`, which is parsed as:
            //       `v: vector[i32, 2] = build<unknown>((i32)1, (i32)2)`
            value = ir::Build::make(type_label, std::move(build->values));
        }

        // TODO: do type-forcing here!
        if (type_label.defined() && type.defined()) {
            internal_assert(ir::equals(type_label, value.type()))
                << "Mismatching assignment: " << loc
                << " is labelled with type: " << type_label << " but " << value
                << " has type " << type;
        }
        ir::Type write_type = type_label.defined() ? type_label : type;

        add_type_to_frame(loc.base, write_type, _mutable);

        if (!_mutable) {
            return ir::LetStmt::make(std::move(loc), std::move(value));
        } else {
            loc = ir::WriteLoc(loc.base, std::move(write_type));
            return ir::Assign::make(std::move(loc), std::move(value),
                                    /*mutating*/ false);
        }
    }

    ir::Stmt parseAssign(ir::WriteLoc loc) {
        ir::Expr value = parseExpr();
        expect(Token::Type::SEMICOL);

        // TODO: do type forcing here!
        if (loc.type.defined() && value.type().defined()) {
            internal_assert(ir::equals(loc.type, value.type()))
                << "Mismatching assignment: " << loc
                << " has type: " << loc.type << " but " << value << " has type "
                << value.type();
        }

        const bool mutating = name_in_scope(loc.base);

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

    ir::Stmt parseAccumulate(ir::WriteLoc loc) {
        ir::Accumulate::OpType op = ir::Accumulate::OpType::Add;
        // Try to parse an accumulate
        if (consume(Token::Type::PLUS)) {
            op = ir::Accumulate::OpType::Add;
        } else if (consume(Token::Type::STAR)) {
            op = ir::Accumulate::OpType::Mul;
        } else {
            internal_error << "Unknown token when parsing Accumulate at line: "
                           << peek().lineBegin;
        }
        expect(Token::Type::ASSIGN);
        ir::Expr value = parseExpr();
        expect(Token::Type::SEMICOL);
        return ir::Accumulate::make(std::move(loc), op, std::move(value));
    }

    // We follow C++'s operator precedence.
    // https://en.cppreference.com/w/cpp/language/operator_precedence

    // TODO: precedence 2: member access...
    // precedence 5: mul, div, mod
    // precedence 6: addition/subtraction
    // precedence 7: bit shifts (TODO: support)
    // precedence 9: relational operators
    // precedence 10: equality operators
    // TODO: we parse and/xor/or and treat them all the same for now.
    // precedence 11: bitwise and (TODO: support)
    // precedence 12: bitwise xor (TODO: support)
    // precedence 13: bitwise or (TODO: support)
    // precedence 14: logical and
    // precedence 15: logical or

    // TODO: parse member access first somehow...?
    // expr := muldivmod_expr
    ir::Expr parseExpr() { return parseOr(); }

    struct BinOperator {
        const ir::BinOp::OpType op;
        const Token::Type token;
        const bool flip = false;
    };

    template <size_t N, typename F>
    ir::Expr parseBinOpWithPrecedence(F parseSubExpr,
                                      const std::array<BinOperator, N> &ops) {
        ir::Expr expr = parseSubExpr();
        while (true) {
            // Find the first matching operation
            auto it =
                std::find_if(ops.begin(), ops.end(), [this](const auto &op) {
                    return consume(op.token);
                });

            // If no operation is found, we're done here.
            if (it == ops.end()) {
                return expr;
            }

            // Parse the right-hand side and create the binary operation
            ir::Expr rhs = parseSubExpr();
            if (it->flip) {
                expr = ir::BinOp::make(it->op, std::move(rhs), std::move(expr));
            } else {
                expr = ir::BinOp::make(it->op, std::move(expr), std::move(rhs));
            }
        }
    }

    // muldivmod_expr := addsub_expr (('*' | '/' | '%') addsub_expr)*
    ir::Expr parseMulDivMod() {
        return parseBinOpWithPrecedence<3>(
            [this]() { return parseBaseExpr(); },
            {{
                {ir::BinOp::Mul, Token::Type::STAR},
                {ir::BinOp::Div, Token::Type::SLASH},
                {ir::BinOp::Mod, Token::Type::MOD},
            }});
    }

    // addsub_expr := rel_expr (('+' | '-') rel_expr)*
    ir::Expr parseAddSub() {
        return parseBinOpWithPrecedence<2>(
            [this]() { return parseMulDivMod(); },
            {{
                {ir::BinOp::Add, Token::Type::PLUS},
                {ir::BinOp::Sub, Token::Type::MINUS},
            }});
    }

    // rel_expr := eq_expr (('<=' | '<') eq_expr)*
    ir::Expr parseRels() {
        return parseBinOpWithPrecedence<4>(
            [this]() { return parseAddSub(); },
            {{
                {ir::BinOp::Lt, Token::Type::LT},
                {ir::BinOp::Le, Token::Type::LEQ},
                {ir::BinOp::Lt, Token::Type::GT, /* flip */ true},
                {ir::BinOp::Le, Token::Type::GEQ, /* flip */ true},
            }});
    }

    // eq_expr := and_expr (('==' | '!=') and_expr)*
    ir::Expr parseEqs() {
        return parseBinOpWithPrecedence<2>(
            [this]() { return parseRels(); },
            {{
                {ir::BinOp::Eq, Token::Type::EQ},
                {ir::BinOp::Neq, Token::Type::NEQ},
            }});
    }

    // and_expr := xor_expr ('^' xor_expr)*
    ir::Expr parseAnd() {
        return parseBinOpWithPrecedence<1>(
            [this]() { return parseEqs(); },
            {{{ir::BinOp::And, Token::Type::AND}}});
    }

    // xor_expr := or_expr ('^' or_expr)*
    ir::Expr parseXor() {
        return parseBinOpWithPrecedence<1>(
            [this]() { return parseAnd(); },
            {{{ir::BinOp::Xor, Token::Type::XOR}}});
    }

    // or_expr := base_expr ('||' base_expr)*
    ir::Expr parseOr() {
        return parseBinOpWithPrecedence<1>(
            // TODO: logical and!
            [this]() { return parseXor(); },
            {{{ir::BinOp::Or, Token::Type::LOR}}});
    }

    // base_expr := '(' expr ')' | name (('.' field) | ('[' index (, index)* ']'
    // |
    // () ) | lambda args? : expr
    ir::Expr parseBaseExpr() {
        if (consume(Token::Type::LPAREN)) {
            ir::Expr inner = parseExpr();
            expect(Token::Type::RPAREN);
            return inner;
            // TODO: do these have the correct precedence?
        } else if (consume(Token::Type::MINUS)) {
            ir::Expr inner = parseExpr();
            return ir::UnOp::make(ir::UnOp::Neg, std::move(inner));
        } else if (consume(Token::Type::NOT)) {
            ir::Expr inner = parseExpr();
            return ir::UnOp::make(ir::UnOp::Not, std::move(inner));
        } else if (peek().type == Token::Type::IDENTIFIER) {
            return parseIdentifier();
            // Parse literals.
        } else if (consume(Token::Type::TRUE)) {
            return ir::BoolImm::make(true);
        } else if (consume(Token::Type::FALSE)) {
            return ir::BoolImm::make(false);
        } else if (peek().type == Token::Type::INT_LITERAL) {
            // can't know concrete type yet, let type inference figure it out.
            const Token token = expect(Token::Type::INT_LITERAL);
            const int64_t value = std::get<int64_t>(token.value);
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
            std::vector<ir::Lambda::Argument> args = parseLambdaArgs();
            new_frame();
            for (const auto &arg : args) {
                add_type_to_frame(arg.name, arg.type, /* mutable */ false);
            }
            // Optionally allow squiggles for lambda expression body.
            const bool hasSquiggles =
                consume(Token::Type::LSQUIGGLE).has_value();
            ir::Expr expr = parseExpr();
            if (hasSquiggles)
                expect(Token::Type::RSQUIGGLE);
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
                    parseExprListUntil(Token::Type::RSQUIGGLE);
                // TODO: can we know the type?
                return ir::Build::make(ir::Type(), std::move(args));
            }
            // TODO: should also support e.g. Type{} notation.
        } else if (consume(Token::Type::STAR)) {
            // Option dereference.
            ir::Expr arg = parseIdentifier();
            ir::Type atype = arg.type();
            internal_assert(atype.defined() && atype.is<ir::Option_t>())
                << "Parsed dereference of non-option: " << arg;
            ir::Type etype = atype.as<ir::Option_t>()->etype;
            // TODO(ajr): do we want an explicit Deref IR node?
            return ir::Cast::make(std::move(etype), std::move(arg));
        } else {
            internal_error << "Unknown token in parseBaseExpr: "
                           << peek().to_string()
                           << " at line: " << peek().lineBegin;
        }
    }

    // TODO(cgyurgyik): clean up parsing methods and add error reporting
    // throughout.
    template <typename OpType, typename Container>
    std::optional<OpType>
    try_match_pattern(const std::string &name, const size_t arg_count,
                      const Container &patterns, size_t n_args, size_t line,
                      size_t col) {
        for (const auto &p : patterns) {
            if (name == p.name) {
                if constexpr (requires { p.skippable; }) {
                    if (p.skippable && arg_count != p.n_args) {
                        return {};
                    }
                }
                if constexpr (requires { p.n_args; }) {
                    internal_assert(arg_count == p.n_args)
                        << p.name << " takes " << p.n_args
                        << " argument(s), received " << arg_count
                        << " instead, on line " << line << ":" << col;
                } else {
                    internal_assert(arg_count == n_args)
                        << p.name << " takes " << n_args
                        << " argument(s), received " << arg_count
                        << " instead, on line " << line << ":" << col;
                }

                return p.op;
            }
        }
        return {};
    }

    ir::Expr try_match_intrinsics(const std::string &name,
                                  std::vector<ir::Expr> args, size_t line,
                                  size_t col) {
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
            {"fma", 3, ir::Intrinsic::fma},
            // These two are skippable because they might be parsed as
            // single-argument reductions below.
            {"max", 2, ir::Intrinsic::max, .skippable = true},
            {"min", 2, ir::Intrinsic::min, .skippable = true},
            {"sin", 1, ir::Intrinsic::sin},
            {"sqrt", 1, ir::Intrinsic::sqrt},
        });

        if (auto op = try_match_pattern<ir::Intrinsic::OpType>(
                name, args.size(), IPATTERNS, 0, line, col)) {
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

        if (auto op = try_match_pattern<ir::SetOp::OpType>(
                name, args.size(), SPATTERNS, 2, line, col)) {
            return ir::SetOp::make(*op, std::move(args[0]), std::move(args[1]));
        }

        // Geometry operations
        struct GeomPattern {
            const std::string_view name;
            ir::GeomOp::OpType op;
        };

        static const auto GPATTERNS = std::to_array<GeomPattern>({
            {"distance", ir::GeomOp::distance},
            {"intersects", ir::GeomOp::intersects},
            {"contains", ir::GeomOp::contains},
        });

        if (auto op = try_match_pattern<ir::GeomOp::OpType>(
                name, args.size(), GPATTERNS, 2, line, col)) {
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
                name, args.size(), RPATTERNS, 2, line, col)) {
            return ir::VectorReduce::make(*op, std::move(args[0]));
        }

        return ir::Expr();
    }

    ir::Expr parseIdentifier() {
        const Token token = expect(Token::Type::IDENTIFIER);
        const std::string name = std::get<std::string>(token.value);
        std::vector<std::string> fields; // possibly nested
        while (consume(Token::Type::PERIOD)) {
            // Member access.
            const Token field_token = expect(Token::Type::IDENTIFIER);
            const std::string field_name =
                std::get<std::string>(field_token.value);
            fields.push_back(field_name);
        }

        // Only use if not an intrinsic!
        auto makeExpr = [&]() {
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
                template_types = parseTypeListUntil(Token::Type::RBRACKET);
                expect(Token::Type::RBRACKET);
                internal_assert(!template_types.empty())
                    << "Template syntax expects type arguments, but did not "
                       "receive "
                       "any for name: "
                    << name << " at line: " << token.lineBegin;
                internal_assert(peek().type == Token::Type::LPAREN)
                    << "Template syntax supported only for function calls, "
                       "found on "
                       "name: "
                    << name << " at line: " << token.lineBegin;
            } else {
                std::vector<ir::Expr> idxs =
                    parseExprListUntil(Token::Type::RBRACKET);
                internal_assert(!idxs.empty())
                    << "Indexing into array/vector expects at least one index "
                       "for "
                       "name: "
                    << name << " at line: " << token.lineBegin;
                ir::Expr expr = makeExpr();
                for (auto &idx : idxs) {
                    expr = ir::Extract::make(std::move(expr), std::move(idx));
                }
                return expr;
            }
        }

        if (consume(Token::Type::LPAREN)) {
            std::vector<ir::Expr> args =
                parseExprListUntil(Token::Type::RPAREN);

            if (fields.empty()) {
                // Checking program.funcs first means that users can override
                // built-in functions. That could be dangerous.
                if (program.funcs.contains(name)) {
                    const auto &func = program.funcs[name];
                    // TODO: handle default params!
                    internal_assert(args.size() == func->args.size())
                        << "Call to: " << name << " at line " << token.lineBegin
                        << " has incorrect number of arguments.\n"
                        << "Expected: " << func->args.size() << " but parsed "
                        << args.size() << " at line: " << token.lineBegin;

                    internal_assert(func->interfaces.size() ==
                                    template_types.size())
                        << "Call to: " << name << " at line " << token.lineBegin
                        << " has incorrect number of template paramters.\n"
                        << "Expected: " << func->interfaces.size()
                        << " but parsed " << template_types.size()
                        << " at line: " << token.lineBegin;

                    const size_t n_generics = func->interfaces.size();

                    std::map<std::string, ir::Type> instantiations;
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
                        internal_assert(
                            !args[i].type().defined() ||
                            ir::equals(expected_type, args[i].type()))
                            << "Argument " << i << " of call to function "
                            << name << " on line " << token.lineBegin
                            << " has incorrect type. Expected " << expected_type
                            << " but parsed: " << args[i].type();
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
                    internal_assert(template_types.empty())
                        << "Error: cannot pass template parameters to lambda "
                        << name << " definition on line: " << token.lineBegin
                        << ":" << token.colBegin;
                    ir::Type var_type =
                        get_type_from_frame(name); // never undefined.
                    ir::Expr expr = ir::Var::make(var_type, name);
                    return ir::Call::make(std::move(expr), std::move(args));
                }

                // Special built-ins with template parameters.
                if (name == "cast") {
                    internal_assert(template_types.size() == 1)
                        << "cast() expects a template parameter, instead "
                           "received: "
                        << template_types.size()
                        << " at line: " << token.lineBegin;
                    internal_assert(args.size() == 1)
                        << "cast takes a single argument, received: "
                        << args.size() << " at line: " << token.lineBegin;
                    return ir::Cast::make(std::move(template_types[0]),
                                          std::move(args[0]));
                } else if (name == "eps") {
                    internal_assert(template_types.size() == 1)
                        << "eps() expects a template parameter, instead "
                           "received: "
                        << template_types.size()
                        << " at line: " << token.lineBegin;
                    internal_assert(args.size() == 0)
                        << "eps takes no arguments, received: " << args.size()
                        << " at line: " << token.lineBegin;
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

                internal_assert(template_types.empty())
                    << name
                    << " does not take template parameters, but received: "
                    << template_types.size() << " at line: " << token.lineBegin;

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
                }

                ir::Expr intrinsic = try_match_intrinsics(
                    name, std::move(args), token.lineBegin, token.colBegin);
                if (intrinsic.defined()) {
                    return intrinsic;
                }

                // Not intrinsic or set op, not sure what this is.
                // TODO: could be a ctor of a type?
                internal_error << "Unknown function call " << name;
            } else {
                // method access!
                // TODO: type inference via interface?
                ir::Expr expr = makeExpr();
                internal_assert(template_types.empty())
                    << "TODO: support passing template types to a method "
                       "access: "
                    << expr << " received " << template_types.size()
                    << " at line: " << token.lineBegin;
                return ir::Call::make(std::move(expr), std::move(args));
            }
        }

        internal_assert(template_types.empty())
            << "TODO: support template arguments in constructors for: " << name
            << " at line: " << token.lineBegin;

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
                        const Token token = expect(Token::Type::IDENTIFIER);
                        const std::string field =
                            std::get<std::string>(token.value);
                        expect(Token::Type::ASSIGN);
                        ir::Expr value = parseExpr();
                        internal_assert(value.defined());
                        args[field] = std::move(value);
                    } while (consume(Token::Type::COMMA) &&
                             consume(Token::Type::PERIOD));
                    expect(Token::Type::RSQUIGGLE);
                    return ir::Build::make(std::move(type), std::move(args));
                }
                // Otherwise just a regular list-like struct build.
                auto args = parseExprListUntil(Token::Type::RSQUIGGLE);
                return ir::Build::make(std::move(type), std::move(args));
            }
            // otherwise ignore, not a struct build, e.g. maybe `if` expr { body
            // }
        }

        return makeExpr();
    }

    std::vector<ir::Expr> parseExprListUntil(const Token::Type &token) {
        std::vector<ir::Expr> exprs;
        if (consume(token)) {
            return exprs;
        }
        do {
            ir::Expr expr = parseExpr();
            exprs.emplace_back(std::move(expr));
        } while (consume(Token::Type::COMMA));
        expect(token);
        return exprs;
    }

    std::vector<ir::Type> parseTypeListUntil(const Token::Type &token) {
        std::vector<ir::Type> types;
        if (consume(token)) {
            return types;
        }
        do {
            ir::Type type = parseType();
            types.emplace_back(std::move(type));
        } while (consume(Token::Type::COMMA));
        expect(token);
        return types;
    }

    std::vector<ir::Lambda::Argument> parseLambdaArgs() {
        // arg := name (':' type)?
        // args := arg (',' arg)*
        // TODO: should we allow no arg lambdas?
        // not sure I want that for now. doesn't
        // that imply some sort of side effects?
        // but maybe we need that for rng?
        std::vector<ir::Lambda::Argument> args;
        do {
            auto def = parseNameDef<false, true>(false);
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
    NameDef parseNameDef(const bool expr_allowed) {
        NameDef def;

        const Token token = expect(Token::Type::IDENTIFIER);
        def.name = std::get<std::string>(token.value);

        if constexpr (T_REQUIRED) {
            expect(Token::Type::COL);
            if (consume(Token::Type::MUT)) {
                def.mut = true;
            }
            def.type = parseType();
        } else if (consume(Token::Type::COL)) {
            if (consume(Token::Type::MUT)) {
                def.mut = true;
                // might have no type label, just mut
                if (peek().type != Token::Type::ASSIGN) {
                    def.type = parseType();
                }
            } else {
                // no mut label, must be type.
                def.type = parseType();
            }
        }

        if (expr_allowed && consume(Token::Type::ASSIGN)) {
            def.value = parseExpr();
            if constexpr (E_CONST) {
                internal_assert(ir::is_constant_expr(def.value))
                    << "Expected constant value for name: " << def.name
                    << " but instead received: " << def.value;
            }
        }
        return def;
    }

    ir::WriteLoc parseWriteLoc() {
        const Token token = expect(Token::Type::IDENTIFIER);
        const std::string base = std::get<std::string>(token.value);
        ir::Type base_type;
        if (name_in_scope(base)) {
            base_type = get_type_from_frame(base);
        }

        ir::WriteLoc loc(base, base_type);

        while ((peek().type == Token::Type::PERIOD) ||
               (peek().type == Token::Type::LBRACKET)) {
            if (consume(Token::Type::PERIOD)) {
                const Token field = expect(Token::Type::IDENTIFIER);
                const std::string field_name =
                    std::get<std::string>(field.value);
                loc.add_struct_access(field_name);
            } else {
                expect(Token::Type::LBRACKET);
                ir::Expr index = parseExpr();
                loc.add_index_access(std::move(index));
            }
        }
        return loc;
    }

    // type := i[N] | u[N]
    //         | f[N] | bf[N] | f[N]_[N]
    //         | bool | vector[type, int]
    //         | option[type] | declared_type
    ir::Type parseType() {
        // TODO: support tuples of types! AKA unnamed structs.
        const Token id = expect(Token::Type::IDENTIFIER);
        const std::string name = std::get<std::string>(id.value);
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
                internal_assert(bits == 16)
                    << "brain float (bfloat) only comes in width 16";
                return ir::Float_t::make_bf16();
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
                internal_error << "unsupported floating point type: f" << bits;
            }
        } else if (name == "bool") {
            return ir::Bool_t::make();
        } else if (name == "void") {
            return ir::Void_t::make();
        }
        // Now look for built-ins
        else if (name == "vector") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parseType();
            expect(Token::Type::COMMA);
            const int64_t lanes = parseIntLiteral();
            // TODO: this upper bound is arbitrary. Can't imagine needing a
            // larger one though?
            internal_assert(lanes > 0 && lanes < 1025)
                << "Vector lane count is invalid: " << lanes;
            expect(Token::Type::RBRACKET);
            return ir::Vector_t::make(std::move(etype),
                                      static_cast<uint32_t>(lanes));
        } else if (name == "option") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parseType();
            expect(Token::Type::RBRACKET);
            internal_assert(!etype.is_bool())
                << "Bonsai does not support option[bool] because the semantics "
                   "are "
                   "confusing.";
            return ir::Option_t::make(std::move(etype));
        } else if (name == "set") {
            expect(Token::Type::LBRACKET);
            ir::Type etype = parseType();
            expect(Token::Type::RBRACKET);
            // TODO: assert etype is a struct_t? or volume_t?
            return ir::Set_t::make(std::move(etype));
        } else if (current_generics.contains(name)) {
            return current_generics[name];
        }
        // Must be a user-defined type, or an error.
        internal_assert(program.types.contains(name))
            << "Unknown type: " << name;
        return program.types[name];
    }

    // interface = iPrimitive | iVector[[interface]] | interface (`|`
    // interface)*
    ir::Interface parseInterface() {
        // TODO: Support interface unions and UDIs!
        return parsePrimitiveInterface();
    }

    ir::Interface parsePrimitiveInterface() {
        const Token id = expect(Token::Type::IDENTIFIER);
        const std::string name = std::get<std::string>(id.value);
        if (name == "IFloat") {
            return ir::IFloat::make();
        } else if (name == "IVector") {
            ir::Interface inner;
            if (consume(Token::Type::LBRACKET)) {
                expect(Token::Type::LBRACKET);
                inner = parseInterface();
                expect(Token::Type::RBRACKET);
                expect(Token::Type::RBRACKET);
            } else {
                inner = ir::IEmpty::make();
            }
            return ir::IVector::make(std::move(inner));
        } else if (current_generics.contains(name)) {
            return current_generics[name].as<ir::Generic_t>()->interface;
        }
        internal_error << "Unrecognized primitive interface: " << name;
    }

    int64_t parseIntLiteral() {
        // TODO: might need a "tryParseIntLiteral"...
        const Token _int = expect(Token::Type::INT_LITERAL);
        return std::get<int64_t>(_int.value);
    }
};

} // namespace

ir::Program parse(const std::string &filename) {
    TokenStream tokens = lex(filename);
    // Don't enforce types when building ASTs, might need to do some inference.
    ir::global_disable_type_enforcement();
    return Parser(tokens).parseProgram();
}

} // namespace parser
} // namespace bonsai
