#include "Parser/Token.h"

#include <sstream>

#include "Error.h"

namespace bonsai {
namespace parser {

namespace {
std::string escape(const std::string &str) {
    std::stringstream oss;

    for (const auto s : str) {
        switch (s) {
        case '\a':
            oss << "\\a";
            break;
        case '\b':
            oss << "\\b";
            break;
        case '\f':
            oss << "\\f";
            break;
        case '\n':
            oss << "\\n";
            break;
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        case '\v':
            oss << "\\v";
            break;
        case '\\':
            oss << "\\\\";
            break;
        case '\'':
            oss << "\\'";
            break;
        case '\"':
            oss << "\\\"";
            break;
        case '\?':
            oss << "\\?";
            break;
        default:
            oss << s;
            break;
        }
    }

    return oss.str();
}
} // namespace

std::string Token::token_type_string(Token::Type type) {
    switch (type) {
    case Token::Type::INT_LITERAL:
        return "int";
    case Token::Type::UINT_LITERAL:
        return "uint";
    case Token::Type::FLOAT_LITERAL:
        return "float";
    case Token::Type::STRING_LITERAL:
        return "string";
    case Token::Type::IDENTIFIER:
        return "id";
    case Token::Type::IMPORT:
        return "import";
    case Token::Type::ELEMENT:
        return "element";
    case Token::Type::INTERFACE:
        return "interface";
    case Token::Type::EXTERN:
        return "extern";
    case Token::Type::FUNC:
        return "func";
    case Token::Type::MUT:
        return "mut";
    case Token::Type::RARROW:
        return "rarrow";
    case Token::Type::RETURN:
        return "return";
    case Token::Type::PRINT:
        return "print";
    case Token::Type::FOR:
        return "for";
    case Token::Type::IN:
        return "in";
    case Token::Type::IF:
        return "if";
    case Token::Type::ELIF:
        return "elif";
    case Token::Type::ELSE:
        return "else";
    case Token::Type::TRUE:
        return "true";
    case Token::Type::FALSE:
        return "false";
    case Token::Type::LPAREN:
        return "lparen";
    case Token::Type::RPAREN:
        return "rparen";
    case Token::Type::LBRACKET:
        return "lbracket";
    case Token::Type::RBRACKET:
        return "rbracket";
    case Token::Type::LSQUIGGLE:
        return "lsquiggle";
    case Token::Type::RSQUIGGLE:
        return "rsquiggle";
    case Token::Type::BAR:
        return "bar";
    case Token::Type::COMMA:
        return "comma";
    case Token::Type::PERIOD:
        return "period";
    case Token::Type::COL:
        return "col";
    case Token::Type::SEMICOL:
        return "semicol";
    case Token::Type::ASSIGN:
        return "assign";
    case Token::Type::AND:
        return "and";
    case Token::Type::AT:
        return "at";
    case Token::Type::LOR:
        return "logical-or";
    case Token::Type::XOR:
        return "xor";
    case Token::Type::NOT:
        return "not";
    case Token::Type::PLUS:
        return "plus";
    case Token::Type::INC:
        return "inc";
    case Token::Type::MINUS:
        return "minus";
    case Token::Type::DEC:
        return "dec";
    case Token::Type::STAR:
        return "star";
    case Token::Type::SLASH:
        return "slash";
    case Token::Type::MOD:
        return "mod";
    case Token::Type::EQ:
        return "eq";
    case Token::Type::NEQ:
        return "neq";
    case Token::Type::LEQ:
        return "leq";
    case Token::Type::GEQ:
        return "geq";
    case Token::Type::LT:
        return "lt";
    case Token::Type::GT:
        return "gt";
    case Token::Type::ERROR:
        return "error";
    default:
        internal_error << "Unknown token type: " << static_cast<int>(type);
        return "";
    }
}

std::string Token::to_string() const {
    switch (type) {
    case Token::Type::INT_LITERAL: {
        internal_assert(std::holds_alternative<int64_t>(value))
            << "Expected INT_LITERAL to hold integer";
        return "'" + std::to_string(std::get<int64_t>(value)) + "'";
    }
    case Token::Type::UINT_LITERAL: {
        internal_assert(std::holds_alternative<uint64_t>(value))
            << "Expected UINT_LITERAL to hold unsigned integer";
        return "'" + std::to_string(std::get<uint64_t>(value)) + "'";
    }
    case Token::Type::FLOAT_LITERAL: {
        internal_assert(std::holds_alternative<double>(value))
            << "Expected FLOAT_LITERAL to hold double";
        return "'" + std::to_string(std::get<double>(value)) + "'";
    }
    case Token::Type::STRING_LITERAL: {
        internal_assert(std::holds_alternative<std::string>(value))
            << "Expected STRING_LITERAL to hold string";
        const std::string &str = std::get<std::string>(value);
        return "'\"" + escape(str) + "\"'";
    }
    case Token::Type::IDENTIFIER: {
        internal_assert(std::holds_alternative<std::string>(value))
            << "Expected INDENTIFIER to hold string";
        return std::get<std::string>(value);
    }
    default:
        return token_type_string(type);
    }
}

std::ostream &operator<<(std::ostream &out, const Token &token) {
    out << "(" << Token::token_type_string(token.type);
    switch (token.type) {
    case Token::Type::INT_LITERAL:
        out << ", " << std::get<int64_t>(token.value);
        break;
    case Token::Type::UINT_LITERAL:
        out << ", " << std::get<uint64_t>(token.value);
        break;
    case Token::Type::FLOAT_LITERAL:
        out << ", " << std::get<double>(token.value);
        break;
    case Token::Type::STRING_LITERAL:
        out << ", \"" << std::get<std::string>(token.value) << "\"";
        break;
    case Token::Type::IDENTIFIER:
        out << ", " << std::get<std::string>(token.value);
        break;
    default:
        break;
    }
    out << ", " << token.lineBegin << ":" << token.colBegin << "-"
        << token.lineEnd << ":" << token.colEnd << ")";
    return out;
}

void TokenStream::add_token(Token::Type type, uint64_t line, uint64_t column,
                            uint32_t length) {
    tokens.push_back(Token{
        .type = type,
        .lineBegin = line,
        .colBegin = column,
        .lineEnd = line,
        .colEnd = column + length - 1,
    });
}

bool TokenStream::consume(Token::Type type) {
    if (tokens.front().type == type) {
        tokens.pop_front();
        return true;
    }

    return false;
}

Token TokenStream::peek(uint32_t count) const {
    if (count == 0) {
        if (tokens.empty()) {
            return Token{.type = Token::Type::ERROR};
        }
        return tokens.front();
    }

    std::list<Token>::const_iterator it = tokens.cbegin();
    for (unsigned i = 0; i < count && it != tokens.cend(); ++i, ++it) {
    }

    if (it == tokens.cend()) {
        Token end_token = Token();
        end_token.type = Token::Type::ERROR;
        return end_token;
    }

    return *it;
}

std::ostream &operator<<(std::ostream &out, const TokenStream &tokens) {
    for (auto it = tokens.tokens.cbegin(); it != tokens.tokens.cend(); ++it) {
        out << *it << std::endl;
    }
    return out;
}

} // namespace parser
} // namespace bonsai
