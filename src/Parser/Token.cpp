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

std::string Token::tokenTypeString(TokenType type) {
    switch (type) {
    case TokenType::INT_LITERAL:
        return "int";
    case TokenType::UINT_LITERAL:
        return "uint";
    case TokenType::FLOAT_LITERAL:
        return "float";
    case TokenType::STRING_LITERAL:
        return "string";
    case TokenType::IDENTIFIER:
        return "id";
    case TokenType::IMPORT:
        return "import";
    case TokenType::ELEMENT:
        return "element";
    case TokenType::INTERFACE:
        return "interface";
    case TokenType::EXTERN:
        return "extern";
    case TokenType::FUNC:
        return "func";
    case TokenType::MUT:
        return "mut";
    case TokenType::RARROW:
        return "rarrow";
    case TokenType::RETURN:
        return "return";
    case TokenType::FOR:
        return "for";
    case TokenType::IN:
        return "in";
    case TokenType::IF:
        return "if";
    case TokenType::ELIF:
        return "elif";
    case TokenType::ELSE:
        return "else";
    case TokenType::TRUE:
        return "true";
    case TokenType::FALSE:
        return "false";
    case TokenType::LPAREN:
        return "lparen";
    case TokenType::RPAREN:
        return "rparen";
    case TokenType::LBRACKET:
        return "lbracket";
    case TokenType::RBRACKET:
        return "rbracket";
    case TokenType::LSQUIGGLE:
        return "lsquiggle";
    case TokenType::RSQUIGGLE:
        return "rsquiggle";
    case TokenType::BAR:
        return "bar";
    case TokenType::COMMA:
        return "comma";
    case TokenType::PERIOD:
        return "period";
    case TokenType::COL:
        return "col";
    case TokenType::SEMICOL:
        return "semicol";
    case TokenType::ASSIGN:
        return "assign";
    case TokenType::AND:
        return "and";
    case TokenType::AT:
        return "at";
    case TokenType::LOR:
        return "logical-or";
    case TokenType::XOR:
        return "xor";
    case TokenType::NOT:
        return "not";
    case TokenType::PLUS:
        return "plus";
    case TokenType::INC:
        return "inc";
    case TokenType::MINUS:
        return "minus";
    case TokenType::DEC:
        return "dec";
    case TokenType::STAR:
        return "star";
    case TokenType::SLASH:
        return "slash";
    case TokenType::MOD:
        return "mod";
    case TokenType::EQ:
        return "eq";
    case TokenType::NEQ:
        return "neq";
    case TokenType::LEQ:
        return "leq";
    case TokenType::GEQ:
        return "geq";
    case TokenType::LT:
        return "lt";
    case TokenType::GT:
        return "gt";
    case TokenType::ERROR:
        return "error";
    default:
        internal_error << "Unknown token type: " << static_cast<int>(type);
        return "";
    }
}

std::string Token::toString() const {
    switch (type) {
    case TokenType::INT_LITERAL: {
        internal_assert(std::holds_alternative<int64_t>(value))
            << "Expected INT_LITERAL to hold integer";
        return "'" + std::to_string(std::get<int64_t>(value)) + "'";
    }
    case TokenType::UINT_LITERAL: {
        internal_assert(std::holds_alternative<uint64_t>(value))
            << "Expected UINT_LITERAL to hold unsigned integer";
        return "'" + std::to_string(std::get<uint64_t>(value)) + "'";
    }
    case TokenType::FLOAT_LITERAL: {
        internal_assert(std::holds_alternative<double>(value))
            << "Expected FLOAT_LITERAL to hold double";
        return "'" + std::to_string(std::get<double>(value)) + "'";
    }
    case TokenType::STRING_LITERAL: {
        internal_assert(std::holds_alternative<std::string>(value))
            << "Expected STRING_LITERAL to hold string";
        const std::string &str = std::get<std::string>(value);
        return "'\"" + escape(str) + "\"'";
    }
    case TokenType::IDENTIFIER: {
        internal_assert(std::holds_alternative<std::string>(value))
            << "Expected INDENTIFIER to hold string";
        return std::get<std::string>(value);
    }
    default:
        return tokenTypeString(type);
    }
}

std::ostream &operator<<(std::ostream &out, const Token &token) {
    out << "(" << Token::tokenTypeString(token.type);
    switch (token.type) {
    case TokenType::INT_LITERAL:
        out << ", " << std::get<int64_t>(token.value);
        break;
    case TokenType::UINT_LITERAL:
        out << ", " << std::get<uint64_t>(token.value);
        break;
    case TokenType::FLOAT_LITERAL:
        out << ", " << std::get<double>(token.value);
        break;
    case TokenType::STRING_LITERAL:
        out << ", \"" << std::get<std::string>(token.value) << "\"";
        break;
    case TokenType::IDENTIFIER:
        out << ", " << std::get<std::string>(token.value);
        break;
    default:
        break;
    }
    out << ", " << token.lineBegin << ":" << token.colBegin << "-"
        << token.lineEnd << ":" << token.colEnd << ")";
    return out;
}

void TokenStream::addToken(Token::Type type, uint64_t line, uint64_t column,
                           uint32_t length) {
    tokens.push_back(Token{
        .type = type,
        .lineBegin = line,
        .colBegin = column,
        .lineEnd = line,
        .colEnd = column + length - 1,
    });
}

bool TokenStream::consume(TokenType type) {
    if (tokens.front().type == type) {
        tokens.pop_front();
        return true;
    }

    return false;
}

Token TokenStream::peek(uint32_t count) const {
    if (count == 0) {
        if (tokens.empty()) {
            Token endToken = Token();
            endToken.type = TokenType::ERROR;
            return endToken;
        }
        return tokens.front();
    }

    std::list<Token>::const_iterator it = tokens.cbegin();
    for (unsigned i = 0; i < count && it != tokens.cend(); ++i, ++it) {
    }

    if (it == tokens.cend()) {
        Token endToken = Token();
        endToken.type = TokenType::ERROR;
        return endToken;
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
