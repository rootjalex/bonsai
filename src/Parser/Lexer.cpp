#include "Parser/Lexer.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "Error.h"

namespace bonsai {
namespace parser {

class Lexer {
  public:
    Lexer() {}

    // TODO: support better error handling?
    void lex(std::istream &);

    // Returns the current line number.
    uint64_t getLineNo() { return line; }

    // Returns the current column number.
    uint64_t getColumnNo() { return column; }

    void resetColumnNo() { column = 1; }
    void resetLineNo() { line = 1; }

    void incrColumnNo(uint32_t value = 1) { column += value; }
    void incrLineNo(uint32_t value = 1) { line += value; }

    void addToken(TokenType type, uint32_t length = 1) {
        stream.addToken(type, getLineNo(), getColumnNo(), length);
    }

    void addToken(Token token) { stream.addToken(token); }

    TokenStream getTokens() { return stream; }

  private:
    TokenStream stream;
    // The current column and line number during the tokenization phase.
    uint64_t column = 1;
    uint64_t line = 1;

    enum class ScanState { INITIAL, SLTEST, MLTEST };

    static TokenType getTokenType(std::string_view);

    void reportError(std::string_view message) {
        internal_error << "Parser error: " << message << "\n  on line "
                       << getLineNo() << ", column " << getColumnNo();
    }

    char handleEscapedChar(std::istream &programStream);
};

TokenType Lexer::getTokenType(const std::string_view token) {
    if (token == "import")
        return TokenType::IMPORT;
    if (token == "element")
        return TokenType::ELEMENT;
    if (token == "interface")
        return TokenType::INTERFACE;
    if (token == "extern")
        return TokenType::EXTERN;
    if (token == "func")
        return TokenType::FUNC;
    if (token == "mut")
        return Token::Type::MUT;
    if (token == "return")
        return TokenType::RETURN;
    if (token == "for")
        return TokenType::FOR;
    if (token == "if")
        return TokenType::IF;
    if (token == "elif")
        return TokenType::ELIF;
    if (token == "else")
        return TokenType::ELSE;
    if (token == "true")
        return TokenType::TRUE;
    if (token == "false")
        return TokenType::FALSE;

    // If string does not correspond to a keyword, assume it is an identifier.
    return TokenType::IDENTIFIER;
}

// Returns whether this is a valid start character of an identifier or keyword,
// i.e., [A-Za-z_]
static bool isValidIdentifierStart(int32_t c) {
    return c == '_' || std::isalpha(c);
}

void Lexer::lex(std::istream &programStream) {
    ScanState state = ScanState::INITIAL;

    while (programStream.peek() != EOF) {
        // Try to parse a name of the form [A-Za-z_][A-Za-z0-9_].
        if (isValidIdentifierStart(programStream.peek())) {
            std::string tokenString(1, programStream.get());
            // [A-Za-z0-9_]
            while (isValidIdentifierStart(programStream.peek()) ||
                   std::isdigit(programStream.peek())) {
                tokenString += programStream.get();
            }

            Token newToken{
                .type = getTokenType(tokenString),
                .lineBegin = getLineNo(),
                .colBegin = getColumnNo(),
                .lineEnd = getLineNo(),
                .colEnd = getColumnNo() + tokenString.length() - 1,
            };
            if (newToken.type == TokenType::IDENTIFIER) {
                newToken.value = tokenString;
            }
            addToken(newToken);
        } else {
            switch (programStream.peek()) {
            case '(':
                programStream.get();
                addToken(TokenType::LPAREN);
                break;
            case ')':
                programStream.get();
                addToken(TokenType::RPAREN);
                break;
            case '[':
                programStream.get();
                addToken(TokenType::LBRACKET);
                break;
            case ']':
                programStream.get();
                addToken(TokenType::RBRACKET);
                break;
            case '{':
                programStream.get();
                addToken(TokenType::LSQUIGGLE);
                break;
            case '}':
                programStream.get();
                addToken(TokenType::RSQUIGGLE);
                break;
            case ',':
                programStream.get();
                addToken(TokenType::COMMA);
                break;
            case '.':
                // NOTE: this means float literals like .0f are illegal!
                programStream.get();
                addToken(TokenType::PERIOD);
                break;
            case ':':
                programStream.get();
                addToken(TokenType::COL);
                break;
            case ';':
                programStream.get();
                addToken(TokenType::SEMICOL);
                break;
            case '@':
                programStream.get();
                addToken(TokenType::AT);
                break;
            case '=':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(TokenType::EQ, /*length=*/2);
                } else {
                    addToken(TokenType::ASSIGN);
                }
                break;
            case '&':
                programStream.get();
                if (programStream.peek() == '&') {
                    programStream.get();
                    addToken(TokenType::AND, /*length=*/2);
                } else {
                    reportError("SINGLE `&` not implemented");
                    addToken(TokenType::ERROR);
                }
                break;
            case '|': {
                programStream.get();
                if (programStream.peek() == '|') {
                    programStream.get();
                    addToken(TokenType::LOR, /*length=*/2);
                    break;
                }
                addToken(TokenType::BAR);
            } break;
            case '^':
                programStream.get();
                addToken(TokenType::XOR);
                break;
            case '!':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(TokenType::NEQ, /*length=*/2);
                } else {
                    addToken(TokenType::NOT);
                }
                break;
            case '+':
                programStream.get();
                if (programStream.peek() == '+') {
                    programStream.get();
                    addToken(TokenType::INC, /*length=*/2);
                } else {
                    addToken(TokenType::PLUS);
                }
                break;
            case '-':
                programStream.get();
                if (programStream.peek() == '>') {
                    programStream.get();
                    addToken(TokenType::RARROW, /*length=*/2);

                } else if (programStream.peek() == '-') {
                    programStream.get();
                    addToken(TokenType::DEC, /*length=*/2);

                } else {
                    addToken(TokenType::MINUS);
                }
                break;
            case '*':
                programStream.get();
                addToken(TokenType::STAR);
                break;
            case '/':
                programStream.get();
                if (programStream.peek() == '/') {
                    while (programStream.peek() != EOF &&
                           programStream.peek() != '\n') {
                        programStream.get();
                        incrColumnNo();
                    }
                    if (programStream.peek() != '\n') {
                        programStream.get();
                        incrLineNo();
                        resetColumnNo();
                    }
                    // TODO: emit comment token?
                } else {
                    addToken(TokenType::SLASH);
                }
                break;
            case '%':
                programStream.get();
                addToken(TokenType::MOD);
                break;
            // EQ, NEQ already handled
            case '<':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(TokenType::LEQ, /*length=*/2);
                } else {
                    addToken(TokenType::LT);
                }
                break;
            case '>':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(TokenType::GEQ, /*length=*/2);
                } else {
                    addToken(TokenType::GT);
                }
                break;
            case '"': {
                programStream.get();
                Token newToken;
                newToken.type = TokenType::STRING_LITERAL;
                newToken.lineBegin = getLineNo();
                newToken.colBegin = getColumnNo();
                std::string str;

                while (programStream.peek() != EOF &&
                       programStream.peek() != '"') {
                    if (programStream.peek() == '\\') {
                        char escapedChar = handleEscapedChar(programStream);
                        if (escapedChar != ' ') {
                            str += escapedChar;
                            programStream.get();

                        } else {
                            // error case.
                            incrColumnNo();
                        }
                    } else {
                        str += programStream.get();
                        incrColumnNo();
                    }
                }

                newToken.lineEnd = getLineNo();
                newToken.colEnd = getColumnNo();
                newToken.value = str;
                addToken(newToken);

                if (programStream.peek() == '"') {
                    programStream.get();
                    incrColumnNo();
                } else {
                    reportError("unclosed string literal");
                }
                break;
            }
            // Whitespace
            case '\v':
                internal_error << "what is \\v = \v ???\n";
            case '\f':
                internal_error << "what is \\f = \f ???\n";
            case '\r': // ???
                internal_error << "what is \\r = \r ???\n";
            case '\n':
                programStream.get();
                if (state == ScanState::SLTEST) {
                    state = ScanState::INITIAL;
                }
                incrLineNo();
                resetColumnNo();
                break;
            case ' ':
            case '\t':
                programStream.get();
                incrColumnNo();
                break;
            default: {
                // Try to parse a(n) (int uint, float) literal.
                // TODO: currently float literals like .0f are illegal due to
                // PERIOD parsing above. this might be fixable in the parser
                // itself though, .int -> float?
                if (!std::isdigit(programStream.peek())) {
                    std::stringstream errMsg;
                    errMsg << "unexpected symbol (expected digit) '"
                           << (char)programStream.peek() << "'";
                    reportError(errMsg.str());
                    while (programStream.peek() != EOF &&
                           !std::isspace(programStream.peek())) {
                        programStream.get();
                        incrColumnNo();
                    }
                    break;
                }

                Token newToken{
                    .lineBegin = getLineNo(),
                    .colBegin = getColumnNo(),
                    .type = TokenType::INT_LITERAL,
                };
                std::string tokenString;
                while (std::isdigit(programStream.peek())) {
                    tokenString += programStream.get();
                    incrColumnNo();
                }

                // Handle decimal.
                if (programStream.peek() == '.') {
                    newToken.type = TokenType::FLOAT_LITERAL;
                    tokenString += programStream.get();
                    incrColumnNo();

                    if (!std::isdigit(programStream.peek())) {
                        std::stringstream errMsg;
                        errMsg << "unexpected symbol (expected digit for "
                                  "decimal) '"
                               << static_cast<char>(programStream.peek())
                               << "'";
                        reportError(errMsg.str());

                        while (programStream.peek() != EOF &&
                               !std::isspace(programStream.peek())) {
                            programStream.get();
                            incrColumnNo();
                        }
                        break;
                    }
                    do {
                        tokenString += programStream.get();
                        incrColumnNo();
                    } while (std::isdigit(programStream.peek()));
                }

                // handle exponent
                if (programStream.peek() == 'e' ||
                    programStream.peek() == 'E') {
                    newToken.type = TokenType::FLOAT_LITERAL;
                    tokenString += programStream.get();
                    incrColumnNo();

                    if (programStream.peek() == '+' ||
                        programStream.peek() == '-') {
                        tokenString += programStream.get();
                        incrColumnNo();
                    }

                    if (!std::isdigit(programStream.peek())) {
                        std::stringstream errMsg;
                        errMsg << "unexpected symbol (expected digit for "
                                  "exponent) '"
                               << static_cast<char>(programStream.peek())
                               << "'";
                        reportError(errMsg.str());

                        while (programStream.peek() != EOF &&
                               !std::isspace(programStream.peek())) {
                            programStream.get();
                            incrColumnNo();
                        }
                        break;
                    }
                    do {
                        tokenString += programStream.get();
                        incrColumnNo();
                    } while (std::isdigit(programStream.peek()));
                }

                // TODO: Handle f (float), h (half) modifiers.

                // Handle u (unsigned) modifier.
                if (programStream.peek() == 'u') {
                    programStream.get();
                    newToken.type = TokenType::UINT_LITERAL;
                    incrColumnNo();
                    newToken.value =
                        static_cast<uint64_t>(std::stoull(tokenString));
                } else if (newToken.type == TokenType::INT_LITERAL) {
                    newToken.type = TokenType::INT_LITERAL;
                    newToken.value =
                        static_cast<int64_t>(std::stoll(tokenString));
                } else {
                    internal_assert(newToken.type == TokenType::FLOAT_LITERAL)
                        << "State error in literal parsing: " << tokenString;
                    newToken.value =
                        static_cast<double>(std::stold(tokenString));
                }
                newToken.lineEnd = getLineNo();
                newToken.colEnd = getColumnNo() - 1;
                addToken(newToken);
                break;
            }
            }
        }
    }
    if (state != ScanState::INITIAL) {
        reportError("unclosed test");
    }
}

char Lexer::handleEscapedChar(std::istream &programStream) {
    switch (programStream.peek()) {
    case 'a':
        return '\a';
    case 'b':
        return '\b';
    case 'f':
        return '\f';
    case 'n':
        return '\n';
    case 'r':
        return '\r';
    case 't':
        return '\t';
    case 'v':
        return '\v';
    case '\\':
        return '\\';
    case '\'':
        return '\'';
    case '"':
        return '\"';
    case '?':
        return '\?';
    default:
        reportError("Unrecognized escape sequence");
        return ' ';
    }
}

TokenStream lex(const std::string &filename) {
    std::ifstream inputFile(filename);
    internal_assert(inputFile.is_open())
        << "Error: Could not open file " << filename;

    // Lexical analysis
    Lexer L;
    L.lex(inputFile);
    return L.getTokens();
}

} // namespace parser
} // namespace bonsai
