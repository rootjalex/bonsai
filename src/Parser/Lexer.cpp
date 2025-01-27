#include "Parser/Lexer.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "Error.h"

namespace bonsai {
namespace parser {

class Lexer {
  public:
    Lexer(std::string filename) : filename(std::move(filename)) {}

    // TODO: support better error handling?
    void lex();

    // Returns the current line number.
    uint64_t getLineNo() { return line; }

    // Returns the current column number.
    uint64_t getColumnNo() { return column; }

    void resetColumnNo() { column = 1; }
    void resetLineNo() { line = 1; }

    // Increments the column number.
    void incrColumnNo(uint32_t value = 1) { column += value; }

    // Increments the line number and resets the column number.
    void incrLineNo(uint32_t value = 1) {
        line += value;
        resetColumnNo();
    }

    void addToken(Token::Type type, uint32_t length = 1) {
        stream.addToken(type, getLineNo(), getColumnNo(), length);
    }

    void addToken(Token token) { stream.addToken(token); }

    const TokenStream &getTokens() { return stream; }

    std::string getFileName() { return filename; }

  private:
    std::string filename;
    TokenStream stream;
    // The current column and line number during the tokenization phase.
    uint64_t column = 1;
    uint64_t line = 1;

    enum class ScanState { INITIAL, SLTEST, MLTEST };

    static Token::Type getTokenType(std::string_view);

    // TODO(cgyurgyik): Column number isn't always lined up correctly.
    // This probably needs to be looked at on a case-by-case basis.
    void reportError(std::string_view message) {
        // Point past the last legally lex'd token.
        if (std::optional<Token> last = getTokens().back()) {
            incrColumnNo(last->colEnd - last->colBegin + 1);
        }
        incrColumnNo(); // Point to this token.
        // Add an error token.
        addToken(Token::Type::ERROR);

        std::ifstream file(getFileName());

        std::string line;
        for (int i = 1; i <= getLineNo(); ++i) {
            if (!std::getline(file, line))
                return;
        }

        // filename:line:column: lex error: <error-message>
        // <line>
        //   ^
        std::cerr << getFileName() << ":" << getLineNo() << ":" << getColumnNo()
                  << ": lex error: " << message << "\n"
                  << line << "\n"
                  << std::string(getColumnNo(), ' ') << std::string(1, '^')
                  << "\n";
    }

    char handleEscapedChar(std::istream &programStream);
};

Token::Type Lexer::getTokenType(const std::string_view token) {
    if (token == "import")
        return Token::Type::IMPORT;
    if (token == "element")
        return Token::Type::ELEMENT;
    if (token == "interface")
        return Token::Type::INTERFACE;
    if (token == "extern")
        return Token::Type::EXTERN;
    if (token == "func")
        return Token::Type::FUNC;
    if (token == "mut")
        return Token::Type::MUT;
    if (token == "return")
        return Token::Type::RETURN;
    if (token == "for")
        return Token::Type::FOR;
    if (token == "if")
        return Token::Type::IF;
    if (token == "elif")
        return Token::Type::ELIF;
    if (token == "else")
        return Token::Type::ELSE;
    if (token == "true")
        return Token::Type::TRUE;
    if (token == "false")
        return Token::Type::FALSE;

    // If string does not correspond to a keyword, assume it is an identifier.
    return Token::Type::IDENTIFIER;
}

// Returns whether this is a valid start character of an identifier or keyword,
// i.e., [A-Za-z_]
static bool isValidIdentifierStart(int32_t c) {
    return c == '_' || std::isalpha(c);
}

void Lexer::lex() {
    std::ifstream programStream(getFileName());
    internal_assert(programStream.is_open())
        << "Error: Could not open file " << getFileName();

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
            if (newToken.type == Token::Type::IDENTIFIER) {
                newToken.value = tokenString;
            }
            addToken(newToken);
        } else {
            switch (programStream.peek()) {
            case '(':
                programStream.get();
                addToken(Token::Type::LPAREN);
                break;
            case ')':
                programStream.get();
                addToken(Token::Type::RPAREN);
                break;
            case '[':
                programStream.get();
                addToken(Token::Type::LBRACKET);
                break;
            case ']':
                programStream.get();
                addToken(Token::Type::RBRACKET);
                break;
            case '{':
                programStream.get();
                addToken(Token::Type::LSQUIGGLE);
                break;
            case '}':
                programStream.get();
                addToken(Token::Type::RSQUIGGLE);
                break;
            case ',':
                programStream.get();
                addToken(Token::Type::COMMA);
                break;
            case '.':
                // NOTE: this means float literals like .0f are illegal!
                programStream.get();
                addToken(Token::Type::PERIOD);
                break;
            case ':':
                programStream.get();
                addToken(Token::Type::COL);
                break;
            case ';':
                programStream.get();
                addToken(Token::Type::SEMICOL);
                break;
            case '@':
                programStream.get();
                addToken(Token::Type::AT);
                break;
            case '=':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(Token::Type::EQ, /*length=*/2);
                } else {
                    addToken(Token::Type::ASSIGN);
                }
                break;
            case '&':
                programStream.get();
                if (programStream.peek() == '&') {
                    programStream.get();
                    addToken(Token::Type::AND, /*length=*/2);
                } else {
                    reportError("SINGLE `&` not implemented");
                }
                break;
            case '|': {
                programStream.get();
                if (programStream.peek() == '|') {
                    programStream.get();
                    addToken(Token::Type::LOR, /*length=*/2);
                    break;
                }
                addToken(Token::Type::BAR);
            } break;
            case '^':
                programStream.get();
                addToken(Token::Type::XOR);
                break;
            case '!':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(Token::Type::NEQ, /*length=*/2);
                } else {
                    addToken(Token::Type::NOT);
                }
                break;
            case '+':
                programStream.get();
                if (programStream.peek() == '+') {
                    programStream.get();
                    addToken(Token::Type::INC, /*length=*/2);
                } else {
                    addToken(Token::Type::PLUS);
                }
                break;
            case '-':
                programStream.get();
                if (programStream.peek() == '>') {
                    programStream.get();
                    addToken(Token::Type::RARROW, /*length=*/2);

                } else if (programStream.peek() == '-') {
                    programStream.get();
                    addToken(Token::Type::DEC, /*length=*/2);

                } else {
                    addToken(Token::Type::MINUS);
                }
                break;
            case '*':
                programStream.get();
                addToken(Token::Type::STAR);
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
                    }
                    // TODO: emit comment token?
                } else {
                    addToken(Token::Type::SLASH);
                }
                break;
            case '%':
                programStream.get();
                addToken(Token::Type::MOD);
                break;
            // EQ, NEQ already handled
            case '<':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(Token::Type::LEQ, /*length=*/2);
                } else {
                    addToken(Token::Type::LT);
                }
                break;
            case '>':
                programStream.get();
                if (programStream.peek() == '=') {
                    programStream.get();
                    addToken(Token::Type::GEQ, /*length=*/2);
                } else {
                    addToken(Token::Type::GT);
                }
                break;
            case '"': {
                programStream.get();
                Token newToken;
                newToken.type = Token::Type::STRING_LITERAL;
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
                            incrColumnNo(2);
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
                    .type = Token::Type::INT_LITERAL,
                };
                std::string tokenString;
                while (std::isdigit(programStream.peek())) {
                    tokenString += programStream.get();
                    incrColumnNo();
                }

                // Handle decimal.
                if (programStream.peek() == '.') {
                    newToken.type = Token::Type::FLOAT_LITERAL;
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
                    newToken.type = Token::Type::FLOAT_LITERAL;
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
                    newToken.type = Token::Type::UINT_LITERAL;
                    incrColumnNo();
                    newToken.value =
                        static_cast<uint64_t>(std::stoull(tokenString));
                } else if (newToken.type == Token::Type::INT_LITERAL) {
                    newToken.type = Token::Type::INT_LITERAL;
                    newToken.value =
                        static_cast<int64_t>(std::stoll(tokenString));
                } else {
                    internal_assert(newToken.type == Token::Type::FLOAT_LITERAL)
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
    // Lexical analysis
    Lexer L(filename);
    L.lex();
    return L.getTokens();
}

} // namespace parser
} // namespace bonsai
