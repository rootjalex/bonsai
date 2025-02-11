#include "Parser/Lexer.h"

#include <fstream>
#include <iostream>
#include <optional>
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
    uint64_t line_no() { return line; }

    // Returns the current column number.
    uint64_t column_no() { return column; }

    void reset_column() { column = 1; }
    void reset_line() { line = 1; }

    // Increments the column number.
    void incr_column(uint32_t value = 1) { column += value; }

    // Increments the line number and resets the column number.
    void incr_line(uint32_t value = 1) {
        line += value;
        reset_column();
    }

    void add_token(Token::Type type, uint32_t length = 1) {
        stream.add_token(type, line_no(), column_no(), length);
    }

    void add_token(Token token) { stream.add_token(std::move(token)); }

    const TokenStream &get_tokens() { return stream; }

    std::string file_name() { return filename; }

  private:
    std::string filename;
    TokenStream stream;
    // The current column and line number during the tokenization phase.
    uint64_t column = 1;
    uint64_t line = 1;

    enum class ScanState { INITIAL, SLTEST, MLTEST };

    static Token::Type get_token_type(std::string_view);

    // TODO(cgyurgyik): Column number isn't always lined up correctly.
    // This probably needs to be looked at on a case-by-case basis.
    void report_error(std::string_view message) {
        // Point past the last legally lex'd token.
        if (std::optional<Token> last = get_tokens().back()) {
            incr_column(last->colEnd - last->colBegin + 1);
        }
        incr_column(); // Point to this token.
        // Add an error token.
        add_token(Token::Type::ERROR);

        std::ifstream file(file_name());

        std::string line;
        for (int i = 1; i <= line_no(); ++i) {
            if (!std::getline(file, line))
                return;
        }

        // filename:line:column: lex error: <error-message>
        // <line>
        //   ^
        std::cerr << file_name() << ":" << line_no() << ":" << column_no()
                  << ": lex error: " << message << "\n"
                  << line << "\n"
                  << std::string(column_no(), ' ') << std::string(1, '^')
                  << "\n";
    }

    char handle_escaped_char(std::istream &program_stream);
};

Token::Type Lexer::get_token_type(const std::string_view token) {
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
    if (token == "print")
        return Token::Type::PRINT;
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
static bool is_valid_identifier_start(int32_t c) {
    return c == '_' || std::isalpha(c);
}

static bool is_valid_identifier_token(int32_t c) {
    return is_valid_identifier_start(c) || std::isdigit(c);
}

void Lexer::lex() {
    std::ifstream program_stream(file_name());
    internal_assert(program_stream.is_open())
        << "Error: Could not open file " << file_name();

    ScanState state = ScanState::INITIAL;

    while (program_stream.peek() != EOF) {
        // Try to parse a name of the form [A-Za-z_][A-Za-z0-9_].
        if (is_valid_identifier_start(program_stream.peek())) {
            std::string token_string(1, program_stream.get());
            while (is_valid_identifier_token(program_stream.peek())) {
                token_string += program_stream.get();
            }
            Token new_token{
                .type = get_token_type(token_string),
                .lineBegin = line_no(),
                .colBegin = column_no(),
                .lineEnd = line_no(),
                .colEnd = column_no() + token_string.length() - 1,
            };
            if (new_token.type == Token::Type::IDENTIFIER) {
                new_token.value = token_string;
            }
            add_token(new_token);
        } else {
            switch (program_stream.peek()) {
            case '(':
                program_stream.get();
                add_token(Token::Type::LPAREN);
                break;
            case ')':
                program_stream.get();
                add_token(Token::Type::RPAREN);
                break;
            case '[':
                program_stream.get();
                add_token(Token::Type::LBRACKET);
                break;
            case ']':
                program_stream.get();
                add_token(Token::Type::RBRACKET);
                break;
            case '{':
                program_stream.get();
                add_token(Token::Type::LSQUIGGLE);
                break;
            case '}':
                program_stream.get();
                add_token(Token::Type::RSQUIGGLE);
                break;
            case ',':
                program_stream.get();
                add_token(Token::Type::COMMA);
                break;
            case '.':
                // NOTE: this means float literals like .0f are illegal!
                program_stream.get();
                add_token(Token::Type::PERIOD);
                break;
            case ':':
                program_stream.get();
                add_token(Token::Type::COL);
                break;
            case ';':
                program_stream.get();
                add_token(Token::Type::SEMICOL);
                break;
            case '@':
                program_stream.get();
                add_token(Token::Type::AT);
                break;
            case '=':
                program_stream.get();
                if (program_stream.peek() == '=') {
                    program_stream.get();
                    add_token(Token::Type::EQ, /*length=*/2);
                } else {
                    add_token(Token::Type::ASSIGN);
                }
                break;
            case '&':
                program_stream.get();
                if (program_stream.peek() == '&') {
                    program_stream.get();
                    add_token(Token::Type::AND, /*length=*/2);
                } else {
                    report_error("SINGLE `&` not implemented");
                }
                break;
            case '|': {
                program_stream.get();
                if (program_stream.peek() == '|') {
                    program_stream.get();
                    add_token(Token::Type::LOR, /*length=*/2);
                    break;
                }
                add_token(Token::Type::BAR);
            } break;
            case '^':
                program_stream.get();
                add_token(Token::Type::XOR);
                break;
            case '!':
                program_stream.get();
                if (program_stream.peek() == '=') {
                    program_stream.get();
                    add_token(Token::Type::NEQ, /*length=*/2);
                } else {
                    add_token(Token::Type::NOT);
                }
                break;
            case '+':
                program_stream.get();
                if (program_stream.peek() == '+') {
                    program_stream.get();
                    add_token(Token::Type::INC, /*length=*/2);
                } else {
                    add_token(Token::Type::PLUS);
                }
                break;
            case '-':
                program_stream.get();
                if (program_stream.peek() == '>') {
                    program_stream.get();
                    add_token(Token::Type::RARROW, /*length=*/2);

                } else if (program_stream.peek() == '-') {
                    program_stream.get();
                    add_token(Token::Type::DEC, /*length=*/2);

                } else {
                    add_token(Token::Type::MINUS);
                }
                break;
            case '*':
                program_stream.get();
                add_token(Token::Type::STAR);
                break;
            case '/':
                program_stream.get();
                if (program_stream.peek() == '/') {
                    while (program_stream.peek() != EOF &&
                           program_stream.peek() != '\n') {
                        program_stream.get();
                        incr_column();
                    }
                    if (program_stream.peek() != '\n') {
                        program_stream.get();
                        incr_line();
                    }
                    // TODO: emit comment token?
                } else {
                    add_token(Token::Type::SLASH);
                }
                break;
            case '%':
                program_stream.get();
                add_token(Token::Type::MOD);
                break;
            // EQ, NEQ already handled
            case '<':
                program_stream.get();
                if (program_stream.peek() == '=') {
                    program_stream.get();
                    add_token(Token::Type::LEQ, /*length=*/2);
                } else {
                    add_token(Token::Type::LT);
                }
                break;
            case '>':
                program_stream.get();
                if (program_stream.peek() == '=') {
                    program_stream.get();
                    add_token(Token::Type::GEQ, /*length=*/2);
                } else {
                    add_token(Token::Type::GT);
                }
                break;
            case '"': {
                program_stream.get();
                Token new_token;
                new_token.type = Token::Type::STRING_LITERAL;
                new_token.lineBegin = line_no();
                new_token.colBegin = column_no();
                std::string str;

                while (program_stream.peek() != EOF &&
                       program_stream.peek() != '"') {
                    if (program_stream.peek() == '\\') {
                        char escapedChar = handle_escaped_char(program_stream);
                        if (escapedChar != ' ') {
                            str += escapedChar;
                            program_stream.get();
                            incr_column(2);
                        } else {
                            // error case.
                            incr_column();
                        }
                    } else {
                        str += program_stream.get();
                        incr_column();
                    }
                }

                new_token.lineEnd = line_no();
                new_token.colEnd = column_no();
                new_token.value = str;
                add_token(new_token);

                if (program_stream.peek() == '"') {
                    program_stream.get();
                    incr_column();
                } else {
                    report_error("unclosed string literal");
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
                program_stream.get();
                if (state == ScanState::SLTEST) {
                    state = ScanState::INITIAL;
                }
                incr_line();
                break;
            case ' ':
            case '\t':
                program_stream.get();
                incr_column();
                break;
            default: {
                // Try to parse a(n) (int uint, float) literal.
                // TODO: currently float literals like .0f are illegal due to
                // PERIOD parsing above. this might be fixable in the parser
                // itself though, .int -> float?
                if (!std::isdigit(program_stream.peek())) {
                    std::stringstream error_message;
                    error_message << "unexpected symbol (expected digit) '"
                                  << (char)program_stream.peek() << "'";
                    report_error(error_message.str());
                    while (program_stream.peek() != EOF &&
                           !std::isspace(program_stream.peek())) {
                        program_stream.get();
                        incr_column();
                    }
                    break;
                }

                Token new_token{
                    .type = Token::Type::INT_LITERAL,
                    .lineBegin = line_no(),
                    .colBegin = column_no(),
                };
                std::string token_string;
                while (std::isdigit(program_stream.peek())) {
                    token_string += program_stream.get();
                    incr_column();
                }

                // Handle decimal.
                if (program_stream.peek() == '.') {
                    new_token.type = Token::Type::FLOAT_LITERAL;
                    token_string += program_stream.get();
                    incr_column();

                    if (!std::isdigit(program_stream.peek())) {
                        std::stringstream error_message;
                        error_message
                            << "unexpected symbol (expected digit for "
                               "decimal) '"
                            << static_cast<char>(program_stream.peek()) << "'";
                        report_error(error_message.str());

                        while (program_stream.peek() != EOF &&
                               !std::isspace(program_stream.peek())) {
                            program_stream.get();
                            incr_column();
                        }
                        break;
                    }
                    do {
                        token_string += program_stream.get();
                        incr_column();
                    } while (std::isdigit(program_stream.peek()));
                }

                // handle exponent
                if (program_stream.peek() == 'e' ||
                    program_stream.peek() == 'E') {
                    new_token.type = Token::Type::FLOAT_LITERAL;
                    token_string += program_stream.get();
                    incr_column();

                    if (program_stream.peek() == '+' ||
                        program_stream.peek() == '-') {
                        token_string += program_stream.get();
                        incr_column();
                    }

                    if (!std::isdigit(program_stream.peek())) {
                        std::stringstream error_message;
                        error_message
                            << "unexpected symbol (expected digit for "
                               "exponent) '"
                            << static_cast<char>(program_stream.peek()) << "'";
                        report_error(error_message.str());

                        while (program_stream.peek() != EOF &&
                               !std::isspace(program_stream.peek())) {
                            program_stream.get();
                            incr_column();
                        }
                        break;
                    }
                    do {
                        token_string += program_stream.get();
                        incr_column();
                    } while (std::isdigit(program_stream.peek()));
                }

                // TODO: Handle f (float), h (half) modifiers.

                // Handle u (unsigned) modifier.
                if (program_stream.peek() == 'u') {
                    program_stream.get();
                    new_token.type = Token::Type::UINT_LITERAL;
                    incr_column();
                    new_token.value =
                        static_cast<uint64_t>(std::stoull(token_string));
                } else if (new_token.type == Token::Type::INT_LITERAL) {
                    new_token.type = Token::Type::INT_LITERAL;
                    new_token.value =
                        static_cast<int64_t>(std::stoll(token_string));
                } else {
                    internal_assert(new_token.type ==
                                    Token::Type::FLOAT_LITERAL)
                        << "State error in literal parsing: " << token_string;
                    new_token.value =
                        static_cast<double>(std::stold(token_string));
                }
                new_token.lineEnd = line_no();
                new_token.colEnd = column_no() - 1;
                add_token(new_token);
                break;
            }
            }
        }
    }
    if (state != ScanState::INITIAL) {
        report_error("unclosed test");
    }
}

char Lexer::handle_escaped_char(std::istream &program_stream) {
    switch (program_stream.peek()) {
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
        report_error("Unrecognized escape sequence");
        return ' ';
    }
}

TokenStream lex(const std::string &filename) {
    // Lexical analysis
    Lexer lexer(filename);
    lexer.lex();
    return lexer.get_tokens();
}

} // namespace parser
} // namespace bonsai
