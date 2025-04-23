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
    Lexer(std::string filename) : stream(filename) {}

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

    void add_token(Token::Type type) {
        stream.add_token(type, line_no(), column_no());
        std::optional<Token> t = stream.back();
        incr_column(t->size());
    }

    void add_token(Token token) {
        stream.add_token(std::move(token));
        std::optional<Token> t = stream.back();
        incr_column(t->size());
    }

    const TokenStream &get_tokens() { return stream; }

    bool consume_if(std::ifstream &ifs, char c) {
        if (ifs.peek() == c) {
            ifs.get();
            return true;
        }
        return false;
    }

    // Returns whether the tokens parsed are currently error-free.
    const bool is_valid() const { return stream.is_valid(); }

    std::string file_name() const { return stream.file_name(); }

  private:
    TokenStream stream;
    // The current column and line number during the tokenization phase.
    uint64_t column = 1;
    uint64_t line = 1;

    enum class ScanState { INITIAL, SLTEST, MLTEST };

    static Token::Type get_token_type(std::string_view);

    // Outputs an error message to error I/O and adds an ERROR token to stream.
    void report_error(std::string_view message) {
        add_token(Token::Type::ERROR);
        std::ifstream file(file_name());
        std::string line;
        for (int i = 1; i <= line_no(); ++i) {
            if (!std::getline(file, line))
                return;
        }

        // filename:line:column: [lex error] <error-message>
        // <line>
        //   ^
        std::cerr << file_name() << ":" << line_no() << ":" << column_no()
                  << ": [lex error] " << message << "\n"
                  << line << "\n"
                  << std::string(std::max<int64_t>(column_no() - 1, 0), ' ')
                  << std::string(1, '^') << "\n";
    }

    // Handles escaped characters, e.g., `\r` or `\t`. If this is an invalid
    // escaped character, reports an error and returns `std::nullopt`.
    std::optional<char> handle_escaped_char(std::ifstream &program_stream);

    char consume(std::ifstream &program_stream);
    void consume_until_space(std::ifstream &program_stream);
    std::string consume_digits(std::ifstream &program_stream);

    std::optional<Token> lex_number(std::ifstream &program_stream);
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
    if (token == "schedule")
        return Token::Type::SCHEDULE;
    if (token == "func")
        return Token::Type::FUNC;
    if (token == "mut")
        return Token::Type::MUT;
    if (token == "return")
        return Token::Type::RETURN;
    if (token == "print")
        return Token::Type::PRINT;
    if (token == "tree")
        return Token::Type::TREE;
    if (token == "with")
        return Token::Type::WITH;
    if (token == "layout")
        return Token::Type::LAYOUT;
    if (token == "group")
        return Token::Type::GROUP;
    if (token == "switch")
        return Token::Type::SWITCH;
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

char Lexer::consume(std::ifstream &program_stream) {
    char c = program_stream.get();
    incr_column();
    return c;
}

void Lexer::consume_until_space(std::ifstream &program_stream) {
    while (program_stream.peek() != EOF &&
           !std::isspace(program_stream.peek())) {
        consume(program_stream);
    }
}

std::string Lexer::consume_digits(std::ifstream &program_stream) {
    std::string token_string;
    do {
        token_string += consume(program_stream);
    } while (std::isdigit(program_stream.peek()));
    return token_string;
}

std::optional<Token> Lexer::lex_number(std::ifstream &program_stream) {
    // Try to parse a(n) (int uint, float) literal.
    // TODO: currently float literals like .0f are illegal due to
    // PERIOD parsing. this might be fixable in the parser itself though,
    // .int -> float?
    if (!std::isdigit(program_stream.peek())) {
        std::stringstream error_message;
        error_message << "unexpected symbol (expected digit) '"
                      << (char)program_stream.peek() << "'";
        report_error(error_message.str());
        consume_until_space(program_stream);
        return {};
    }

    Token new_token(Token::Type::INT_LITERAL,
                    /*line_begin=*/line_no(),
                    /*column_begin=*/column_no());
    std::string token_string = consume_digits(program_stream);

    // Handle decimal.
    if (program_stream.peek() == '.') {
        new_token.type = Token::Type::FLOAT_LITERAL;
        token_string += consume(program_stream);

        if (!std::isdigit(program_stream.peek())) {
            std::stringstream error_message;
            error_message << "unexpected symbol (expected digit for "
                             "decimal) '"
                          << static_cast<char>(program_stream.peek()) << "'";
            report_error(error_message.str());
            consume_until_space(program_stream);
            return Token::ErrorToken();
        }
        token_string += consume_digits(program_stream);
    }

    // handle exponent
    if (program_stream.peek() == 'e' || program_stream.peek() == 'E') {
        new_token.type = Token::Type::FLOAT_LITERAL;
        token_string += consume(program_stream);

        if (program_stream.peek() == '+' || program_stream.peek() == '-') {
            token_string += consume(program_stream);
        }

        if (!std::isdigit(program_stream.peek())) {
            std::stringstream error_message;
            error_message << "unexpected symbol (expected digit for "
                             "exponent) '"
                          << static_cast<char>(program_stream.peek()) << "'";
            report_error(error_message.str());
            consume_until_space(program_stream);
            return Token::ErrorToken();
        }
        token_string += consume_digits(program_stream);
    }

    // TODO: Handle f (float), h (half) modifiers.

    // Handle u (unsigned) modifier.
    if (program_stream.peek() == 'u') {
        consume(program_stream);
        new_token.type = Token::Type::UINT_LITERAL;
        new_token.value = static_cast<uint64_t>(std::stoull(token_string));
    } else if (new_token.type == Token::Type::INT_LITERAL) {
        new_token.type = Token::Type::INT_LITERAL;
        new_token.value = static_cast<int64_t>(std::stoll(token_string));
    } else {
        internal_assert(new_token.type == Token::Type::FLOAT_LITERAL)
            << "State error in literal parsing: " << token_string;
        new_token.value = static_cast<double>(std::stold(token_string));
    }

    new_token.update_line_end(line_no());
    return new_token;
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
            Token new_token(get_token_type(token_string),
                            /*line_begin=*/line_no(),
                            /*column_begin=*/column_no());
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
                if (consume_if(program_stream, '=')) {
                    add_token(Token::Type::EQ);
                } else {
                    add_token(Token::Type::ASSIGN);
                }
                break;
            case '&':
                program_stream.get();
                if (consume_if(program_stream, '&')) {
                    add_token(Token::Type::LOGICAL_AND);
                } else {
                    add_token(Token::Type::BITWISE_AND);
                }
                break;
            case '|': {
                program_stream.get();
                if (consume_if(program_stream, '|')) {
                    add_token(Token::Type::LOGICAL_OR);
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
                if (consume_if(program_stream, '=')) {
                    add_token(Token::Type::NEQ);
                } else {
                    add_token(Token::Type::NOT);
                }
                break;
            case '+':
                program_stream.get();
                if (consume_if(program_stream, '+')) {
                    add_token(Token::Type::INC);
                } else {
                    add_token(Token::Type::PLUS);
                }
                break;
            case '-':
                program_stream.get();
                if (consume_if(program_stream, '>')) {
                    add_token(Token::Type::RARROW);
                } else if (consume_if(program_stream, '-')) {
                    add_token(Token::Type::DEC);
                } else if (std::isdigit(program_stream.peek())) {
                    std::optional<Token> token = lex_number(program_stream);
                    if (!token.has_value()) {
                        report_error("Negated literal, unsure how to parse.");
                    }
                    if (std::holds_alternative<int64_t>(token->value)) {
                        token->value = -std::get<int64_t>(token->value);
                    } else if (std::holds_alternative<uint64_t>(token->value)) {
                        report_error("negated unsigned integer");
                    } else if (std::holds_alternative<double>(token->value)) {
                        token->value = -std::get<double>(token->value);
                    }
                    add_token(*token);
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
                if (consume_if(program_stream, '=')) {
                    add_token(Token::Type::LEQ);
                } else {
                    add_token(Token::Type::LT);
                }
                break;
            case '>':
                program_stream.get();
                if (consume_if(program_stream, '=')) {
                    add_token(Token::Type::GEQ);
                } else {
                    add_token(Token::Type::GT);
                }
                break;
            case '"': {
                program_stream.get();
                Token new_token(Token::Type::STRING_LITERAL,
                                /*line_begin=*/line_no(),
                                /*column_begin=*/column_no());
                std::string token_value;

                while (program_stream.peek() != EOF &&
                       program_stream.peek() != '"') {
                    if (program_stream.peek() != '\\') {
                        token_value += program_stream.get();
                        continue;
                    }
                    std::optional<char> c = handle_escaped_char(program_stream);
                    if (!c.has_value()) {
                        continue;
                    }
                    token_value += *c;
                    // This is technically two characters in the input stream
                    // being lexed so we consume an additional character here.
                    consume(program_stream);
                }

                new_token.update_line_end(line_no());
                new_token.value = token_value;
                add_token(new_token);
                if (program_stream.peek() != '"') {
                    report_error("unclosed string literal");
                }
                consume(program_stream);
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
                std::optional<Token> token = lex_number(program_stream);
                internal_assert(token.has_value());
                add_token(*token);
                break;
            }
            }
        }
    }
    if (state != ScanState::INITIAL) {
        report_error("unclosed test");
    }
}

std::optional<char> Lexer::handle_escaped_char(std::ifstream &program_stream) {
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
        const char c = consume(program_stream);
        report_error("unrecognized escape sequence: \\" + std::string{c});
        return std::nullopt;
    }
}

TokenStream lex(const std::string &filename) {
    // Lexical analysis
    Lexer lexer(filename);
    lexer.lex();
    internal_assert(lexer.is_valid()) << "Failed to tokenize " << filename;
    TokenStream stream = lexer.get_tokens();
    stream.finalize_for_consumption();
    return stream;
}

} // namespace parser
} // namespace bonsai
