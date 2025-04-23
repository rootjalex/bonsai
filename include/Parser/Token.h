#pragma once

#include "Error.h"

#include <list>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace bonsai {
namespace parser {

// Similar to Simit's (lots of borrowed code)

class Token {
  public:
    enum class Type {
        // Constants
        INT_LITERAL,
        UINT_LITERAL,
        FLOAT_LITERAL,
        STRING_LITERAL,
        // Names
        IDENTIFIER,

        // Special words/characters
        IMPORT,    // import
        ELEMENT,   // element
        INTERFACE, // interface
        EXTERN,    // extern
        SCHEDULE,  // schedule

        // Function related keywords
        FUNC,   // func
        MUT,    // mut
        RARROW, // ->
        RETURN, // return
        PRINT,  // print

        // Data structure specification related keywords
        TREE,   // tree
        WITH,   // with
        LAYOUT, // layout
        GROUP,  // group
        SWITCH, // switch

        // WHILE, // while
        FOR,   // for
        IN,    // in
        IF,    // if
        ELIF,  // elif
        ELSE,  // else
        TRUE,  // true
        FALSE, // false

        LPAREN,    // (
        RPAREN,    // )
        LBRACKET,  // [
        RBRACKET,  // ]
        LSQUIGGLE, // {
        RSQUIGGLE, // }
        COMMA,     // ,
        PERIOD,    // .
        COL,       // :
        SEMICOL,   // ;
        BAR,       // |

        ASSIGN,      // =
        LOGICAL_AND, // &&
        AT,          // @
        LOGICAL_OR,  // ||
        XOR,         // ^
        BITWISE_AND, // &
        NOT,         // !
        PLUS,        // +
        INC,         // ++
        MINUS,       // -
        DEC,         // --
        STAR,        // *
        SLASH,       // /
        MOD,         // %
        // EXP, // ^ TODO: OR IS THIS XOR?
        EQ,  // ==
        NEQ, // !=
        LEQ, // <=
        GEQ, // >=
        LT,  // <
        GT,  // >

        ERROR,
    };

    static Token ErrorToken() {
        return Token(Type::ERROR, /*line_begin=*/0, /*column_begin=*/0);
    }

    Token(Type type, uint64_t line_begin, uint64_t line_end,
          uint64_t column_begin)
        : type(type), begin_line(line_begin), begin_column(column_begin),
          line_offset(line_end - line_begin) {}

    // Most of the time, the token is on the same line.
    Token(Type type, uint64_t line_begin, uint64_t column_begin)
        : type(type), begin_line(line_begin), begin_column(column_begin),
          line_offset(0) {}

    // Provides a common interface for accessing line/column information.
    inline uint64_t line_begin() const { return begin_line; }
    inline uint64_t line_end() const { return begin_line + line_offset; }
    inline uint64_t column_begin() const { return begin_column; }
    inline uint64_t column_end() const { return begin_column + size(); }

    // Updates the end line index for this token.
    void update_line_end(uint64_t line) { line_offset = line - begin_line; }

    static std::string token_type_string(Token::Type);
    static std::string token_type_string(const Token &);

    // Returns the number of characters in this token.
    uint64_t size() const;

    // Converts this token to a string.
    std::string to_string() const;

    // The value associated with this token (if any).
    std::variant<std::monostate, int64_t, uint64_t, double, std::string> value;

    // The type of this token.
    Type type;

    friend std::ostream &operator<<(std::ostream &, const Token &);

  private:
    uint64_t begin_line;
    uint64_t begin_column;
    // The offset used to calculate `end_line = begin_line + line_offset`.
    int32_t line_offset;
};

struct TokenStream {
    TokenStream(std::string filename) : filename(std::move(filename)) {}

    void add_token(Token new_token) { tokens.push_back(std::move(new_token)); }
    void add_token(Token::Type, uint64_t, uint64_t);

    Token peek(uint32_t count) const;

    std::optional<Token> back() const {
        if (tokens.empty())
            return std::nullopt;
        return tokens.back();
    }

    void skip() { consume(tokens.back().type); }

    bool consume(Token::Type);

    bool empty() const { return tokens.empty(); }

    // Applies any necessary changes required for consumption. Note: this
    // method is *not* idempotent.
    void finalize_for_consumption() {
        std::reverse(tokens.begin(), tokens.end());
    }

    // Returns the current token. This is useful for error message handling.
    const Token &current_token() const {
        internal_assert(current.has_value());
        return *current;
    }

    // Returns the file name for this token stream.
    const std::string &file_name() const { return filename; }

    // Returns whether this is a valid token stream.
    bool is_valid() const {
        return std::none_of(tokens.begin(), tokens.end(),
                            [](const Token &token) {
                                return token.type == Token::Type::ERROR;
                            });
    }

    friend std::ostream &operator<<(std::ostream &, const TokenStream &);

  private:
    // The current token being visited.
    std::optional<Token> current = std::nullopt;

    // The list of tokens in this stream. These are lexed in normal order, i.e.,
    // by pushing to the back, and then reversed before consumption.
    std::vector<Token> tokens;

    // The file name associated with this token stream. This assumes every token
    // stream is associated with exactly one file.
    std::string filename;
};

} // namespace parser
} // namespace bonsai
