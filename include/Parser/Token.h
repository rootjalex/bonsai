#pragma once

#include <list>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace bonsai {
namespace parser {

// Similar to Simit's (lots of borrowed code)

struct Token {
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

        FUNC,   // func
        MUT,    // mut
        RARROW, // ->
        RETURN, // return
        PRINT,  // print

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

        ASSIGN, // =
        AND,    // &&
        AT,     // @
        LOR,    // ||
        XOR,    // ^
        NOT,    // !
        PLUS,   // +
        INC,    // ++
        MINUS,  // -
        DEC,    // --
        STAR,   // *
        SLASH,  // /
        MOD,    // %
        // EXP, // ^ TODO: OR IS THIS XOR?
        EQ,  // ==
        NEQ, // !=
        LEQ, // <=
        GEQ, // >=
        LT,  // <
        GT,  // >

        ERROR,
    };

    uint64_t lineBegin;
    uint64_t colBegin;
    uint64_t lineEnd;
    std::string fileName;
    Type type;

    std::variant<std::monostate, int64_t, uint64_t, double, std::string> value;

    // Provides a common interface for accessing line/column information.
    inline uint64_t line_begin() const { return lineBegin; }
    inline uint64_t line_end() const { return lineEnd; }
    inline uint64_t column_begin() const { return colBegin; }
    inline uint64_t column_end() const { return colBegin + size(); }
    // TODO(cgyurgyik): we probably want to reduce this to some pointer/index
    // instead of copying the file name to *every* token.
    inline std::string file_name() const { return fileName; }

    static std::string token_type_string(Token::Type);

    // Returns the number of characters in this token.
    uint64_t size() const;

    std::string to_string() const;

    friend std::ostream &operator<<(std::ostream &, const Token &);
};

struct TokenStream {
    void add_token(Token new_token) { tokens.push_back(std::move(new_token)); }
    void add_token(Token::Type, uint64_t, uint64_t, std::string);

    Token peek(uint32_t count) const;

    std::optional<Token> back() const {
        if (tokens.empty())
            return std::nullopt;
        return tokens.back();
    }

    void skip() { consume(tokens.front().type); }

    bool consume(Token::Type);

    bool empty() const { return tokens.empty(); }

    // Returns the current token. This is useful for error message handling.
    const Token &current_token() const { return current; }

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
    Token current;
    // The list of tokens in this stream.
    // TODO(cgyurgyik): This probably doesn't need to be a linked list?
    std::list<Token> tokens;
};

} // namespace parser
} // namespace bonsai
