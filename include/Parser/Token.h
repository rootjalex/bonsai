#pragma once

#include <list>
#include <string>
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

    Type type;
    std::variant<std::monostate, int64_t, uint64_t, double, std::string> value;

    uint64_t lineBegin;
    uint64_t colBegin;
    uint64_t lineEnd;
    uint64_t colEnd;

    static std::string token_type_string(Token::Type);

    std::string to_string() const;

    friend std::ostream &operator<<(std::ostream &, const Token &);
};

struct TokenStream {
    void add_token(Token new_token) { tokens.push_back(std::move(new_token)); }
    void add_token(Token::Type, uint64_t, uint64_t, uint32_t);

    Token peek(uint32_t count) const;

    std::optional<Token> back() const {
        if (tokens.empty())
            return std::nullopt;
        return tokens.back();
    }

    void skip() { tokens.pop_front(); }

    bool consume(Token::Type);

    bool empty() const { return tokens.empty(); }

    friend std::ostream &operator<<(std::ostream &, const TokenStream &);

  private:
    std::list<Token> tokens;
};

} // namespace parser
} // namespace bonsai
