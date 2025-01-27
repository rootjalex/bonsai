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

    uint32_t lineBegin;
    uint32_t colBegin;
    uint32_t lineEnd;
    uint32_t colEnd;

    static std::string tokenTypeString(Token::Type);

    std::string toString() const;

    friend std::ostream &operator<<(std::ostream &, const Token &);
};

struct TokenStream {
    void addToken(Token newToken) { tokens.push_back(newToken); }
    void addToken(Token::Type, uint32_t line, uint32_t col,
                  uint32_t length = 1);

    Token peek(uint32_t count) const;

    void skip() { tokens.pop_front(); }
    Token back() { return tokens.back(); }
    bool consume(Token::Type);

    bool empty() const { return tokens.empty(); }

    friend std::ostream &operator<<(std::ostream &, const TokenStream &);

  private:
    std::list<Token> tokens;
};

} // namespace parser
} // namespace bonsai
