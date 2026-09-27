package de.hjstephan86.jcl.phases;

/**
 * Repraesentiert ein lexikalisches Token des Java-Quellcodes.
 *
 * @author Stephan Epp
 */
public record Token(Kind kind, String value, int line, int col) {

    public enum Kind {
        // Literale
        INT_LIT, LONG_LIT, FLOAT_LIT, DOUBLE_LIT, CHAR_LIT,
        STRING_LIT, BOOL_LIT, NULL_LIT,

        // Bezeichner
        IDENT,

        // Schluesselwoerter
        KW_CLASS, KW_INTERFACE, KW_ENUM, KW_RECORD, KW_SEALED,
        KW_PERMITS, KW_EXTENDS, KW_IMPLEMENTS,
        KW_PUBLIC, KW_PRIVATE, KW_PROTECTED, KW_STATIC, KW_FINAL,
        KW_ABSTRACT, KW_NATIVE, KW_SYNCHRONIZED, KW_TRANSIENT,
        KW_VOLATILE, KW_DEFAULT, KW_STRICTFP,
        KW_NEW, KW_THIS, KW_SUPER, KW_INSTANCEOF,
        KW_IF, KW_ELSE, KW_WHILE, KW_DO, KW_FOR,
        KW_SWITCH, KW_CASE, KW_YIELD, KW_BREAK, KW_CONTINUE,
        KW_RETURN, KW_THROW, KW_TRY, KW_CATCH, KW_FINALLY,
        KW_IMPORT, KW_PACKAGE,
        KW_VOID, KW_BOOLEAN, KW_BYTE, KW_SHORT, KW_INT,
        KW_LONG, KW_FLOAT, KW_DOUBLE, KW_CHAR,
        KW_VAR,

        // Operatoren
        PLUS, MINUS, STAR, SLASH, PERCENT,
        AMP, PIPE, CARET, TILDE, LSHIFT, RSHIFT, URSHIFT,
        ASSIGN, PLUS_ASSIGN, MINUS_ASSIGN, STAR_ASSIGN,
        SLASH_ASSIGN, PERCENT_ASSIGN, AMP_ASSIGN, PIPE_ASSIGN,
        CARET_ASSIGN, LSHIFT_ASSIGN, RSHIFT_ASSIGN, URSHIFT_ASSIGN,
        EQ, NE, LT, GT, LE, GE,
        AND, OR, NOT, INC, DEC,
        QUESTION, COLON, ARROW, COLONCOLON,

        // Trennzeichen
        LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET,
        SEMI, COMMA, DOT, DOTDOTDOT, AT,

        // Sonstiges
        EOF, ERROR
    }

    @Override
    public String toString() {
        return String.format("Token[%s, \"%s\", %d:%d]", kind, value, line, col);
    }
}
