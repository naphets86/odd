package de.hjstephan86.jcl.phases;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Lexer fuer Java-Quellcode.
 * Laufzeit: O(|src|) -- linearer Scan.
 *
 * @author Stephan Epp
 */
public class Lexer {

    private final String src;
    private int pos  = 0;
    private int line = 1;
    private int col  = 1;

    /** Schluesselwort-Tabelle. */
    private static final Map<String, Token.Kind> KEYWORDS = Map.ofEntries(
        Map.entry("class",        Token.Kind.KW_CLASS),
        Map.entry("interface",    Token.Kind.KW_INTERFACE),
        Map.entry("enum",         Token.Kind.KW_ENUM),
        Map.entry("record",       Token.Kind.KW_RECORD),
        Map.entry("sealed",       Token.Kind.KW_SEALED),
        Map.entry("permits",      Token.Kind.KW_PERMITS),
        Map.entry("extends",      Token.Kind.KW_EXTENDS),
        Map.entry("implements",   Token.Kind.KW_IMPLEMENTS),
        Map.entry("public",       Token.Kind.KW_PUBLIC),
        Map.entry("private",      Token.Kind.KW_PRIVATE),
        Map.entry("protected",    Token.Kind.KW_PROTECTED),
        Map.entry("static",       Token.Kind.KW_STATIC),
        Map.entry("final",        Token.Kind.KW_FINAL),
        Map.entry("abstract",     Token.Kind.KW_ABSTRACT),
        Map.entry("native",       Token.Kind.KW_NATIVE),
        Map.entry("synchronized", Token.Kind.KW_SYNCHRONIZED),
        Map.entry("new",          Token.Kind.KW_NEW),
        Map.entry("this",         Token.Kind.KW_THIS),
        Map.entry("super",        Token.Kind.KW_SUPER),
        Map.entry("instanceof",   Token.Kind.KW_INSTANCEOF),
        Map.entry("if",           Token.Kind.KW_IF),
        Map.entry("else",         Token.Kind.KW_ELSE),
        Map.entry("while",        Token.Kind.KW_WHILE),
        Map.entry("do",           Token.Kind.KW_DO),
        Map.entry("for",          Token.Kind.KW_FOR),
        Map.entry("switch",       Token.Kind.KW_SWITCH),
        Map.entry("case",         Token.Kind.KW_CASE),
        Map.entry("yield",        Token.Kind.KW_YIELD),
        Map.entry("break",        Token.Kind.KW_BREAK),
        Map.entry("continue",     Token.Kind.KW_CONTINUE),
        Map.entry("return",       Token.Kind.KW_RETURN),
        Map.entry("throw",        Token.Kind.KW_THROW),
        Map.entry("try",          Token.Kind.KW_TRY),
        Map.entry("catch",        Token.Kind.KW_CATCH),
        Map.entry("finally",      Token.Kind.KW_FINALLY),
        Map.entry("import",       Token.Kind.KW_IMPORT),
        Map.entry("package",      Token.Kind.KW_PACKAGE),
        Map.entry("void",         Token.Kind.KW_VOID),
        Map.entry("boolean",      Token.Kind.KW_BOOLEAN),
        Map.entry("byte",         Token.Kind.KW_BYTE),
        Map.entry("short",        Token.Kind.KW_SHORT),
        Map.entry("int",          Token.Kind.KW_INT),
        Map.entry("long",         Token.Kind.KW_LONG),
        Map.entry("float",        Token.Kind.KW_FLOAT),
        Map.entry("double",       Token.Kind.KW_DOUBLE),
        Map.entry("char",         Token.Kind.KW_CHAR),
        Map.entry("var",          Token.Kind.KW_VAR),
        Map.entry("true",         Token.Kind.BOOL_LIT),
        Map.entry("false",        Token.Kind.BOOL_LIT),
        Map.entry("null",         Token.Kind.NULL_LIT)
    );

    public Lexer(String src) {
        this.src = src;
    }

    /** Tokenisiert den gesamten Quellcode. */
    public List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();
        while (pos < src.length()) {
            skipWhitespaceAndComments();
            if (pos >= src.length()) break;
            tokens.add(nextToken());
        }
        tokens.add(new Token(Token.Kind.EOF, "", line, col));
        return tokens;
    }

    // ----------------------------------------------------------------
    // Interner Scan
    // ----------------------------------------------------------------

    private Token nextToken() {
        int startLine = line;
        int startCol  = col;
        char c = peek();

        // Bezeichner / Schluesselwort
        if (Character.isJavaIdentifierStart(c)) {
            return scanIdentOrKeyword(startLine, startCol);
        }
        // Zahl
        if (Character.isDigit(c)) {
            return scanNumber(startLine, startCol);
        }
        // String
        if (c == '"') return scanString(startLine, startCol);
        // Char
        if (c == '\'') return scanChar(startLine, startCol);
        // Operatoren und Trennzeichen
        return scanOperatorOrDelim(startLine, startCol);
    }

    private Token scanIdentOrKeyword(int l, int c2) {
        int start = pos;
        while (pos < src.length() &&
               Character.isJavaIdentifierPart(src.charAt(pos))) {
            advance();
        }
        String text = src.substring(start, pos);
        Token.Kind kind = KEYWORDS.getOrDefault(text, Token.Kind.IDENT);
        return new Token(kind, text, l, c2);
    }

    private Token scanNumber(int l, int c2) {
        int start = pos;
        boolean isFloat = false;
        while (pos < src.length() &&
               (Character.isDigit(src.charAt(pos)) ||
                src.charAt(pos) == '_' || src.charAt(pos) == '.')) {
            if (src.charAt(pos) == '.') isFloat = true;
            advance();
        }
        char suffix = pos < src.length() ? src.charAt(pos) : 0;
        if (suffix == 'L' || suffix == 'l') { advance(); return new Token(Token.Kind.LONG_LIT,   src.substring(start, pos), l, c2); }
        if (suffix == 'F' || suffix == 'f') { advance(); return new Token(Token.Kind.FLOAT_LIT,  src.substring(start, pos), l, c2); }
        if (suffix == 'D' || suffix == 'd') { advance(); return new Token(Token.Kind.DOUBLE_LIT, src.substring(start, pos), l, c2); }
        return new Token(isFloat ? Token.Kind.DOUBLE_LIT : Token.Kind.INT_LIT,
                         src.substring(start, pos), l, c2);
    }

    private Token scanString(int l, int c2) {
        int start = pos;
        advance(); // "
        while (pos < src.length() && src.charAt(pos) != '"') {
            if (src.charAt(pos) == '\\') advance(); // escape
            advance();
        }
        if (pos < src.length()) advance(); // closing "
        return new Token(Token.Kind.STRING_LIT, src.substring(start, pos), l, c2);
    }

    private Token scanChar(int l, int c2) {
        int start = pos;
        advance(); // '
        if (pos < src.length() && src.charAt(pos) == '\\') advance();
        if (pos < src.length()) advance();
        if (pos < src.length() && src.charAt(pos) == '\'') advance();
        return new Token(Token.Kind.CHAR_LIT, src.substring(start, pos), l, c2);
    }

    private Token scanOperatorOrDelim(int l, int c2) {
        char c = advance();
        return switch (c) {
            case '(' -> tok(Token.Kind.LPAREN,   "(", l, c2);
            case ')' -> tok(Token.Kind.RPAREN,   ")", l, c2);
            case '{' -> tok(Token.Kind.LBRACE,   "{", l, c2);
            case '}' -> tok(Token.Kind.RBRACE,   "}", l, c2);
            case '[' -> tok(Token.Kind.LBRACKET, "[", l, c2);
            case ']' -> tok(Token.Kind.RBRACKET, "]", l, c2);
            case ';' -> tok(Token.Kind.SEMI,     ";", l, c2);
            case ',' -> tok(Token.Kind.COMMA,    ",", l, c2);
            case '@' -> tok(Token.Kind.AT,       "@", l, c2);
            case '~' -> tok(Token.Kind.TILDE,    "~", l, c2);
            case '?' -> tok(Token.Kind.QUESTION, "?", l, c2);
            case '+' -> match('=') ? tok(Token.Kind.PLUS_ASSIGN,  "+=", l, c2)
                      : match('+') ? tok(Token.Kind.INC,          "++", l, c2)
                      :              tok(Token.Kind.PLUS,          "+",  l, c2);
            case '-' -> match('=') ? tok(Token.Kind.MINUS_ASSIGN, "-=", l, c2)
                      : match('-') ? tok(Token.Kind.DEC,          "--", l, c2)
                      : match('>') ? tok(Token.Kind.ARROW,        "->", l, c2)
                      :              tok(Token.Kind.MINUS,         "-",  l, c2);
            case '*' -> match('=') ? tok(Token.Kind.STAR_ASSIGN,  "*=", l, c2)
                      :              tok(Token.Kind.STAR,          "*",  l, c2);
            case '/' -> match('=') ? tok(Token.Kind.SLASH_ASSIGN, "/=", l, c2)
                      :              tok(Token.Kind.SLASH,         "/",  l, c2);
            case '%' -> match('=') ? tok(Token.Kind.PERCENT_ASSIGN,"%=",l, c2)
                      :              tok(Token.Kind.PERCENT,       "%",  l, c2);
            case '=' -> match('=') ? tok(Token.Kind.EQ,  "==", l, c2)
                      :              tok(Token.Kind.ASSIGN,"=",  l, c2);
            case '!' -> match('=') ? tok(Token.Kind.NE,  "!=", l, c2)
                      :              tok(Token.Kind.NOT,  "!",  l, c2);
            case '<' -> match('=') ? tok(Token.Kind.LE,  "<=", l, c2)
                      :              tok(Token.Kind.LT,   "<",  l, c2);
            case '>' -> match('=') ? tok(Token.Kind.GE,  ">=", l, c2)
                      :              tok(Token.Kind.GT,   ">",  l, c2);
            case '&' -> match('&') ? tok(Token.Kind.AND, "&&", l, c2)
                      : match('=') ? tok(Token.Kind.AMP_ASSIGN,"&=",l,c2)
                      :              tok(Token.Kind.AMP,  "&",  l, c2);
            case '|' -> match('|') ? tok(Token.Kind.OR,  "||", l, c2)
                      : match('=') ? tok(Token.Kind.PIPE_ASSIGN,"|=",l,c2)
                      :              tok(Token.Kind.PIPE, "|",  l, c2);
            case '^' -> match('=') ? tok(Token.Kind.CARET_ASSIGN,"^=",l,c2)
                      :              tok(Token.Kind.CARET,"^",  l, c2);
            case ':' -> match(':') ? tok(Token.Kind.COLONCOLON,"::",l,c2)
                      :              tok(Token.Kind.COLON, ":", l, c2);
            case '.' -> match('.') && match('.')
                      ? tok(Token.Kind.DOTDOTDOT, "...", l, c2)
                      : tok(Token.Kind.DOT, ".", l, c2);
            default  -> tok(Token.Kind.ERROR, String.valueOf(c), l, c2);
        };
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private void skipWhitespaceAndComments() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (Character.isWhitespace(c)) {
                if (c == '\n') { line++; col = 1; } else col++;
                pos++;
            } else if (pos + 1 < src.length() &&
                       src.charAt(pos) == '/' && src.charAt(pos+1) == '/') {
                while (pos < src.length() && src.charAt(pos) != '\n') pos++;
            } else if (pos + 1 < src.length() &&
                       src.charAt(pos) == '/' && src.charAt(pos+1) == '*') {
                pos += 2; col += 2;
                while (pos + 1 < src.length() &&
                       !(src.charAt(pos) == '*' && src.charAt(pos+1) == '/')) {
                    if (src.charAt(pos) == '\n') { line++; col = 1; } else col++;
                    pos++;
                }
                if (pos + 1 < src.length()) { pos += 2; col += 2; }
            } else {
                break;
            }
        }
    }

    private char peek() { return src.charAt(pos); }

    private char advance() {
        char c = src.charAt(pos++);
        if (c == '\n') { line++; col = 1; } else col++;
        return c;
    }

    private boolean match(char expected) {
        if (pos < src.length() && src.charAt(pos) == expected) {
            advance(); return true;
        }
        return false;
    }

    private static Token tok(Token.Kind k, String v, int l, int c) {
        return new Token(k, v, l, c);
    }
}
