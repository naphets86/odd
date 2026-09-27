package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.ir.ASTNode;
import de.hjstephan86.jcl.ir.ASTNode.Kind;

import java.util.List;

/**
 * Rekursiv-absteigender Parser fuer einen Subset von Java SE 21.
 * Laufzeit: O(n) mit n = Token-Anzahl.
 *
 * @author Stephan Epp
 */
public class Parser {

    private final List<Token> tokens;
    private int pos = 0;

    public Parser(List<Token> tokens) {
        this.tokens = tokens;
    }

    // ----------------------------------------------------------------
    // Einstiegspunkt
    // ----------------------------------------------------------------

    public ASTNode parseCompilationUnit() {
        ASTNode cu = node(Kind.COMPILATION_UNIT, "");

        // optionaler package-Bezeichner
        if (peek().kind() == Token.Kind.KW_PACKAGE) {
            cu.addChild(parsePackageDecl());
        }
        // import-Deklarationen
        while (peek().kind() == Token.Kind.KW_IMPORT) {
            cu.addChild(parseImportDecl());
        }
        // Typdeklarationen
        while (peek().kind() != Token.Kind.EOF) {
            cu.addChild(parseTypeDecl());
        }
        return cu;
    }

    // ----------------------------------------------------------------
    // Package / Import
    // ----------------------------------------------------------------

    private ASTNode parsePackageDecl() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_PACKAGE);
        String name = parseQualifiedName();
        consume(Token.Kind.SEMI);
        ASTNode n = new ASTNode(Kind.PACKAGE_DECL, name, l, c);
        return n;
    }

    private ASTNode parseImportDecl() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_IMPORT);
        boolean isStatic = matchKind(Token.Kind.KW_STATIC);
        String name = parseQualifiedName();
        if (matchKind(Token.Kind.DOT)) {
            consume(Token.Kind.STAR);
            name += ".*";
        }
        consume(Token.Kind.SEMI);
        return new ASTNode(Kind.IMPORT_DECL, (isStatic ? "static " : "") + name, l, c);
    }

    // ----------------------------------------------------------------
    // Typdeklarationen
    // ----------------------------------------------------------------

    private ASTNode parseTypeDecl() {
        // Modifikatoren
        ASTNode mods = parseModifiers();

        Token t = peek();
        return switch (t.kind()) {
            case KW_CLASS     -> parseClassDecl(mods);
            case KW_INTERFACE -> parseInterfaceDecl(mods);
            case KW_ENUM      -> parseEnumDecl(mods);
            case KW_RECORD    -> parseRecordDecl(mods);
            default           -> errorNode("Unerwartetes Token: " + t);
        };
    }

    private ASTNode parseClassDecl(ASTNode mods) {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_CLASS);
        String name = consumeIdent();
        ASTNode n = new ASTNode(Kind.CLASS_DECL, name, l, c);
        n.addChild(mods);
        // optionaler Typparameter, extends, implements -- vereinfacht
        skipTo(Token.Kind.LBRACE);
        n.addChild(parseClassBody());
        return n;
    }

    private ASTNode parseInterfaceDecl(ASTNode mods) {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_INTERFACE);
        String name = consumeIdent();
        ASTNode n = new ASTNode(Kind.INTERFACE_DECL, name, l, c);
        n.addChild(mods);
        skipTo(Token.Kind.LBRACE);
        n.addChild(parseClassBody());
        return n;
    }

    private ASTNode parseEnumDecl(ASTNode mods) {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_ENUM);
        String name = consumeIdent();
        ASTNode n = new ASTNode(Kind.ENUM_DECL, name, l, c);
        n.addChild(mods);
        skipTo(Token.Kind.LBRACE);
        n.addChild(parseClassBody());
        return n;
    }

    private ASTNode parseRecordDecl(ASTNode mods) {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_RECORD);
        String name = consumeIdent();
        ASTNode n = new ASTNode(Kind.RECORD_DECL, name, l, c);
        n.addChild(mods);
        skipTo(Token.Kind.LBRACE);
        n.addChild(parseClassBody());
        return n;
    }

    // ----------------------------------------------------------------
    // Klassenrumpf
    // ----------------------------------------------------------------

    private ASTNode parseClassBody() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.LBRACE);
        ASTNode body = new ASTNode(Kind.BLOCK, "", l, c);
        while (peek().kind() != Token.Kind.RBRACE &&
               peek().kind() != Token.Kind.EOF) {
            body.addChild(parseMember());
        }
        consume(Token.Kind.RBRACE);
        return body;
    }

    private ASTNode parseMember() {
        ASTNode mods = parseModifiers();
        Token t = peek();

        // Konstruktor: Name + '('
        if (t.kind() == Token.Kind.IDENT &&
            pos + 1 < tokens.size() &&
            tokens.get(pos + 1).kind() == Token.Kind.LPAREN) {
            return parseConstructor(mods);
        }
        // Methode oder Feld: Typ + Name
        return parseMethodOrField(mods);
    }

    private ASTNode parseConstructor(ASTNode mods) {
        int l = peek().line(), c = peek().col();
        String name = consumeIdent();
        ASTNode n = new ASTNode(Kind.CONSTRUCTOR_DECL, name, l, c);
        n.addChild(mods);
        n.addChild(parseParamList());
        n.addChild(parseBlock());
        return n;
    }

    private ASTNode parseMethodOrField(ASTNode mods) {
        int l = peek().line(), c = peek().col();
        ASTNode type = parseType();
        String name = consumeIdent();

        if (peek().kind() == Token.Kind.LPAREN) {
            // Methode
            ASTNode n = new ASTNode(Kind.METHOD_DECL, name, l, c);
            n.addChild(mods);
            n.addChild(type);
            n.addChild(parseParamList());
            if (peek().kind() == Token.Kind.LBRACE) {
                n.addChild(parseBlock());
            } else {
                consume(Token.Kind.SEMI);
            }
            return n;
        } else {
            // Feld
            ASTNode n = new ASTNode(Kind.FIELD_DECL, name, l, c);
            n.addChild(mods);
            n.addChild(type);
            if (matchKind(Token.Kind.ASSIGN)) {
                n.addChild(parseExpr());
            }
            consume(Token.Kind.SEMI);
            return n;
        }
    }

    // ----------------------------------------------------------------
    // Anweisungen
    // ----------------------------------------------------------------

    private ASTNode parseBlock() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.LBRACE);
        ASTNode block = new ASTNode(Kind.BLOCK, "", l, c);
        while (peek().kind() != Token.Kind.RBRACE &&
               peek().kind() != Token.Kind.EOF) {
            block.addChild(parseStatement());
        }
        consume(Token.Kind.RBRACE);
        return block;
    }

    private ASTNode parseStatement() {
        Token t = peek();
        return switch (t.kind()) {
            case LBRACE     -> parseBlock();
            case KW_IF      -> parseIf();
            case KW_WHILE   -> parseWhile();
            case KW_DO      -> parseDo();
            case KW_FOR     -> parseFor();
            case KW_SWITCH  -> parseSwitch();
            case KW_RETURN  -> parseReturn();
            case KW_THROW   -> parseThrow();
            case KW_BREAK   -> { advance(); consume(Token.Kind.SEMI);
                                  yield node(Kind.BREAK_STMT, ""); }
            case KW_CONTINUE-> { advance(); consume(Token.Kind.SEMI);
                                  yield node(Kind.CONTINUE_STMT, ""); }
            case KW_TRY     -> parseTry();
            case SEMI       -> { advance(); yield node(Kind.EMPTY_STMT, ""); }
            default         -> parseExprOrVarDeclStmt();
        };
    }

    private ASTNode parseIf() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_IF);
        consume(Token.Kind.LPAREN);
        ASTNode cond = parseExpr();
        consume(Token.Kind.RPAREN);
        ASTNode n = new ASTNode(Kind.IF_STMT, "", l, c);
        n.addChild(cond);
        n.addChild(parseStatement());
        if (peek().kind() == Token.Kind.KW_ELSE) {
            advance();
            n.addChild(parseStatement());
        }
        return n;
    }

    private ASTNode parseWhile() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_WHILE);
        consume(Token.Kind.LPAREN);
        ASTNode cond = parseExpr();
        consume(Token.Kind.RPAREN);
        ASTNode n = new ASTNode(Kind.WHILE_STMT, "", l, c);
        n.addChild(cond);
        n.addChild(parseStatement());
        return n;
    }

    private ASTNode parseDo() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_DO);
        ASTNode body = parseStatement();
        consume(Token.Kind.KW_WHILE);
        consume(Token.Kind.LPAREN);
        ASTNode cond = parseExpr();
        consume(Token.Kind.RPAREN);
        consume(Token.Kind.SEMI);
        ASTNode n = new ASTNode(Kind.DO_STMT, "", l, c);
        n.addChild(body);
        n.addChild(cond);
        return n;
    }

    private ASTNode parseFor() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_FOR);
        consume(Token.Kind.LPAREN);
        ASTNode n = new ASTNode(Kind.FOR_STMT, "", l, c);

        if (peek().kind() != Token.Kind.SEMI) {
            if (isTypeName(peek()) && pos + 1 < tokens.size()) {
                Token typeToken = peek();
                ASTNode type = parseType();
                if (peek().kind() == Token.Kind.IDENT) {
                    String name = consumeIdent();
                    ASTNode decl = new ASTNode(Kind.VAR_DECL_STMT, name, typeToken.line(), typeToken.col());
                    decl.addChild(type);
                    if (matchKind(Token.Kind.ASSIGN)) decl.addChild(parseExpr());
                    n.addChild(decl);
                } else {
                    pos = pos - 1;
                    n.addChild(parseExpr());
                }
            } else {
                n.addChild(parseExpr());
            }
        }
        consume(Token.Kind.SEMI);
        if (peek().kind() != Token.Kind.SEMI) n.addChild(parseExpr());
        consume(Token.Kind.SEMI);
        if (peek().kind() != Token.Kind.RPAREN) n.addChild(parseExpr());
        consume(Token.Kind.RPAREN);
        n.addChild(parseStatement());
        return n;
    }

    private ASTNode parseSwitch() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_SWITCH);
        consume(Token.Kind.LPAREN);
        ASTNode expr = parseExpr();
        consume(Token.Kind.RPAREN);
        ASTNode n = new ASTNode(Kind.SWITCH_STMT, "", l, c);
        n.addChild(expr);
        consume(Token.Kind.LBRACE);
        while (peek().kind() != Token.Kind.RBRACE &&
               peek().kind() != Token.Kind.EOF) {
            n.addChild(parseStatement());
        }
        consume(Token.Kind.RBRACE);
        return n;
    }

    private ASTNode parseReturn() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_RETURN);
        ASTNode n = new ASTNode(Kind.RETURN_STMT, "", l, c);
        if (peek().kind() != Token.Kind.SEMI) n.addChild(parseExpr());
        consume(Token.Kind.SEMI);
        return n;
    }

    private ASTNode parseThrow() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_THROW);
        ASTNode n = new ASTNode(Kind.THROW_STMT, "", l, c);
        n.addChild(parseExpr());
        consume(Token.Kind.SEMI);
        return n;
    }

    private ASTNode parseTry() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_TRY);
        ASTNode n = new ASTNode(Kind.TRY_STMT, "", l, c);
        n.addChild(parseBlock());
        while (peek().kind() == Token.Kind.KW_CATCH) {
            advance();
            consume(Token.Kind.LPAREN);
            String exType = parseQualifiedName();
            String exName = consumeIdent();
            consume(Token.Kind.RPAREN);
            ASTNode cc = new ASTNode(Kind.CATCH_CLAUSE, exType + " " + exName, l, c);
            cc.addChild(parseBlock());
            n.addChild(cc);
        }
        if (peek().kind() == Token.Kind.KW_FINALLY) {
            advance();
            n.addChild(parseBlock());
        }
        return n;
    }

    private ASTNode parseExprOrVarDeclStmt() {
        // Heuristik: lokale Variablendeklaration wie "int x = 1;"
        if (isTypeName(peek()) && pos + 1 < tokens.size()) {
            Token typeToken = peek();
            int startPos = pos;
            ASTNode type = parseType();
            Token next = peek();
            if (next.kind() == Token.Kind.IDENT) {
                String name = consumeIdent();
                ASTNode n = new ASTNode(Kind.VAR_DECL_STMT, name, typeToken.line(), typeToken.col());
                n.addChild(type);
                if (matchKind(Token.Kind.ASSIGN)) {
                    n.addChild(parseExpr());
                }
                consume(Token.Kind.SEMI);
                return n;
            }
            pos = startPos;
        }

        ASTNode expr = parseExpr();
        consume(Token.Kind.SEMI);
        ASTNode n = node(Kind.EXPR_STMT, "");
        n.addChild(expr);
        return n;
    }

    // ----------------------------------------------------------------
    // Ausdruecke (vereinfacht, linksassoziativ)
    // ----------------------------------------------------------------

    private ASTNode parseExpr() { return parseAssign(); }

    private ASTNode parseAssign() {
        ASTNode left = parseTernary();
        if (isAssignOp()) {
            String op = advance().value();
            ASTNode right = parseAssign();
            ASTNode n = node(Kind.ASSIGN_EXPR, op);
            n.addChild(left); n.addChild(right);
            return n;
        }
        return left;
    }

    private ASTNode parseTernary() {
        ASTNode cond = parseOr();
        if (matchKind(Token.Kind.QUESTION)) {
            ASTNode then = parseExpr();
            consume(Token.Kind.COLON);
            ASTNode els = parseExpr();
            ASTNode n = node(Kind.TERNARY, "?:");
            n.addChild(cond); n.addChild(then); n.addChild(els);
            return n;
        }
        return cond;
    }

    private ASTNode parseOr()  { return parseBinary(this::parseAnd,  Token.Kind.OR); }
    private ASTNode parseAnd() { return parseBinary(this::parseEq,   Token.Kind.AND); }
    private ASTNode parseEq()  { return parseBinary(this::parseRel,  Token.Kind.EQ, Token.Kind.NE); }
    private ASTNode parseRel() { return parseBinary(this::parseAdd,  Token.Kind.LT, Token.Kind.GT, Token.Kind.LE, Token.Kind.GE); }
    private ASTNode parseAdd() { return parseBinary(this::parseMul,  Token.Kind.PLUS, Token.Kind.MINUS); }
    private ASTNode parseMul() { return parseBinary(this::parseUnary,Token.Kind.STAR, Token.Kind.SLASH, Token.Kind.PERCENT); }

    @FunctionalInterface
    private interface ParseFn { ASTNode parse(); }

    private ASTNode parseBinary(ParseFn sub, Token.Kind... ops) {
        ASTNode left = sub.parse();
        while (isOneOf(ops)) {
            String op = advance().value();
            ASTNode right = sub.parse();
            ASTNode n = node(Kind.BINARY, op);
            n.addChild(left); n.addChild(right);
            left = n;
        }
        return left;
    }

    private ASTNode parseUnary() {
        Token t = peek();
        if (t.kind() == Token.Kind.NOT || t.kind() == Token.Kind.TILDE ||
            t.kind() == Token.Kind.MINUS || t.kind() == Token.Kind.INC ||
            t.kind() == Token.Kind.DEC) {
            String op = advance().value();
            ASTNode n = node(Kind.UNARY, op);
            n.addChild(parseUnary());
            return n;
        }
        return parsePostfix();
    }

    private ASTNode parsePostfix() {
        ASTNode expr = parsePrimary();
        while (true) {
            if (peek().kind() == Token.Kind.DOT) {
                advance();
                String name = consumeIdent();
                if (peek().kind() == Token.Kind.LPAREN) {
                    ASTNode n = node(Kind.METHOD_CALL, name);
                    n.addChild(expr);
                    n.addChild(parseArgs());
                    expr = n;
                } else {
                    ASTNode n = node(Kind.FIELD_ACCESS, name);
                    n.addChild(expr);
                    expr = n;
                }
            } else if (peek().kind() == Token.Kind.LBRACKET) {
                advance();
                ASTNode idx = parseExpr();
                consume(Token.Kind.RBRACKET);
                ASTNode n = node(Kind.ARRAY_ACCESS, "[]");
                n.addChild(expr); n.addChild(idx);
                expr = n;
            } else if (peek().kind() == Token.Kind.INC ||
                       peek().kind() == Token.Kind.DEC) {
                String op = advance().value() + "(post)";
                ASTNode n = node(Kind.UNARY, op);
                n.addChild(expr);
                expr = n;
            } else {
                break;
            }
        }
        return expr;
    }

    private ASTNode parsePrimary() {
        Token t = peek();
        return switch (t.kind()) {
            case INT_LIT, LONG_LIT, FLOAT_LIT, DOUBLE_LIT,
                 CHAR_LIT, STRING_LIT, BOOL_LIT, NULL_LIT -> {
                advance();
                yield node(Kind.LITERAL, t.value());
            }
            case IDENT -> {
                String name = advance().value();
                if (peek().kind() == Token.Kind.LPAREN) {
                    ASTNode n = node(Kind.METHOD_CALL, name);
                    n.addChild(parseArgs());
                    yield n;
                }
                yield node(Kind.NAME, name);
            }
            case KW_THIS  -> { advance(); yield node(Kind.NAME, "this"); }
            case KW_SUPER -> { advance(); yield node(Kind.NAME, "super"); }
            case KW_NEW   -> parseNewExpr();
            case LPAREN   -> {
                advance();
                ASTNode e = parseExpr();
                consume(Token.Kind.RPAREN);
                yield e;
            }
            default -> {
                advance();
                yield node(Kind.ERROR, t.value());
            }
        };
    }

    private ASTNode parseNewExpr() {
        int l = peek().line(), c = peek().col();
        consume(Token.Kind.KW_NEW);
        String type = parseQualifiedName();
        ASTNode n = new ASTNode(Kind.NEW_OBJECT, type, l, c);
        consume(Token.Kind.LPAREN);
        while (peek().kind() != Token.Kind.RPAREN &&
               peek().kind() != Token.Kind.EOF) {
            n.addChild(parseExpr());
            matchKind(Token.Kind.COMMA);
        }
        consume(Token.Kind.RPAREN);
        return n;
    }

    private ASTNode parseArgs() {
        ASTNode args = node(Kind.BLOCK, "args");
        consume(Token.Kind.LPAREN);
        while (peek().kind() != Token.Kind.RPAREN &&
               peek().kind() != Token.Kind.EOF) {
            args.addChild(parseExpr());
            matchKind(Token.Kind.COMMA);
        }
        consume(Token.Kind.RPAREN);
        return args;
    }

    // ----------------------------------------------------------------
    // Typ-Parsing
    // ----------------------------------------------------------------

    private ASTNode parseType() {
        Token t = peek();
        Kind k = switch (t.kind()) {
            case KW_INT     -> Kind.PRIMITIVE_TYPE;
            case KW_LONG    -> Kind.PRIMITIVE_TYPE;
            case KW_DOUBLE  -> Kind.PRIMITIVE_TYPE;
            case KW_FLOAT   -> Kind.PRIMITIVE_TYPE;
            case KW_BOOLEAN -> Kind.PRIMITIVE_TYPE;
            case KW_CHAR    -> Kind.PRIMITIVE_TYPE;
            case KW_BYTE    -> Kind.PRIMITIVE_TYPE;
            case KW_SHORT   -> Kind.PRIMITIVE_TYPE;
            case KW_VOID    -> Kind.VOID_TYPE;
            default         -> Kind.CLASS_TYPE;
        };
        String name = (k == Kind.CLASS_TYPE) ? parseQualifiedName()
                                              : advance().value();
        // Array-Suffix
        while (peek().kind() == Token.Kind.LBRACKET) {
            advance(); consume(Token.Kind.RBRACKET);
            name += "[]";
            k = Kind.ARRAY_TYPE;
        }
        return new ASTNode(k, name, t.line(), t.col());
    }

    private ASTNode parseParamList() {
        ASTNode params = node(Kind.BLOCK, "params");
        consume(Token.Kind.LPAREN);
        while (peek().kind() != Token.Kind.RPAREN &&
               peek().kind() != Token.Kind.EOF) {
            ASTNode type = parseType();
            String name = consumeIdent();
            ASTNode p = node(Kind.PARAM, name);
            p.addChild(type);
            params.addChild(p);
            matchKind(Token.Kind.COMMA);
        }
        consume(Token.Kind.RPAREN);
        return params;
    }

    private ASTNode parseModifiers() {
        ASTNode mods = node(Kind.MODIFIER, "");
        while (isModifier()) {
            mods.addChild(node(Kind.MODIFIER, advance().value()));
        }
        return mods;
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private String parseQualifiedName() {
        StringBuilder sb = new StringBuilder(consumeIdent());
        while (peek().kind() == Token.Kind.DOT &&
               pos + 1 < tokens.size() &&
               tokens.get(pos + 1).kind() == Token.Kind.IDENT) {
            advance();
            sb.append('.').append(consumeIdent());
        }
        return sb.toString();
    }

    private Token peek() {
        return pos < tokens.size() ? tokens.get(pos)
               : new Token(Token.Kind.EOF, "", 0, 0);
    }

    private Token advance() {
        Token t = peek();
        if (pos < tokens.size()) pos++;
        return t;
    }

    private Token consume(Token.Kind expected) {
        Token t = advance();
        if (t.kind() != expected) {
            throw new RuntimeException(
                "Syntaxfehler: Erwartet " + expected +
                ", gefunden " + t + " (Zeile " + t.line() + ")");
        }
        return t;
    }

    private boolean matchKind(Token.Kind k) {
        if (peek().kind() == k) { advance(); return true; }
        return false;
    }

    private String consumeIdent() {
        Token t = advance();
        if (t.kind() != Token.Kind.IDENT) {
            return t.value(); // Tolerant: Schluesselwort als Name
        }
        return t.value();
    }

    private void skipTo(Token.Kind k) {
        while (peek().kind() != k && peek().kind() != Token.Kind.EOF) advance();
    }

    private boolean isOneOf(Token.Kind... kinds) {
        Token.Kind cur = peek().kind();
        for (Token.Kind k : kinds) if (cur == k) return true;
        return false;
    }

    private boolean isAssignOp() {
        return switch (peek().kind()) {
            case ASSIGN, PLUS_ASSIGN, MINUS_ASSIGN, STAR_ASSIGN,
                 SLASH_ASSIGN, PERCENT_ASSIGN, AMP_ASSIGN, PIPE_ASSIGN,
                 CARET_ASSIGN, LSHIFT_ASSIGN, RSHIFT_ASSIGN, URSHIFT_ASSIGN
                 -> true;
            default -> false;
        };
    }

    private boolean isModifier() {
        return switch (peek().kind()) {
            case KW_PUBLIC, KW_PRIVATE, KW_PROTECTED, KW_STATIC,
                 KW_FINAL, KW_ABSTRACT, KW_NATIVE, KW_SYNCHRONIZED,
                 KW_TRANSIENT, KW_VOLATILE, KW_STRICTFP, KW_SEALED,
                 KW_DEFAULT -> true;
            default -> false;
        };
    }

    private boolean isTypeName(Token t) {
        return switch (t.kind()) {
            case KW_INT, KW_LONG, KW_DOUBLE, KW_FLOAT, KW_BOOLEAN,
                 KW_CHAR, KW_BYTE, KW_SHORT, KW_VOID, IDENT -> true;
            default -> false;
        };
    }

    private ASTNode node(Kind k, String v) {
        Token t = peek();
        return new ASTNode(k, v, t.line(), t.col());
    }

    private ASTNode errorNode(String msg) {
        return new ASTNode(Kind.ERROR, msg, peek().line(), peek().col());
    }
}
