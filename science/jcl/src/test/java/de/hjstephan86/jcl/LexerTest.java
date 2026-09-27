package de.hjstephan86.jcl;

import de.hjstephan86.jcl.phases.Lexer;
import de.hjstephan86.jcl.phases.Token;
import de.hjstephan86.jcl.phases.Token.Kind;

import java.util.List;

/**
 * Tests fuer den Lexer.
 *
 * @author Stephan Epp
 */
public class LexerTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testEmpty();
        testIntLiteral();
        testKeywords();
        testOperators();
        testIdentifier();
        testStringLiteral();
        testCommentSkip();
        testBlockCommentSkip();
        testFullClass();
        printSummary();
    }

    static void testEmpty() {
        List<Token> tokens = new Lexer("").tokenize();
        assertEqual("Nur EOF", 1, tokens.size());
        assertEqual("EOF-Kind", Kind.EOF, tokens.get(0).kind());
    }

    static void testIntLiteral() {
        List<Token> tokens = new Lexer("42").tokenize();
        assertEqual("Int + EOF", 2, tokens.size());
        assertEqual("INT_LIT", Kind.INT_LIT, tokens.get(0).kind());
        assertEqual("Wert 42", "42", tokens.get(0).value());
    }

    static void testKeywords() {
        List<Token> tokens = new Lexer("class public static void").tokenize();
        assertEqual("4 Keywords + EOF", 5, tokens.size());
        assertEqual("class", Kind.KW_CLASS,  tokens.get(0).kind());
        assertEqual("public",Kind.KW_PUBLIC, tokens.get(1).kind());
        assertEqual("static",Kind.KW_STATIC, tokens.get(2).kind());
        assertEqual("void",  Kind.KW_VOID,   tokens.get(3).kind());
    }

    static void testOperators() {
        List<Token> tokens = new Lexer("+ - * / == != <= >=").tokenize();
        assertEqual("8 Operatoren + EOF", 9, tokens.size());
        assertEqual("PLUS",  Kind.PLUS,  tokens.get(0).kind());
        assertEqual("MINUS", Kind.MINUS, tokens.get(1).kind());
        assertEqual("EQ",    Kind.EQ,    tokens.get(4).kind());
        assertEqual("NE",    Kind.NE,    tokens.get(5).kind());
    }

    static void testIdentifier() {
        List<Token> tokens = new Lexer("myVar _count x123").tokenize();
        assertEqual("3 Idents + EOF", 4, tokens.size());
        for (int i = 0; i < 3; i++)
            assertEqual("IDENT", Kind.IDENT, tokens.get(i).kind());
    }

    static void testStringLiteral() {
        List<Token> tokens = new Lexer("\"hello world\"").tokenize();
        assertEqual("STRING + EOF", 2, tokens.size());
        assertEqual("STRING_LIT", Kind.STRING_LIT, tokens.get(0).kind());
    }

    static void testCommentSkip() {
        List<Token> tokens = new Lexer("// this is a comment\n42").tokenize();
        assertEqual("Kommentar uebersprungen: INT + EOF", 2, tokens.size());
        assertEqual("INT_LIT nach Kommentar", Kind.INT_LIT, tokens.get(0).kind());
    }

    static void testBlockCommentSkip() {
        List<Token> tokens = new Lexer("/* block */ 99").tokenize();
        assertEqual("Block-Kommentar uebersprungen: INT + EOF", 2, tokens.size());
        assertEqual("INT_LIT nach Block-Kommentar", Kind.INT_LIT, tokens.get(0).kind());
    }

    static void testFullClass() {
        String src = """
            public class Hello {
                public static void main(String[] args) {
                    int x = 42;
                }
            }
            """;
        List<Token> tokens = new Lexer(src).tokenize();
        assertTrue("Mehr als 10 Token", tokens.size() > 10);
        assertEqual("Erstes Token: public", Kind.KW_PUBLIC, tokens.get(0).kind());
        assertEqual("Letztes Token: EOF", Kind.EOF, tokens.get(tokens.size()-1).kind());
    }

    static void assertEqual(String msg, int exp, int act) {
        if (exp == act) { System.out.println("[PASS] " + msg); passed++; }
        else { System.out.println("[FAIL] " + msg + ": exp=" + exp + " act=" + act); failed++; }
    }

    static void assertEqual(String msg, Kind exp, Kind act) {
        if (exp == act) { System.out.println("[PASS] " + msg); passed++; }
        else { System.out.println("[FAIL] " + msg + ": exp=" + exp + " act=" + act); failed++; }
    }

    static void assertEqual(String msg, String exp, String act) {
        if (exp.equals(act)) { System.out.println("[PASS] " + msg); passed++; }
        else { System.out.println("[FAIL] " + msg + ": exp=" + exp + " act=" + act); failed++; }
    }

    static void assertTrue(String msg, boolean cond) {
        if (cond) { System.out.println("[PASS] " + msg); passed++; }
        else      { System.out.println("[FAIL] " + msg); failed++; }
    }

    static void printSummary() {
        System.out.printf("%nErgebnis: %d/%d Tests bestanden.%n", passed, passed + failed);
        if (failed > 0) System.exit(1);
    }
}
