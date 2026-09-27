package de.hjstephan86.jcl;

import de.hjstephan86.jcl.ir.ASTNode;
import de.hjstephan86.jcl.ir.ASTNode.Kind;
import de.hjstephan86.jcl.phases.Lexer;
import de.hjstephan86.jcl.phases.Parser;
import de.hjstephan86.jcl.phases.Token;

import java.util.List;

/**
 * Tests fuer den Parser.
 *
 * @author Stephan Epp
 */
public class ParserTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testEmptyClass();
        testClassWithMethod();
        testIfStatement();
        testWhileStatement();
        testBinaryExpression();
        testMethodCall();
        testReturnStatement();
        testNestedBlocks();
        printSummary();
    }

    static ASTNode parse(String src) {
        List<Token> tokens = new Lexer(src).tokenize();
        return new Parser(tokens).parseCompilationUnit();
    }

    static void testEmptyClass() {
        ASTNode ast = parse("class Empty {}");
        assertEqual("COMPILATION_UNIT", Kind.COMPILATION_UNIT, ast.getKind());
        assertTrue("Hat Kinder", ast.childCount() > 0);
        assertEqual("CLASS_DECL", Kind.CLASS_DECL,
            ast.getChild(0).getKind());
        assertEqual("Name: Empty", "Empty",
            ast.getChild(0).getValue());
    }

    static void testClassWithMethod() {
        ASTNode ast = parse("""
            class MyClass {
                public int add(int a, int b) {
                    return a;
                }
            }
            """);
        ASTNode cls = ast.getChild(0);
        assertEqual("CLASS_DECL", Kind.CLASS_DECL, cls.getKind());
        // Kinder des Klassenrumpfs (BLOCK) enthalten METHOD_DECL
        boolean hasMethod = containsKind(cls, Kind.METHOD_DECL);
        assertTrue("Methode 'add' gefunden", hasMethod);
    }

    static void testIfStatement() {
        ASTNode ast = parse("""
            class T {
                void test() {
                    if (x) { }
                }
            }
            """);
        assertTrue("IF_STMT im AST", containsKind(ast, Kind.IF_STMT));
    }

    static void testWhileStatement() {
        ASTNode ast = parse("""
            class T {
                void test() {
                    while (true) { }
                }
            }
            """);
        assertTrue("WHILE_STMT im AST", containsKind(ast, Kind.WHILE_STMT));
    }

    static void testBinaryExpression() {
        ASTNode ast = parse("""
            class T {
                int f() { return 1 + 2; }
            }
            """);
        assertTrue("BINARY-Knoten im AST", containsKind(ast, Kind.BINARY));
    }

    static void testMethodCall() {
        ASTNode ast = parse("""
            class T {
                void f() { g(); }
                void g() {}
            }
            """);
        assertTrue("METHOD_CALL im AST", containsKind(ast, Kind.METHOD_CALL));
    }

    static void testReturnStatement() {
        ASTNode ast = parse("""
            class T {
                int f() { return 42; }
            }
            """);
        assertTrue("RETURN_STMT im AST", containsKind(ast, Kind.RETURN_STMT));
    }

    static void testNestedBlocks() {
        ASTNode ast = parse("""
            class T {
                void f() {
                    if (a) {
                        while (b) {
                            return;
                        }
                    }
                }
            }
            """);
        assertTrue("IF_STMT vorhanden",    containsKind(ast, Kind.IF_STMT));
        assertTrue("WHILE_STMT vorhanden", containsKind(ast, Kind.WHILE_STMT));
        assertTrue("RETURN_STMT vorhanden",containsKind(ast, Kind.RETURN_STMT));
    }

    // ----------------------------------------------------------------

    static boolean containsKind(ASTNode node, Kind target) {
        if (node.getKind() == target) return true;
        for (ASTNode c : node.getChildren())
            if (containsKind(c, target)) return true;
        return false;
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
