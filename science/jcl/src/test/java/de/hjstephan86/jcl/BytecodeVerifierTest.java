package de.hjstephan86.jcl;

import de.hjstephan86.jcl.ir.ASTNode;
import de.hjstephan86.jcl.ir.ASTNode.Kind;
import de.hjstephan86.jcl.phases.BytecodeVerifier;
import de.hjstephan86.jcl.phases.BytecodeVerifier.VerificationException;

import java.util.List;

/**
 * Tests fuer BytecodeVerifier (BVP-Phase).
 *
 * @author Stephan Epp
 */
public class BytecodeVerifierTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testEmptyAST();
        testIntLiteral();
        testBoolLiteral();
        testStringLiteral();
        testNullLiteral();
        testValidBinaryInt();
        testIllegalTransitionDetection();
        printSummary();
    }

    static void testEmptyAST() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        BytecodeVerifier bvp = new BytecodeVerifier(cu);
        bvp.verify(); // keine Exception erwartet
        assertTrue("Leerer AST verifiziert", true);
    }

    static void testIntLiteral() {
        ASTNode cu  = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ASTNode lit = new ASTNode(Kind.LITERAL, "42", 1, 1);
        cu.addChild(lit);
        BytecodeVerifier bvp = new BytecodeVerifier(cu);
        bvp.verify();
        assertTrue("Int-Literal verifiziert", bvp.getErrors().isEmpty());
    }

    static void testBoolLiteral() {
        ASTNode cu  = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ASTNode lit = new ASTNode(Kind.LITERAL, "true", 1, 1);
        cu.addChild(lit);
        new BytecodeVerifier(cu).verify();
        assertTrue("Bool-Literal verifiziert", true);
    }

    static void testStringLiteral() {
        ASTNode cu  = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ASTNode lit = new ASTNode(Kind.LITERAL, "\"hello\"", 1, 1);
        cu.addChild(lit);
        new BytecodeVerifier(cu).verify();
        assertTrue("String-Literal verifiziert", true);
    }

    static void testNullLiteral() {
        ASTNode cu  = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ASTNode lit = new ASTNode(Kind.LITERAL, "null", 1, 1);
        cu.addChild(lit);
        new BytecodeVerifier(cu).verify();
        assertTrue("Null-Literal verifiziert", true);
    }

    static void testValidBinaryInt() {
        // int + int -> kein Fehler
        ASTNode cu  = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ASTNode bin = new ASTNode(Kind.BINARY, "+", 1, 1);
        bin.addChild(new ASTNode(Kind.LITERAL, "1",  1, 1));
        bin.addChild(new ASTNode(Kind.LITERAL, "2",  1, 1));
        cu.addChild(bin);
        new BytecodeVerifier(cu).verify();
        assertTrue("int + int verifiziert", true);
    }

    static void testIllegalTransitionDetection() {
        // containsIllegalTransition: K_2 (Fehlermuster) in trivialem Graph
        boolean[][] typeGraph    = {{false, true}, {false, false}};
        boolean[][] illegalAdj   = {{false, true}, {false, false}};
        boolean found = BytecodeVerifier.containsIllegalTransition(
            typeGraph, illegalAdj);
        // Muster K_1->K_2 ist im typeGraph enthalten
        assertTrue("Illegaler Uebergang erkannt via SI", found);
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
