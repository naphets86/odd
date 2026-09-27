package de.hjstephan86.jcl;

import de.hjstephan86.jcl.phases.ConstantPropagator;
import de.hjstephan86.jcl.phases.ConstantPropagator.SSAVar;

import java.util.List;
import java.util.Map;

/**
 * Tests fuer ConstantPropagator (CPP-Phase).
 *
 * @author Stephan Epp
 */
public class ConstantPropagatorTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testNoConstants();
        testDirectConstant();
        testPropagatedConstant();
        testContradiction();
        testGetAllConstants();
        printSummary();
    }

    static void testNoConstants() {
        // Alle Variablen ohne bekannten Wert
        List<SSAVar> vars = List.of(
            new SSAVar(0, "x", null),
            new SSAVar(1, "y", null)
        );
        boolean[][] du = {{false, false}, {false, false}};
        ConstantPropagator cp = new ConstantPropagator(vars, du);
        int count = cp.propagate();
        assertEqual("Keine Konstanten propagiert", 0, count);
    }

    static void testDirectConstant() {
        // x = 42 (direkt konstant)
        List<SSAVar> vars = List.of(
            new SSAVar(0, "x", 42),
            new SSAVar(1, "y", null)
        );
        boolean[][] du = {{false, false}, {false, false}};
        ConstantPropagator cp = new ConstantPropagator(vars, du);
        cp.propagate();
        assertTrue("x == 42", Integer.valueOf(42).equals(cp.getConstant(0)));
    }

    static void testPropagatedConstant() {
        // x = 5; y haengt von x ab -> y == 5
        List<SSAVar> vars = List.of(
            new SSAVar(0, "x", 5),
            new SSAVar(1, "y", null)
        );
        // x -> y
        boolean[][] du = {{false, true}, {false, false}};
        ConstantPropagator cp = new ConstantPropagator(vars, du);
        int count = cp.propagate();
        assertEqual("1 Konstante propagiert", 1, count);
        assertTrue("y == 5", Integer.valueOf(5).equals(cp.getConstant(1)));
    }

    static void testContradiction() {
        // z haengt von x=1 und y=2 ab -> Widerspruch -> nicht konstant
        List<SSAVar> vars = List.of(
            new SSAVar(0, "x", 1),
            new SSAVar(1, "y", 2),
            new SSAVar(2, "z", null)
        );
        boolean[][] du = {
            {false, false, true},
            {false, false, true},
            {false, false, false}
        };
        ConstantPropagator cp = new ConstantPropagator(vars, du);
        cp.propagate();
        assertTrue("z nicht konstant (Widerspruch)", cp.getConstant(2) == null);
    }

    static void testGetAllConstants() {
        List<SSAVar> vars = List.of(
            new SSAVar(0, "a", 10),
            new SSAVar(1, "b", 20),
            new SSAVar(2, "c", null)
        );
        boolean[][] du = new boolean[3][3];
        ConstantPropagator cp = new ConstantPropagator(vars, du);
        cp.propagate();
        Map<String, Object> consts = cp.getAllConstants();
        assertTrue("a in Konstanten", consts.containsKey("a"));
        assertTrue("b in Konstanten", consts.containsKey("b"));
        assertTrue("c nicht in Konstanten", !consts.containsKey("c"));
    }

    static void assertEqual(String msg, int exp, int act) {
        if (exp == act) { System.out.println("[PASS] " + msg); passed++; }
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
