package de.hjstephan86.jcl;

import de.hjstephan86.jcl.ir.ASTNode;
import de.hjstephan86.jcl.ir.ASTNode.Kind;
import de.hjstephan86.jcl.phases.ModuleResolver;
import de.hjstephan86.jcl.phases.ModuleResolver.JavaModule;

import java.util.List;

/**
 * Tests fuer ModuleResolver (LAP-Phase).
 *
 * @author Stephan Epp
 */
public class ModuleResolverTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testNoModules();
        testSingleModule();
        testLinearDependency();
        testCycleDetected();
        testContainsPattern();
        testMaxDepth();
        printSummary();
    }

    static void testNoModules() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ModuleResolver mr = new ModuleResolver(cu);
        List<JavaModule> order = mr.resolve();
        assertTrue("Leere Modulliste", order.isEmpty());
    }

    static void testSingleModule() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ModuleResolver mr = new ModuleResolver(cu);
        mr.addModule(new JavaModule("de.hjstephan.jcl",
            List.of("java.base"), List.of(), List.of(), false));
        List<JavaModule> order = mr.resolve();
        // java.base nicht registriert -> nur das eine Modul
        assertEqual("1 Modul aufgeloest", 1, order.size());
    }

    static void testLinearDependency() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ModuleResolver mr = new ModuleResolver(cu);
        mr.addModule(new JavaModule("A", List.of("B"), List.of(), List.of(), false));
        mr.addModule(new JavaModule("B", List.of("C"), List.of(), List.of(), false));
        mr.addModule(new JavaModule("C", List.of(),    List.of(), List.of(), false));
        List<JavaModule> order = mr.resolve();
        assertEqual("3 Module aufgeloest", 3, order.size());
        // C muss vor B, B vor A kommen
        int ia = indexOf(order, "A"), ib = indexOf(order, "B"), ic = indexOf(order, "C");
        assertTrue("C vor B", ic < ib);
        assertTrue("B vor A", ib < ia);
    }

    static void testCycleDetected() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ModuleResolver mr = new ModuleResolver(cu);
        mr.addModule(new JavaModule("X", List.of("Y"), List.of(), List.of(), false));
        mr.addModule(new JavaModule("Y", List.of("X"), List.of(), List.of(), false));
        boolean caught = false;
        try { mr.resolve(); }
        catch (IllegalStateException e) { caught = true; }
        assertTrue("Zirkulaere Abhaengigkeit erkannt", caught);
    }

    static void testContainsPattern() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ModuleResolver mr = new ModuleResolver(cu);
        mr.addModule(new JavaModule("A", List.of("B"), List.of(), List.of(), false));
        mr.addModule(new JavaModule("B", List.of(),    List.of(), List.of(), false));
        mr.resolve();
        // Muster: eine Kante (2-Knoten-Graph)
        boolean[][] pattern = {{false, true}, {false, false}};
        assertTrue("Muster enthalten", mr.containsPattern(pattern));
    }

    static void testMaxDepth() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ModuleResolver mr = new ModuleResolver(cu);
        mr.addModule(new JavaModule("A", List.of("B"), List.of(), List.of(), false));
        mr.addModule(new JavaModule("B", List.of("C"), List.of(), List.of(), false));
        mr.addModule(new JavaModule("C", List.of(),    List.of(), List.of(), false));
        mr.resolve();
        int depth = mr.maxDepth();
        assertTrue("Tiefe >= 2", depth >= 2);
    }

    static int indexOf(List<JavaModule> list, String name) {
        for (int i = 0; i < list.size(); i++)
            if (list.get(i).name().equals(name)) return i;
        return -1;
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
