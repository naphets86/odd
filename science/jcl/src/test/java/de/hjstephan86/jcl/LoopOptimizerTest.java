package de.hjstephan86.jcl;

import de.hjstephan86.jcl.ir.CFGBuilder.MethodCFG;
import de.hjstephan86.jcl.ir.SSABuilder.SSAMethod;
import de.hjstephan86.jcl.phases.DeadCodeEliminator.BasicBlock;
import de.hjstephan86.jcl.phases.LoopOptimizer;

import java.util.List;

/**
 * Tests fuer LoopOptimizer (SEP-Phase).
 *
 * @author Stephan Epp
 */
public class LoopOptimizerTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testNoLoop();
        testSimpleLoop();
        testNestedLoop();
        testLoopWithInvariant();
        printSummary();
    }

    static void testNoLoop() {
        // Linearer CFG ohne Rückkante -- keine Schleifen
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"), List.of(1)),
            new BasicBlock(1, List.of("x = 1"), List.of(2)),
            new BasicBlock(2, List.of("return"), List.of())
        );
        MethodCFG cfg = new MethodCFG("testNoLoop", blocks, 0, 2);
        SSAMethod ssa = new SSAMethod("testNoLoop", List.of(),
            new boolean[0][0], new boolean[0][0], 0, new int[0]);
        LoopOptimizer opt = new LoopOptimizer(List.of(cfg), List.of(ssa));
        int hoisted = opt.optimize();
        assertEqual("Kein Hoist ohne Schleife", 0, hoisted);
    }

    static void testSimpleLoop() {
        // Schleife: header(1) -> body(2) -> header(1), exit(3)
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"),    List.of(1)),
            new BasicBlock(1, List.of("i < 10"),   List.of(2, 3)),
            new BasicBlock(2, List.of("i = i + 1"),List.of(1)),  // Rueckkante
            new BasicBlock(3, List.of("return"),   List.of())
        );
        MethodCFG cfg = new MethodCFG("testSimpleLoop", blocks, 0, 3);
        SSAMethod ssa = new SSAMethod("testSimpleLoop", List.of(),
            new boolean[0][0], new boolean[0][0], 0, new int[0]);
        LoopOptimizer opt = new LoopOptimizer(List.of(cfg), List.of(ssa));
        int hoisted = opt.optimize();
        assertTrue("Optimierung laeuft ohne Fehler", hoisted >= 0);
    }

    static void testNestedLoop() {
        // Aussen: 0->1->2->1 (exit 3); Innen: 1->2->2 (kein echter Innen-Loop)
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"),  List.of(1)),
            new BasicBlock(1, List.of("outer"),  List.of(2, 3)),
            new BasicBlock(2, List.of("inner"),  List.of(1)),
            new BasicBlock(3, List.of("exit"),   List.of())
        );
        MethodCFG cfg = new MethodCFG("testNested", blocks, 0, 3);
        SSAMethod ssa = new SSAMethod("testNested", List.of(),
            new boolean[0][0], new boolean[0][0], 0, new int[0]);
        LoopOptimizer opt = new LoopOptimizer(List.of(cfg), List.of(ssa));
        int hoisted = opt.optimize();
        assertTrue("Kein Fehler bei geschachtelter Struktur", hoisted >= 0);
    }

    static void testLoopWithInvariant() {
        // Schleife mit konstantem Ausdruck: "c = 42" ist invariant
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"),    List.of(1)),
            new BasicBlock(1, List.of("i < n"),    List.of(2, 3)),
            new BasicBlock(2, List.of("c = 42", "i = i + 1"), List.of(1)),
            new BasicBlock(3, List.of("return"),   List.of())
        );
        MethodCFG cfg = new MethodCFG("testInvariant", blocks, 0, 3);
        SSAMethod ssa = new SSAMethod("testInvariant", List.of(),
            new boolean[0][0], new boolean[0][0], 0, new int[0]);
        LoopOptimizer opt = new LoopOptimizer(List.of(cfg), List.of(ssa));
        int hoisted = opt.optimize();
        // "c = 42" ist invariant und wird gezaehlt
        assertTrue("Mindestens eine invariante Definition erkannt", hoisted >= 1);
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
