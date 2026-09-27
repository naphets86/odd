package de.hjstephan86.jcl;

import de.hjstephan86.jcl.phases.DeadCodeEliminator;
import de.hjstephan86.jcl.phases.DeadCodeEliminator.BasicBlock;

import java.util.List;

/**
 * Tests fuer DeadCodeEliminator (DCE-Phase).
 *
 * @author Stephan Epp
 */
public class DCETest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testNoDeadCode();
        testOneDeadBlock();
        testAllDead();
        testLinearChain();
        testCountDeadBlocks();
        printSummary();
    }

    static void testNoDeadCode() {
        // entry -> B1 -> B2 -> exit
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"), List.of(1)),
            new BasicBlock(1, List.of("x = 1"), List.of(2)),
            new BasicBlock(2, List.of("return"), List.of())
        );
        DeadCodeEliminator dce = new DeadCodeEliminator(blocks, 0);
        List<BasicBlock> live = dce.eliminate();
        assertEqual("Alle 3 Bloecke lebendig", 3, live.size());
    }

    static void testOneDeadBlock() {
        // entry -> B1; B2 nicht erreichbar
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"), List.of(1)),
            new BasicBlock(1, List.of("return"), List.of()),
            new BasicBlock(2, List.of("dead"),   List.of())  // tot
        );
        DeadCodeEliminator dce = new DeadCodeEliminator(blocks, 0);
        List<BasicBlock> live = dce.eliminate();
        assertEqual("2 lebendige Bloecke", 2, live.size());
        assertTrue("Block 2 nicht in live",
            live.stream().noneMatch(b -> b.id() == 2));
    }

    static void testAllDead() {
        // Nur entry-Block erreichbar, alle anderen tot
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"), List.of()),
            new BasicBlock(1, List.of("dead1"), List.of()),
            new BasicBlock(2, List.of("dead2"), List.of())
        );
        DeadCodeEliminator dce = new DeadCodeEliminator(blocks, 0);
        List<BasicBlock> live = dce.eliminate();
        assertEqual("Nur entry lebendig", 1, live.size());
    }

    static void testLinearChain() {
        // entry -> 1 -> 2 -> 3 -> 4
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"), List.of(1)),
            new BasicBlock(1, List.of("b1"),    List.of(2)),
            new BasicBlock(2, List.of("b2"),    List.of(3)),
            new BasicBlock(3, List.of("b3"),    List.of(4)),
            new BasicBlock(4, List.of("exit"),  List.of())
        );
        DeadCodeEliminator dce = new DeadCodeEliminator(blocks, 0);
        List<BasicBlock> live = dce.eliminate();
        assertEqual("Alle 5 Bloecke lebendig", 5, live.size());
    }

    static void testCountDeadBlocks() {
        List<BasicBlock> blocks = List.of(
            new BasicBlock(0, List.of("entry"), List.of(1)),
            new BasicBlock(1, List.of("b1"),    List.of()),
            new BasicBlock(2, List.of("dead1"), List.of(3)),
            new BasicBlock(3, List.of("dead2"), List.of())
        );
        DeadCodeEliminator dce = new DeadCodeEliminator(blocks, 0);
        int dead = dce.countDeadBlocks();
        assertEqual("2 tote Bloecke gezaehlt", 2, dead);
    }

    // ----------------------------------------------------------------

    static void assertEqual(String msg, int expected, int actual) {
        if (expected == actual) {
            System.out.println("[PASS] " + msg);
            passed++;
        } else {
            System.out.println("[FAIL] " + msg +
                ": erwartet=" + expected + ", erhalten=" + actual);
            failed++;
        }
    }

    static void assertTrue(String msg, boolean cond) {
        if (cond) { System.out.println("[PASS] " + msg); passed++; }
        else       { System.out.println("[FAIL] " + msg); failed++; }
    }

    static void printSummary() {
        System.out.printf("%nErgebnis: %d/%d Tests bestanden.%n",
            passed, passed + failed);
        if (failed > 0) System.exit(1);
    }
}
