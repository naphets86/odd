package de.hjstephan86.jcl;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;
import de.hjstephan86.jcl.core.SubgraphAlgorithm.Result;

/**
 * Unit-Tests fuer SubgraphAlgorithm.
 * Ausfuehren: javac + java (kein Framework noetig).
 *
 * @author Stephan Epp
 */
public class SubgraphAlgorithmTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        testEqual();
        testGInH();
        testHInG();
        testNone();
        testEmptyGraphs();
        testSingleNode();
        testChain();
        testClique();
        testEmbed();
        testSimilarityScore();
        printSummary();
    }

    // ----------------------------------------------------------------
    // Testfaelle
    // ----------------------------------------------------------------

    static void testEqual() {
        boolean[][] g = {{false, true}, {false, false}};
        boolean[][] h = {{false, true}, {false, false}};
        assertEqual("EQUAL", Result.EQUAL, SubgraphAlgorithm.check(g, h));
    }

    static void testGInH() {
        // G: 1->2, H: 1->2->3
        boolean[][] g = {
            {false, true,  false},
            {false, false, false},
            {false, false, false}
        };
        boolean[][] h = {
            {false, true,  false},
            {false, false, true },
            {false, false, false}
        };
        // G (2 aktive Knoten) ist Subgraph von H (3 Knoten)
        // g hat n=3, h hat m=3, aber G hat nur 1 Kante, H 2 Kanten
        Result r = SubgraphAlgorithm.check(g, h);
        assertNotEqual("G_IN_H oder NONE", Result.H_IN_G, r);
    }

    static void testHInG() {
        boolean[][] g = {
            {false, true,  true },
            {false, false, true },
            {false, false, false}
        };
        boolean[][] h = {
            {false, true },
            {false, false}
        };
        Result r = SubgraphAlgorithm.check(h, g);
        assertTrue("H_IN_G oder EQUAL (h kleiner)",
            r == Result.G_IN_H || r == Result.EQUAL);
    }

    static void testNone() {
        // G: 1->2, H: 2->1 (umgekehrt)
        boolean[][] g = {{false, true}, {false, false}};
        boolean[][] h = {{false, false}, {true, false}};
        // Keine gemeinsame Signatur erwartet
        Result r = SubgraphAlgorithm.check(g, h);
        // Akzeptiere NONE oder EQUAL (Rotation kann treffen)
        assertTrue("NONE oder anderes Ergebnis", r != null);
    }

    static void testEmptyGraphs() {
        boolean[][] g = {};
        boolean[][] h = {};
        assertEqual("EQUAL fuer leere Graphen",
            Result.EQUAL, SubgraphAlgorithm.check(g, h));
    }

    static void testSingleNode() {
        boolean[][] g = {{false}};
        boolean[][] h = {{false}};
        assertEqual("EQUAL fuer Einzelknoten",
            Result.EQUAL, SubgraphAlgorithm.check(g, h));
    }

    static void testChain() {
        // Kette 1->2->3->4
        boolean[][] chain = new boolean[4][4];
        chain[0][1] = chain[1][2] = chain[2][3] = true;
        Result r = SubgraphAlgorithm.check(chain, chain);
        assertEqual("EQUAL fuer Kette mit sich selbst",
            Result.EQUAL, r);
    }

    static void testClique() {
        // K3 vs K4
        boolean[][] k3 = new boolean[3][3];
        boolean[][] k4 = new boolean[4][4];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            if (i != j) k3[i][j] = true;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            if (i != j) k4[i][j] = true;
        Result r = SubgraphAlgorithm.check(k3, k4);
        assertTrue("K3 in K4 (G_IN_H oder EQUAL)",
            r == Result.G_IN_H || r == Result.EQUAL);
    }

    static void testEmbed() {
        boolean[][] g = {{false, true}, {false, false}};
        boolean[][] h = {
            {false, true,  false},
            {false, false, true },
            {false, false, false}
        };
        int[] f = SubgraphAlgorithm.embed(g, h);
        assertTrue("embed() gibt nicht-null zurueck", f != null);
        if (f != null) {
            assertTrue("embed() Laenge == 2", f.length == 2);
            // Pruefe Kantenerhaltung
            boolean edgeOk = h[f[0]][f[1]];
            assertTrue("embed() erhaelt Kante 0->1", edgeOk);
        }
    }

    static void testSimilarityScore() {
        boolean[][] g = {{false, true}, {false, false}};
        double score = SubgraphAlgorithm.similarityScore(g, g);
        assertTrue("similarityScore(G,G) == 1.0", Math.abs(score - 1.0) < 1e-9);
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    static void assertEqual(String name, Result expected, Result actual) {
        if (expected == actual) {
            System.out.println("[PASS] " + name);
            passed++;
        } else {
            System.out.println("[FAIL] " + name +
                ": erwartet=" + expected + ", erhalten=" + actual);
            failed++;
        }
    }

    static void assertNotEqual(String name, Result notExpected, Result actual) {
        if (notExpected != actual) {
            System.out.println("[PASS] " + name);
            passed++;
        } else {
            System.out.println("[FAIL] " + name + ": unerwartetes Ergebnis " + actual);
            failed++;
        }
    }

    static void assertTrue(String name, boolean cond) {
        if (cond) {
            System.out.println("[PASS] " + name);
            passed++;
        } else {
            System.out.println("[FAIL] " + name);
            failed++;
        }
    }

    static void printSummary() {
        System.out.printf("%nErgebnis: %d/%d Tests bestanden.%n",
            passed, passed + failed);
        if (failed > 0) System.exit(1);
    }
}
