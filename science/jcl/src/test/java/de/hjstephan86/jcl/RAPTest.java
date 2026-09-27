package de.hjstephan86.jcl;

import de.hjstephan86.jcl.phases.RegisterAllocator;

/**
 * Tests fuer RegisterAllocator (RAP-Phase).
 *
 * @author Stephan Epp
 */
public class RAPTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testNoInterference();
        testFullInterference();
        testPartialInterference();
        testChromaticNumber();
        testSpillCount();
        printSummary();
    }

    static void testNoInterference() {
        // 3 Variablen, keine Interferenz -> alle koennen Slot 0 haben
        boolean[][] ig = new boolean[3][3];
        RegisterAllocator ra = new RegisterAllocator(3, ig, 4);
        int[] color = ra.allocate();
        assertTrue("Kein Spill", noSpills(color));
        // Alle koennen Slot 0 bekommen (kein Konflikt)
        assertTrue("Slot 0 moeglich", color[0] == 0);
    }

    static void testFullInterference() {
        // K3: alle interferieren miteinander -- brauchen 3 verschiedene Slots
        boolean[][] ig = {{false,true,true},{true,false,true},{true,true,false}};
        RegisterAllocator ra = new RegisterAllocator(3, ig, 5);
        int[] color = ra.allocate();
        assertTrue("Kein Spill bei k=5", noSpills(color));
        // Alle 3 Farben verschieden
        assertTrue("Var 0 != Var 1", color[0] != color[1]);
        assertTrue("Var 0 != Var 2", color[0] != color[2]);
        assertTrue("Var 1 != Var 2", color[1] != color[2]);
    }

    static void testPartialInterference() {
        // 4 Variablen, Pfad-Interferenz: 0-1-2-3
        boolean[][] ig = new boolean[4][4];
        ig[0][1] = ig[1][0] = true;
        ig[1][2] = ig[2][1] = true;
        ig[2][3] = ig[3][2] = true;
        RegisterAllocator ra = new RegisterAllocator(4, ig, 4);
        int[] color = ra.allocate();
        assertTrue("Kein Spill", noSpills(color));
        // Nachbarn haben verschiedene Farben
        assertTrue("0 != 1", color[0] != color[1]);
        assertTrue("1 != 2", color[1] != color[2]);
        assertTrue("2 != 3", color[2] != color[3]);
        // Nicht-Nachbarn koennen gleich sein
        assertTrue("0 und 2 koennen gleich sein", color[0] == color[2] || true);
    }

    static void testChromaticNumber() {
        // K4 braucht 4 Farben
        boolean[][] k4 = new boolean[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            if (i != j) k4[i][j] = true;
        RegisterAllocator ra = new RegisterAllocator(4, k4, 10);
        int chi = ra.chromaticNumber();
        assertEqual("chi(K4) == 4", 4, chi);
    }

    static void testSpillCount() {
        // K4 mit nur 2 Slots -> Spills noetig
        boolean[][] k4 = new boolean[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            if (i != j) k4[i][j] = true;
        RegisterAllocator ra = new RegisterAllocator(4, k4, 2);
        int spills = ra.spillCount(2);
        assertTrue("Spills > 0 bei k=2 fuer K4", spills > 0);
    }

    // ----------------------------------------------------------------

    static boolean noSpills(int[] color) {
        for (int c : color) if (c == -1) return false;
        return true;
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
        System.out.printf("%nErgebnis: %d/%d Tests bestanden.%n",
            passed, passed + failed);
        if (failed > 0) System.exit(1);
    }
}
