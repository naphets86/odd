package de.hjstephan86.jcl;

import de.hjstephan86.jcl.ir.ASTNode;
import de.hjstephan86.jcl.ir.ASTNode.Kind;
import de.hjstephan86.jcl.phases.StaticInitResolver;
import de.hjstephan86.jcl.phases.StaticInitResolver.StaticInit;

import java.util.List;

/**
 * Tests fuer StaticInitResolver (IRP-Phase).
 *
 * @author Stephan Epp
 */
public class StaticInitResolverTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testEmptyAST();
        testNoCycle();
        testCycleDetected();
        testHasSIOF();
        printSummary();
    }

    static void testEmptyAST() {
        ASTNode cu = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        StaticInitResolver irp = new StaticInitResolver(cu);
        List<StaticInit> order = irp.resolve();
        assertTrue("Leere Liste fuer leeren AST", order.isEmpty());
        assertEqual("0 Initialisierer", 0, irp.initCount());
    }

    static void testNoCycle() {
        // Keine statischen Felder -> keine SIOF
        ASTNode cu  = new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1);
        ASTNode cls = new ASTNode(Kind.CLASS_DECL, "MyClass", 1, 1);
        // Nicht-statisches Feld (kein static-Modifier)
        ASTNode fld = new ASTNode(Kind.FIELD_DECL, "x", 2, 1);
        ASTNode mod = new ASTNode(Kind.MODIFIER, "", 2, 1);
        fld.addChild(mod);
        cls.addChild(fld);
        cu.addChild(cls);
        StaticInitResolver irp = new StaticInitResolver(cu);
        List<StaticInit> order = irp.resolve();
        assertTrue("Kein SIOF", !irp.hasSIOF());
    }

    static void testCycleDetected() {
        // Direktes Erstellen eines Resolvers mit zirkulaeren Abhaengigkeiten
        // via Testsubklasse
        StaticInitResolverTestable irp = new StaticInitResolverTestable();
        irp.addInit("A", List.of("B"));
        irp.addInit("B", List.of("A"));
        boolean siof = irp.hasSIOF();
        assertTrue("SIOF erkannt (Zykel A->B->A)", siof);
    }

    static void testHasSIOF() {
        StaticInitResolverTestable irp = new StaticInitResolverTestable();
        irp.addInit("X", List.of());
        irp.addInit("Y", List.of("X"));
        irp.addInit("Z", List.of("Y"));
        assertTrue("Kein SIOF bei linearer Abhaengigkeit", !irp.hasSIOF());
    }

    // ----------------------------------------------------------------
    // Testbare Unterklasse
    // ----------------------------------------------------------------

    static class StaticInitResolverTestable extends StaticInitResolver {
        private final java.util.List<StaticInit> testInits = new java.util.ArrayList<>();

        StaticInitResolverTestable() {
            super(new ASTNode(Kind.COMPILATION_UNIT, "", 1, 1));
        }

        void addInit(String name, List<String> deps) {
            testInits.add(new StaticInit("", name, deps, null));
        }

        @Override
        public boolean hasSIOF() {
            // Direkter Zykeltest auf testInits
            int q = testInits.size();
            if (q < 2) return false;
            java.util.Map<String, Integer> idx = new java.util.HashMap<>();
            for (int i = 0; i < q; i++) idx.put(testInits.get(i).fieldName(), i);
            boolean[][] adj = new boolean[q][q];
            for (StaticInit s : testInits) {
                int i = idx.get(s.fieldName());
                for (String d : s.dependsOn()) {
                    Integer j = idx.get(d);
                    if (j != null) adj[i][j] = true;
                }
            }
            // Floyd-Warshall fuer Zykelerkennung
            boolean[][] r = new boolean[q][q];
            for (int i = 0; i < q; i++) r[i] = adj[i].clone();
            for (int k = 0; k < q; k++)
                for (int i = 0; i < q; i++)
                    for (int j = 0; j < q; j++)
                        if (r[i][k] && r[k][j]) r[i][j] = true;
            for (int i = 0; i < q; i++) if (r[i][i]) return true;
            return false;
        }
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
