package de.hjstephan86.jcl;

import de.hjstephan86.jcl.phases.TypeResolver;
import de.hjstephan86.jcl.phases.TypeResolver.JavaType;

import java.util.List;

/**
 * Tests fuer TypeResolver (TAP-Phase).
 *
 * @author Stephan Epp
 */
public class TypeResolverTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testSimpleHierarchy();
        testIsSubtype();
        testCycleDetection();
        testEmptyResolver();
        printSummary();
    }

    static void testSimpleHierarchy() {
        TypeResolver tr = new TypeResolver();
        tr.register(new JavaType("Object",    null,       List.of(), null));
        tr.register(new JavaType("Animal",    "Object",   List.of(), null));
        tr.register(new JavaType("Dog",       "Animal",   List.of(), null));
        tr.register(new JavaType("Cat",       "Animal",   List.of(), null));

        List<JavaType> order = tr.resolve();
        assertTrue("Nicht leer", !order.isEmpty());
        // Object muss vor Animal kommen
        int idxObj    = indexOf(order, "Object");
        int idxAnimal = indexOf(order, "Animal");
        int idxDog    = indexOf(order, "Dog");
        assertTrue("Object vor Animal", idxObj < idxAnimal);
        assertTrue("Animal vor Dog",    idxAnimal < idxDog);
        passed("testSimpleHierarchy");
    }

    static void testIsSubtype() {
        TypeResolver tr = new TypeResolver();
        tr.register(new JavaType("Object",  null,      List.of(), null));
        tr.register(new JavaType("Shape",   "Object",  List.of(), null));
        tr.register(new JavaType("Circle",  "Shape",   List.of(), null));
        tr.resolve();
        assertTrue("Circle isSubtype Shape",  tr.isSubtype("Circle", "Shape"));
        assertTrue("Circle isSubtype Object", tr.isSubtype("Circle", "Object"));
        assertTrue("Shape nicht Sub v Circle", !tr.isSubtype("Shape", "Circle"));
        passed("testIsSubtype");
    }

    static void testCycleDetection() {
        TypeResolver tr = new TypeResolver();
        tr.register(new JavaType("A", "B", List.of(), null));
        tr.register(new JavaType("B", "A", List.of(), null));
        boolean caught = false;
        try { tr.resolve(); }
        catch (IllegalStateException e) { caught = true; }
        assertTrue("Zykel erkannt", caught);
        passed("testCycleDetection");
    }

    static void testEmptyResolver() {
        TypeResolver tr = new TypeResolver();
        List<JavaType> order = tr.resolve();
        assertTrue("Leere Liste", order.isEmpty());
        passed("testEmptyResolver");
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    static int indexOf(List<JavaType> list, String name) {
        for (int i = 0; i < list.size(); i++)
            if (list.get(i).name().equals(name)) return i;
        return -1;
    }

    static void assertTrue(String msg, boolean cond) {
        if (!cond) {
            System.out.println("[FAIL] " + msg);
            failed++;
        }
    }

    static void passed(String test) {
        System.out.println("[PASS] " + test);
        passed++;
    }

    static void printSummary() {
        System.out.printf("%nErgebnis: %d/%d Tests bestanden.%n",
            passed, passed + failed);
        if (failed > 0) System.exit(1);
    }
}
