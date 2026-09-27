package de.hjstephan86.jcl;

/**
 * Integrationstests fuer die JCL-Pipeline.
 *
 * @author Stephan Epp
 */
public class JCLCompilerTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) {
        testEmptyClass();
        testClassWithFields();
        testClassWithMethod();
        testClassWithIf();
        testClassWithLoop();
        testBytecodeHeader();
        printSummary();
    }

    static byte[] compile(String src) {
        JCLCompiler.Options opts = new JCLCompiler.Options();
        opts.verbose = false;
        return new JCLCompiler(opts).compile(src);
    }

    static void testEmptyClass() {
        byte[] bc = compile("class Empty {}");
        assertTrue("Bytecode nicht null",  bc != null);
        assertTrue("Bytecode nicht leer",  bc.length > 0);
        // Magic Number CAFEBABE
        assertTrue("Magic: CA", bc[0] == (byte)0xCA);
        assertTrue("Magic: FE", bc[1] == (byte)0xFE);
        assertTrue("Magic: BA", bc[2] == (byte)0xBA);
        assertTrue("Magic: BE", bc[3] == (byte)0xBE);
    }

    static void testClassWithFields() {
        byte[] bc = compile("""
            class WithFields {
                int x;
                String name;
            }
            """);
        assertTrue("Bytecode fuer Klasse mit Feldern", bc != null && bc.length > 0);
    }

    static void testClassWithMethod() {
        byte[] bc = compile("""
            class WithMethod {
                public int add(int a, int b) {
                    return a;
                }
            }
            """);
        assertTrue("Bytecode fuer Klasse mit Methode", bc != null && bc.length > 0);
    }

    static void testClassWithIf() {
        byte[] bc = compile("""
            class WithIf {
                void test(boolean flag) {
                    if (flag) {
                        return;
                    }
                }
            }
            """);
        assertTrue("Bytecode fuer if-Anweisung", bc != null && bc.length > 0);
    }

    static void testClassWithLoop() {
        byte[] bc = compile("""
            class WithLoop {
                int sum(int n) {
                    int s = 0;
                    while (n > 0) {
                        n = n - 1;
                    }
                    return s;
                }
            }
            """);
        assertTrue("Bytecode fuer while-Schleife", bc != null && bc.length > 0);
    }

    static void testBytecodeHeader() {
        byte[] bc = compile("class H {}");
        // Java 21 = Major Version 65 = 0x41
        assertTrue("Minor Version 0 (high)",  bc[4] == 0x00);
        assertTrue("Minor Version 0 (low)",   bc[5] == 0x00);
        assertTrue("Major Version 0 (high)",  bc[6] == 0x00);
        assertTrue("Major Version 65 (low)",  bc[7] == 0x41);
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
