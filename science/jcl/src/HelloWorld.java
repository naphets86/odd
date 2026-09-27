/**
 * Beispiel-Java-Programm fuer den JCL-Compiler.
 * Demonstriert alle unterstuetzten Sprachkonstrukte.
 *
 * Ausfuehren mit JCL:
 *   java -jar target/jcl.jar --verbose HelloWorld.java
 *
 * @author Stephan Epp
 */
public class HelloWorld {

    // Statisches Feld (IRP: statische Initialisierungsreihenfolge)
    public static final int MAX = 100;
    public static final String GREETING = "Hallo von JCL!";

    // Instanzfelder (TAP: Typaufloesung)
    private int value;
    private String name;

    // Konstruktor
    public HelloWorld(int value, String name) {
        this.value = value;
        this.name  = name;
    }

    // Einfache Methode mit Rueckgabewert (CPP: Konstantenpropagation)
    public int add(int a, int b) {
        int result = a + b;
        return result;
    }

    // Methode mit Schleife (SEP: Schleifenoptimierung)
    public int sumTo(int n) {
        int sum = 0;
        int i   = 0;
        while (i <= n) {
            sum = sum + i;
            i   = i + 1;
        }
        return sum;
    }

    // Methode mit for-Schleife
    public int factorial(int n) {
        int result = 1;
        for (int k = 2; k <= n; k = k + 1) {
            result = result * k;
        }
        return result;
    }

    // Methode mit if-else (DCE: Dead-Code-Elimination)
    public String classify(int x) {
        if (x < 0) {
            return "negativ";
        } else if (x == 0) {
            return "null";
        } else {
            return "positiv";
        }
    }

    // Methode mit try-catch
    public int safeDivide(int a, int b) {
        try {
            return a / b;
        } catch (Exception e) {
            return 0;
        }
    }

    // Rekursive Methode (CFG: Kontrollflussgraph mit Rekursion)
    public int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    // Getter / Setter
    public int  getValue() { return value;  }
    public String getName() { return name;  }
    public void setValue(int v) { this.value = v; }
    public void setName(String n) { this.name = n; }

    // main-Methode
    public static void main(String[] args) {
        HelloWorld hw = new HelloWorld(42, "JCL");
        int sum = hw.sumTo(10);
        int fac = hw.factorial(5);
        String cls = hw.classify(-3);
        int fib = hw.fibonacci(7);
    }
}
