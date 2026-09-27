package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;

import java.util.*;

/**
 * JVM-Slot-Allokation via Interferenzgraph-Faerbung + SI-Cliquencheck (RAP).
 * Laufzeit: O(n^3) mit n = Variablenanzahl.
 *
 * @author Stephan Epp
 */
public class RegisterAllocator {

    private final int       numVars;
    private final boolean[][] interference;   // Interferenzgraph G_I
    private final int       maxSlots;         // k verfuegbare JVM-Slots

    /**
     * @param numVars      Anzahl lokaler Variablen
     * @param interference Interferenzgraph (numVars x numVars, symmetrisch)
     * @param maxSlots     Maximale JVM-Slot-Anzahl (typisch 65535)
     */
    public RegisterAllocator(int numVars, boolean[][] interference, int maxSlots) {
        this.numVars      = numVars;
        this.interference = interference;
        this.maxSlots     = maxSlots;
    }

    // ----------------------------------------------------------------
    // Allokation
    // ----------------------------------------------------------------

    /**
     * Alloziert JVM-lokale Slots.
     *
     * @return color[i] = Slot-Index fuer Variable i (0-basiert),
     *         oder -1 falls Variable gespillt werden muss.
     */
    public int[] allocate() {
        // SI: pruefe ob K_{maxSlots+1} als Clique in G_I vorhanden
        boolean needsSpill = cliqueExistsLargerThan(maxSlots);
        return needsSpill ? greedyColor(true) : greedyColor(false);
    }

    /**
     * Gibt die chromatische Zahl chi(G_I) zurueck.
     * Da G_I unter SSA chordal ist, gilt chi = omega (Cliquenzahl).
     */
    public int chromaticNumber() {
        for (int k = 1; k <= numVars; k++) {
            if (!cliqueExistsLargerThan(k)) return k;
        }
        return numVars;
    }

    /**
     * Anzahl der Spill-Variablen fuer gegebene Slot-Anzahl k.
     */
    public int spillCount(int k) {
        int[] color = new RegisterAllocator(numVars, interference, k).allocate();
        int spills = 0;
        for (int c : color) if (c == -1) spills++;
        return spills;
    }

    // ----------------------------------------------------------------
    // Interne Methoden
    // ----------------------------------------------------------------

    /**
     * Prueft via SI ob K_{k+1} als Subgraph in G_I einbettbar ist.
     * Falls ja, ist chi(G_I) > k und Spills sind noetig.
     */
    private boolean cliqueExistsLargerThan(int k) {
        if (k + 1 > numVars) return false;
        boolean[][] clique = buildClique(k + 1);
        SubgraphAlgorithm.Result r = SubgraphAlgorithm.check(clique, interference);
        return r == SubgraphAlgorithm.Result.G_IN_H ||
               r == SubgraphAlgorithm.Result.EQUAL;
    }

    /** Greedy-Faerbung; -1 = Spill wenn kein Slot mehr frei. */
    private int[] greedyColor(boolean allowSpill) {
        int[] color = new int[numVars];
        Arrays.fill(color, -1);

        // Sortiere Variablen nach Grad (hoehere Grade zuerst)
        Integer[] order = new Integer[numVars];
        for (int i = 0; i < numVars; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> degree(b) - degree(a));

        for (int v : order) {
            Set<Integer> forbidden = new HashSet<>();
            for (int u = 0; u < numVars; u++) {
                if (interference[v][u] && color[u] >= 0) {
                    forbidden.add(color[u]);
                }
            }
            boolean assigned = false;
            for (int c = 0; c < maxSlots; c++) {
                if (!forbidden.contains(c)) {
                    color[v] = c;
                    assigned = true;
                    break;
                }
            }
            if (!assigned && !allowSpill) {
                throw new IllegalStateException(
                    "Keine freien Slots fuer Variable " + v);
            }
            // allowSpill: color[v] bleibt -1
        }
        return color;
    }

    private int degree(int v) {
        int d = 0;
        for (int u = 0; u < numVars; u++) if (interference[v][u]) d++;
        return d;
    }

    private boolean[][] buildClique(int k) {
        boolean[][] c = new boolean[k][k];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
                if (i != j) c[i][j] = true;
        return c;
    }
}
