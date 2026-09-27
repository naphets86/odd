package de.hjstephan86.jcl.core;

import java.util.Arrays;

/**
 * Subgraph Algorithmus nach Epp (2026).
 *
 * Bestimmt fuer zwei Graphen G und H mit je n Knoten in Zeit O(n^3),
 * ob G Subgraph von H ist, H Subgraph von G, beide gleich oder keiner
 * im anderen enthalten ist.
 *
 * Grundlage: Spaltensignaturen sigma_j = sum_i A[i][j] * 2^i (injektiv
 * fuer n <= 60) und zyklische Rotation der sortierten Signatur-Arrays.
 *
 * @author Stephan Epp
 * @version 1.0 (2026)
 */
public final class SubgraphAlgorithm {

    // ---------------------------------------------------------------
    // Ergebnis-Enum
    // ---------------------------------------------------------------

    /**
     * Moegliche Ergebnisse eines Subgraph-Vergleichs.
     */
    public enum Result {
        /** G und H sind isomorph (strukturell gleich). */
        EQUAL,
        /** G ist Subgraph von H. */
        G_IN_H,
        /** H ist Subgraph von G. */
        H_IN_G,
        /** Keiner ist Subgraph des anderen. */
        NONE
    }

    private SubgraphAlgorithm() {}

    // ---------------------------------------------------------------
    // Oeffentliche API
    // ---------------------------------------------------------------

    /**
     * Prueft die Subgraph-Relation zwischen G (adjG) und H (adjH).
     *
     * @param adjG Adjazenzmatrix von G (n x n, boolean)
     * @param adjH Adjazenzmatrix von H (m x m, boolean)
     * @return {@link Result}
     * @throws IllegalArgumentException wenn Matrizen nicht quadratisch sind
     */
    public static Result check(boolean[][] adjG, boolean[][] adjH) {
        validate(adjG, "G");
        validate(adjH, "H");

        int n = adjG.length;
        int m = adjH.length;

        long[] sigG = columnSignatures(adjG, n);
        long[] sigH = columnSignatures(adjH, m);

        boolean gInH = (n <= m) && isSubgraphViaRotation(sigG, sigH, n, m);
        boolean hInG = (m <= n) && isSubgraphViaRotation(sigH, sigG, m, n);

        if (gInH && hInG) return Result.EQUAL;
        if (gInH)         return Result.G_IN_H;
        if (hInG)         return Result.H_IN_G;
        return Result.NONE;
    }

    /**
     * Berechnet eine explizite Einbettung f: V(G) -> V(H) via Backtracking.
     *
     * @param adjG Adjazenzmatrix von G
     * @param adjH Adjazenzmatrix von H
     * @return Einbettungsarray f[] mit f[i] = Bildknoten in H,
     *         oder null falls kein Subgraph-Isomorphismus existiert
     */
    public static int[] embed(boolean[][] adjG, boolean[][] adjH) {
        int n = adjG.length;
        int m = adjH.length;
        if (n > m) return null;

        int[] f     = new int[n];
        Arrays.fill(f, -1);
        boolean[] used = new boolean[m];

        return backtrack(adjG, adjH, f, used, 0, n, m) ? f : null;
    }

    /**
     * Berechnet den SI-Aehnlichkeitsscore im Intervall [0, 1].
     * 1.0 = identisch, 0.0 = kein gemeinsames Teilmuster.
     *
     * @param adjG Adjazenzmatrix von G
     * @param adjH Adjazenzmatrix von H
     * @return Aehnlichkeitsscore in [0, 1]
     */
    public static double similarityScore(boolean[][] adjG, boolean[][] adjH) {
        Result r = check(adjG, adjH);
        int n = adjG.length;
        int m = adjH.length;
        return switch (r) {
            case EQUAL  -> 1.0;
            case G_IN_H -> (double) n / m;
            case H_IN_G -> (double) m / n;
            case NONE   -> 0.0;
        };
    }

    // ---------------------------------------------------------------
    // Interne Hilfsmethoden
    // ---------------------------------------------------------------

    /**
     * Berechnet Spaltensignaturen: sigma_j = sum_i A[i][j] * 2^i.
     * Injektiv fuer n <= 60 (passt in long).
     */
    static long[] columnSignatures(boolean[][] adj, int n) {
        long[] sig = new long[n];
        for (int j = 0; j < n; j++) {
            long s = 0L;
            for (int i = 0; i < n; i++) {
                if (adj[i][j]) s |= (1L << i);
            }
            sig[j] = s;
        }
        return sig;
    }

    /**
     * Prueft via zyklischer Rotation und LCS-Matching, ob das Signatur-
     * Multiset von sigSmall als Teilmenge in sigLarge eingebettet werden kann.
     *
     * Laufzeit: O(small * large) pro Rotation, O(large) Rotationen
     * => O(small * large^2) = O(n^3) insgesamt.
     */
    static boolean isSubgraphViaRotation(long[] sigSmall, long[] sigLarge,
                                          int small, int large) {
        long[] sortedSmall = sigSmall.clone();
        long[] sortedLarge = sigLarge.clone();
        Arrays.sort(sortedSmall);
        Arrays.sort(sortedLarge);

        for (int r = 0; r < large; r++) {
            int si = 0;
            for (int li = r; li < r + large && si < small; li++) {
                if (sortedLarge[li % large] == sortedSmall[si]) {
                    si++;
                }
            }
            if (si == small) return true;
        }
        return false;
    }

    /** Rekursives Backtracking fuer die explizite Einbettung. */
    private static boolean backtrack(boolean[][] adjG, boolean[][] adjH,
                                      int[] f, boolean[] used,
                                      int k, int n, int m) {
        if (k == n) return true;
        for (int v = 0; v < m; v++) {
            if (!used[v] && consistent(adjG, adjH, f, k, v)) {
                f[k] = v;
                used[v] = true;
                if (backtrack(adjG, adjH, f, used, k + 1, n, m)) return true;
                f[k] = -1;
                used[v] = false;
            }
        }
        return false;
    }

    /** Prueft Kantenerhaltung fuer Backtracking-Schritt k -> v. */
    private static boolean consistent(boolean[][] adjG, boolean[][] adjH,
                                       int[] f, int k, int v) {
        for (int i = 0; i < k; i++) {
            if (f[i] < 0) continue;
            if (adjG[i][k] && !adjH[f[i]][v]) return false;
            if (adjG[k][i] && !adjH[v][f[i]]) return false;
        }
        return true;
    }

    /** Validiert quadratische Adjazenzmatrix. */
    private static void validate(boolean[][] adj, String name) {
        if (adj == null) throw new IllegalArgumentException(name + " ist null.");
        for (int i = 0; i < adj.length; i++) {
            if (adj[i].length != adj.length) {
                throw new IllegalArgumentException(
                    name + ": Zeile " + i + " hat falsche Laenge.");
            }
        }
    }
}
