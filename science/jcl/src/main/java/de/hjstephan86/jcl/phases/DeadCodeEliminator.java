package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;

import java.util.*;

/**
 * Dead-Code-Elimination (DCE) via Erreichbarkeitsanalyse im CFG.
 * Laufzeit: O(|B|^3) mit |B| = Anzahl Basisblöcke.
 *
 * @author Stephan Epp
 */
public class DeadCodeEliminator {

    // ----------------------------------------------------------------
    // Datenmodell Basisblock
    // ----------------------------------------------------------------

    public record BasicBlock(
        int id,
        List<String> instructions,
        List<Integer> successors
    ) {}

    private final List<BasicBlock> blocks;
    private final int entry;

    public DeadCodeEliminator(List<BasicBlock> blocks, int entry) {
        this.blocks = new ArrayList<>(blocks);
        this.entry  = entry;
    }

    // ----------------------------------------------------------------
    // DCE-Analyse
    // ----------------------------------------------------------------

    /**
     * Identifiziert und entfernt tote Basisblöcke.
     *
     * @return Liste lebendiger Blöcke in originaler Reihenfolge
     */
    public List<BasicBlock> eliminate() {
        int n = blocks.size();
        if (n == 0) return List.of();

        boolean[][] adj   = buildCFG(n);
        boolean[][] reach = transitiveClosure(adj, n);

        List<BasicBlock> live = new ArrayList<>();
        Set<Integer>     dead = new LinkedHashSet<>();

        for (int b = 0; b < n; b++) {
            if (b == entry || reach[entry][b]) {
                live.add(blocks.get(b));
            } else {
                dead.add(b);
            }
        }

        if (!dead.isEmpty()) {
            System.out.printf("[DCE] %d tote Basisblöcke eliminiert: %s%n",
                              dead.size(), dead);
        }
        return live;
    }

    /**
     * Zaehlt die Anzahl toter Bloecke ohne sie zu entfernen.
     */
    public int countDeadBlocks() {
        int n     = blocks.size();
        boolean[][] reach = transitiveClosure(buildCFG(n), n);
        int count = 0;
        for (int b = 0; b < n; b++) {
            if (b != entry && !reach[entry][b]) count++;
        }
        return count;
    }

    /**
     * Gibt eine Erreichbarkeits-Matrix zurueck:
     * reach[i][j] == true <==> Block j ist von Block i erreichbar.
     */
    public boolean[][] reachabilityMatrix() {
        return transitiveClosure(buildCFG(blocks.size()), blocks.size());
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private boolean[][] buildCFG(int n) {
        boolean[][] a = new boolean[n][n];
        for (BasicBlock b : blocks) {
            for (int s : b.successors()) {
                if (s >= 0 && s < n) a[b.id()][s] = true;
            }
        }
        return a;
    }

    /** Floyd-Warshall transitiver Abschluss in O(n^3). */
    private boolean[][] transitiveClosure(boolean[][] a, int n) {
        boolean[][] r = new boolean[n][n];
        for (int i = 0; i < n; i++) r[i] = a[i].clone();
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (r[i][k])
                    for (int j = 0; j < n; j++)
                        if (r[k][j]) r[i][j] = true;
        return r;
    }
}
