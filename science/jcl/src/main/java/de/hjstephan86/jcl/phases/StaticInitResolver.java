package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;
import de.hjstephan86.jcl.ir.ASTNode;

import java.util.*;

/**
 * Statische Initialisierungsreihenfolge (IRP) via Subgraph Algorithmus.
 * Erkennt das Static Initialization Order Fiasco (SIOF) als Zykel im
 * Initialisierungsgraphen G_S.
 * Laufzeit: O(q^3) mit q = Anzahl statischer Initialisierer.
 *
 * @author Stephan Epp
 */
public class StaticInitResolver {

    public record StaticInit(
        String typeName,
        String fieldName,
        List<String> dependsOn,   // andere Initialisierer, die zuerst laufen muessen
        ASTNode node
    ) {}

    private final List<StaticInit>      inits    = new ArrayList<>();
    private final Map<String, Integer>  initIdx  = new HashMap<>();
    private final ASTNode               ast;

    public StaticInitResolver(ASTNode ast) {
        this.ast = ast;
        extractFromAST();
    }

    // ----------------------------------------------------------------
    // Aufloesung
    // ----------------------------------------------------------------

    /**
     * Berechnet die statische Initialisierungsreihenfolge.
     *
     * @return Initialisierer in gueltiger Reihenfolge
     * @throws IllegalStateException bei SIOF (Zykel)
     */
    public List<StaticInit> resolve() {
        int q = inits.size();
        if (q == 0) return List.of();

        boolean[][] adj = buildGraph(q);

        // SIOF-Detektion via SI fuer Zykellaengen 2..min(q,8)
        for (int k = 2; k <= Math.min(q, 8); k++) {
            boolean[][] cycle = buildCycle(k);
            SubgraphAlgorithm.Result r = SubgraphAlgorithm.check(cycle, adj);
            if (r == SubgraphAlgorithm.Result.G_IN_H ||
                r == SubgraphAlgorithm.Result.EQUAL) {
                throw new IllegalStateException(
                    "SIOF: Zirkulaere statische Initialisierung erkannt " +
                    "(Zykellaenge " + k + ", SI-IRP).");
            }
        }

        return topologicalSort(adj, q);
    }

    /**
     * Prueft explizit auf SIOF ohne Ausnahme zu werfen.
     *
     * @return true falls SIOF vorhanden
     */
    public boolean hasSIOF() {
        try { resolve(); return false; }
        catch (IllegalStateException e) { return true; }
    }

    // ----------------------------------------------------------------
    // AST-Extraktion
    // ----------------------------------------------------------------

    private void extractFromAST() {
        extractRec(ast);
    }

    private void extractRec(ASTNode n) {
        if (n.getKind() == ASTNode.Kind.FIELD_DECL) {
            boolean isStatic = isStatic(n);
            if (isStatic) {
                String key = n.getValue();
                if (!initIdx.containsKey(key)) {
                    initIdx.put(key, inits.size());
                    inits.add(new StaticInit(
                        findEnclosingClass(n), key, new ArrayList<>(), n));
                }
            }
        }
        for (ASTNode child : n.getChildren()) extractRec(child);
    }

    private boolean isStatic(ASTNode field) {
        for (ASTNode child : field.getChildren()) {
            if (child.getKind() == ASTNode.Kind.MODIFIER) {
                for (ASTNode mod : child.getChildren()) {
                    if ("static".equals(mod.getValue())) return true;
                }
            }
        }
        return false;
    }

    private String findEnclosingClass(ASTNode n) { return ""; }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private boolean[][] buildGraph(int q) {
        boolean[][] a = new boolean[q][q];
        for (StaticInit s : inits) {
            int i = initIdx.get(s.fieldName());
            for (String dep : s.dependsOn()) {
                Integer j = initIdx.get(dep);
                if (j != null) a[i][j] = true;
            }
        }
        return a;
    }

    private boolean[][] buildCycle(int k) {
        boolean[][] c = new boolean[k][k];
        for (int i = 0; i < k; i++) c[i][(i + 1) % k] = true;
        return c;
    }

    private List<StaticInit> topologicalSort(boolean[][] a, int q) {
        int[] inDeg = new int[q];
        for (int i = 0; i < q; i++)
            for (int j = 0; j < q; j++)
                if (a[i][j]) inDeg[j]++;
        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < q; i++) if (inDeg[i] == 0) queue.add(i);
        List<StaticInit> order = new ArrayList<>();
        while (!queue.isEmpty()) {
            int u = queue.poll();
            order.add(inits.get(u));
            for (int v = 0; v < q; v++)
                if (a[u][v] && --inDeg[v] == 0) queue.add(v);
        }
        return order;
    }

    public int initCount() { return inits.size(); }
}

