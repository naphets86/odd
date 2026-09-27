package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;
import de.hjstephan86.jcl.ir.ASTNode;

import java.util.*;

/**
 * Modulabhaengigkeitsaufloesung (LAP) via Subgraph Algorithmus.
 * Unterstuetzt das Java Platform Module System (JPMS, Java 9+).
 * Laufzeit: O(p^3) mit p = Anzahl Module.
 *
 * @author Stephan Epp
 */
public class ModuleResolver {

    public record JavaModule(
        String name,
        List<String> requires,       // requires-Deklarationen
        List<String> exports,        // exports-Pakete
        List<String> opens,          // opens-Pakete
        boolean isAutomatic
    ) {}

    private final List<JavaModule>       modules   = new ArrayList<>();
    private final Map<String, Integer>   moduleIdx = new HashMap<>();
    private final ASTNode                ast;

    public ModuleResolver(ASTNode ast) {
        this.ast = ast;
    }

    // ----------------------------------------------------------------
    // Modulregistrierung
    // ----------------------------------------------------------------

    public void addModule(JavaModule m) {
        if (!moduleIdx.containsKey(m.name())) {
            moduleIdx.put(m.name(), modules.size());
            modules.add(m);
        }
    }

    // ----------------------------------------------------------------
    // Aufloesung
    // ----------------------------------------------------------------

    /**
     * Loest Modulabhaengigkeiten auf (LAP).
     *
     * @return Kompilierungsreihenfolge (topologisch sortiert)
     * @throws IllegalStateException bei zirkulaeren Abhaengigkeiten
     */
    public List<JavaModule> resolve() {
        int p = modules.size();
        if (p == 0) return List.of();

        boolean[][] adj = buildModuleGraph(p);

        // Zykeldetektion via SI
        detectCycles(adj, p);

        // Topologische Sortierung
        return topologicalSort(adj, p);
    }

    /**
     * Prueft ob ein Muster-Abhaengigkeitsgraph H als Subgraph
     * im Modulabhaengigkeitsgraph G_M vorkommt (LAP Teilproblem 3).
     *
     * @param pattern Adjazenzmatrix des gesuchten Musters
     * @return true falls Muster enthalten
     */
    public boolean containsPattern(boolean[][] pattern) {
        int p = modules.size();
        boolean[][] adj = buildModuleGraph(p);
        SubgraphAlgorithm.Result r = SubgraphAlgorithm.check(pattern, adj);
        return r == SubgraphAlgorithm.Result.G_IN_H ||
               r == SubgraphAlgorithm.Result.EQUAL;
    }

    /**
     * Gibt die Modulauflösungstiefe zurueck (laengster Abhaengigkeitspfad).
     */
    public int maxDepth() {
        int p = modules.size();
        boolean[][] reach = transitiveClosure(buildModuleGraph(p), p);
        int max = 0;
        for (int i = 0; i < p; i++) {
            int depth = 0;
            for (int j = 0; j < p; j++) if (reach[i][j]) depth++;
            max = Math.max(max, depth);
        }
        return max;
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private void detectCycles(boolean[][] adj, int p) {
        // Zykel k=2: suche gerichteten 2-Zykel als SI-Muster
        for (int k = 2; k <= Math.min(p, 6); k++) {
            boolean[][] cycle = buildCycle(k);
            SubgraphAlgorithm.Result r = SubgraphAlgorithm.check(cycle, adj);
            if (r == SubgraphAlgorithm.Result.G_IN_H ||
                r == SubgraphAlgorithm.Result.EQUAL) {
                throw new IllegalStateException(
                    "Zirkulaere Modulabhaengigkeit (Laenge " + k + ") erkannt (SI-LAP).");
            }
        }
    }

    private boolean[][] buildModuleGraph(int p) {
        boolean[][] a = new boolean[p][p];
        for (JavaModule m : modules) {
            int i = moduleIdx.get(m.name());
            for (String req : m.requires()) {
                Integer j = moduleIdx.get(req);
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

    private boolean[][] transitiveClosure(boolean[][] a, int n) {
        boolean[][] r = new boolean[n][n];
        for (int i = 0; i < n; i++) r[i] = a[i].clone();
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (r[i][k] && r[k][j]) r[i][j] = true;
        return r;
    }

    private List<JavaModule> topologicalSort(boolean[][] a, int p) {
        int[] inDeg = new int[p];
        for (int i = 0; i < p; i++)
            for (int j = 0; j < p; j++)
                if (a[i][j]) inDeg[j]++;
        Queue<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < p; i++) if (inDeg[i] == 0) q.add(i);
        List<JavaModule> order = new ArrayList<>();
        while (!q.isEmpty()) {
            int u = q.poll();
            order.add(modules.get(u));
            for (int v = 0; v < p; v++)
                if (a[u][v] && --inDeg[v] == 0) q.add(v);
        }
        return order;
    }
}

