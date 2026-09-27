package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;
import de.hjstephan86.jcl.ir.ASTNode;

import java.util.*;

/**
 * Typaufloesung und Vererbungshierarchie (TAP) via Subgraph Algorithmus.
 * Laufzeit: O(|T|^3) mit |T| = Anzahl Typen.
 *
 * @author Stephan Epp
 */
public class TypeResolver {

    // ----------------------------------------------------------------
    // Interne Typdefinition
    // ----------------------------------------------------------------

    public record JavaType(
        String name,
        String superClass,             // null fuer Object-Wurzel
        List<String> interfaces,
        ASTNode declaration
    ) {}

    private final List<JavaType>         types     = new ArrayList<>();
    private final Map<String, Integer>   typeIndex = new HashMap<>();
    private final Map<String, JavaType>  typeMap   = new HashMap<>();

    // ----------------------------------------------------------------
    // Registrierung
    // ----------------------------------------------------------------

    public void register(JavaType t) {
        if (!typeMap.containsKey(t.name())) {
            typeIndex.put(t.name(), types.size());
            types.add(t);
            typeMap.put(t.name(), t);
        }
    }

    public void registerFromAST(ASTNode cu) {
        for (ASTNode child : cu.getChildren()) {
            switch (child.getKind()) {
                case CLASS_DECL, INTERFACE_DECL, ENUM_DECL, RECORD_DECL ->
                    extractType(child);
                default -> {}
            }
        }
    }

    private void extractType(ASTNode n) {
        String name = n.getValue();
        register(new JavaType(name, "Object", List.of(), n));
    }

    // ----------------------------------------------------------------
    // Aufloesung via SI
    // ----------------------------------------------------------------

    /**
     * Loest die Vererbungshierarchie auf.
     * Wirft IllegalStateException bei Zykeln (via SI-Detektion).
     *
     * @return Typen in topologischer Reihenfolge (Supertypen zuerst)
     */
    public List<JavaType> resolve() {
        int n = types.size();
        if (n == 0) return List.of();

        boolean[][] adj = buildAdjacency(n);

        // Zykeldetektion: suche K_2 (ungerichtete Kante) als Muster
        boolean[][] sym = symmetrize(adj, n);
        if (n >= 2) {
            boolean[][] k2 = {{false, true}, {true, false}};
            SubgraphAlgorithm.Result r = SubgraphAlgorithm.check(k2, sym);
            if (r == SubgraphAlgorithm.Result.G_IN_H ||
                r == SubgraphAlgorithm.Result.EQUAL) {
                throw new IllegalStateException(
                    "Zirkulaere Vererbung im Typgraphen erkannt (SI-TAP).");
            }
        }

        return topologicalSort(adj, n);
    }

    /**
     * Prueft, ob Typ {@code sub} ein Untertyp von {@code sup} ist.
     * Verwendet Subgraph-Einbettung im Vererbungsgraphen.
     */
    public boolean isSubtype(String sub, String sup) {
        Integer si = typeIndex.get(sub);
        Integer pi = typeIndex.get(sup);
        if (si == null || pi == null) return false;
        if (si.equals(pi)) return true;

        int n = types.size();
        boolean[][] adj = buildAdjacency(n);
        // Transitiver Abschluss
        boolean[][] reach = transitiveClosure(adj, n);
        return reach[si][pi];
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private boolean[][] buildAdjacency(int n) {
        boolean[][] a = new boolean[n][n];
        for (JavaType t : types) {
            int i = typeIndex.get(t.name());
            if (t.superClass() != null) {
                Integer j = typeIndex.get(t.superClass());
                if (j != null) a[i][j] = true;
            }
            for (String iface : t.interfaces()) {
                Integer j = typeIndex.get(iface);
                if (j != null) a[i][j] = true;
            }
        }
        return a;
    }

    private boolean[][] symmetrize(boolean[][] a, int n) {
        boolean[][] s = new boolean[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                s[i][j] = a[i][j] || a[j][i];
        return s;
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

    private List<JavaType> topologicalSort(boolean[][] a, int n) {
        int[] inDeg = new int[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (a[i][j]) inDeg[j]++;

        Queue<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) if (inDeg[i] == 0) q.add(i);

        List<JavaType> order = new ArrayList<>();
        while (!q.isEmpty()) {
            int u = q.poll();
            order.add(types.get(u));
            for (int v = 0; v < n; v++)
                if (a[u][v] && --inDeg[v] == 0) q.add(v);
        }
        return order;
    }

    public Optional<JavaType> lookup(String name) {
        return Optional.ofNullable(typeMap.get(name));
    }

    public int typeCount() { return types.size(); }
}

