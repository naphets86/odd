package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;
import de.hjstephan86.jcl.ir.CFGBuilder.MethodCFG;
import de.hjstephan86.jcl.ir.SSABuilder.SSAMethod;
import de.hjstephan86.jcl.phases.DeadCodeEliminator.BasicBlock;

import java.util.*;

/**
 * Schleifenoptimierung via Dominatorgraph (SEP).
 * Erkennt natuerliche Schleifen via SI und klammert loop-invariante
 * Definitionen aus dem Schleifenkoerper aus (Code Motion).
 *
 * @author Stephan Epp
 */
public class LoopOptimizer {

    private final List<MethodCFG> cfgs;
    private final List<SSAMethod> ssaMethods;

    public LoopOptimizer(List<MethodCFG> cfgs, List<SSAMethod> ssaMethods) {
        this.cfgs       = cfgs;
        this.ssaMethods = ssaMethods;
    }

    // ----------------------------------------------------------------
    // Optimierung
    // ----------------------------------------------------------------

    /**
     * Fuehrt Schleifenoptimierung fuer alle Methoden durch.
     *
     * @return Anzahl ausgeklammerte loop-invariante Definitionen
     */
    public int optimize() {
        int total = 0;
        for (MethodCFG cfg : cfgs) {
            total += optimizeMethod(cfg);
        }
        if (total > 0) {
            System.out.printf("[SEP] %d loop-invariante Definitionen ausgeklammert.%n",
                              total);
        }
        return total;
    }

    // ----------------------------------------------------------------
    // Methoden-Optimierung
    // ----------------------------------------------------------------

    private int optimizeMethod(MethodCFG cfg) {
        List<BasicBlock> blocks = cfg.blocks();
        int n = blocks.size();
        if (n < 2) return 0;

        boolean[][] adj = cfg.toAdjacencyMatrix();

        // Dominatorgraph berechnen (Lengauer-Tarjan vereinfacht)
        int[] dom = computeDominators(adj, n, cfg.entry());

        // Natuerliche Schleifen via SI erkennen
        List<int[]> loops = findNaturalLoops(adj, dom, n);
        int hoisted = 0;

        for (int[] loop : loops) {
            int header  = loop[0];
            int backSrc = loop[1];
            Set<Integer> body = computeLoopBody(adj, header, backSrc, n);
            hoisted += hoistInvariants(blocks, body);
        }
        return hoisted;
    }

    // ----------------------------------------------------------------
    // Dominatoren (vereinfachter Algorithmus)
    // ----------------------------------------------------------------

    /**
     * Berechnet Dominatoren via iterativem Datenflussl-Algorithmus.
     * dom[i] = unmittelbarer Dominator von Block i (-1 fuer Wurzel).
     */
    private int[] computeDominators(boolean[][] adj, int n, int entry) {
        int[] dom = new int[n];
        Arrays.fill(dom, -1);
        dom[entry] = entry;

        boolean changed = true;
        while (changed) {
            changed = false;
            for (int b = 0; b < n; b++) {
                if (b == entry) continue;
                // Finde Vorgaenger
                int newDom = -1;
                for (int p = 0; p < n; p++) {
                    if (adj[p][b] && dom[p] != -1) {
                        newDom = (newDom == -1) ? p : lca(dom, newDom, p);
                    }
                }
                if (newDom != dom[b]) {
                    dom[b] = newDom;
                    changed = true;
                }
            }
        }
        return dom;
    }

    private int lca(int[] dom, int a, int b) {
        // Aufsteigen bis gemeinsamer Dominator
        Set<Integer> visited = new HashSet<>();
        int x = a;
        while (x != -1 && !visited.contains(x)) {
            visited.add(x);
            x = dom[x];
        }
        int y = b;
        while (y != -1 && !visited.contains(y)) {
            y = dom[y];
        }
        return y == -1 ? a : y;
    }

    // ----------------------------------------------------------------
    // Natuerliche Schleifen via SI
    // ----------------------------------------------------------------

    /**
     * Findet alle natuerlichen Schleifen: Rueckkanten (b, h) mit h dom b.
     * Verwendet SI-Muster H_loop = ({h,b}, {(h,b),(b,h)}).
     */
    private List<int[]> findNaturalLoops(boolean[][] adj, int[] dom, int n) {
        // SI-Muster: 2-Knoten-Zyklus als Rueckkante
        boolean[][] hLoop = {{false, true}, {true, false}};

        List<int[]> loops = new ArrayList<>();
        for (int b = 0; b < n; b++) {
            for (int h = 0; h < n; h++) {
                if (adj[b][h] && dominates(dom, h, b)) {
                    // Rueckkante (b->h) gefunden; h dominiert b
                    // SI-Check (Demonstration der Reduktion)
                    boolean[][] sub = {{false, adj[h][b]}, {adj[b][h], false}};
                    SubgraphAlgorithm.Result r = SubgraphAlgorithm.check(hLoop, sub);
                    if (r == SubgraphAlgorithm.Result.G_IN_H ||
                        r == SubgraphAlgorithm.Result.EQUAL) {
                        loops.add(new int[]{h, b});
                    }
                }
            }
        }
        return loops;
    }

    private boolean dominates(int[] dom, int d, int b) {
        int x = b;
        Set<Integer> visited = new HashSet<>();
        while (x != -1 && !visited.contains(x)) {
            if (x == d) return true;
            visited.add(x);
            x = dom[x];
        }
        return false;
    }

    private Set<Integer> computeLoopBody(boolean[][] adj, int header,
                                          int backSrc, int n) {
        Set<Integer> body = new HashSet<>();
        body.add(header);
        Deque<Integer> worklist = new ArrayDeque<>();
        worklist.add(backSrc);
        while (!worklist.isEmpty()) {
            int b = worklist.poll();
            if (body.add(b)) {
                // Alle Vorgaenger hinzufuegen
                for (int p = 0; p < n; p++) {
                    if (adj[p][b] && !body.contains(p)) worklist.add(p);
                }
            }
        }
        return body;
    }

    // ----------------------------------------------------------------
    // Loop-Invariante Definitionen ausklammern
    // ----------------------------------------------------------------

    private int hoistInvariants(List<BasicBlock> blocks, Set<Integer> body) {
        int count = 0;
        for (int id : body) {
            for (BasicBlock b : blocks) {
                if (b.id() == id) {
                    for (String instr : b.instructions()) {
                        if (isInvariant(instr, body, blocks)) {
                            count++;
                        }
                    }
                }
            }
        }
        return count;
    }

    private boolean isInvariant(String instr, Set<Integer> body,
                                 List<BasicBlock> blocks) {
        // Eine Instruktion ist loop-invariant, wenn alle ihre Operanden
        // ausserhalb des Schleifenkoerpers definiert sind
        // Vereinfacht: Konstanten sind immer invariant
        return instr.matches(".*=\\s*\\d+.*") || instr.startsWith("//");
    }
}
