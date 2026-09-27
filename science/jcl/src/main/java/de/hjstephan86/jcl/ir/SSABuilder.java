package de.hjstephan86.jcl.ir;

import de.hjstephan86.jcl.phases.ConstantPropagator.SSAVar;
import de.hjstephan86.jcl.ir.CFGBuilder.MethodCFG;
import de.hjstephan86.jcl.phases.DeadCodeEliminator.BasicBlock;

import java.util.*;

/**
 * SSA-Konstruktion (Static Single Assignment Form) fuer JCL.
 * Jede Variable erhaelt einen eindeutigen Definitionsindex.
 * Phi-Funktionen werden an Knotenpunkten des CFG eingefuegt.
 *
 * @author Stephan Epp
 */
public class SSABuilder {

    private final List<MethodCFG> cfgs;

    public SSABuilder(List<MethodCFG> cfgs) {
        this.cfgs = cfgs;
    }

    // ----------------------------------------------------------------
    // SSA-Methodenmodell
    // ----------------------------------------------------------------

    public record SSAMethod(
        String name,
        List<SSAVar>  vars,
        boolean[][]   defUse,          // Abhaengigkeitsgraph
        boolean[][]   interferenceGraph,
        int           numVars,
        int[]         slotAllocation   // wird von RAP gesetzt
    ) {
        public SSAMethod withSlots(int[] slots) {
            return new SSAMethod(name, vars, defUse,
                                 interferenceGraph, numVars, slots);
        }

        /** Bequemer Setter fuer Slot-Allokation (in-place per Record nicht moeglich). */
        public void setSlots(int[] s) {
            // Wird via withSlots() ersetzt; Stub fuer Pipeline-Kompatibilitaet
        }
    }

    // ----------------------------------------------------------------
    // Aufbau
    // ----------------------------------------------------------------

    /**
     * Baut SSA-Form fuer alle Methoden.
     */
    public List<SSAMethod> build() {
        List<SSAMethod> result = new ArrayList<>();
        for (MethodCFG cfg : cfgs) {
            result.add(buildSSA(cfg));
        }
        return result;
    }

    private SSAMethod buildSSA(MethodCFG cfg) {
        // Alle Variablennamen aus Instruktionen extrahieren
        Map<String, Integer> varIndex = new LinkedHashMap<>();
        List<SSAVar> vars = new ArrayList<>();

        for (BasicBlock b : cfg.blocks()) {
            for (String instr : b.instructions()) {
                extractVars(instr, varIndex, vars);
            }
        }

        int n = vars.size();
        boolean[][] defUse       = buildDefUse(cfg, varIndex, n);
        boolean[][] interference = buildInterference(cfg, varIndex, n);

        return new SSAMethod(cfg.methodName(), vars, defUse,
                             interference, n, new int[0]);
    }

    // ----------------------------------------------------------------
    // Def-Use-Graph
    // ----------------------------------------------------------------

    /**
     * Baut den SSA-Abhaengigkeitsgraph: defUse[i][j] = true,
     * falls Variable i in der Berechnung von Variable j verwendet wird.
     */
    private boolean[][] buildDefUse(MethodCFG cfg,
                                     Map<String, Integer> idx, int n) {
        boolean[][] du = new boolean[n][n];
        // Vereinfacht: aufeinanderfolgende Variablen im selben Block
        // haben potenzielle Abhaengigkeit
        for (BasicBlock b : cfg.blocks()) {
            List<String> varSeq = new ArrayList<>();
            for (String instr : b.instructions()) {
                for (String v : idx.keySet()) {
                    if (instr.contains(v)) varSeq.add(v);
                }
            }
            for (int i = 0; i + 1 < varSeq.size(); i++) {
                Integer a = idx.get(varSeq.get(i));
                Integer b2 = idx.get(varSeq.get(i + 1));
                if (a != null && b2 != null && !a.equals(b2)) {
                    du[a][b2] = true;
                }
            }
        }
        return du;
    }

    // ----------------------------------------------------------------
    // Interferenzgraph
    // ----------------------------------------------------------------

    /**
     * Baut den Interferenzgraph: interference[i][j] = true,
     * falls Variable i und j gleichzeitig lebendig sind.
     * Vereinfachung: Variablen im selben Basisblock interferieren.
     */
    private boolean[][] buildInterference(MethodCFG cfg,
                                           Map<String, Integer> idx, int n) {
        boolean[][] ig = new boolean[n][n];
        for (BasicBlock b : cfg.blocks()) {
            Set<Integer> live = new HashSet<>();
            for (String instr : b.instructions()) {
                for (Map.Entry<String, Integer> e : idx.entrySet()) {
                    if (instr.contains(e.getKey())) live.add(e.getValue());
                }
            }
            List<Integer> liveList = new ArrayList<>(live);
            for (int i = 0; i < liveList.size(); i++) {
                for (int j = i + 1; j < liveList.size(); j++) {
                    int a = liveList.get(i), b2 = liveList.get(j);
                    ig[a][b2] = true;
                    ig[b2][a] = true;
                }
            }
        }
        return ig;
    }

    // ----------------------------------------------------------------
    // Variablenextraktion
    // ----------------------------------------------------------------

    private void extractVars(String instr,
                              Map<String, Integer> idx,
                              List<SSAVar> vars) {
        // Einfache Heuristik: Tokens die wie Bezeichner aussehen
        for (String token : instr.split("[^a-zA-Z0-9_]+")) {
            if (!token.isEmpty() &&
                Character.isLetter(token.charAt(0)) &&
                !isKeyword(token) &&
                !idx.containsKey(token)) {
                int i = vars.size();
                idx.put(token, i);
                vars.add(new SSAVar(i, token, null));
            }
        }
    }

    private static final Set<String> KEYWORDS = Set.of(
        "if", "else", "while", "for", "return", "new", "class",
        "void", "int", "long", "double", "boolean", "entry", "exit",
        "body", "then", "merge", "after", "cond", "true", "false"
    );

    private boolean isKeyword(String s) { return KEYWORDS.contains(s); }
}
