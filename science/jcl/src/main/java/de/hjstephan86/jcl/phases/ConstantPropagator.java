package de.hjstephan86.jcl.phases;

import java.util.*;

/**
 * Konstantenpropagation in SSA-Form (CPP).
 * Iterativer Algorithmus; konvergiert in O(m) Iterationen.
 *
 * @author Stephan Epp
 */
public class ConstantPropagator {

    public record SSAVar(int id, String name, Object value) {
        public boolean isConstant() { return value != null; }
    }

    private final List<SSAVar>     vars;
    private final boolean[][]      defUse;    // defUse[i][j]: i definiert j
    private final Object[]         constants; // constants[i]: Wert oder null

    public ConstantPropagator(List<SSAVar> vars, boolean[][] defUse) {
        this.vars      = vars;
        this.defUse    = defUse;
        this.constants = new Object[vars.size()];
        for (SSAVar v : vars) {
            constants[v.id()] = v.value();
        }
    }

    // ----------------------------------------------------------------
    // Propagation
    // ----------------------------------------------------------------

    /**
     * Propagiert Konstanten iterativ bis Fixpunkt.
     *
     * @return Anzahl propagierter Konstanten
     */
    public int propagate() {
        boolean changed = true;
        int iterations  = 0;
        int propagated  = 0;

        while (changed) {
            changed = false;
            iterations++;
            for (int i = 0; i < vars.size(); i++) {
                if (constants[i] != null) continue; // schon konstant
                Object val = evalDependencies(i);
                if (val != null) {
                    constants[i] = val;
                    changed = true;
                    propagated++;
                }
            }
        }
        System.out.printf("[CPP] %d Konstanten propagiert in %d Iterationen.%n",
                          propagated, iterations);
        return propagated;
    }

    /** Gibt den propagierten Wert fuer Variable i zurueck, oder null. */
    public Object getConstant(int i) { return constants[i]; }

    /** Gibt Map aller bekannten Konstanten zurueck. */
    public Map<String, Object> getAllConstants() {
        Map<String, Object> result = new LinkedHashMap<>();
        for (SSAVar v : vars) {
            if (constants[v.id()] != null) {
                result.put(v.name(), constants[v.id()]);
            }
        }
        return result;
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    /**
     * Bewertet alle Definitionsquellen von Variable i.
     * Gibt ihren gemeinsamen Wert zurueck, oder null.
     */
    private Object evalDependencies(int i) {
        Object candidate = null;
        for (int j = 0; j < vars.size(); j++) {
            if (defUse[j][i]) {
                Object c = constants[j];
                if (c == null) return null;      // Quelle nicht konstant
                if (candidate == null) {
                    candidate = c;
                } else if (!candidate.equals(c)) {
                    return null;                 // Widerspruch
                }
            }
        }
        return candidate;
    }
}
