package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.core.SubgraphAlgorithm;
import de.hjstephan86.jcl.ir.ASTNode;

import java.util.*;

/**
 * Bytecode-Verifikation (BVP) via Typzustandsgraph und Subgraph Algorithmus.
 * Prueft Typkonsistenz aller AST-Ausdruecke vor der Codegenerierung.
 * Laufzeit: O(r^3) mit r = Anzahl Instruktionen/Ausdruecke.
 *
 * @author Stephan Epp
 */
public class BytecodeVerifier {

    // Typzustaende (vereinfacht auf JVM-Grundtypen)
    public enum TypeState {
        INT, LONG, FLOAT, DOUBLE, REFERENCE, BOOLEAN,
        BYTE, SHORT, CHAR, VOID, TOP, UNINITIALIZED
    }

    /** Legale Typuebergaenge (von -> nach). */
    private static final Set<String> LEGAL_TRANSITIONS = Set.of(
        "INT->INT", "LONG->LONG", "FLOAT->FLOAT", "DOUBLE->DOUBLE",
        "REFERENCE->REFERENCE", "BOOLEAN->INT",
        "BYTE->INT", "SHORT->INT", "CHAR->INT",
        "INT->LONG", "INT->FLOAT", "INT->DOUBLE",
        "LONG->FLOAT", "LONG->DOUBLE", "FLOAT->DOUBLE"
    );

    private final ASTNode compilationUnit;
    private final List<String> errors = new ArrayList<>();

    public BytecodeVerifier(ASTNode compilationUnit) {
        this.compilationUnit = compilationUnit;
    }

    // ----------------------------------------------------------------
    // Verifikation
    // ----------------------------------------------------------------

    /**
     * Verifiziert alle Methoden im AST auf Typkonsistenz.
     *
     * @throws VerificationException bei Typfehlern
     */
    public void verify() {
        verifyNode(compilationUnit, TypeState.TOP);
        if (!errors.isEmpty()) {
            throw new VerificationException(errors);
        }
    }

    private TypeState verifyNode(ASTNode n, TypeState incoming) {
        return switch (n.getKind()) {
            case BINARY       -> verifyBinary(n, incoming);
            case LITERAL      -> inferLiteralType(n.getValue());
            case RETURN_STMT  -> verifyReturn(n, incoming);
            case METHOD_DECL,
                 CONSTRUCTOR_DECL -> verifyMethod(n);
            default -> {
                TypeState t = incoming;
                for (ASTNode c : n.getChildren()) t = verifyNode(c, t);
                yield t;
            }
        };
    }

    private TypeState verifyBinary(ASTNode n, TypeState incoming) {
        if (n.childCount() < 2) return incoming;
        TypeState left  = verifyNode(n.getChild(0), TypeState.TOP);
        TypeState right = verifyNode(n.getChild(1), TypeState.TOP);
        if (!isLegalTransition(left, right)) {
            errors.add(String.format(
                "BVP Typfehler Zeile %d: illegaler Uebergang %s -> %s in Operator '%s'",
                n.getLine(), left, right, n.getValue()));
            return TypeState.TOP;
        }
        return promoteType(left, right);
    }

    private TypeState verifyReturn(ASTNode n, TypeState incoming) {
        if (n.childCount() == 0) return TypeState.VOID;
        return verifyNode(n.getChild(0), incoming);
    }

    private TypeState verifyMethod(ASTNode n) {
        TypeState t = TypeState.TOP;
        for (ASTNode c : n.getChildren()) t = verifyNode(c, t);
        return t;
    }

    // ----------------------------------------------------------------
    // SI-basierte Mustersuche fuer illegale Uebergaenge
    // ----------------------------------------------------------------

    /**
     * Prueft via SI, ob ein illegaler Typuebergang als Muster im
     * Typzustandsgraph auftaucht. Demonstriert BVP <= SI.
     *
     * @param typeGraph  Adjazenzmatrix des Typzustandsgraphen
     * @param illegalAdj Adjazenzmatrix des illegalen Musters (Kantenmuster)
     * @return true falls illegaler Uebergang gefunden
     */
    public static boolean containsIllegalTransition(
            boolean[][] typeGraph, boolean[][] illegalAdj) {
        SubgraphAlgorithm.Result r =
            SubgraphAlgorithm.check(illegalAdj, typeGraph);
        return r == SubgraphAlgorithm.Result.G_IN_H ||
               r == SubgraphAlgorithm.Result.EQUAL;
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private TypeState inferLiteralType(String val) {
        if (val == null || val.isEmpty()) return TypeState.TOP;
        if (val.equals("true") || val.equals("false")) return TypeState.BOOLEAN;
        if (val.equals("null"))                         return TypeState.REFERENCE;
        if (val.endsWith("L") || val.endsWith("l"))    return TypeState.LONG;
        if (val.endsWith("F") || val.endsWith("f"))    return TypeState.FLOAT;
        if (val.endsWith("D") || val.endsWith("d"))    return TypeState.DOUBLE;
        if (val.startsWith("\""))                       return TypeState.REFERENCE;
        if (val.startsWith("'"))                        return TypeState.CHAR;
        try { Integer.parseInt(val); return TypeState.INT; } catch (NumberFormatException e) {}
        try { Double.parseDouble(val); return TypeState.DOUBLE; } catch (NumberFormatException e) {}
        return TypeState.REFERENCE;
    }

    private boolean isLegalTransition(TypeState from, TypeState to) {
        if (from == TypeState.TOP || to == TypeState.TOP) return true;
        if (from == to) return true;
        return LEGAL_TRANSITIONS.contains(from + "->" + to);
    }

    private TypeState promoteType(TypeState a, TypeState b) {
        if (a == TypeState.DOUBLE || b == TypeState.DOUBLE) return TypeState.DOUBLE;
        if (a == TypeState.FLOAT  || b == TypeState.FLOAT)  return TypeState.FLOAT;
        if (a == TypeState.LONG   || b == TypeState.LONG)   return TypeState.LONG;
        return TypeState.INT;
    }

    public List<String> getErrors() { return Collections.unmodifiableList(errors); }

    // ----------------------------------------------------------------
    // Ausnahme
    // ----------------------------------------------------------------

    public static class VerificationException extends RuntimeException {
        private final List<String> errors;
        public VerificationException(List<String> errors) {
            super("Verifikationsfehler:\n" + String.join("\n", errors));
            this.errors = errors;
        }
        public List<String> getErrors() { return errors; }
    }
}

