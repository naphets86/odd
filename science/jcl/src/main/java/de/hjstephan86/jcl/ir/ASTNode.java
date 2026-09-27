package de.hjstephan86.jcl.ir;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Knoten des Abstract Syntax Tree (AST) fuer Java-Quellcode.
 *
 * @author Stephan Epp
 */
public class ASTNode {

    public enum Kind {
        // Kompilationseinheit
        COMPILATION_UNIT, PACKAGE_DECL, IMPORT_DECL,

        // Typdeklarationen
        CLASS_DECL, INTERFACE_DECL, ENUM_DECL, RECORD_DECL,
        ANNOTATION_TYPE_DECL,

        // Member
        FIELD_DECL, METHOD_DECL, CONSTRUCTOR_DECL,
        ENUM_CONSTANT, RECORD_COMPONENT,

        // Typen
        PRIMITIVE_TYPE, ARRAY_TYPE, CLASS_TYPE, VOID_TYPE,
        TYPE_PARAM, WILDCARD,

        // Anweisungen
        BLOCK, EXPR_STMT, IF_STMT, WHILE_STMT, DO_STMT, FOR_STMT,
        ENHANCED_FOR_STMT, SWITCH_STMT, SWITCH_EXPR,
        CASE_LABEL, YIELD_STMT,
        RETURN_STMT, THROW_STMT, BREAK_STMT, CONTINUE_STMT,
        TRY_STMT, CATCH_CLAUSE, VAR_DECL_STMT,
        LABELED_STMT, EMPTY_STMT,

        // Ausdruecke
        LITERAL, NAME, FIELD_ACCESS, ARRAY_ACCESS,
        METHOD_CALL, NEW_OBJECT, NEW_ARRAY,
        UNARY, BINARY, TERNARY, ASSIGN_EXPR,
        CAST, INSTANCEOF_EXPR, LAMBDA, METHOD_REF,
        ARRAY_INIT,

        // Sonstiges
        MODIFIER, ANNOTATION, PARAM, VARARGS_PARAM,
        THROWS_CLAUSE, ERROR
    }

    private final Kind   kind;
    private final String value;      // Bezeichner / Literalwert
    private final int    line;
    private final int    col;

    private final List<ASTNode> children = new ArrayList<>();

    // Verweise fuer Semantik (werden spaeter gesetzt)
    private String  resolvedType;
    private ASTNode declaration;

    public ASTNode(Kind kind, String value, int line, int col) {
        this.kind  = kind;
        this.value = value;
        this.line  = line;
        this.col   = col;
    }

    // ----------------------------------------------------------------
    // Kindknoten
    // ----------------------------------------------------------------

    public void addChild(ASTNode child) {
        if (child != null) children.add(child);
    }

    public List<ASTNode> getChildren() {
        return Collections.unmodifiableList(children);
    }

    public ASTNode getChild(int i) { return children.get(i); }
    public int     childCount()    { return children.size(); }

    // ----------------------------------------------------------------
    // Getter
    // ----------------------------------------------------------------

    public Kind   getKind()         { return kind; }
    public String getValue()        { return value; }
    public int    getLine()         { return line; }
    public int    getCol()          { return col; }
    public String getResolvedType() { return resolvedType; }
    public ASTNode getDeclaration() { return declaration; }

    // ----------------------------------------------------------------
    // Setter (Semantik)
    // ----------------------------------------------------------------

    public void setResolvedType(String t) { this.resolvedType = t; }
    public void setDeclaration(ASTNode d) { this.declaration  = d; }

    // ----------------------------------------------------------------
    // Graphkonvertierung (fuer Subgraph Algorithmus)
    // ----------------------------------------------------------------

    /**
     * Konvertiert den Teilbaum ab diesem Knoten in eine Adjazenzmatrix.
     * Knotennummerierung per BFS-Reihenfolge.
     *
     * @return boolean[n][n] Adjazenzmatrix
     */
    public boolean[][] toAdjacencyMatrix() {
        List<ASTNode> bfs = bfsOrder();
        int n = bfs.size();
        boolean[][] adj = new boolean[n][n];

        java.util.IdentityHashMap<ASTNode, Integer> idx =
            new java.util.IdentityHashMap<>();
        for (int i = 0; i < n; i++) idx.put(bfs.get(i), i);

        for (ASTNode node : bfs) {
            int i = idx.get(node);
            for (ASTNode child : node.children) {
                Integer j = idx.get(child);
                if (j != null) adj[i][j] = true;
            }
        }
        return adj;
    }

    /** BFS-Traversal ab diesem Knoten. */
    public List<ASTNode> bfsOrder() {
        List<ASTNode> result = new ArrayList<>();
        java.util.Deque<ASTNode> queue = new java.util.ArrayDeque<>();
        queue.add(this);
        while (!queue.isEmpty()) {
            ASTNode n = queue.poll();
            result.add(n);
            queue.addAll(n.children);
        }
        return result;
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    /** Gibt den Subtree als eingerueckten String aus. */
    public String dump(int indent) {
        StringBuilder sb = new StringBuilder();
        sb.append(" ".repeat(indent))
          .append(kind)
          .append(value.isEmpty() ? "" : "(\"" + value + "\")")
          .append("\n");
        for (ASTNode child : children) sb.append(child.dump(indent + 2));
        return sb.toString();
    }

    @Override
    public String toString() {
        return kind + (value.isEmpty() ? "" : "(\"" + value + "\")");
    }
}

