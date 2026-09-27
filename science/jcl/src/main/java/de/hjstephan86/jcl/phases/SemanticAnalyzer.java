package de.hjstephan86.jcl.phases;

import de.hjstephan86.jcl.ir.ASTNode;

import java.util.*;

/**
 * Semantische Analyse: Namensaufloesung, Typcheck, Sichtbarkeitspruefung.
 * Arbeitet zusammen mit TypeResolver (TAP).
 *
 * @author Stephan Epp
 */
public class SemanticAnalyzer {

    private final ASTNode      ast;
    private final TypeResolver typeResolver;
    private final List<String> errors = new ArrayList<>();

    /** Scope-Stack: Name -> Typ-String */
    private final Deque<Map<String, String>> scopes = new ArrayDeque<>();

    public SemanticAnalyzer(ASTNode ast, TypeResolver typeResolver) {
        this.ast          = ast;
        this.typeResolver = typeResolver;
    }

    // ----------------------------------------------------------------
    // Analyse-Einstieg
    // ----------------------------------------------------------------

    /**
     * Fuehrt die vollstaendige semantische Analyse durch.
     *
     * @throws SemanticException wenn Fehler gefunden wurden
     */
    public void analyze() {
        // Typen aus AST registrieren
        typeResolver.registerFromAST(ast);

        // Vererbungshierarchie aufloesen (TAP via SI)
        try {
            typeResolver.resolve();
        } catch (IllegalStateException e) {
            errors.add("TAP: " + e.getMessage());
        }

        pushScope();
        visitNode(ast);
        popScope();

        if (!errors.isEmpty()) {
            throw new SemanticException(errors);
        }
    }

    // ----------------------------------------------------------------
    // Knotenbesucher
    // ----------------------------------------------------------------

    private void visitNode(ASTNode n) {
        switch (n.getKind()) {
            case METHOD_DECL, CONSTRUCTOR_DECL -> visitMethod(n);
            case FIELD_DECL                   -> visitField(n);
            case VAR_DECL_STMT                -> visitVarDecl(n);
            case NAME                         -> resolveNameRef(n);
            case METHOD_CALL                  -> visitMethodCall(n);
            case BINARY                       -> visitBinary(n);
            case IF_STMT, WHILE_STMT, DO_STMT -> visitChildren(n);
            default                           -> visitChildren(n);
        }
    }

    private void visitMethod(ASTNode n) {
        pushScope();
        // Parameter in Scope eintragen
        for (ASTNode child : n.getChildren()) {
            if (child.getKind() == ASTNode.Kind.PARAM) {
                String paramType = child.childCount() > 0
                    ? child.getChild(0).getValue() : "Object";
                defineVar(child.getValue(), paramType);
            }
        }
        visitChildren(n);
        popScope();
    }

    private void visitField(ASTNode n) {
        String type = n.childCount() > 1 ? n.getChild(1).getValue() : "Object";
        defineVar(n.getValue(), type);
        visitChildren(n);
    }

    private void visitVarDecl(ASTNode n) {
        String type = n.childCount() > 0 ? n.getChild(0).getValue() : "Object";
        defineVar(n.getValue(), type);
        visitChildren(n);
    }

    private void resolveNameRef(ASTNode n) {
        String name = n.getValue();
        if (name.equals("this") || name.equals("super") ||
            name.equals("null") || name.isEmpty()) return;
        String type = lookupVar(name);
        if (type != null) {
            n.setResolvedType(type);
        }
        // Kein Fehler: Felder koennen spaeter definiert sein
    }

    private void visitMethodCall(ASTNode n) {
        visitChildren(n);
    }

    private void visitBinary(ASTNode n) {
        visitChildren(n);
        if (n.childCount() >= 2) {
            String lt = n.getChild(0).getResolvedType();
            String rt = n.getChild(1).getResolvedType();
            if (lt != null && rt != null && !lt.equals(rt)) {
                // Typwarnung (nicht Fehler -- Java hat Promotionen)
                n.setResolvedType(promoteNumeric(lt, rt));
            } else if (lt != null) {
                n.setResolvedType(lt);
            }
        }
    }

    private void visitChildren(ASTNode n) {
        for (ASTNode child : n.getChildren()) visitNode(child);
    }

    // ----------------------------------------------------------------
    // Scope-Verwaltung
    // ----------------------------------------------------------------

    private void pushScope() { scopes.push(new LinkedHashMap<>()); }
    private void popScope()  { scopes.pop(); }

    private void defineVar(String name, String type) {
        if (!scopes.isEmpty()) scopes.peek().put(name, type);
    }

    private String lookupVar(String name) {
        for (Map<String, String> scope : scopes) {
            String t = scope.get(name);
            if (t != null) return t;
        }
        return null;
    }

    // ----------------------------------------------------------------
    // Typpromotion
    // ----------------------------------------------------------------

    private String promoteNumeric(String a, String b) {
        if ("double".equals(a) || "double".equals(b)) return "double";
        if ("float".equals(a)  || "float".equals(b))  return "float";
        if ("long".equals(a)   || "long".equals(b))   return "long";
        return "int";
    }

    // ----------------------------------------------------------------
    // Ausnahme
    // ----------------------------------------------------------------

    public static class SemanticException extends RuntimeException {
        private final List<String> errors;
        public SemanticException(List<String> errors) {
            super("Semantikfehler:\n" + String.join("\n", errors));
            this.errors = errors;
        }
        public List<String> getErrors() { return errors; }
    }
}
