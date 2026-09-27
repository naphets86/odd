package de.hjstephan86.jcl.ir;

import de.hjstephan86.jcl.phases.DeadCodeEliminator.BasicBlock;

import java.util.*;

/**
 * Baut den Kontrollflussgraphen (CFG) aus dem AST.
 * Jede Methode erhaelt einen eigenen CFG.
 *
 * @author Stephan Epp
 */
public class CFGBuilder {

    private final ASTNode ast;

    public CFGBuilder(ASTNode ast) {
        this.ast = ast;
    }

    // ----------------------------------------------------------------
    // CFG-Modell fuer eine Methode
    // ----------------------------------------------------------------

    public record MethodCFG(
        String methodName,
        List<BasicBlock> blocks,
        int entry,
        int exit
    ) {
        public void setBlocks(List<BasicBlock> b) {
            // Immutable Record – wird per Neukonstruktion ersetzt
        }

        public MethodCFG withBlocks(List<BasicBlock> newBlocks) {
            return new MethodCFG(methodName, newBlocks, entry, exit);
        }

        /** Adjazenzmatrix des CFG (fuer Subgraph Algorithmus). */
        public boolean[][] toAdjacencyMatrix() {
            int n = blocks.size();
            boolean[][] adj = new boolean[n][n];
            for (BasicBlock b : blocks) {
                for (int s : b.successors()) {
                    if (s >= 0 && s < n) adj[b.id()][s] = true;
                }
            }
            return adj;
        }
    }

    // ----------------------------------------------------------------
    // Aufbau
    // ----------------------------------------------------------------

    /**
     * Baut CFGs fuer alle Methoden im AST.
     */
    public List<MethodCFG> build() {
        List<MethodCFG> cfgs = new ArrayList<>();
        collectMethods(ast, cfgs);
        return cfgs;
    }

    private void collectMethods(ASTNode n, List<MethodCFG> out) {
        if (n.getKind() == ASTNode.Kind.METHOD_DECL ||
            n.getKind() == ASTNode.Kind.CONSTRUCTOR_DECL) {
            out.add(buildCFG(n));
        }
        for (ASTNode child : n.getChildren()) collectMethods(child, out);
    }

    // ----------------------------------------------------------------
    // CFG einer Methode
    // ----------------------------------------------------------------

    private int nextId;

    private MethodCFG buildCFG(ASTNode method) {
        nextId = 0;
        List<BasicBlock> blocks = new ArrayList<>();

        // Entry-Block
        BasicBlock entry = block(List.of("// entry: " + method.getValue()));
        blocks.add(entry);

        // Rumpf
        ASTNode body = findBlock(method);
        int lastId = buildBlock(body, entry.id(), blocks);

        // Exit-Block
        BasicBlock exit = block(List.of("// exit"));
        blocks.add(exit);

        // Letzter Block -> Exit verknuepfen (falls nicht schon RETURN)
        if (lastId != exit.id()) {
            blocks = rewire(blocks, lastId, exit.id());
        }

        return new MethodCFG(method.getValue(), blocks, entry.id(), exit.id());
    }

    private int buildBlock(ASTNode n, int currentId, List<BasicBlock> blocks) {
        if (n == null) return currentId;
        return switch (n.getKind()) {
            case BLOCK -> {
                int id = currentId;
                for (ASTNode child : n.getChildren()) {
                    id = buildBlock(child, id, blocks);
                }
                yield id;
            }
            case IF_STMT -> buildIf(n, currentId, blocks);
            case WHILE_STMT -> buildWhile(n, currentId, blocks);
            case FOR_STMT   -> buildFor(n, currentId, blocks);
            case RETURN_STMT -> {
                BasicBlock b = block(List.of("return"));
                blocks.add(b);
                connect(blocks, currentId, b.id());
                yield b.id();
            }
            default -> {
                appendInstr(blocks, currentId, n.toString());
                yield currentId;
            }
        };
    }

    private int buildIf(ASTNode n, int currentId, List<BasicBlock> blocks) {
        appendInstr(blocks, currentId, "if(" + condStr(n) + ")");
        BasicBlock thenBlock = block(List.of("// then"));
        BasicBlock elseBlock = block(List.of("// else"));
        BasicBlock mergeBlock= block(List.of("// merge"));
        blocks.add(thenBlock); blocks.add(elseBlock); blocks.add(mergeBlock);
        connect(blocks, currentId, thenBlock.id());
        connect(blocks, currentId, elseBlock.id());

        int thenEnd = n.childCount() > 1
            ? buildBlock(n.getChild(1), thenBlock.id(), blocks) : thenBlock.id();
        int elseEnd = n.childCount() > 2
            ? buildBlock(n.getChild(2), elseBlock.id(), blocks) : elseBlock.id();

        connect(blocks, thenEnd, mergeBlock.id());
        connect(blocks, elseEnd, mergeBlock.id());
        return mergeBlock.id();
    }

    private int buildWhile(ASTNode n, int currentId, List<BasicBlock> blocks) {
        BasicBlock head  = block(List.of("while(" + condStr(n) + ")"));
        BasicBlock body  = block(List.of("// body"));
        BasicBlock after = block(List.of("// after"));
        blocks.add(head); blocks.add(body); blocks.add(after);
        connect(blocks, currentId, head.id());
        connect(blocks, head.id(), body.id());
        connect(blocks, head.id(), after.id());
        int bodyEnd = n.childCount() > 1
            ? buildBlock(n.getChild(1), body.id(), blocks) : body.id();
        connect(blocks, bodyEnd, head.id());  // Rueckkante
        return after.id();
    }

    private int buildFor(ASTNode n, int currentId, List<BasicBlock> blocks) {
        BasicBlock head  = block(List.of("for(cond)"));
        BasicBlock body  = block(List.of("// for-body"));
        BasicBlock after = block(List.of("// for-after"));
        blocks.add(head); blocks.add(body); blocks.add(after);
        connect(blocks, currentId, head.id());
        connect(blocks, head.id(), body.id());
        connect(blocks, head.id(), after.id());
        int bodyEnd = n.childCount() > 3
            ? buildBlock(n.getChild(3), body.id(), blocks) : body.id();
        connect(blocks, bodyEnd, head.id());
        return after.id();
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private BasicBlock block(List<String> instrs) {
        return new BasicBlock(nextId++, instrs, new ArrayList<>());
    }

    private void appendInstr(List<BasicBlock> blocks, int id, String instr) {
        for (int i = 0; i < blocks.size(); i++) {
            if (blocks.get(i).id() == id) {
                List<String> newInstrs = new ArrayList<>(blocks.get(i).instructions());
                newInstrs.add(instr);
                List<Integer> succs = new ArrayList<>(blocks.get(i).successors());
                blocks.set(i, new BasicBlock(id, newInstrs, succs));
                return;
            }
        }
    }

    private void connect(List<BasicBlock> blocks, int from, int to) {
        for (int i = 0; i < blocks.size(); i++) {
            BasicBlock b = blocks.get(i);
            if (b.id() == from && !b.successors().contains(to)) {
                List<Integer> succs = new ArrayList<>(b.successors());
                succs.add(to);
                blocks.set(i, new BasicBlock(b.id(), b.instructions(), succs));
                return;
            }
        }
    }

    private List<BasicBlock> rewire(List<BasicBlock> blocks, int from, int to) {
        connect(blocks, from, to);
        return blocks;
    }

    private ASTNode findBlock(ASTNode method) {
        for (ASTNode c : method.getChildren()) {
            if (c.getKind() == ASTNode.Kind.BLOCK) return c;
        }
        return null;
    }

    private String condStr(ASTNode n) {
        return n.childCount() > 0 ? n.getChild(0).toString() : "?";
    }
}

