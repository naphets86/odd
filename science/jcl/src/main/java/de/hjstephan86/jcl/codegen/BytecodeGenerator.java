package de.hjstephan86.jcl.codegen;

import de.hjstephan86.jcl.ir.ASTNode;

import java.io.*;
import java.util.*;

/**
 * Einfacher JVM-Bytecode-Generator fuer den JCL-Demonstrationscompiler.
 * Erzeugt validen .class-File-Header (Magic Number + Versionsnummer)
 * sowie vereinfachten Methodenbytecode.
 *
 * JVM SE 21 = Major Version 65 (0x41).
 *
 * @author Stephan Epp
 */
public class BytecodeGenerator {

    // JVM .class Konstanten
    private static final int MAGIC         = 0xCAFEBABE;
    private static final int MINOR_VERSION = 0;
    private static final int MAJOR_VERSION = 65; // Java 21

    // Bytecode-Opcodes (Auswahl)
    public static final byte BIPUSH   = 0x10;
    public static final byte SIPUSH   = 0x11;
    public static final byte ILOAD_0  = 0x1A;
    public static final byte ILOAD_1  = 0x1B;
    public static final byte ILOAD_2  = 0x1C;
    public static final byte ISTORE_0 = 0x3B;
    public static final byte ISTORE_1 = 0x3C;
    public static final byte IADD     = 0x60;
    public static final byte ISUB     = 0x64;
    public static final byte IMUL     = 0x68;
    public static final byte IDIV     = 0x6C;
    public static final byte IRETURN  = (byte) 0xAC;
    public static final byte RETURN   = (byte) 0xB1;
    public static final byte INVOKEVIRTUAL   = (byte) 0xB6;
    public static final byte INVOKESTATIC    = (byte) 0xB8;
    public static final byte GETSTATIC       = (byte) 0xB2;
    public static final byte NOP      = 0x00;

    private final ASTNode   compilationUnit;
    private final List<int[]> slotAllocations;  // pro Methode

    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    public BytecodeGenerator(ASTNode compilationUnit,
                              List<int[]> slotAllocations) {
        this.compilationUnit = compilationUnit;
        this.slotAllocations = slotAllocations;
    }

    // ----------------------------------------------------------------
    // Hauptmethode
    // ----------------------------------------------------------------

    /**
     * Generiert eine minimale .class-Datei als Byte-Array.
     *
     * @return Bytecode-Array (gueltiger .class-File-Header)
     */
    public byte[] generate() {
        out.reset();
        try {
            writeClassFile();
        } catch (IOException e) {
            throw new RuntimeException("Bytecode-Generierung fehlgeschlagen.", e);
        }
        return out.toByteArray();
    }

    /**
     * Schreibt die generierte .class-Datei nach {@code path}.
     */
    public void writeTo(String path) throws IOException {
        byte[] bc = generate();
        try (FileOutputStream fos = new FileOutputStream(path)) {
            fos.write(bc);
        }
        System.out.printf("[Codegen] %s erzeugt (%d Bytes).%n", path, bc.length);
    }

    // ----------------------------------------------------------------
    // .class-File-Aufbau
    // ----------------------------------------------------------------

    private void writeClassFile() throws IOException {
        // Magic Number
        writeU4(MAGIC);
        // Version
        writeU2(MINOR_VERSION);
        writeU2(MAJOR_VERSION);

        // Vereinfachter Constant Pool (Groesse 1 = leer)
        writeU2(1);

        // Access Flags: public (0x0001)
        writeU2(0x0001);

        // This class / super class (Index 0 = Platzhalter)
        writeU2(0);
        writeU2(0);

        // Interfaces
        writeU2(0);
        // Fields
        writeU2(0);
        // Methods -- fuer jeden erkannten AST-Methodenknoten
        List<ASTNode> methods = collectMethods(compilationUnit);
        writeU2(methods.size());
        for (int i = 0; i < methods.size(); i++) {
            int[] slots = (i < slotAllocations.size())
                          ? slotAllocations.get(i) : new int[0];
            writeMethodInfo(methods.get(i), slots);
        }
        // Attributes
        writeU2(0);
    }

    private void writeMethodInfo(ASTNode method, int[] slots)
            throws IOException {
        // Access flags: public (0x0001)
        writeU2(0x0001);
        // Name index (0 = unresolved in diesem Stub)
        writeU2(0);
        // Descriptor index
        writeU2(0);
        // Attributes: Code-Attribut
        writeU2(1);
        writeCodeAttribute(method, slots);
    }

    private void writeCodeAttribute(ASTNode method, int[] slots)
            throws IOException {
        // Attributname-Index
        writeU2(0);
        // Code generieren
        byte[] code = generateMethodCode(method, slots);
        // Attributlaenge: 2(max_stack)+2(max_locals)+4(code_length)+code+2(eh)+2(attrs)
        long attrLen = 2 + 2 + 4 + code.length + 2 + 2;
        writeU4((int) attrLen);
        writeU2(8);               // max_stack (konservativ)
        writeU2(Math.max(1, slots.length)); // max_locals
        writeU4(code.length);
        out.write(code);
        writeU2(0);               // exception table length
        writeU2(0);               // attributes count
    }

    /**
     * Erzeugt vereinfachten Bytecode fuer eine Methode.
     * Fuer jede RETURN-Anweisung im AST wird ireturn / return emittiert.
     */
    private byte[] generateMethodCode(ASTNode method, int[] slots) {
        ByteArrayOutputStream code = new ByteArrayOutputStream();
        emitMethodBody(method, code);
        // Sicherheits-return am Ende
        code.write(RETURN & 0xFF);
        return code.toByteArray();
    }

    private void emitMethodBody(ASTNode node, ByteArrayOutputStream code) {
        switch (node.getKind()) {
            case RETURN_STMT -> {
                if (node.childCount() > 0) {
                    emitExpression(node.getChild(0), code);
                    code.write(IRETURN & 0xFF);
                } else {
                    code.write(RETURN & 0xFF);
                }
                return;
            }
            case BINARY -> {
                if (node.childCount() >= 2) {
                    emitExpression(node.getChild(0), code);
                    emitExpression(node.getChild(1), code);
                    byte op = switch (node.getValue()) {
                        case "+" -> IADD;
                        case "-" -> ISUB;
                        case "*" -> IMUL;
                        case "/" -> IDIV;
                        default  -> NOP;
                    };
                    code.write(op & 0xFF);
                }
                return;
            }
            default -> {}
        }
        for (ASTNode child : node.getChildren()) {
            emitMethodBody(child, code);
        }
    }

    private void emitExpression(ASTNode expr, ByteArrayOutputStream code) {
        switch (expr.getKind()) {
            case LITERAL -> {
                try {
                    int val = Integer.parseInt(expr.getValue());
                    if (val >= -1 && val <= 5) {
                        code.write(0x03 + val); // iconst_N
                    } else if (val >= -128 && val <= 127) {
                        code.write(BIPUSH & 0xFF);
                        code.write(val & 0xFF);
                    } else {
                        code.write(SIPUSH & 0xFF);
                        code.write((val >> 8) & 0xFF);
                        code.write(val & 0xFF);
                    }
                } catch (NumberFormatException e) {
                    code.write(NOP & 0xFF);
                }
            }
            case NAME -> {
                // Heuristik: Variable 0-3 -> iload_N
                code.write(ILOAD_0 & 0xFF);
            }
            default -> emitMethodBody(expr, code);
        }
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private List<ASTNode> collectMethods(ASTNode root) {
        List<ASTNode> methods = new ArrayList<>();
        collectMethodsRec(root, methods);
        return methods;
    }

    private void collectMethodsRec(ASTNode n, List<ASTNode> out) {
        if (n.getKind() == ASTNode.Kind.METHOD_DECL ||
            n.getKind() == ASTNode.Kind.CONSTRUCTOR_DECL) {
            out.add(n);
        }
        for (ASTNode child : n.getChildren()) collectMethodsRec(child, out);
    }

    private void writeU4(int v) throws IOException {
        out.write((v >> 24) & 0xFF);
        out.write((v >> 16) & 0xFF);
        out.write((v >>  8) & 0xFF);
        out.write( v        & 0xFF);
    }

    private void writeU2(int v) throws IOException {
        out.write((v >> 8) & 0xFF);
        out.write( v       & 0xFF);
    }
}
