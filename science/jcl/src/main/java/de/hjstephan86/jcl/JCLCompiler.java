package de.hjstephan86.jcl;

import de.hjstephan86.jcl.codegen.BytecodeGenerator;
import de.hjstephan86.jcl.ir.ASTNode;
import de.hjstephan86.jcl.ir.CFGBuilder;
import de.hjstephan86.jcl.ir.CFGBuilder.MethodCFG;
import de.hjstephan86.jcl.ir.SSABuilder;
import de.hjstephan86.jcl.ir.SSABuilder.SSAMethod;
import de.hjstephan86.jcl.phases.*;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * JCL -- Java Compiler via Language-Graph.
 *
 * Vollstaendige Pipeline mit acht SI-Reduktionen:
 *   TAP  Typaufloesung          (TypeResolver)
 *   DCE  Dead-Code-Elimination  (DeadCodeEliminator)
 *   CPP  Konstantenpropagation  (ConstantPropagator)
 *   RAP  Registerallokation     (RegisterAllocator)
 *   SEP  Schleifenoptimierung   (LoopOptimizer)
 *   LAP  Modulabhaengigkeit     (ModuleResolver)
 *   IRP  Statische Init.        (StaticInitResolver)
 *   BVP  Bytecode-Verifikation  (BytecodeVerifier)
 *
 * Aufruf:
 *   java -jar jcl.jar [Optionen] Datei.java ...
 *
 * Optionen:
 *   --dump-ast        AST-Dump ausgeben
 *   --dump-cfg        CFG-Dump ausgeben
 *   --no-dce          Dead-Code-Elimination deaktivieren
 *   --no-cpp          Konstantenpropagation deaktivieren
 *   --no-sep          Schleifenoptimierung deaktivieren
 *   --verbose         Ausfuehrliche Ausgabe
 *   --out <verz>      Ausgabeverzeichnis (Standard: .)
 *
 * @author Stephan Epp
 * @version 1.0 (2026)
 */
public class JCLCompiler {

    // ----------------------------------------------------------------
    // Compiler-Optionen
    // ----------------------------------------------------------------

    public static class Options {
        public boolean dumpAST  = false;
        public boolean dumpCFG  = false;
        public boolean dce      = true;
        public boolean cpp      = true;
        public boolean sep      = true;
        public boolean verbose  = false;
        public String  outDir   = ".";
    }

    // ----------------------------------------------------------------
    // Kompilierungsergebnis
    // ----------------------------------------------------------------

    public record CompileResult(
        String sourceFile,
        byte[] bytecode,
        long   timeMs,
        int    astNodes,
        int    deadBlocksEliminated,
        int    constantsPropagated,
        int    loopInvariantsHoisted
    ) {}

    // ----------------------------------------------------------------
    // main()
    // ----------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        Options opts  = new Options();
        List<String> files = parseArgs(args, opts);

        if (files.isEmpty()) {
            printUsage();
            System.exit(1);
        }

        JCLCompiler compiler = new JCLCompiler(opts);
        int errors = 0;

        for (String file : files) {
            try {
                CompileResult result = compiler.compileFile(file);
                String outPath = Paths.get(opts.outDir,
                    Paths.get(file).getFileName().toString()
                         .replace(".java", ".class")).toString();
                Files.write(Paths.get(outPath), result.bytecode());
                if (opts.verbose) printResult(result);
                else System.out.printf("JCL: %s -> %s (%d Bytes, %d ms)%n",
                    file, outPath, result.bytecode().length, result.timeMs());
            } catch (Exception e) {
                System.err.println("FEHLER: " + file + ": " + e.getMessage());
                errors++;
            }
        }
        System.exit(errors);
    }

    // ----------------------------------------------------------------
    // Kompilierung einer Datei
    // ----------------------------------------------------------------

    private final Options opts;

    public JCLCompiler(Options opts) {
        this.opts = opts;
    }

    public JCLCompiler() {
        this(new Options());
    }

    public CompileResult compileFile(String path) throws IOException {
        String source = Files.readString(Path.of(path));
        long start = System.currentTimeMillis();
        byte[] bc  = compile(source);
        long ms    = System.currentTimeMillis() - start;
        return new CompileResult(path, bc, ms,
            lastASTNodes, lastDeadBlocks, lastConstants, lastHoisted);
    }

    // Statistiken der letzten Kompilierung
    private int lastASTNodes   = 0;
    private int lastDeadBlocks = 0;
    private int lastConstants  = 0;
    private int lastHoisted    = 0;

    // ----------------------------------------------------------------
    // Haupt-Pipeline
    // ----------------------------------------------------------------

    /**
     * Kompiliert Java-Quellcode und gibt JVM-Bytecode zurueck.
     *
     * @param source Java-Quellcode als String
     * @return JVM-Bytecode (.class)
     */
    public byte[] compile(String source) {
        // ── Phase 1: Lexer ─────────────────────────────────────────
        Lexer lexer  = new Lexer(source);
        List<Token> tokens = lexer.tokenize();
        log("Phase 1 (Lexer): " + tokens.size() + " Token");

        // ── Phase 2: Parser -> AST ──────────────────────────────────
        Parser  parser = new Parser(tokens);
        ASTNode ast    = parser.parseCompilationUnit();
        lastASTNodes   = ast.bfsOrder().size();
        log("Phase 2 (Parser): " + lastASTNodes + " AST-Knoten");

        if (opts.dumpAST) System.out.println(ast.dump(0));

        // ── Phase 3: Semantik + TAP ─────────────────────────────────
        TypeResolver typeResolver = new TypeResolver();
        SemanticAnalyzer sema = new SemanticAnalyzer(ast, typeResolver);
        sema.analyze();
        log("Phase 3 (TAP/Semantik): " + typeResolver.typeCount() + " Typen aufgeloest");

        // ── Phase 4: BVP ────────────────────────────────────────────
        BytecodeVerifier bvp = new BytecodeVerifier(ast);
        bvp.verify();
        log("Phase 4 (BVP): Bytecode-Typkonsistenz verifiziert");

        // ── Phase 5: CFG-Konstruktion ───────────────────────────────
        CFGBuilder cfgBuilder = new CFGBuilder(ast);
        List<MethodCFG> cfgs  = cfgBuilder.build();
        log("Phase 5 (CFG): " + cfgs.size() + " Methoden-CFGs erstellt");

        if (opts.dumpCFG) cfgs.forEach(c -> System.out.println(cfgDump(c)));

        // ── Phase 6: DCE ────────────────────────────────────────────
        List<MethodCFG> optimizedCFGs = new ArrayList<>();
        lastDeadBlocks = 0;
        for (MethodCFG cfg : cfgs) {
            if (opts.dce) {
                DeadCodeEliminator dce =
                    new DeadCodeEliminator(cfg.blocks(), cfg.entry());
                lastDeadBlocks += dce.countDeadBlocks();
                optimizedCFGs.add(cfg.withBlocks(dce.eliminate()));
            } else {
                optimizedCFGs.add(cfg);
            }
        }
        log("Phase 6 (DCE): " + lastDeadBlocks + " tote Bloecke eliminiert");

        // ── Phase 7: SSA-Konstruktion ───────────────────────────────
        SSABuilder ssaBuilder     = new SSABuilder(optimizedCFGs);
        List<SSAMethod> ssaMethods = ssaBuilder.build();
        log("Phase 7 (SSA): " + ssaMethods.stream()
            .mapToInt(SSAMethod::numVars).sum() + " SSA-Variablen");

        // ── Phase 8: CPP ────────────────────────────────────────────
        lastConstants = 0;
        if (opts.cpp) {
            for (SSAMethod m : ssaMethods) {
                ConstantPropagator cpp =
                    new ConstantPropagator(m.vars(), m.defUse());
                lastConstants += cpp.propagate();
            }
        }
        log("Phase 8 (CPP): " + lastConstants + " Konstanten propagiert");

        // ── Phase 9: SEP ────────────────────────────────────────────
        lastHoisted = 0;
        if (opts.sep) {
            LoopOptimizer sep = new LoopOptimizer(optimizedCFGs, ssaMethods);
            lastHoisted = sep.optimize();
        }
        log("Phase 9 (SEP): " + lastHoisted + " loop-invariante Def. ausgeklammert");

        // ── Phase 10: RAP ───────────────────────────────────────────
        List<int[]> slotAllocations = new ArrayList<>();
        for (SSAMethod m : ssaMethods) {
            RegisterAllocator rap = new RegisterAllocator(
                m.numVars(), m.interferenceGraph(), 65535);
            slotAllocations.add(rap.allocate());
        }
        log("Phase 10 (RAP): Slot-Allokation fuer " + ssaMethods.size() + " Methoden");

        // ── Phase 11: LAP ───────────────────────────────────────────
        ModuleResolver lap = new ModuleResolver(ast);
        try { lap.resolve(); } catch (IllegalStateException e) {
            System.err.println("LAP Warnung: " + e.getMessage());
        }
        log("Phase 11 (LAP): Modulabhaengigkeiten aufgeloest");

        // ── Phase 12: IRP ───────────────────────────────────────────
        StaticInitResolver irp = new StaticInitResolver(ast);
        try { irp.resolve(); } catch (IllegalStateException e) {
            throw new RuntimeException("IRP: " + e.getMessage(), e);
        }
        log("Phase 12 (IRP): " + irp.initCount() + " statische Initialisierer geordnet");

        // ── Phase 13: Bytecode-Generierung ─────────────────────────
        BytecodeGenerator gen = new BytecodeGenerator(ast, slotAllocations);
        byte[] bytecode = gen.generate();
        log("Phase 13 (Codegen): " + bytecode.length + " Bytes Bytecode erzeugt");

        return bytecode;
    }

    // ----------------------------------------------------------------
    // Hilfsmethoden
    // ----------------------------------------------------------------

    private void log(String msg) {
        if (opts.verbose) System.out.println("[JCL] " + msg);
    }

    private static String cfgDump(MethodCFG cfg) {
        StringBuilder sb = new StringBuilder();
        sb.append("CFG ").append(cfg.methodName()).append(":\n");
        for (var b : cfg.blocks()) {
            sb.append("  B").append(b.id()).append(" -> ").append(b.successors())
              .append(" | ").append(b.instructions()).append("\n");
        }
        return sb.toString();
    }

    private static void printResult(CompileResult r) {
        System.out.printf("""
            ── JCL Kompilierungsbericht ─────────────────────────
            Quelldatei:      %s
            Bytecode:        %d Bytes
            Zeit:            %d ms
            AST-Knoten:      %d
            Tote Bloecke:    %d eliminiert
            Konstanten:      %d propagiert
            Loop-Invariante: %d ausgeklammert
            ─────────────────────────────────────────────────────%n""",
            r.sourceFile(), r.bytecode().length, r.timeMs(),
            r.astNodes(), r.deadBlocksEliminated(),
            r.constantsPropagated(), r.loopInvariantsHoisted());
    }

    private static List<String> parseArgs(String[] args, Options opts) {
        List<String> files = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--dump-ast"  -> opts.dumpAST = true;
                case "--dump-cfg"  -> opts.dumpCFG = true;
                case "--no-dce"    -> opts.dce     = false;
                case "--no-cpp"    -> opts.cpp     = false;
                case "--no-sep"    -> opts.sep     = false;
                case "--verbose"   -> opts.verbose  = true;
                case "--out"       -> { if (++i < args.length) opts.outDir = args[i]; }
                default            -> files.add(args[i]);
            }
        }
        return files;
    }

    private static void printUsage() {
        System.err.println("""
            Verwendung: jcl [Optionen] Datei.java ...
            Optionen:
              --dump-ast        AST ausgeben
              --dump-cfg        CFG ausgeben
              --no-dce          Dead-Code-Elimination deaktivieren
              --no-cpp          Konstantenpropagation deaktivieren
              --no-sep          Schleifenoptimierung deaktivieren
              --verbose         Ausfuehrliche Ausgabe
              --out <verzeichnis>  Ausgabeverzeichnis
            """);
    }
}
