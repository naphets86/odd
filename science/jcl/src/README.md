# JCL – Java Compiler via Language-Graph

> **Stephan Epp · 2026**  
> Wissenschaftliche Arbeit: `jcl.tex` / `jcl.pdf`

---

## Überblick

JCL ist ein Demonstrationscompiler, der acht fundamentale Probleme beim
Java-Compilerbau auf das **Subgraph-Isomorphismusproblem** reduziert und mit
dem **Subgraph Algorithmus** (Epp 2026) in $O(n^3)$ löst.

| Phase | Abkürzung | Klasse                  | Reduktion            |
|-------|-----------|-------------------------|----------------------|
| 1     | Lexer     | `Lexer`                 | –                    |
| 2     | Parser    | `Parser`                | –                    |
| 3     | TAP       | `TypeResolver`          | Typauflösung → SI    |
| 4     | BVP       | `BytecodeVerifier`      | Bytecode-Verifik. → SI |
| 5     | CFG       | `CFGBuilder`            | –                    |
| 6     | DCE       | `DeadCodeEliminator`    | Dead Code → SI       |
| 7     | SSA       | `SSABuilder`            | –                    |
| 8     | CPP       | `ConstantPropagator`    | Konstantenprop. → SI |
| 9     | SEP       | `LoopOptimizer`         | Schleifenopt. → SI   |
| 10    | RAP       | `RegisterAllocator`     | Registerallok. → SI  |
| 11    | LAP       | `ModuleResolver`        | Modulabh. → SI       |
| 12    | IRP       | `StaticInitResolver`    | Statische Init. → SI |
| 13    | Codegen   | `BytecodeGenerator`     | –                    |

---

## Projektstruktur

```
jcl/
  pom.xml
  README.md
  src/main/java/
    module-info.java
    de/hjstephan/jcl/
      JCLCompiler.java              ← Haupt-Pipeline
      core/
        SubgraphAlgorithm.java      ← SI-Kern (Epp 2026)
      phases/
        Lexer.java
        Token.java
        Parser.java
        SemanticAnalyzer.java
        TypeResolver.java           ← TAP
        BytecodeVerifier.java       ← BVP
        DeadCodeEliminator.java     ← DCE
        ConstantPropagator.java     ← CPP
        RegisterAllocator.java      ← RAP
        LoopOptimizer.java          ← SEP
        ModuleResolver.java         ← LAP
        StaticInitResolver.java     ← IRP
      ir/
        ASTNode.java
        CFGBuilder.java
        SSABuilder.java
      codegen/
        BytecodeGenerator.java
  src/test/java/de/hjstephan/jcl/
    SubgraphAlgorithmTest.java
    TypeResolverTest.java
    DCETest.java
    RAPTest.java
```

---

## Build & Ausführen

**Voraussetzung:** Java 21+, Maven 3.8+

```bash
# Kompilieren + Tests
mvn clean package

# Ausfuehren
java -jar target/jcl.jar --verbose HelloWorld.java

# Mit Optionen
java -jar target/jcl.jar --dump-ast --dump-cfg --out /tmp MyClass.java
```

**Ohne Maven:**
```bash
# Kompilieren
find src/main/java -name "*.java" | \
  xargs javac --enable-preview --release 21 -d out/

# Tests
javac --enable-preview --release 21 \
  -cp out/ src/test/java/de/hjstephan/jcl/*.java -d out/
java --enable-preview -cp out/ de.hjstephan.jcl.SubgraphAlgorithmTest
java --enable-preview -cp out/ de.hjstephan.jcl.TypeResolverTest
java --enable-preview -cp out/ de.hjstephan.jcl.DCETest
java --enable-preview -cp out/ de.hjstephan.jcl.RAPTest
```

---

## Laufzeitgarantien

| Problem | Laufzeit       | Klasse                |
|---------|----------------|-----------------------|
| TAP     | O(\|T\|³)      | TypeResolver          |
| DCE     | O(\|B\|³)      | DeadCodeEliminator    |
| CPP     | O(m·\|SSA\|²)  | ConstantPropagator    |
| RAP     | O(\|V\|³)      | RegisterAllocator     |
| SEP     | O(\|B\|³)      | LoopOptimizer         |
| LAP     | O(p³)          | ModuleResolver        |
| IRP     | O(q³)          | StaticInitResolver    |
| BVP     | O(r³)          | BytecodeVerifier      |

**Gesamt:** $O(n^3 + m^3 + p^3 + r^3)$

---

## Literatur

- S. Epp, *Der Subgraph Algorithmus*, 2026. https://github.com/hjstephan86/science
- J. Gosling et al., *The Java Language Specification, SE 21*, Oracle, 2021.
- A. V. Aho et al., *Compilers: Principles, Techniques, and Tools*, 2nd ed., 2006.
