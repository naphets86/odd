/*
 * main.c  --  Testprogramm fuer den Subgraph-basierten Compiler+Linker
 *
 * Demonstriert alle 7 Reduktionen aus ccl.tex:
 *   TAP, DCE, LAP, RAP, SEP, CPP, IRP
 */
#include "compiler.h"
#include "linker.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ================================================================== */
/* Testprogramme                                                      */
/* ================================================================== */

/* Testprogramm 1: Demonstriert TAP, DCE, SEP, CPP, RAP              */
static const char *SOURCE_MAIN =
    "/* Testprogramm: Subgraph-Compiler Demo */\n"
    "\n"
    "/* Statische Objekte (IRP) */\n"
    "static int global_counter;\n"
    "static int global_max;\n"
    "\n"
    "int factorial(int n) {\n"
    "    int result;\n"
    "    int i;\n"
    "    result = 1;\n"
    "    i = 1;\n"
    "    while (i < n) {\n" /* Schleife => SEP */
    "        result = result * i;\n"
    "        i = i + 1;\n"
    "    }\n"
    "    return result;\n"
    "}\n"
    "\n"
    "int main() {\n"
    "    int x;\n"
    "    int y;\n"
    "    int z;\n"
    "    x = 5;\n"     /* Konstante => CPP */
    "    y = 3;\n"     /* Konstante => CPP */
    "    z = x + y;\n" /* z = 8 (konstant) */
    "    if (z < 10) {\n"
    "        x = factorial(z);\n"
    "    } else {\n"
    "        x = 0;\n" /* dead code if z<10 immer? => DCE */
    "    }\n"
    "    return x;\n"
    "}\n";

/* Testprogramm 2: Hilfsmodul (fuer LAP) */
static const char *SOURCE_HELPER =
    "int add(int a, int b) {\n"
    "    int result;\n"
    "    result = a + b;\n"
    "    return result;\n"
    "}\n"
    "\n"
    "int multiply(int a, int b) {\n"
    "    int result;\n"
    "    result = a * b;\n"
    "    return result;\n"
    "}\n";

/* ================================================================== */
/* TAP-Einzeltest: Typabhaengigkeiten                                 */
/* ================================================================== */

static void test_tap(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  TAP-Test: Typabhaengigkeiten        ║\n");
    printf("╚══════════════════════════════════════╝\n");

    TypeSystem ts;
    memset(&ts, 0, sizeof(ts));

    /* Typen: A -> B -> C  (azyklisch), D -> A (azyklisch) */
    strcpy(ts.types[0].name, "TypeA");
    ts.types[0].deps[0] = 1;
    ts.types[0].ndeps = 1; /* A haengt von B ab */
    strcpy(ts.types[1].name, "TypeB");
    ts.types[1].deps[0] = 2;
    ts.types[1].ndeps = 1; /* B haengt von C ab */
    strcpy(ts.types[2].name, "TypeC");
    ts.types[2].ndeps = 0;
    strcpy(ts.types[3].name, "TypeD");
    ts.types[3].deps[0] = 0;
    ts.types[3].ndeps = 1; /* D haengt von A ab */
    ts.count = 4;

    bool ok = tap_resolve(&ts);
    printf("  Azyklisch: %s\n", ok ? "JA" : "NEIN (Zyklus!)");
    if (ok)
    {
        printf("  Auflösungsreihenfolge:\n");
        for (int i = 0; i < ts.topo_count; i++)
            printf("    %d. %s\n", i + 1, ts.types[ts.topo_order[i]].name);
    }

    /* Zyklus-Test: A -> B -> A */
    printf("\n  Zyklus-Test (A->B->A):\n");
    TypeSystem ts2;
    memset(&ts2, 0, sizeof(ts2));
    strcpy(ts2.types[0].name, "CycleA");
    ts2.types[0].deps[0] = 1;
    ts2.types[0].ndeps = 1;
    strcpy(ts2.types[1].name, "CycleB");
    ts2.types[1].deps[0] = 0;
    ts2.types[1].ndeps = 1;
    ts2.count = 2;
    bool ok2 = tap_resolve(&ts2);
    printf("  Azyklisch: %s\n", ok2 ? "JA" : "NEIN (Zyklus erkannt!)");
}

/* ================================================================== */
/* IRP-Einzeltest: Statische Initialisierung                          */
/* ================================================================== */

static void test_irp(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  IRP-Test: Statische Initialisierung ║\n");
    printf("╚══════════════════════════════════════╝\n");

    InitSystem is;
    memset(&is, 0, sizeof(is));

    /* Logger haengt von Config ab, Config haengt von Defaults ab */
    strcpy(is.objs[0].name, "Logger");
    is.objs[0].deps[0] = 1;
    is.objs[0].ndeps = 1;
    strcpy(is.objs[1].name, "Config");
    is.objs[1].deps[0] = 2;
    is.objs[1].ndeps = 1;
    strcpy(is.objs[2].name, "Defaults");
    is.objs[2].ndeps = 0;
    is.count = 3;

    irp_resolve(&is);

    /* SIOF-Test: Logger <-> Config (gegenseitig) */
    printf("\n  SIOF-Test (Logger <-> Config):\n");
    InitSystem is2;
    memset(&is2, 0, sizeof(is2));
    strcpy(is2.objs[0].name, "SLogger");
    is2.objs[0].deps[0] = 1;
    is2.objs[0].ndeps = 1;
    strcpy(is2.objs[1].name, "SConfig");
    is2.objs[1].deps[0] = 0;
    is2.objs[1].ndeps = 1;
    is2.count = 2;
    irp_resolve(&is2);
}

/* ================================================================== */
/* Subgraph Algorithmus Direkttest                                    */
/* ================================================================== */

static void test_subgraph_core(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  Subgraph Algorithmus Kerntest       ║\n");
    printf("╚══════════════════════════════════════╝\n");

    /* Beispiel aus subgraph.tex: G ist Subgraph von G' */
    Graph A, B;
    graph_init(&A, 4);
    graph_set_label(&A, 0, "a0");
    graph_set_label(&A, 1, "a1");
    graph_set_label(&A, 2, "a2");
    graph_set_label(&A, 3, "a3");
    graph_add_edge(&A, 0, 1);
    graph_add_edge(&A, 1, 2);
    graph_add_edge(&A, 2, 3);

    graph_init(&B, 4);
    graph_set_label(&B, 0, "b0");
    graph_set_label(&B, 1, "b1");
    graph_set_label(&B, 2, "b2");
    graph_set_label(&B, 3, "b3");
    graph_add_edge(&B, 0, 1);
    graph_add_edge(&B, 0, 2); /* Extra-Kante */
    graph_add_edge(&B, 1, 2);
    graph_add_edge(&B, 2, 3);

    graph_print(&A, "Graph A");
    graph_print(&B, "Graph B");

    SI_Match m = subgraph_check(&A, &B);
    const char *result_str[] = {"A ⊆ B", "B ⊆ A", "A = B", "Disjunkt"};
    printf("  Ergebnis: %s (LCS=%d, Rotation=%d)\n",
           result_str[m.result], m.lcs_value, m.best_rotation);

    /* DAG-Test fuer TAP/LAP/IRP */
    printf("\n  DAG-Test (Pfad A->B->C):\n");
    Graph dag;
    graph_init(&dag, 3);
    graph_add_edge(&dag, 0, 1);
    graph_add_edge(&dag, 1, 2);
    printf("  Ist DAG: %s\n", graph_is_dag(&dag) ? "JA" : "NEIN");

    Graph cycle;
    graph_init(&cycle, 3);
    graph_add_edge(&cycle, 0, 1);
    graph_add_edge(&cycle, 1, 2);
    graph_add_edge(&cycle, 2, 0);
    printf("  Zyklus-Graph ist DAG: %s\n",
           graph_is_dag(&cycle) ? "JA" : "NEIN (Zyklus erkannt)");
}

/* ================================================================== */
/* HAUPTPROGRAMM                                                      */
/* ================================================================== */

int main(void)
{
    printf("╔═════════════════════════════════════════════════════╗\n");
    printf("║  Subgraph-basierter Compiler+Linker  (ccl.tex)      ║\n");
    printf("║  Polynomielle Reduktionen: TAP DCE LAP RAP SEP      ║\n");
    printf("║  CPP IRP  ->  SI  (Epp 2026)                        ║\n");
    printf("╚═════════════════════════════════════════════════════╝\n");

    /* ---- 1. Subgraph-Kerntest ---- */
    test_subgraph_core();

    /* ---- 2. TAP-Test ---- */
    test_tap();

    /* ---- 3. IRP-Test ---- */
    test_irp();

    /* ---- 4. Kompiliere Hauptmodul (TAP+CFG+DCE+SEP+SSA+CPP+RAP) ---- */
    ObjFile *obj_main = (ObjFile *)calloc(1, sizeof(ObjFile));
    ObjFile *obj_helper = (ObjFile *)calloc(1, sizeof(ObjFile));
    if (!obj_main || !obj_helper)
    {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  Compiler: Vollstaendige Pipeline    ║\n");
    printf("╚══════════════════════════════════════╝\n");

    bool ok1 = compile(SOURCE_MAIN, "main.c", obj_main);
    bool ok2 = compile(SOURCE_HELPER, "helper.c", obj_helper);

    if (!ok1 || !ok2)
    {
        fprintf(stderr, "Kompilierung fehlgeschlagen\n");
        return 1;
    }

    /* Symbole aus main: "main", "factorial" als definiert markieren */
    obj_main->syms[obj_main->nsyms].kind = SYM_DEFINED;
    strncpy(obj_main->syms[obj_main->nsyms].name, "main", 63);
    obj_main->nsyms++;
    obj_main->syms[obj_main->nsyms].kind = SYM_DEFINED;
    strncpy(obj_main->syms[obj_main->nsyms].name, "factorial", 63);
    obj_main->nsyms++;
    /* main.c benoetigt "add" aus helper.c */
    obj_main->syms[obj_main->nsyms].kind = SYM_UNDEFINED;
    strncpy(obj_main->syms[obj_main->nsyms].name, "add", 63);
    obj_main->nsyms++;

    /* Symbole aus helper.c */
    obj_helper->syms[obj_helper->nsyms].kind = SYM_DEFINED;
    strncpy(obj_helper->syms[obj_helper->nsyms].name, "add", 63);
    obj_helper->nsyms++;
    obj_helper->syms[obj_helper->nsyms].kind = SYM_DEFINED;
    strncpy(obj_helper->syms[obj_helper->nsyms].name, "multiply", 63);
    obj_helper->nsyms++;

    /* ---- 5. Linken (LAP + IRP) ---- */
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  Linker: LAP + IRP                   ║\n");
    printf("╚══════════════════════════════════════╝\n");

    Linker lnk;
    linker_init(&lnk);
    linker_add_obj(&lnk, obj_main);
    linker_add_obj(&lnk, obj_helper);

    bool link_ok = linker_link(&lnk, "program.out");

    if (!link_ok)
    {
        fprintf(stderr, "Linking fehlgeschlagen\n");
        return 1;
    }

    /* ---- 6. Ausgabe des erzeugten Assemblercodes ---- */
    printf("\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║  Generierter Assemblercode (Auszug)  ║\n");
    printf("╚══════════════════════════════════════╝\n");

    /* Zeige ersten 2000 Zeichen */
    int show = (int)strlen(lnk.exec_asm);
    if (show > 2000)
        show = 2000;
    printf("%.*s", show, lnk.exec_asm);
    if ((int)strlen(lnk.exec_asm) > 2000)
        printf("\n  ... [gekuerzt] ...\n");

    /* ---- 7. Zusammenfassung ---- */
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Zusammenfassung der Reduktionen (ccl.tex)       ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  TAP  Typabhaengigkeitsaufloesung  -> SI  O(n^3) ║\n");
    printf("║  DCE  Dead-Code-Elimination        -> SI  O(k^3) ║\n");
    printf("║  LAP  Modulabhaengigkeitsaufloesung-> SI  O(k^3) ║\n");
    printf("║  RAP  Registerallokation           -> SI  O(n^3) ║\n");
    printf("║  SEP  Schleifenerkennung           -> SI  O(n^3) ║\n");
    printf("║  CPP  Konstantenpropagation        -> SI  O(n^4) ║\n");
    printf("║  IRP  Statische Initialisierung    -> SI  O(n^3) ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Gesamtpipeline:  O(n^3 + k^3)   SETH-optimal    ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");

    return 0;
}
