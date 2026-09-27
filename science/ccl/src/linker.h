/*
 * linker.h  --  Subgraph-basierter Linker (ccl.tex, Epp 2026)
 *
 * Implementiert LAP (Problem 3) und IRP (Problem 7) fuer den Linker:
 *   LAP  Modulabhaengigkeitsaufloesung    Abschnitt 5
 *   IRP  Statische Initialisierung        Abschnitt 9
 */
#ifndef LINKER_H
#define LINKER_H

#include "compiler.h"

/* ================================================================== */
/* LINKER-TYPEN                                                       */
/* ================================================================== */

#define MAX_OBJ_FILES 32
#define MAX_RELOCS 512

typedef struct
{
    char sym_name[64]; /* Referenziertes Symbol    */
    int offset;        /* Offset in der Textsektion */
    int obj_idx;       /* Welche Objektdatei?       */
} Relocation;

typedef struct
{
    ObjFile *objs[MAX_OBJ_FILES];
    int nobj;
    Relocation relocs[MAX_RELOCS];
    int nrelocs;

    /* Ergebnis nach LAP */
    int link_order[MAX_OBJ_FILES]; /* topologische Reihenfolge */
    int link_order_count;
    bool has_circular;

    /* Symbol-Tabelle des Executables */
    Symbol sym_table[MAX_SYMBOLS];
    int nsyms;

    /* Ausgabe-Executable */
    uint8_t exec_code[1024 * 1024];
    int exec_len;
    char exec_asm[1024 * 1024]; /* lesbare ASM-Ausgabe */
} Linker;

/* ================================================================== */
/* LINKER-API                                                         */
/* ================================================================== */

void linker_init(Linker *lnk);
void linker_add_obj(Linker *lnk, ObjFile *obj);
bool linker_link(Linker *lnk, const char *output_name);

#endif /* LINKER_H */
