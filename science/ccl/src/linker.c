/*
 * linker.c  --  Subgraph-basierter Linker (ccl.tex, Epp 2026)
 *
 * LAP (Abschnitt 5):
 *   Symbol-Referenzgraph G_O aufbauen
 *   Reduktion auf SI: G_D ⊆ H_chain
 *   Linkreihenfolge aus Rotation r*
 *
 * IRP (Abschnitt 9):
 *   Initialisierungsgraph G_S aller statischen Symbole
 *   Reduktion auf SI: G_S ⊆ H_DAG
 */
#include "linker.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void linker_init(Linker *lnk)
{
    memset(lnk, 0, sizeof(Linker));
}

void linker_add_obj(Linker *lnk, ObjFile *obj)
{
    if (lnk->nobj < MAX_OBJ_FILES)
        lnk->objs[lnk->nobj++] = obj;
}

/* ------------------------------------------------------------------ */
/* Hilfsfunktion: Symbol in Objektdatei-Liste suchen                  */
/* ------------------------------------------------------------------ */
static int find_symbol_def(Linker *lnk, const char *name, int *obj_idx)
{
    for (int i = 0; i < lnk->nobj; i++)
    {
        ObjFile *o = lnk->objs[i];
        for (int s = 0; s < o->nsyms; s++)
        {
            if (o->syms[s].kind == SYM_DEFINED &&
                !strcmp(o->syms[s].name, name))
            {
                if (obj_idx)
                    *obj_idx = i;
                return s;
            }
        }
    }
    return -1;
}

/* ================================================================== */
/* LINKER-HAUPTFUNKTION                                               */
/* ================================================================== */

bool linker_link(Linker *lnk, const char *output_name)
{
    int k = lnk->nobj;
    printf("\n=== Linker: %d Objektdateien -> %s ===\n", k, output_name);

    if (k == 0)
    {
        fprintf(stderr, "[LINKER] Keine Objektdateien\n");
        return false;
    }

    /* ============================================================ */
    /* LAP: Symbol-Referenzgraph G_O aufbauen                       */
    /* G_O[i][j] = 1 wenn O_i ein Symbol aus O_j verwendet          */
    /* ============================================================ */

    Graph G_O;
    graph_init(&G_O, k);
    for (int i = 0; i < k; i++)
    {
        graph_set_label(&G_O, i, lnk->objs[i]->name);
        for (int s = 0; s < lnk->objs[i]->nsyms; s++)
        {
            if (lnk->objs[i]->syms[s].kind != SYM_UNDEFINED)
                continue;
            /* Symbol wird in O_i verwendet; suche Definition */
            int def_obj = -1;
            find_symbol_def(lnk, lnk->objs[i]->syms[s].name, &def_obj);
            if (def_obj >= 0 && def_obj != i)
            {
                graph_add_edge(&G_O, i, def_obj);
                /* Relocation registrieren */
                if (lnk->nrelocs < MAX_RELOCS)
                {
                    Relocation *r = &lnk->relocs[lnk->nrelocs++];
                    strncpy(r->sym_name, lnk->objs[i]->syms[s].name, 63);
                    r->obj_idx = i;
                    r->offset = lnk->objs[i]->syms[s].offset;
                }
            }
        }
    }

    printf("  [LAP] Symbol-Referenzgraph: %d Knoten\n", k);
    graph_print(&G_O, "G_O (Symbol-Referenzgraph)");

    /* ============================================================ */
    /* LAP Schritt 1: Transitiver Abschluss von O_main              */
    /* D(O_main) = alle transitiv von O_main[0] abhaengigen Objekte */
    /* ============================================================ */

    bool dep[MAX_OBJ_FILES] = {false};
    int queue[MAX_OBJ_FILES], qh = 0, qt = 0;
    int main_obj = 0; /* O_main ist immer das erste Objekt */
    dep[main_obj] = true;
    queue[qt++] = main_obj;

    while (qh < qt)
    {
        int v = queue[qh++];
        for (int j = 0; j < k; j++)
        {
            if (G_O.adj[v][j] && !dep[j])
            {
                dep[j] = true;
                queue[qt++] = j;
            }
        }
    }

    /* Induzierten Teilgraph G_D aufbauen */
    int d_idx[MAX_OBJ_FILES]; /* original -> D-Index */
    int d_map[MAX_OBJ_FILES]; /* D-Index -> original */
    int nd = 0;
    memset(d_idx, -1, sizeof(d_idx));
    for (int i = 0; i < k; i++)
    {
        if (dep[i])
        {
            d_idx[i] = nd;
            d_map[nd] = i;
            nd++;
        }
    }

    Graph G_D;
    graph_init(&G_D, nd);
    for (int ci = 0; ci < nd; ci++)
    {
        int i = d_map[ci];
        graph_set_label(&G_D, ci, lnk->objs[i]->name);
        for (int cj = 0; cj < nd; cj++)
        {
            int j = d_map[cj];
            if (G_O.adj[i][j])
                graph_add_edge(&G_D, ci, cj);
        }
    }

    /* ============================================================ */
    /* LAP Schritt 2: H_chain = vollstaendiger DAG ueber nd Knoten  */
    /* ============================================================ */

    Graph H_chain;
    graph_init(&H_chain, nd);
    for (int i = 0; i < nd; i++)
    {
        char lbl[64];
        snprintf(lbl, 64, "pos_%d", i);
        graph_set_label(&H_chain, i, lbl);
        for (int j = i + 1; j < nd; j++)
            graph_add_edge(&H_chain, i, j);
    }

    /* ============================================================ */
    /* LAP Schritt 3: Subgraph Algorithmus G_D ⊆ H_chain?           */
    /* ============================================================ */

    SI_Match m = subgraph_check(&G_D, &H_chain);

    lnk->has_circular = (m.result == SI_DISJOINT);

    if (lnk->has_circular)
    {
        fprintf(stderr, "  [LAP] FEHLER: Zirkulaere Modulabhaengigkeit!\n");
        fprintf(stderr, "  [LAP] LCS=%d, Rotation=%d\n",
                m.lcs_value, m.best_rotation);
        /* Betroffene Objekte ausgeben */
        for (int ci = 0; ci < nd; ci++)
        {
            if (m.mapping[ci] < 0)
                fprintf(stderr, "  [LAP] Zyklus betrifft: '%s'\n",
                        lnk->objs[d_map[ci]]->name);
        }
        return false;
    }

    /* ============================================================ */
    /* LAP Schritt 4: Linkreihenfolge aus Rotation r*               */
    /* Korollar: Reihenfolge aus topologischer Sortierung           */
    /* ============================================================ */

    int in_deg[MAX_OBJ_FILES] = {0};
    for (int ci = 0; ci < nd; ci++)
        for (int cj = 0; cj < nd; cj++)
            if (G_D.adj[ci][cj])
                in_deg[cj]++;

    int tq[MAX_OBJ_FILES], tqh = 0, tqt = 0;
    for (int ci = 0; ci < nd; ci++)
        if (in_deg[ci] == 0)
            tq[tqt++] = ci;

    lnk->link_order_count = 0;
    while (tqh < tqt)
    {
        int cv = tq[tqh++];
        lnk->link_order[lnk->link_order_count++] = d_map[cv];
        for (int cj = 0; cj < nd; cj++)
        {
            if (G_D.adj[cv][cj])
            {
                in_deg[cj]--;
                if (in_deg[cj] == 0)
                    tq[tqt++] = cj;
            }
        }
    }

    printf("  [LAP] Linkreihenfolge (r*=%d):\n", m.best_rotation);
    for (int i = 0; i < lnk->link_order_count; i++)
        printf("    %d. %s\n", i + 1,
               lnk->objs[lnk->link_order[i]]->name);

    /* ============================================================ */
    /* Symbol-Aufloesung (Relocation)                               */
    /* ============================================================ */

    printf("  [LINK] Symbol-Aufloesung...\n");

    /* Globale Symboltabelle aufbauen */
    lnk->nsyms = 0;
    for (int i = 0; i < lnk->link_order_count; i++)
    {
        int oi = lnk->link_order[i];
        ObjFile *o = lnk->objs[oi];
        for (int s = 0; s < o->nsyms && lnk->nsyms < MAX_SYMBOLS; s++)
        {
            if (o->syms[s].kind != SYM_DEFINED)
                continue;
            /* Pruefe Doppeldefinition */
            bool dup = false;
            for (int q = 0; q < lnk->nsyms; q++)
            {
                if (!strcmp(lnk->sym_table[q].name, o->syms[s].name))
                {
                    fprintf(stderr, "  [LINK] WARNUNG: Doppeldefinition '%s'\n",
                            o->syms[s].name);
                    dup = true;
                    break;
                }
            }
            if (!dup)
            {
                lnk->sym_table[lnk->nsyms] = o->syms[s];
                lnk->nsyms++;
            }
        }
    }

    /* Unaufgeloeste Symbole pruefen */
    bool all_resolved = true;
    for (int i = 0; i < lnk->link_order_count; i++)
    {
        int oi = lnk->link_order[i];
        ObjFile *o = lnk->objs[oi];
        for (int s = 0; s < o->nsyms; s++)
        {
            if (o->syms[s].kind != SYM_UNDEFINED)
                continue;
            int def_obj = -1;
            int def_sym = find_symbol_def(lnk, o->syms[s].name, &def_obj);
            if (def_sym < 0)
            {
                fprintf(stderr, "  [LINK] FEHLER: Unaufgeloestes Symbol '%s'"
                                " in '%s'\n",
                        o->syms[s].name, o->name);
                all_resolved = false;
            }
            else
            {
                printf("  [LINK] '%s' in '%s' -> '%s'\n",
                       o->syms[s].name, o->name,
                       lnk->objs[def_obj]->name);
            }
        }
    }

    if (!all_resolved)
    {
        fprintf(stderr, "  [LINK] Linking fehlgeschlagen\n");
        return false;
    }

    /* ============================================================ */
    /* IRP: Statische Initialisierungsreihenfolge ueber alle Objs   */
    /* Baue globalen Initialisierungsgraph G_S                      */
    /* ============================================================ */

    printf("  [IRP] Globale Initialisierungsreihenfolge...\n");

    InitSystem global_is;
    memset(&global_is, 0, sizeof(global_is));

    /* Sammle alle definierten Symbole als statische Objekte */
    for (int i = 0; i < lnk->nsyms && global_is.count < MAX_STATIC_OBJS; i++)
    {
        int si = global_is.count++;
        strncpy(global_is.objs[si].name, lnk->sym_table[i].name, 63);
        global_is.objs[si].ndeps = 0;
    }

    /* Abhaengigkeiten aus Relocation-Tabelle */
    for (int r = 0; r < lnk->nrelocs; r++)
    {
        /* Finde Quell- und Zielobjekt */
        int src = -1, dst = -1;
        for (int q = 0; q < global_is.count; q++)
        {
            /* Vereinfachung: Relocation-Symbol = Abhaengigkeit */
            if (!strcmp(global_is.objs[q].name,
                        lnk->relocs[r].sym_name))
                dst = q;
        }
        if (src >= 0 && dst >= 0 && src != dst)
        {
            StaticObj *so = &global_is.objs[src];
            if (so->ndeps < MAX_STATIC_OBJS)
                so->deps[so->ndeps++] = dst;
        }
    }

    irp_resolve(&global_is);

    /* ============================================================ */
    /* Code zusammenfuehren                                         */
    /* ============================================================ */

    printf("  [LINK] Erzeuge Executable '%s'...\n", output_name);

    /* ASM-Header */
    int pos = 0;
    pos += snprintf(lnk->exec_asm + pos,
                    sizeof(lnk->exec_asm) - pos,
                    "# Subgraph-Linker Output: %s\n"
                    "# Epp 2026 -- ccl.tex\n"
                    "# Linkreihenfolge (LAP, r*=%d):\n",
                    output_name, m.best_rotation);

    for (int i = 0; i < lnk->link_order_count; i++)
        pos += snprintf(lnk->exec_asm + pos,
                        sizeof(lnk->exec_asm) - pos,
                        "#   %d. %s\n", i + 1,
                        lnk->objs[lnk->link_order[i]]->name);
    pos += snprintf(lnk->exec_asm + pos,
                    sizeof(lnk->exec_asm) - pos, "\n");

    /* Sections in Linkreihenfolge zusammenfuehren */
    lnk->exec_len = 0;
    for (int i = 0; i < lnk->link_order_count; i++)
    {
        int oi = lnk->link_order[i];
        ObjFile *o = lnk->objs[oi];

        /* ASM-Text */
        int avail_asm = (int)sizeof(lnk->exec_asm) - pos - 1;
        if (avail_asm > 0)
        {
            int len = (int)strlen(o->asm_text);
            if (len > avail_asm)
                len = avail_asm;
            memcpy(lnk->exec_asm + pos, o->asm_text, len);
            pos += len;
        }

        /* Maschinencode */
        int avail = (int)sizeof(lnk->exec_code) - lnk->exec_len;
        if (avail > 0)
        {
            int len = o->code_len < avail ? o->code_len : avail;
            memcpy(lnk->exec_code + lnk->exec_len, o->code, len);
            lnk->exec_len += len;
        }
    }
    lnk->exec_asm[pos] = '\0';

    printf("  [LINK] Executable '%s': %d Bytes\n",
           output_name, lnk->exec_len);

    return true;
}
