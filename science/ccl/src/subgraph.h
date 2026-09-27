/*
 * subgraph.h  --  Subgraph Algorithmus (Epp 2026)
 *
 * Kernoperation fuer alle 7 polynomiellen Reduktionen aus ccl.tex:
 *   TAP, DCE, LAP, RAP, SEP, CPP, IRP
 *
 * Laufzeit: O(n^3)  |  Speicher: O(n^2)
 */
#ifndef SUBGRAPH_H
#define SUBGRAPH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Adjazenzmatrix-Graph (n <= SUBGRAPH_MAX_NODES)                     */
/* ------------------------------------------------------------------ */
#define SUBGRAPH_MAX_NODES 256

typedef struct
{
    int n;                                           /* Knotenanzahl            */
    int adj[SUBGRAPH_MAX_NODES][SUBGRAPH_MAX_NODES]; /* Adjazenzmatrix */
    char label[SUBGRAPH_MAX_NODES][64];              /* optionale Knotenbezeichner */
} Graph;

/* Ergebnis des Subgraph Algorithmus */
typedef enum
{
    SI_A_IN_B,  /* G  ist Subgraph von H  (G  ⊆ H)  */
    SI_B_IN_A,  /* H  ist Subgraph von G  (H  ⊆ G)  */
    SI_EQUAL,   /* G  und H  sind aequivalent        */
    SI_DISJOINT /* keiner ist im anderen enthalten   */
} SI_Result;

/* Ergebnis mit Rotationsinformation */
typedef struct
{
    SI_Result result;
    int best_rotation;               /* Rotation r* bei der Match gefunden wurde */
    int lcs_value;                   /* LCS-Laenge des besten Matches            */
    int mapping[SUBGRAPH_MAX_NODES]; /* Knotenzuordnung A->B          */
} SI_Match;

/* ------------------------------------------------------------------ */
/* Oeffentliche API                                                   */
/* ------------------------------------------------------------------ */

/* Graph initialisieren */
void graph_init(Graph *g, int n);
void graph_add_edge(Graph *g, int from, int to);
void graph_set_label(Graph *g, int node, const char *lbl);

/* Subgraph Algorithmus: entscheide ob A ⊆ B, B ⊆ A, gleich, disjunkt */
SI_Match subgraph_check(const Graph *A, const Graph *B);

/* Hilfsfunktionen fuer Reduktionen */
void graph_print(const Graph *g, const char *title);
bool graph_is_dag(const Graph *g);         /* Zykluspruefung via DFS */
int graph_max_clique_size(const Graph *g); /* fuer RAP               */

#endif /* SUBGRAPH_H */
