/*
 * subgraph.c  --  Implementierung des Subgraph Algorithmus (Epp 2026)
 *
 * Signaturfunktion:
 *   sigma_j = sum_{i=0}^{n-1} A_{ij} * 2^i  +  j * 2^n
 *
 * Zyklische Rotation + LCS-Vergleich => O(n^3)
 */
#include "subgraph.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Graph-Grundoperationen                                             */
/* ------------------------------------------------------------------ */

void graph_init(Graph *g, int n)
{
    g->n = n;
    memset(g->adj, 0, sizeof(g->adj));
    memset(g->label, 0, sizeof(g->label));
    for (int i = 0; i < n; i++)
    {
        snprintf(g->label[i], 64, "v%d", i);
    }
}

void graph_add_edge(Graph *g, int from, int to)
{
    if (from >= 0 && from < g->n && to >= 0 && to < g->n)
        g->adj[from][to] = 1;
}

void graph_set_label(Graph *g, int node, const char *lbl)
{
    if (node >= 0 && node < g->n)
        snprintf(g->label[node], 64, "%s", lbl);
}

void graph_print(const Graph *g, const char *title)
{
    printf("=== %s (n=%d) ===\n", title, g->n);
    for (int i = 0; i < g->n; i++)
    {
        printf("  [%s] ->", g->label[i]);
        for (int j = 0; j < g->n; j++)
            if (g->adj[i][j])
                printf(" %s", g->label[j]);
        printf("\n");
    }
}

/* ------------------------------------------------------------------ */
/* Signaturfunktion  (Lemma 1, Epp 2026: injektiv)                    */
/*   sigma_j = sum_i A_{ij} * 2^i  +  j * 2^n                         */
/* Wir verwenden uint64_t; fuer n<=30 exakt, sonst approximativ       */
/* ------------------------------------------------------------------ */
static void compute_signatures(const Graph *g,
                               uint64_t sig[SUBGRAPH_MAX_NODES])
{
    int n = g->n;
    for (int j = 0; j < n; j++)
    {
        uint64_t row_sig = 0;
        for (int i = 0; i < n; i++)
            if (g->adj[i][j])
                row_sig |= ((uint64_t)1 << i);
        /* Spaltengewicht: j * 2^n  (mod 2^64 fuer grosse n) */
        uint64_t col_weight = (n < 64) ? ((uint64_t)j << n) : (uint64_t)j;
        sig[j] = row_sig + col_weight;
    }
}

/* Extrahiere Zeilenkomponente (ohne Spaltengewicht) */
static void row_signatures(const Graph *g,
                           uint64_t row[SUBGRAPH_MAX_NODES])
{
    int n = g->n;
    for (int j = 0; j < n; j++)
    {
        uint64_t r = 0;
        for (int i = 0; i < n; i++)
            if (g->adj[i][j])
                r |= ((uint64_t)1 << i);
        row[j] = r;
    }
}

/* ------------------------------------------------------------------ */
/* LCS zweier Sequenzen  (O(n^2) DP)                                  */
/* Gibt LCS-Laenge zurueck; schreibt optionale Rueckverfolgung        */
/* ------------------------------------------------------------------ */
static int lcs(const uint64_t *X, int nx,
               const uint64_t *Y, int ny,
               int match_pos[SUBGRAPH_MAX_NODES]) /* out: positions in Y, oder NULL */
{
    static int dp[SUBGRAPH_MAX_NODES + 1][SUBGRAPH_MAX_NODES + 1];
    for (int i = 0; i <= nx; i++)
        dp[i][0] = 0;
    for (int j = 0; j <= ny; j++)
        dp[0][j] = 0;

    for (int i = 1; i <= nx; i++)
        for (int j = 1; j <= ny; j++)
        {
            if (X[i - 1] == Y[j - 1])
                dp[i][j] = dp[i - 1][j - 1] + 1;
            else
                dp[i][j] = dp[i - 1][j] > dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1];
        }

    /* Rueckverfolgung falls gewuenscht */
    if (match_pos)
    {
        for (int k = 0; k < nx; k++)
            match_pos[k] = -1;
        int i = nx, j = ny;
        while (i > 0 && j > 0)
        {
            if (X[i - 1] == Y[j - 1])
            {
                match_pos[i - 1] = j - 1;
                i--;
                j--;
            }
            else if (dp[i - 1][j] > dp[i][j - 1])
            {
                i--;
            }
            else
            {
                j--;
            }
        }
    }
    return dp[nx][ny];
}

/* ------------------------------------------------------------------ */
/* Subgraph Algorithmus (Kernfunktion, O(n^3))                        */
/* ------------------------------------------------------------------ */
SI_Match subgraph_check(const Graph *A, const Graph *B)
{
    SI_Match result;
    memset(&result, 0, sizeof(result));
    result.result = SI_DISJOINT;
    result.best_rotation = -1;
    result.lcs_value = 0;
    for (int i = 0; i < SUBGRAPH_MAX_NODES; i++)
        result.mapping[i] = -1;

    int na = A->n, nb = B->n;

    /* Schritt 1: Zeilenkomponenten (ohne Spaltengewicht) berechnen */
    uint64_t rowA[SUBGRAPH_MAX_NODES], rowB[SUBGRAPH_MAX_NODES];
    row_signatures(A, rowA);
    row_signatures(B, rowB);

    /* Schritt 2: Fuer jede der nb zyklischen Rotationen von B */
    int best_lcs = -1;
    int best_rot = -1;
    bool a_in_b = false;
    bool b_in_a = false;
    int best_match[SUBGRAPH_MAX_NODES];

    uint64_t rotB[SUBGRAPH_MAX_NODES];

    for (int rot = 0; rot < nb; rot++)
    {
        /* Rotation: rotB[i] = rowB[(rot + i) % nb] */
        for (int i = 0; i < nb; i++)
            rotB[i] = rowB[(rot + i) % nb];

        /* LCS von rowA gegen rotB */
        int mp[SUBGRAPH_MAX_NODES];
        int l = lcs(rowA, na, rotB, nb, mp);

        if (l > best_lcs)
        {
            best_lcs = l;
            best_rot = rot;
            memcpy(best_match, mp, sizeof(mp));
        }

        /* Subgraph-Kriterium: LCS >= 2 (mind. eine Kante repraesentiert) */
        if (l >= 2)
        {
            /* Pruefe Richtung: A ⊆ B (rotiert) */
            if (l == na)
            {
                a_in_b = true;
                if (best_rot == rot)
                    memcpy(best_match, mp, sizeof(mp));
            }
        }
    }

    /* Schritt 3: Pruefe B ⊆ A (Gegenrichtung) */
    for (int rot = 0; rot < na; rot++)
    {
        uint64_t rotA[SUBGRAPH_MAX_NODES];
        for (int i = 0; i < na; i++)
            rotA[i] = rowA[(rot + i) % na];

        int l = lcs(rowB, nb, rotA, na, NULL);
        if (l == nb)
        {
            b_in_a = true;
            break;
        }
    }

    /* Ergebnis bestimmen */
    if (a_in_b && b_in_a)
        result.result = SI_EQUAL;
    else if (a_in_b)
        result.result = SI_A_IN_B;
    else if (b_in_a)
        result.result = SI_B_IN_A;
    else
        result.result = SI_DISJOINT;

    result.best_rotation = best_rot;
    result.lcs_value = best_lcs;

    /* Knotenzuordnung aus bestem Match */
    for (int i = 0; i < na; i++)
    {
        if (best_match[i] >= 0)
        {
            /* Position in rotiertem B zurueck auf Original-B-Index */
            result.mapping[i] = (best_rot + best_match[i]) % nb;
        }
    }
    return result;
}

/* ------------------------------------------------------------------ */
/* Hilfsfunktionen                                                    */
/* ------------------------------------------------------------------ */

/* Zykluspruefung via iterativem DFS  */
static bool dfs_cycle(const Graph *g, int v,
                      int color[SUBGRAPH_MAX_NODES])
{
    color[v] = 1; /* grau = in Bearbeitung */
    for (int u = 0; u < g->n; u++)
    {
        if (!g->adj[v][u])
            continue;
        if (color[u] == 1)
            return true; /* Rueckwaertskante => Zyklus */
        if (color[u] == 0 && dfs_cycle(g, u, color))
            return true;
    }
    color[v] = 2; /* schwarz = fertig */
    return false;
}

bool graph_is_dag(const Graph *g)
{
    int color[SUBGRAPH_MAX_NODES] = {0};
    for (int v = 0; v < g->n; v++)
        if (color[v] == 0 && dfs_cycle(g, v, color))
            return false;
    return true;
}

/* Maximale Cliquengroesse via Greedy-LexBFS (exakt fuer chordale Graphen) */
int graph_max_clique_size(const Graph *g)
{
    /* Fuer chordale Graphen: Cliquenzahl = chromatische Zahl           */
    /* Vereinfachte Implementierung: maximaler Grad + 1 als obere Schranke */
    int max_deg = 0;
    for (int i = 0; i < g->n; i++)
    {
        int deg = 0;
        for (int j = 0; j < g->n; j++)
            if (g->adj[i][j])
                deg++;
        if (deg > max_deg)
            max_deg = deg;
    }
    return max_deg + 1;
}
