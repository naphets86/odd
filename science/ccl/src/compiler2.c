/*
 * compiler2.c  --  SSA, CPP, RAP, IRP, Codegen, compile()
 *
 * Fortsetzung von compiler.c
 */
#include "compiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

/* ================================================================== */
/* SSA-TRANSFORMATION (vereinfacht: Rename + Phi-Einfuegung)          */
/* ================================================================== */

void ssa_transform(CFG *cfg, SSAForm *ssa)
{
    memset(ssa, 0, sizeof(SSAForm));
    int n = cfg->nblocks;
    int var_counter = 0;

    /* Sammle alle Variablen aus IR-Anweisungen */
    for (int b = 0; b < n; b++)
    {
        if (!cfg->blocks[b].reachable)
            continue;
        BasicBlock *bb = &cfg->blocks[b];
        for (int k = 0; k < bb->ninstr; k++)
        {
            IRInstr *ir = &bb->instrs[k];
            if (ir->dst[0] && ssa->nvars < MAX_SSA_VARS)
            {
                /* Pruefe ob Variable schon bekannt */
                bool found = false;
                for (int v = 0; v < ssa->nvars; v++)
                {
                    if (!strcmp(ssa->vars[v].name, ir->dst))
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    int vi = ssa->nvars++;
                    snprintf(ssa->vars[vi].name, 64, "%s_%d",
                             ir->dst, var_counter++);
                    ssa->vars[vi].def_block = b;
                    ssa->vars[vi].is_const = (ir->op == IR_LOAD_CONST);
                    ssa->vars[vi].const_val = ir->imm;
                }
            }
        }
    }

    /* SSA-Abhaengigkeitsgraph */
    int nv = ssa->nvars;
    graph_init(&ssa->ssa_graph, nv);
    for (int i = 0; i < nv; i++)
        graph_set_label(&ssa->ssa_graph, i, ssa->vars[i].name);

    /* Einfache Abhaengigkeit: Variable j benutzt Variable i */
    for (int b = 0; b < n; b++)
    {
        BasicBlock *bb = &cfg->blocks[b];
        for (int k = 0; k < bb->ninstr; k++)
        {
            IRInstr *ir = &bb->instrs[k];
            if (!ir->dst[0])
                continue;
            /* Suche dst in SSA-Variablen */
            int dst_v = -1;
            for (int v = 0; v < nv; v++)
                if (strncmp(ssa->vars[v].name, ir->dst,
                            strlen(ir->dst)) == 0)
                {
                    dst_v = v;
                    break;
                }
            if (dst_v < 0)
                continue;
            /* Suche src1, src2 */
            for (int v = 0; v < nv; v++)
            {
                if (ir->src1[0] &&
                    strncmp(ssa->vars[v].name, ir->src1,
                            strlen(ir->src1)) == 0)
                    graph_add_edge(&ssa->ssa_graph, v, dst_v);
                if (ir->src2[0] &&
                    strncmp(ssa->vars[v].name, ir->src2,
                            strlen(ir->src2)) == 0)
                    graph_add_edge(&ssa->ssa_graph, v, dst_v);
            }
        }
    }
}

/* ================================================================== */
/* CPP: Konstantenpropagation  (Abschnitt 8)                          */
/* Reduktion: G_SSA[In(d_j)] ⊆ G_K via Subgraph Algorithmus           */
/* Fixpunktiteration, O(n^4)                                          */
/* ================================================================== */

void cpp_propagate(SSAForm *ssa)
{
    int nv = ssa->nvars;
    /* Initialisiere Konstantenmenge K = {d_j | d_j.is_const} */
    for (int v = 0; v < nv; v++)
    {
        ssa->is_const[v] = ssa->vars[v].is_const;
        ssa->const_val[v] = ssa->vars[v].const_val;
    }

    bool changed = true;
    int iteration = 0;
    while (changed)
    {
        changed = false;
        iteration++;

        /* Konstruiere G_K = SSA-Graph[K] */
        /* Zaehle Konstantenknoten */
        int k_idx[MAX_SSA_VARS]; /* original -> K-Index */
        int k_map[MAX_SSA_VARS]; /* K-Index -> original */
        int nk = 0;
        memset(k_idx, -1, sizeof(k_idx));
        for (int v = 0; v < nv; v++)
        {
            if (ssa->is_const[v])
            {
                k_idx[v] = nk;
                k_map[nk] = v;
                nk++;
            }
        }
        if (nk == 0)
            break;

        graph_init(&ssa->const_graph, nk);
        for (int ci = 0; ci < nk; ci++)
        {
            int i = k_map[ci];
            graph_set_label(&ssa->const_graph, ci, ssa->vars[i].name);
            for (int cj = 0; cj < nk; cj++)
            {
                int j = k_map[cj];
                if (ssa->ssa_graph.adj[i][j])
                    graph_add_edge(&ssa->const_graph, ci, cj);
            }
        }

        /* Pruefe fuer jede nicht-konstante Variable d_j */
        for (int dj = 0; dj < nv; dj++)
        {
            if (ssa->is_const[dj])
                continue;

            /* Sammle Eingaben In(d_j): alle d_i mit (d_i, d_j) in SSA */
            int inputs[MAX_SSA_VARS], ninputs = 0;
            for (int di = 0; di < nv; di++)
                if (ssa->ssa_graph.adj[di][dj])
                    inputs[ninputs++] = di;

            if (ninputs == 0)
                continue;

            /* Konstruiere G_SSA[In(d_j)] */
            /* Pruefe: alle Eingaben konstant? */
            bool all_const = true;
            for (int q = 0; q < ninputs; q++)
            {
                if (!ssa->is_const[inputs[q]])
                {
                    all_const = false;
                    break;
                }
            }
            if (!all_const)
                continue;

            /* G_SSA[In(d_j)]: Teilgraph der Eingaben */
            Graph G_in;
            graph_init(&G_in, ninputs);
            for (int qi = 0; qi < ninputs; qi++)
            {
                graph_set_label(&G_in, qi, ssa->vars[inputs[qi]].name);
                for (int qj = 0; qj < ninputs; qj++)
                {
                    if (ssa->ssa_graph.adj[inputs[qi]][inputs[qj]])
                        graph_add_edge(&G_in, qi, qj);
                }
            }

            /* Subgraph-Pruefung: G_in ⊆ G_K? */
            SI_Match m = subgraph_check(&G_in, &ssa->const_graph);
            if (m.result == SI_A_IN_B || m.result == SI_EQUAL ||
                m.result == SI_B_IN_A)
            {
                /* d_j ist jetzt konstant propagierbar */
                ssa->is_const[dj] = true;
                /* Berechne Konstantenwert (vereinfacht: erster Input) */
                ssa->const_val[dj] = ssa->const_val[inputs[0]];
                printf("  [CPP] Variable '%s' = %d (propagiert)\n",
                       ssa->vars[dj].name, ssa->const_val[dj]);
                changed = true;
            }
        }
    }
    printf("  [CPP] Fixpunkt nach %d Iteration(en)\n", iteration);
}

/* ================================================================== */
/* RAP: Registerallokation  (Abschnitt 6)                             */
/* Reduktion: K_omega ⊆ K_k via Subgraph Algorithmus (chordal)        */
/* Greedy-Faerbung in perfekter Eliminationsreihenfolge               */
/* ================================================================== */

void rap_allocate(const SSAForm *ssa, int k_regs, RegAlloc *out)
{
    int nv = ssa->nvars;
    memset(out, 0, sizeof(RegAlloc));
    for (int i = 0; i < MAX_VARS; i++)
        out->var_to_reg[i] = -1;

    if (nv == 0)
        return;

    /* Interferenzgraph G_I: Variablen, die gleichzeitig leben */
    /* Vereinfachung: Variablen im selben Block interferieren   */
    Graph G_I;
    graph_init(&G_I, nv);
    for (int i = 0; i < nv; i++)
    {
        graph_set_label(&G_I, i, ssa->vars[i].name);
        for (int j = 0; j < nv; j++)
        {
            if (i == j)
                continue;
            if (ssa->vars[i].def_block == ssa->vars[j].def_block)
                graph_add_edge(&G_I, i, j);
        }
    }

    /* Bestimme Cliquenzahl omega via max_clique_size */
    int omega = graph_max_clique_size(&G_I);

    /* Konstruiere K_omega und K_k */
    Graph K_omega_g, K_k;
    int n_omega = (omega < nv) ? omega : nv;
    int n_k = (k_regs < SUBGRAPH_MAX_NODES) ? k_regs : SUBGRAPH_MAX_NODES - 1;
    graph_init(&K_omega_g, n_omega);
    graph_init(&K_k, n_k);
    /* Vollstaendige Cliquen */
    for (int i = 0; i < n_omega; i++)
        for (int j = 0; j < n_omega; j++)
            if (i != j)
                graph_add_edge(&K_omega_g, i, j);
    for (int i = 0; i < n_k; i++)
        for (int j = 0; j < n_k; j++)
            if (i != j)
                graph_add_edge(&K_k, i, j);

    /* Subgraph-Pruefung: K_omega ⊆ K_k? */
    SI_Match m = subgraph_check(&K_omega_g, &K_k);
    bool feasible = (m.result == SI_A_IN_B || m.result == SI_EQUAL);

    if (!feasible)
    {
        printf("  [RAP] %d Register reichen nicht (benoetigt >= %d), "
               "Spilling erforderlich\n",
               k_regs, omega);
    }

    /* Greedy-Faerbung (optimal fuer chordale Graphen) */
    int color[MAX_SSA_VARS];
    memset(color, -1, sizeof(color));

    for (int i = 0; i < nv; i++)
    {
        bool used[MAX_REGS] = {false};
        for (int j = 0; j < nv; j++)
        {
            if (G_I.adj[i][j] && color[j] >= 0 && color[j] < MAX_REGS)
                used[color[j]] = true;
        }
        int chosen = -1;
        for (int c = 0; c < k_regs; c++)
        {
            if (!used[c])
            {
                chosen = c;
                break;
            }
        }
        color[i] = chosen; /* -1 = spill */
        out->var_to_reg[i] = chosen;
        if (chosen >= 0)
            printf("  [RAP] Var '%s' -> %%r%d\n", ssa->vars[i].name, chosen);
        else
            printf("  [RAP] Var '%s' -> SPILL (Stack)\n", ssa->vars[i].name);
    }
    out->nregs_used = omega < k_regs ? omega : k_regs;
}

/* ================================================================== */
/* IRP: Statische Initialisierungsreihenfolge  (Abschnitt 9)          */
/* Reduktion: G_S ⊆ H_DAG via Subgraph Algorithmus                    */
/* ================================================================== */

void irp_resolve(InitSystem *is)
{
    int n = is->count;
    is->has_siof = false;
    is->order_count = 0;
    if (n == 0)
        return;

    /* Konstruiere Initialisierungsgraph G_S */
    Graph G_S;
    graph_init(&G_S, n);
    for (int i = 0; i < n; i++)
    {
        graph_set_label(&G_S, i, is->objs[i].name);
        for (int d = 0; d < is->objs[i].ndeps; d++)
        {
            int j = is->objs[i].deps[d];
            if (j >= 0 && j < n)
                graph_add_edge(&G_S, i, j);
        }
    }

    /* Vollstaendiger DAG H_DAG */
    Graph H_DAG;
    graph_init(&H_DAG, n);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            graph_add_edge(&H_DAG, i, j);

    /* Subgraph-Pruefung: G_S ⊆ H_DAG? */
    SI_Match m = subgraph_check(&G_S, &H_DAG);

    if (m.result == SI_DISJOINT)
    {
        /* Zyklus: Static Initialization Order Fiasco */
        is->has_siof = true;
        printf("  [IRP] WARNUNG: Static Initialization Order Fiasco!\n");
        printf("  [IRP] Zyklus bei LCS-Wert=%d, Rotation=%d\n",
               m.lcs_value, m.best_rotation);
        /* Betroffene Objekte ausgeben */
        for (int i = 0; i < n; i++)
            if (m.mapping[i] < 0)
                printf("  [IRP] Zyklus betrifft: '%s'\n", is->objs[i].name);
        return;
    }

    /* Korrekte Reihenfolge: Kahn's Algorithmus (topologische Sortierung) */
    int in_deg[MAX_STATIC_OBJS] = {0};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (G_S.adj[i][j])
                in_deg[j]++;

    int queue[MAX_STATIC_OBJS], qh = 0, qt = 0;
    for (int i = 0; i < n; i++)
        if (in_deg[i] == 0)
            queue[qt++] = i;

    while (qh < qt)
    {
        int v = queue[qh++];
        is->order[is->order_count++] = v;
        is->objs[v].init_order = is->order_count;
        for (int u = 0; u < n; u++)
        {
            if (G_S.adj[v][u])
            {
                in_deg[u]--;
                if (in_deg[u] == 0)
                    queue[qt++] = u;
            }
        }
    }

    printf("  [IRP] Initialisierungsreihenfolge:\n");
    for (int k = 0; k < is->order_count; k++)
        printf("    %d. %s\n", k + 1, is->objs[is->order[k]].name);
}

/* ================================================================== */
/* CODEGENERATOR  (x86-64 AT&T-Syntax)                                */
/* ================================================================== */

static void emit(CodeBuffer *cb, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int written = vsnprintf(cb->text + cb->pos,
                            sizeof(cb->text) - cb->pos, fmt, ap);
    va_end(ap);
    if (written > 0)
        cb->pos += written;
}

/* Registername nach Konvention */
static const char *reg_name(int r, int size_bytes)
{
    static const char *regs64[] = {
        "%rax", "%rbx", "%rcx", "%rdx",
        "%rsi", "%rdi", "%r8", "%r9",
        "%r10", "%r11", "%r12", "%r13",
        "%r14", "%r15"};
    static const char *regs32[] = {
        "%eax", "%ebx", "%ecx", "%edx",
        "%esi", "%edi", "%r8d", "%r9d",
        "%r10d", "%r11d", "%r12d", "%r13d",
        "%r14d", "%r15d"};
    if (r < 0 || r >= 14)
        return "%rsp"; /* Spill -> Stack */
    return (size_bytes == 8) ? regs64[r] : regs32[r];
}

static int find_var_reg(const SSAForm *ssa, const RegAlloc *ra,
                        const char *name)
{
    for (int v = 0; v < ssa->nvars; v++)
    {
        if (strncmp(ssa->vars[v].name, name, strlen(name)) == 0)
            return ra->var_to_reg[v];
    }
    return -1;
}

void codegen_function(const CFG *cfg, const SSAForm *ssa,
                      const RegAlloc *ra, const char *func_name,
                      CodeBuffer *out)
{
    emit(out, "\t.text\n");
    emit(out, "\t.globl %s\n", func_name);
    emit(out, "\t.type %s, @function\n", func_name);
    emit(out, "%s:\n", func_name);

    /* Prolog */
    emit(out, "\tpushq\t%%rbp\n");
    emit(out, "\tmovq\t%%rsp, %%rbp\n");
    /* Stackplatz fuer Spills (vereinfacht: 128 Byte) */
    emit(out, "\tsubq\t$128, %%rsp\n");

    int spill_offset = -8; /* naechster Stack-Offset fuer Spills */

    for (int b = 0; b < cfg->nblocks; b++)
    {
        if (!cfg->blocks[b].reachable)
            continue;
        const BasicBlock *bb = &cfg->blocks[b];

        emit(out, ".%s:\n", bb->name);

        for (int k = 0; k < bb->ninstr; k++)
        {
            const IRInstr *ir = &bb->instrs[k];

            switch (ir->op)
            {
            case IR_LOAD_CONST:
                if (ir->dst[0])
                {
                    int r = find_var_reg(ssa, ra, ir->dst);
                    if (r >= 0)
                        emit(out, "\tmovl\t$%d, %s\n",
                             ir->imm, reg_name(r, 4));
                    else
                        emit(out, "\tmovl\t$%d, %d(%%rbp)\n",
                             ir->imm, spill_offset -= 4);
                }
                break;

            case IR_COPY:
            case IR_ASSIGN:
            {
                if (!ir->dst[0])
                    break;
                int rd = find_var_reg(ssa, ra, ir->dst);
                int rs = find_var_reg(ssa, ra, ir->src1);

                if (ir->src2[0] == '\0')
                {
                    /* Einfache Zuweisung */
                    if (rs >= 0 && rd >= 0)
                        emit(out, "\tmovl\t%s, %s\n",
                             reg_name(rs, 4), reg_name(rd, 4));
                    else if (rd >= 0)
                        emit(out, "\tmovl\t$0, %s\n", reg_name(rd, 4));
                }
                else
                {
                    /* Binaere Operation */
                    int rs2 = find_var_reg(ssa, ra, ir->src2);
                    if (rd >= 0 && rs >= 0)
                    {
                        emit(out, "\tmovl\t%s, %s\n",
                             reg_name(rs, 4), reg_name(rd, 4));
                        if (!strcmp(ir->op_str, "+"))
                            emit(out, "\taddl\t%s, %s\n",
                                 rs2 >= 0 ? reg_name(rs2, 4) : "$0",
                                 reg_name(rd, 4));
                        else if (!strcmp(ir->op_str, "-"))
                            emit(out, "\tsubl\t%s, %s\n",
                                 rs2 >= 0 ? reg_name(rs2, 4) : "$0",
                                 reg_name(rd, 4));
                        else if (!strcmp(ir->op_str, "*"))
                        {
                            if (rs2 >= 0)
                                emit(out, "\timull\t%s, %s\n",
                                     reg_name(rs2, 4), reg_name(rd, 4));
                        }
                        else if (!strcmp(ir->op_str, "/"))
                        {
                            emit(out, "\tmovl\t%s, %%eax\n", reg_name(rs, 4));
                            emit(out, "\tcdq\n");
                            if (rs2 >= 0)
                                emit(out, "\tidivl\t%s\n", reg_name(rs2, 4));
                            emit(out, "\tmovl\t%%eax, %s\n", reg_name(rd, 4));
                        }
                        else if (!strcmp(ir->op_str, "<"))
                            emit(out, "\tcmpl\t%s, %s\n\tsetl\t%%al\n"
                                      "\tmovzbl\t%%al, %s\n",
                                 rs2 >= 0 ? reg_name(rs2, 4) : "$0",
                                 reg_name(rs, 4), reg_name(rd, 4));
                        else if (!strcmp(ir->op_str, ">"))
                            emit(out, "\tcmpl\t%s, %s\n\tsetg\t%%al\n"
                                      "\tmovzbl\t%%al, %s\n",
                                 rs2 >= 0 ? reg_name(rs2, 4) : "$0",
                                 reg_name(rs, 4), reg_name(rd, 4));
                    }
                }
                break;
            }
            case IR_BRANCH:
            {
                int rc = find_var_reg(ssa, ra, ir->src1);
                if (rc >= 0)
                    emit(out, "\ttestl\t%s, %s\n\tjne\t.%s\n",
                         reg_name(rc, 4), reg_name(rc, 4), ir->label);
                break;
            }
            case IR_JUMP:
                emit(out, "\tjmp\t.%s\n", ir->label);
                break;

            case IR_CALL:
                emit(out, "\tcall\t%s\n", ir->src1);
                if (ir->dst[0])
                {
                    int rd = find_var_reg(ssa, ra, ir->dst);
                    if (rd >= 0)
                        emit(out, "\tmovl\t%%eax, %s\n", reg_name(rd, 4));
                }
                break;

            case IR_RETURN:
            {
                int rv = find_var_reg(ssa, ra, ir->src1);
                if (rv >= 0)
                    emit(out, "\tmovl\t%s, %%eax\n", reg_name(rv, 4));
                else
                    emit(out, "\tmovl\t$0, %%eax\n");
                emit(out, "\tleave\n\tret\n");
                break;
            }
            case IR_PHI:
            case IR_NOP:
            default:
                break;
            }
        }
    }

    /* Epilog (falls kein explizites return) */
    emit(out, "\tleave\n");
    emit(out, "\tret\n");
    emit(out, "\t.size %s, .-%s\n\n", func_name, func_name);
}

/* ================================================================== */
/* OBJEKTDATEI aufbauen                                               */
/* ================================================================== */

static void objfile_build(ObjFile *obj, const char *filename,
                          const CodeBuffer *cb,
                          const TypeSystem *ts)
{
    strncpy(obj->name, filename, 63);
    /* Symboltabelle aus Typsystem */
    for (int i = 0; i < ts->topo_count && obj->nsyms < MAX_SYMBOLS; i++)
    {
        int ti = ts->topo_order[i];
        Symbol *s = &obj->syms[obj->nsyms++];
        strncpy(s->name, ts->types[ti].name, 63);
        s->kind = SYM_DEFINED;
        s->offset = 0;
    }
    /* ASM-Text */
    strncpy(obj->asm_text, cb->text,
            sizeof(obj->asm_text) - 1);
    /* "Maschinencode" = ASM-Text als Bytes (vereinfacht) */
    obj->code_len = (int)strlen(cb->text);
    if (obj->code_len > MAX_OBJ_CODE)
        obj->code_len = MAX_OBJ_CODE - 1;
    memcpy(obj->code, cb->text, obj->code_len);
}

/* ================================================================== */
/* COMPILE: Hauptfunktion                                             */
/* ================================================================== */

bool compile(const char *source, const char *filename, ObjFile *obj)
{
    CompileUnit cu;
    memset(&cu, 0, sizeof(cu));
    cu.ok = true;

    printf("\n=== Kompiliere: %s ===\n", filename);

    /* --- Lexer + Parser --- */
    AST ast;
    memset(&ast, 0, sizeof(AST));
    Parser p;
    p.ast = &ast;
    p.types = &cu.types;
    p.statics = &cu.statics;
    p.ok = true;
    lexer_init(&p.lex, source);
    parse_program(&p);
    if (!p.ok)
    {
        fprintf(stderr, "Parserfehler: %s\n", p.error);
        return false;
    }
    printf("  [Parser] %d AST-Knoten\n", ast.count);

    /* --- TAP: Typabhaengigkeitsaufloesung --- */
    printf("  [TAP] Loesung Typabhaengigkeiten (%d Typen)...\n",
           cu.types.count);
    bool tap_ok = tap_resolve(&cu.types);
    if (!tap_ok)
    {
        fprintf(stderr, "  [TAP] FEHLER: Zirkulaere Typabhaengigkeit!\n");
        /* Fahre trotzdem fort (Forward-Deklarationen moeglich) */
    }
    else
    {
        printf("  [TAP] OK: Typreihenfolge bestimmt\n");
    }

    /* --- CFG aus erster Funktion aufbauen --- */
    /* Suche erste Funktionsdeklaration im AST */
    int fn_node = -1;
    const char *fn_name = "main";
    for (int i = 0; i < ast.count; i++)
    {
        if (ast.nodes[i].kind == AST_FUNC_DECL)
        {
            fn_node = i;
            fn_name = ast.nodes[i].text;
            break;
        }
    }

    if (fn_node >= 0)
    {
        build_cfg_for_function(&cu.cfg, &ast, fn_node);
        printf("  [CFG] %d Basisbloecke aufgebaut\n", cu.cfg.nblocks);

        /* --- DCE: Dead-Code-Elimination --- */
        printf("  [DCE] Dead-Code-Elimination...\n");
        dce_eliminate(&cu.cfg);

        /* --- SEP: Schleifenerkennung --- */
        printf("  [SEP] Schleifenerkennung...\n");
        sep_find_loops(&cu.cfg);

        /* --- SSA-Transformation --- */
        ssa_transform(&cu.cfg, &cu.ssa);
        printf("  [SSA] %d SSA-Variablen\n", cu.ssa.nvars);

        /* --- CPP: Konstantenpropagation --- */
        printf("  [CPP] Konstantenpropagation...\n");
        cpp_propagate(&cu.ssa);

        /* --- RAP: Registerallokation (8 Register) --- */
        printf("  [RAP] Registerallokation (8 Register)...\n");
        rap_allocate(&cu.ssa, 8, &cu.regalloc);

        /* --- Codegenerierung --- */
        printf("  [CG]  Codegenerierung fuer '%s'...\n", fn_name);
        codegen_function(&cu.cfg, &cu.ssa, &cu.regalloc,
                         fn_name, &cu.codebuf);
    }
    else
    {
        printf("  [CG]  Keine Funktion gefunden; leere Objektdatei\n");
    }

    /* --- IRP: Statische Initialisierung --- */
    if (cu.statics.count > 0)
    {
        printf("  [IRP] Statische Initialisierungsreihenfolge (%d Objekte)...\n",
               cu.statics.count);
        irp_resolve(&cu.statics);
    }

    /* --- Objektdatei --- */
    objfile_build(obj, filename, &cu.codebuf, &cu.types);
    printf("  [OBJ] Objektdatei '%s' erstellt (%d Bytes ASM)\n",
           filename, cu.codebuf.pos);

    return true;
}
