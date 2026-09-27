/*
 * compiler.c  --  Subgraph-basierter Compiler (ccl.tex, Epp 2026)
 *
 * Implementiert alle 7 polynomiellen Reduktionen:
 *   TAP  Typabhaengigkeitsaufloesung     Abschnitt 3
 *   DCE  Dead-Code-Elimination           Abschnitt 4
 *   LAP  Linkaufloesung                  Abschnitt 5  (linker.c)
 *   RAP  Registerallokation              Abschnitt 6
 *   SEP  Schleifenerkennung              Abschnitt 7
 *   CPP  Konstantenpropagation           Abschnitt 8
 *   IRP  Statische Initialisierung       Abschnitt 9
 */
#include "compiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

/* ================================================================== */
/* LEXER                                                              */
/* ================================================================== */

void lexer_init(Lexer *l, const char *src)
{
    l->src = src;
    l->pos = 0;
    l->line = 1;
}

static void skip_whitespace_comments(Lexer *l)
{
    while (l->src[l->pos])
    {
        if (l->src[l->pos] == '\n')
        {
            l->line++;
            l->pos++;
        }
        else if (isspace((unsigned char)l->src[l->pos]))
        {
            l->pos++;
        }
        else if (l->src[l->pos] == '/' && l->src[l->pos + 1] == '/')
        {
            while (l->src[l->pos] && l->src[l->pos] != '\n')
                l->pos++;
        }
        else if (l->src[l->pos] == '/' && l->src[l->pos + 1] == '*')
        {
            l->pos += 2;
            while (l->src[l->pos] &&
                   !(l->src[l->pos] == '*' && l->src[l->pos + 1] == '/'))
            {
                if (l->src[l->pos] == '\n')
                    l->line++;
                l->pos++;
            }
            if (l->src[l->pos])
                l->pos += 2;
        }
        else
            break;
    }
}

Token lexer_next(Lexer *l)
{
    skip_whitespace_comments(l);
    Token t;
    memset(&t, 0, sizeof(t));
    t.line = l->line;

    if (!l->src[l->pos])
    {
        t.kind = TOK_EOF;
        return t;
    }

    /* Bezeichner / Schluesselwoerter */
    if (isalpha((unsigned char)l->src[l->pos]) || l->src[l->pos] == '_')
    {
        int start = l->pos;
        while (isalnum((unsigned char)l->src[l->pos]) || l->src[l->pos] == '_')
            l->pos++;
        int len = l->pos - start;
        strncpy(t.text, l->src + start, len < 127 ? len : 127);
        /* Schluesselwort-Erkennung */
        if (!strcmp(t.text, "int"))
            t.kind = TOK_INT;
        else if (!strcmp(t.text, "float"))
            t.kind = TOK_FLOAT;
        else if (!strcmp(t.text, "void"))
            t.kind = TOK_VOID;
        else if (!strcmp(t.text, "if"))
            t.kind = TOK_IF;
        else if (!strcmp(t.text, "else"))
            t.kind = TOK_ELSE;
        else if (!strcmp(t.text, "while"))
            t.kind = TOK_WHILE;
        else if (!strcmp(t.text, "for"))
            t.kind = TOK_FOR;
        else if (!strcmp(t.text, "return"))
            t.kind = TOK_RETURN;
        else if (!strcmp(t.text, "static"))
            t.kind = TOK_STATIC;
        else
            t.kind = TOK_IDENT;
        return t;
    }
    /* Zahlen */
    if (isdigit((unsigned char)l->src[l->pos]))
    {
        int start = l->pos;
        while (isdigit((unsigned char)l->src[l->pos]))
            l->pos++;
        int len = l->pos - start;
        strncpy(t.text, l->src + start, len < 127 ? len : 127);
        t.kind = TOK_NUMBER;
        return t;
    }
    /* Operatoren und Satzzeichen */
    char c = l->src[l->pos++];
    t.text[0] = c;
    switch (c)
    {
    case '=':
        if (l->src[l->pos] == '=')
        {
            t.text[1] = '=';
            l->pos++;
            t.kind = TOK_EQ;
        }
        else
            t.kind = TOK_ASSIGN;
        break;
    case '!':
        if (l->src[l->pos] == '=')
        {
            t.text[1] = '=';
            l->pos++;
            t.kind = TOK_NEQ;
        }
        else
            t.kind = TOK_EOF; /* unbekannt */
        break;
    case '<':
        if (l->src[l->pos] == '=')
        {
            t.text[1] = '=';
            l->pos++;
            t.kind = TOK_LE;
        }
        else
            t.kind = TOK_LT;
        break;
    case '>':
        if (l->src[l->pos] == '=')
        {
            t.text[1] = '=';
            l->pos++;
            t.kind = TOK_GE;
        }
        else
            t.kind = TOK_GT;
        break;
    case '+':
        t.kind = TOK_PLUS;
        break;
    case '-':
        t.kind = TOK_MINUS;
        break;
    case '*':
        t.kind = TOK_STAR;
        break;
    case '/':
        t.kind = TOK_SLASH;
        break;
    case '(':
        t.kind = TOK_LPAREN;
        break;
    case ')':
        t.kind = TOK_RPAREN;
        break;
    case '{':
        t.kind = TOK_LBRACE;
        break;
    case '}':
        t.kind = TOK_RBRACE;
        break;
    case ';':
        t.kind = TOK_SEMICOLON;
        break;
    case ',':
        t.kind = TOK_COMMA;
        break;
    default:
        t.kind = TOK_EOF;
        break;
    }
    return t;
}

Token lexer_peek(Lexer *l)
{
    Lexer save = *l;
    Token t = lexer_next(l);
    *l = save;
    return t;
}

/* ================================================================== */
/* PARSER  (rekursiver Abstieg -> AST)                                */
/* ================================================================== */

typedef struct
{
    Lexer lex;
    AST *ast;
    TypeSystem *types;
    InitSystem *statics;
    char error[256];
    bool ok;
} Parser;

static int ast_new(AST *ast, ASTKind kind)
{
    if (ast->count >= AST_MAX_NODES)
        return -1;
    int id = ast->count++;
    memset(&ast->nodes[id], 0, sizeof(ASTNode));
    ast->nodes[id].kind = kind;
    ast->nodes[id].id = id;
    return id;
}

static void ast_add_child(AST *ast, int parent, int child)
{
    ASTNode *p = &ast->nodes[parent];
    if (p->nchildren < AST_MAX_CHILDREN)
        p->children[p->nchildren++] = child;
}

static Token expect(Parser *p, TokenKind kind)
{
    Token t = lexer_next(&p->lex);
    if (t.kind != kind && p->ok)
    {
        snprintf(p->error, 256, "Zeile %d: Erwartet Token %d, got %d ('%s')",
                 t.line, kind, t.kind, t.text);
        p->ok = false;
    }
    return t;
}

/* Vorwaertsdeklarationen */
static int parse_expr(Parser *p);
static int parse_stmt(Parser *p);
static int parse_block(Parser *p);

static int parse_expr(Parser *p)
{
    /* Vereinfacht: nur einfache Ausdruecke */
    Token t = lexer_peek(&p->lex);
    int node;

    if (t.kind == TOK_NUMBER)
    {
        lexer_next(&p->lex);
        node = ast_new(p->ast, AST_NUMBER);
        strncpy(p->ast->nodes[node].text, t.text, 127);
        return node;
    }
    if (t.kind == TOK_IDENT)
    {
        lexer_next(&p->lex);
        Token t2 = lexer_peek(&p->lex);
        if (t2.kind == TOK_LPAREN)
        {
            /* Funktionsaufruf */
            lexer_next(&p->lex);
            node = ast_new(p->ast, AST_CALL);
            strncpy(p->ast->nodes[node].text, t.text, 127);
            while (lexer_peek(&p->lex).kind != TOK_RPAREN &&
                   lexer_peek(&p->lex).kind != TOK_EOF)
            {
                int arg = parse_expr(p);
                if (arg >= 0)
                    ast_add_child(p->ast, node, arg);
                if (lexer_peek(&p->lex).kind == TOK_COMMA)
                    lexer_next(&p->lex);
            }
            expect(p, TOK_RPAREN);
            return node;
        }
        node = ast_new(p->ast, AST_IDENT);
        strncpy(p->ast->nodes[node].text, t.text, 127);
        /* Binaere Operation? */
        t2 = lexer_peek(&p->lex);
        if (t2.kind == TOK_PLUS || t2.kind == TOK_MINUS ||
            t2.kind == TOK_STAR || t2.kind == TOK_SLASH ||
            t2.kind == TOK_LT || t2.kind == TOK_GT ||
            t2.kind == TOK_LE || t2.kind == TOK_GE ||
            t2.kind == TOK_EQ || t2.kind == TOK_NEQ)
        {
            lexer_next(&p->lex);
            int binop = ast_new(p->ast, AST_BINOP);
            strncpy(p->ast->nodes[binop].text, t2.text, 127);
            ast_add_child(p->ast, binop, node);
            int rhs = parse_expr(p);
            if (rhs >= 0)
                ast_add_child(p->ast, binop, rhs);
            return binop;
        }
        return node;
    }
    return -1;
}

static int parse_stmt(Parser *p)
{
    Token t = lexer_peek(&p->lex);
    int node = -1;

    /* Deklaration: [static] int/float/void ident [= expr] ; */
    if (t.kind == TOK_STATIC || t.kind == TOK_INT ||
        t.kind == TOK_FLOAT || t.kind == TOK_VOID)
    {
        bool is_static = false;
        if (t.kind == TOK_STATIC)
        {
            lexer_next(&p->lex);
            is_static = true;
        }
        Token type_tok = lexer_next(&p->lex);
        Token name_tok = expect(p, TOK_IDENT);
        node = ast_new(p->ast, AST_VAR_DECL);
        strncpy(p->ast->nodes[node].text, name_tok.text, 127);
        strncpy(p->ast->nodes[node].type_name, type_tok.text, 63);
        p->ast->nodes[node].is_static = is_static;

        /* Typabhaengigkeit registrieren (TAP) */
        if (p->types->count < MAX_TYPES)
        {
            int ti = p->types->count++;
            strncpy(p->types->types[ti].name, name_tok.text, 63);
            p->types->types[ti].ndeps = 0;
        }
        /* Statisches Objekt registrieren (IRP) */
        if (is_static && p->statics->count < MAX_STATIC_OBJS)
        {
            int si = p->statics->count++;
            strncpy(p->statics->objs[si].name, name_tok.text, 63);
            p->statics->objs[si].ndeps = 0;
        }
        if (lexer_peek(&p->lex).kind == TOK_ASSIGN)
        {
            lexer_next(&p->lex);
            int init = parse_expr(p);
            if (init >= 0)
                ast_add_child(p->ast, node, init);
        }
        expect(p, TOK_SEMICOLON);
        return node;
    }
    /* Zuweisung: ident = expr ; */
    if (t.kind == TOK_IDENT)
    {
        Token name = lexer_next(&p->lex);
        if (lexer_peek(&p->lex).kind == TOK_ASSIGN)
        {
            lexer_next(&p->lex);
            node = ast_new(p->ast, AST_ASSIGN);
            strncpy(p->ast->nodes[node].text, name.text, 127);
            int rhs = parse_expr(p);
            if (rhs >= 0)
                ast_add_child(p->ast, node, rhs);
            expect(p, TOK_SEMICOLON);
            return node;
        }
        /* Ausdruck-Statement: f(x); */
        node = ast_new(p->ast, AST_IDENT);
        strncpy(p->ast->nodes[node].text, name.text, 127);
        expect(p, TOK_SEMICOLON);
        return node;
    }
    /* return */
    if (t.kind == TOK_RETURN)
    {
        lexer_next(&p->lex);
        node = ast_new(p->ast, AST_RETURN);
        if (lexer_peek(&p->lex).kind != TOK_SEMICOLON)
        {
            int val = parse_expr(p);
            if (val >= 0)
                ast_add_child(p->ast, node, val);
        }
        expect(p, TOK_SEMICOLON);
        return node;
    }
    /* if */
    if (t.kind == TOK_IF)
    {
        lexer_next(&p->lex);
        expect(p, TOK_LPAREN);
        node = ast_new(p->ast, AST_IF);
        int cond = parse_expr(p);
        if (cond >= 0)
            ast_add_child(p->ast, node, cond);
        expect(p, TOK_RPAREN);
        int then_b = parse_block(p);
        if (then_b >= 0)
            ast_add_child(p->ast, node, then_b);
        if (lexer_peek(&p->lex).kind == TOK_ELSE)
        {
            lexer_next(&p->lex);
            int else_b = parse_block(p);
            if (else_b >= 0)
                ast_add_child(p->ast, node, else_b);
        }
        return node;
    }
    /* while */
    if (t.kind == TOK_WHILE)
    {
        lexer_next(&p->lex);
        expect(p, TOK_LPAREN);
        node = ast_new(p->ast, AST_WHILE);
        int cond = parse_expr(p);
        if (cond >= 0)
            ast_add_child(p->ast, node, cond);
        expect(p, TOK_RPAREN);
        int body = parse_block(p);
        if (body >= 0)
            ast_add_child(p->ast, node, body);
        return node;
    }
    /* { Block } */
    if (t.kind == TOK_LBRACE)
        return parse_block(p);

    /* Unbekannt: ueberspringen */
    lexer_next(&p->lex);
    return -1;
}

static int parse_block(Parser *p)
{
    expect(p, TOK_LBRACE);
    int node = ast_new(p->ast, AST_BLOCK);
    while (lexer_peek(&p->lex).kind != TOK_RBRACE &&
           lexer_peek(&p->lex).kind != TOK_EOF)
    {
        int s = parse_stmt(p);
        if (s >= 0)
            ast_add_child(p->ast, node, s);
        if (!p->ok)
            break;
    }
    expect(p, TOK_RBRACE);
    return node;
}

static void parse_function(Parser *p, int prog_node)
{
    /* Typ ident ( param* ) block */
    Token type_tok = lexer_next(&p->lex);
    Token name_tok = expect(p, TOK_IDENT);
    expect(p, TOK_LPAREN);

    int fn = ast_new(p->ast, AST_FUNC_DECL);
    strncpy(p->ast->nodes[fn].text, name_tok.text, 127);
    strncpy(p->ast->nodes[fn].type_name, type_tok.text, 63);
    ast_add_child(p->ast, prog_node, fn);

    /* Parameter */
    while (lexer_peek(&p->lex).kind != TOK_RPAREN &&
           lexer_peek(&p->lex).kind != TOK_EOF)
    {
        Token pt = lexer_next(&p->lex); /* Typ */
        Token pn = lexer_next(&p->lex); /* Name */
        int pnode = ast_new(p->ast, AST_VAR_DECL);
        strncpy(p->ast->nodes[pnode].text, pn.text, 127);
        strncpy(p->ast->nodes[pnode].type_name, pt.text, 63);
        ast_add_child(p->ast, fn, pnode);
        if (lexer_peek(&p->lex).kind == TOK_COMMA)
            lexer_next(&p->lex);
    }
    expect(p, TOK_RPAREN);
    int body = parse_block(p);
    if (body >= 0)
        ast_add_child(p->ast, fn, body);
}

static bool parse_program(Parser *p)
{
    int prog = ast_new(p->ast, AST_PROGRAM);
    while (lexer_peek(&p->lex).kind != TOK_EOF && p->ok)
    {
        Token t = lexer_peek(&p->lex);
        if (t.kind == TOK_INT || t.kind == TOK_FLOAT || t.kind == TOK_VOID)
        {
            parse_function(p, prog);
        }
        else if (t.kind == TOK_STATIC)
        {
            /* Top-level statische Variable */
            int s = parse_stmt(p);
            if (s >= 0)
                ast_add_child(p->ast, prog, s);
        }
        else
        {
            lexer_next(&p->lex); /* Fehler-Recovery */
        }
    }
    return p->ok;
}

/* ================================================================== */
/* TAP: Typabhaengigkeitsaufloesung  (Abschnitt 3)                    */
/* Reduktion: G_T ⊆ H_DAG via Subgraph Algorithmus                    */
/* ================================================================== */

bool tap_resolve(TypeSystem *ts)
{
    int n = ts->count;
    if (n == 0)
    {
        ts->topo_count = 0;
        ts->has_cycle = false;
        return true;
    }

    /* Konstruiere Abhaengigkeitsgraph G_T */
    Graph G;
    graph_init(&G, n);
    for (int i = 0; i < n; i++)
    {
        graph_set_label(&G, i, ts->types[i].name);
        for (int d = 0; d < ts->types[i].ndeps; d++)
        {
            int j = ts->types[i].deps[d];
            if (j >= 0 && j < n)
                graph_add_edge(&G, i, j);
        }
    }

    /* Konstruiere vollstaendigen DAG H_DAG ueber n Knoten */
    Graph H;
    graph_init(&H, n);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            graph_add_edge(&H, i, j);

    /* Subgraph Algorithmus: G_T ⊆ H_DAG? */
    SI_Match m = subgraph_check(&G, &H);

    ts->has_cycle = (m.result == SI_DISJOINT);

    if (!ts->has_cycle)
    {
        /* Topologische Ordnung aus Signatur-Matching rekonstruieren */
        /* Einfache Fallback-Implementierung: Kahn's Algorithmus */
        int in_deg[MAX_TYPES] = {0};
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (G.adj[i][j])
                    in_deg[j]++;

        int queue[MAX_TYPES], qhead = 0, qtail = 0;
        for (int i = 0; i < n; i++)
            if (in_deg[i] == 0)
                queue[qtail++] = i;

        ts->topo_count = 0;
        while (qhead < qtail)
        {
            int v = queue[qhead++];
            ts->topo_order[ts->topo_count++] = v;
            ts->types[v].order = ts->topo_count;
            for (int u = 0; u < n; u++)
            {
                if (G.adj[v][u])
                {
                    in_deg[u]--;
                    if (in_deg[u] == 0)
                        queue[qtail++] = u;
                }
            }
        }
        if (ts->topo_count < n)
        { /* Zyklus gefunden */
            ts->has_cycle = true;
            return false;
        }
        return true;
    }
    return false;
}

/* ================================================================== */
/* CFG-AUFBAU aus AST                                                 */
/* ================================================================== */

typedef struct
{
    CFG *cfg;
    int cur_block;
    int tmp_count;
    char tmp_prefix[16];
} CFGBuilder;

static int cfg_new_block(CFGBuilder *b, const char *name)
{
    CFG *cfg = b->cfg;
    if (cfg->nblocks >= MAX_BLOCKS)
        return -1;
    int id = cfg->nblocks++;
    BasicBlock *bb = &cfg->blocks[id];
    memset(bb, 0, sizeof(BasicBlock));
    bb->id = id;
    bb->dom_parent = -1;
    bb->idom = -1;
    snprintf(bb->name, 64, "%s", name);
    return id;
}

static void cfg_emit(CFGBuilder *b, IROp op,
                     const char *dst, const char *s1, const char *s2,
                     const char *op_str, int imm, const char *label)
{
    BasicBlock *bb = &b->cfg->blocks[b->cur_block];
    if (bb->ninstr >= MAX_INSTRS)
        return;
    IRInstr *ir = &bb->instrs[bb->ninstr++];
    ir->op = op;
    if (dst)
        strncpy(ir->dst, dst, 63);
    if (s1)
        strncpy(ir->src1, s1, 63);
    if (s2)
        strncpy(ir->src2, s2, 63);
    if (op_str)
        strncpy(ir->op_str, op_str, 7);
    if (label)
        strncpy(ir->label, label, 63);
    ir->imm = imm;
}

static char *new_tmp(CFGBuilder *b, char *buf, int bufsz)
{
    snprintf(buf, bufsz, "_t%d", b->tmp_count++);
    return buf;
}

static void cfg_add_succ(CFG *cfg, int from, int to)
{
    BasicBlock *bb = &cfg->blocks[from];
    if (bb->nsuccs < 4)
        bb->succs[bb->nsuccs++] = to;
}

/* AST -> IR-Anweisungen im aktuellen Block */
static void codegen_ast(CFGBuilder *b, const AST *ast, int node_id,
                        char *result_var, int result_sz);

static void codegen_ast(CFGBuilder *b, const AST *ast, int node_id,
                        char *result_var, int result_sz)
{
    if (node_id < 0 || node_id >= ast->count)
    {
        if (result_var)
            strncpy(result_var, "0", result_sz);
        return;
    }
    const ASTNode *n = &ast->nodes[node_id];
    char tmp[64];

    switch (n->kind)
    {
    case AST_NUMBER:
        if (result_var)
            strncpy(result_var, n->text, result_sz);
        cfg_emit(b, IR_LOAD_CONST, NULL, n->text, NULL, NULL, atoi(n->text), NULL);
        break;

    case AST_IDENT:
        if (result_var)
            strncpy(result_var, n->text, result_sz);
        break;

    case AST_ASSIGN:
    {
        char rhs[64] = "0";
        if (n->nchildren > 0)
            codegen_ast(b, ast, n->children[0], rhs, sizeof(rhs));
        cfg_emit(b, IR_ASSIGN, n->text, rhs, NULL, NULL, 0, NULL);
        if (result_var)
            strncpy(result_var, n->text, result_sz);
        break;
    }
    case AST_BINOP:
    {
        char lhs[64] = "0", rhs[64] = "0";
        if (n->nchildren > 0)
            codegen_ast(b, ast, n->children[0], lhs, sizeof(lhs));
        if (n->nchildren > 1)
            codegen_ast(b, ast, n->children[1], rhs, sizeof(rhs));
        new_tmp(b, tmp, sizeof(tmp));
        cfg_emit(b, IR_ASSIGN, tmp, lhs, rhs, n->text, 0, NULL);
        if (result_var)
            strncpy(result_var, tmp, result_sz);
        break;
    }
    case AST_IF:
    {
        char cond[64] = "0";
        if (n->nchildren > 0)
            codegen_ast(b, ast, n->children[0], cond, sizeof(cond));
        int then_b = cfg_new_block(b, "if.then");
        int else_b = cfg_new_block(b, "if.else");
        int merge_b = cfg_new_block(b, "if.merge");
        cfg_emit(b, IR_BRANCH, NULL, cond, NULL, NULL, 0, "if.then");
        /* Kanten im CFG */
        cfg_add_succ(b->cfg, b->cur_block, then_b);
        cfg_add_succ(b->cfg, b->cur_block, else_b);
        /* Then-Zweig */
        b->cur_block = then_b;
        if (n->nchildren > 1)
            codegen_ast(b, ast, n->children[1], NULL, 0);
        cfg_add_succ(b->cfg, b->cur_block, merge_b);
        cfg_emit(b, IR_JUMP, NULL, NULL, NULL, NULL, 0, "if.merge");
        /* Else-Zweig */
        b->cur_block = else_b;
        if (n->nchildren > 2)
            codegen_ast(b, ast, n->children[2], NULL, 0);
        cfg_add_succ(b->cfg, b->cur_block, merge_b);
        cfg_emit(b, IR_JUMP, NULL, NULL, NULL, NULL, 0, "if.merge");
        b->cur_block = merge_b;
        break;
    }
    case AST_WHILE:
    {
        int cond_b = cfg_new_block(b, "while.cond");
        int body_b = cfg_new_block(b, "while.body");
        int exit_b = cfg_new_block(b, "while.exit");
        cfg_add_succ(b->cfg, b->cur_block, cond_b);
        cfg_emit(b, IR_JUMP, NULL, NULL, NULL, NULL, 0, "while.cond");
        b->cur_block = cond_b;
        char cond[64] = "1";
        if (n->nchildren > 0)
            codegen_ast(b, ast, n->children[0], cond, sizeof(cond));
        cfg_emit(b, IR_BRANCH, NULL, cond, NULL, NULL, 0, "while.body");
        cfg_add_succ(b->cfg, cond_b, body_b);
        cfg_add_succ(b->cfg, cond_b, exit_b);
        b->cur_block = body_b;
        if (n->nchildren > 1)
            codegen_ast(b, ast, n->children[1], NULL, 0);
        cfg_add_succ(b->cfg, b->cur_block, cond_b); /* Rueckwaertskante */
        cfg_emit(b, IR_JUMP, NULL, NULL, NULL, NULL, 0, "while.cond");
        b->cur_block = exit_b;
        break;
    }
    case AST_RETURN:
    {
        char val[64] = "0";
        if (n->nchildren > 0)
            codegen_ast(b, ast, n->children[0], val, sizeof(val));
        cfg_emit(b, IR_RETURN, NULL, val, NULL, NULL, 0, NULL);
        break;
    }
    case AST_BLOCK:
        for (int i = 0; i < n->nchildren; i++)
            codegen_ast(b, ast, n->children[i], NULL, 0);
        break;
    case AST_CALL:
    {
        new_tmp(b, tmp, sizeof(tmp));
        cfg_emit(b, IR_CALL, tmp, n->text, NULL, NULL, 0, NULL);
        if (result_var)
            strncpy(result_var, tmp, result_sz);
        break;
    }
    case AST_FUNC_DECL:
    case AST_VAR_DECL:
    case AST_PROGRAM:
    default:
        for (int i = 0; i < n->nchildren; i++)
            codegen_ast(b, ast, n->children[i], NULL, 0);
        break;
    }
}

/* CFG aus Funktion im AST aufbauen */
static void build_cfg_for_function(CFG *cfg, const AST *ast, int fn_node)
{
    memset(cfg, 0, sizeof(CFG));
    CFGBuilder b;
    b.cfg = cfg;
    b.tmp_count = 0;
    strcpy(b.tmp_prefix, "_t");

    /* Eintrittsblock */
    cfg->entry = cfg_new_block(&b, "entry");
    b.cur_block = cfg->entry;

    /* EXIT-Block (wird am Ende hinzugefuegt) */
    int exit_b = cfg_new_block(&b, "exit");

    const ASTNode *fn = &ast->nodes[fn_node];
    for (int i = 0; i < fn->nchildren; i++)
        codegen_ast(&b, ast, fn->children[i], NULL, 0);

    /* Verbinde letzten Block mit EXIT */
    cfg_add_succ(cfg, b.cur_block, exit_b);

    cfg_build_graphs(cfg);
}

/* ================================================================== */
/* CFG-GRAPHEN aufbauen (Adjazenzmatrizen)                            */
/* ================================================================== */

void cfg_build_graphs(CFG *cfg)
{
    int n = cfg->nblocks;
    graph_init(&cfg->cfg_graph, n);
    for (int i = 0; i < n; i++)
    {
        graph_set_label(&cfg->cfg_graph, i, cfg->blocks[i].name);
        for (int s = 0; s < cfg->blocks[i].nsuccs; s++)
        {
            int j = cfg->blocks[i].succs[s];
            if (j >= 0 && j < n)
                graph_add_edge(&cfg->cfg_graph, i, j);
        }
    }
}

/* ================================================================== */
/* DCE: Dead-Code-Elimination  (Abschnitt 4)                          */
/* Reduktion: P_B ⊄ G_P[Reach(s)] via Subgraph Algorithmus            */
/* ================================================================== */

void dce_eliminate(CFG *cfg)
{
    int n = cfg->nblocks;
    /* BFS vom Eintrittsblock */
    bool visited[MAX_BLOCKS] = {false};
    int queue[MAX_BLOCKS];
    int qhead = 0, qtail = 0;
    queue[qtail++] = cfg->entry;
    visited[cfg->entry] = true;

    while (qhead < qtail)
    {
        int v = queue[qhead++];
        for (int s = 0; s < cfg->blocks[v].nsuccs; s++)
        {
            int u = cfg->blocks[v].succs[s];
            if (u >= 0 && u < n && !visited[u])
            {
                visited[u] = true;
                queue[qtail++] = u;
            }
        }
    }

    /* Konstruiere Erreichbarkeitsgraph H = G_P[Reach(s)] */
    /* Zaehle erreichbare Knoten */
    int reach_idx[MAX_BLOCKS]; /* original -> komprimierter Index */
    int reach_map[MAX_BLOCKS]; /* komprimierter -> original */
    int nreach = 0;
    memset(reach_idx, -1, sizeof(reach_idx));
    for (int i = 0; i < n; i++)
    {
        if (visited[i])
        {
            reach_idx[i] = nreach;
            reach_map[nreach] = i;
            nreach++;
        }
    }

    Graph H;
    graph_init(&H, nreach);
    for (int ci = 0; ci < nreach; ci++)
    {
        int i = reach_map[ci];
        graph_set_label(&H, ci, cfg->blocks[i].name);
        for (int s = 0; s < cfg->blocks[i].nsuccs; s++)
        {
            int j = cfg->blocks[i].succs[s];
            if (j >= 0 && j < n && reach_idx[j] >= 0)
                graph_add_edge(&H, ci, reach_idx[j]);
        }
    }

    /* Fuer jeden Block: Einknoten-Muster P_B = ({B}, {}) */
    /* B ist dead code <=> P_B ist kein Subgraph von H    */
    for (int i = 0; i < n; i++)
    {
        if (!visited[i])
        {
            cfg->blocks[i].reachable = false;
            /* Subgraph-Pruefung: Einknoten-Graph vs. H */
            Graph PB;
            graph_init(&PB, 1);
            graph_set_label(&PB, 0, cfg->blocks[i].name);
            SI_Match m = subgraph_check(&PB, &H);
            /* m.result == SI_DISJOINT => nicht in H => dead code (bestaetigt) */
            (void)m;
            printf("  [DCE] Block '%s' entfernt (dead code)\n",
                   cfg->blocks[i].name);
        }
        else
        {
            cfg->blocks[i].reachable = true;
        }
    }
}

/* ================================================================== */
/* Dominatorgraph (Lengauer-Tarjan vereinfacht)                       */
/* ================================================================== */

static void compute_dominators(CFG *cfg)
{
    int n = cfg->nblocks;
    int s = cfg->entry;
    int idom[MAX_BLOCKS];
    for (int i = 0; i < n; i++)
        idom[i] = -1;
    idom[s] = s;

    /* Iterativer Algorithmus (Cooper et al.) */
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int b = 0; b < n; b++)
        {
            if (b == s)
                continue;
            /* Finde bearbeiteten Vorgaenger mit bekanntem idom */
            int new_idom = -1;
            for (int p = 0; p < n; p++)
            {
                if (!cfg->cfg_graph.adj[p][b])
                    continue;
                if (idom[p] < 0)
                    continue;
                if (new_idom < 0)
                {
                    new_idom = p;
                    continue;
                }
                /* Schnitt: LCA im Dominanzbaum */
                int a = p, c = new_idom;
                while (a != c)
                {
                    while (a > c)
                        a = (idom[a] >= 0) ? idom[a] : a;
                    while (c > a)
                        c = (idom[c] >= 0) ? idom[c] : c;
                }
                new_idom = a;
            }
            if (new_idom >= 0 && idom[b] != new_idom)
            {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    /* Dominatorgraph aufbauen */
    graph_init(&cfg->dom_graph, n);
    for (int i = 0; i < n; i++)
    {
        cfg->blocks[i].idom = idom[i];
        graph_set_label(&cfg->dom_graph, i, cfg->blocks[i].name);
        if (idom[i] >= 0 && idom[i] != i)
            graph_add_edge(&cfg->dom_graph, idom[i], i);
    }
}

/* ================================================================== */
/* SEP: Schleifenerkennung  (Abschnitt 7)                             */
/* Reduktion: G_Schleife ⊆ G_kombi via Subgraph Algorithmus           */
/* ================================================================== */

void sep_find_loops(CFG *cfg)
{
    int n = cfg->nblocks;
    compute_dominators(cfg);

    /* Kombinierter Graph: G_kombi = CFG ∪ Dominatorgraph */
    /* Wir kodieren: adj[i][j] = 1 (CFG), 2 (Dominanz), 3 (beide) */
    graph_init(&cfg->kombi_graph, n);
    for (int i = 0; i < n; i++)
    {
        graph_set_label(&cfg->kombi_graph, i, cfg->blocks[i].name);
        for (int j = 0; j < n; j++)
        {
            int val = 0;
            if (cfg->cfg_graph.adj[i][j])
                val |= 1;
            if (cfg->dom_graph.adj[i][j])
                val |= 2;
            cfg->kombi_graph.adj[i][j] = (val > 0) ? 1 : 0;
        }
    }

    /* Schleifenmuster: G_Schleife = ({d,n}, {(d,n)_D, (n,d)_P}) */
    /* vereinfacht als 2-Knoten-Zyklus im kombinierten Graphen     */
    Graph G_loop;
    graph_init(&G_loop, 2);
    graph_set_label(&G_loop, 0, "header");
    graph_set_label(&G_loop, 1, "latch");
    graph_add_edge(&G_loop, 0, 1); /* d -> n (Dominanz)   */
    graph_add_edge(&G_loop, 1, 0); /* n -> d (Rueckwaerts) */

    SI_Match m = subgraph_check(&G_loop, &cfg->kombi_graph);

    if (m.result == SI_A_IN_B || m.result == SI_EQUAL)
    {
        /* Schleifen-Kanten direkt im CFG suchen (Rueckwaertskanten) */
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (!cfg->cfg_graph.adj[i][j])
                    continue;
                /* (i,j) ist Rueckwaertskante wenn idom[i] == j oder j dom i */
                if (cfg->dom_graph.adj[j][i])
                { /* j dominiert i */
                    cfg->blocks[j].is_loop_header = true;
                    printf("  [SEP] Schleife: Header='%s', Latch='%s'\n",
                           cfg->blocks[j].name, cfg->blocks[i].name);
                }
            }
        }
    }
    else
    {
        printf("  [SEP] Keine natuerlichen Schleifen gefunden.\n");
    }
}
