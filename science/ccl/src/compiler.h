/*
 * compiler.h  --  Subgraph-basierter Compiler (ccl.tex, Epp 2026)
 *
 * Pipeline:
 *   Quelltext -> Lexer -> Parser (AST) -> TAP -> CFG -> DCE
 *             -> SSA -> CPP -> SEP -> RAP -> Codegen -> Objektdatei
 *
 * Jede Phase nutzt den Subgraph Algorithmus als Kernoperation.
 */
#ifndef COMPILER_H
#define COMPILER_H

#include "subgraph.h"
#include <stdbool.h>
#include <stdint.h>

/* ================================================================== */
/* 1. TOKEN / LEXER                                                   */
/* ================================================================== */

typedef enum
{
    TOK_EOF = 0,
    TOK_INT,
    TOK_FLOAT,
    TOK_VOID, /* Typen         */
    TOK_IF,
    TOK_ELSE,
    TOK_WHILE,
    TOK_FOR, /* Kontrollfluss */
    TOK_RETURN,
    TOK_IDENT,  /* Bezeichner    */
    TOK_NUMBER, /* Literale      */
    TOK_ASSIGN, /* =             */
    TOK_PLUS,
    TOK_MINUS,
    TOK_STAR,
    TOK_SLASH,
    TOK_EQ,
    TOK_NEQ,
    TOK_LT,
    TOK_GT,
    TOK_LE,
    TOK_GE,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_SEMICOLON,
    TOK_COMMA,
    TOK_STATIC /* fuer IRP      */
} TokenKind;

typedef struct
{
    TokenKind kind;
    char text[128];
    int line;
} Token;

/* Lexer-Zustand */
typedef struct
{
    const char *src;
    int pos;
    int line;
} Lexer;

void lexer_init(Lexer *l, const char *src);
Token lexer_next(Lexer *l);
Token lexer_peek(Lexer *l);

/* ================================================================== */
/* 2. AST                                                             */
/* ================================================================== */

typedef enum
{
    AST_PROGRAM,
    AST_FUNC_DECL,
    AST_VAR_DECL, /* int x; oder static int x; */
    AST_BLOCK,
    AST_IF,
    AST_WHILE,
    AST_FOR,
    AST_RETURN,
    AST_ASSIGN,
    AST_BINOP,
    AST_IDENT,
    AST_NUMBER,
    AST_CALL
} ASTKind;

#define AST_MAX_CHILDREN 16
#define AST_MAX_NODES 512

typedef struct ASTNode
{
    ASTKind kind;
    char text[128];     /* Bezeichner / Operator / Wert */
    char type_name[64]; /* Typname (fuer Deklarationen) */
    bool is_static;     /* fuer IRP */
    int children[AST_MAX_CHILDREN];
    int nchildren;
    int id; /* Index im AST-Array */
} ASTNode;

typedef struct
{
    ASTNode nodes[AST_MAX_NODES];
    int count;
} AST;

/* ================================================================== */
/* 3. TYPEN-ABHAENGIGKEITSGRAPH  (TAP, Abschnitt 3 ccl.tex)           */
/* ================================================================== */

#define MAX_TYPES 64

typedef struct
{
    char name[64];
    int deps[MAX_TYPES]; /* Indices der benoetigten Typen */
    int ndeps;
    int order; /* topologische Ordnung nach TAP */
} TypeInfo;

typedef struct
{
    TypeInfo types[MAX_TYPES];
    int count;
    int topo_order[MAX_TYPES]; /* Ergebnis: azyklische Reihenfolge */
    int topo_count;
    bool has_cycle;
} TypeSystem;

bool tap_resolve(TypeSystem *ts); /* Reduktion auf SI, O(n^3) */

/* ================================================================== */
/* 4. KONTROLLFLUSSGRAPH  (CFG, DCE, SEP)                             */
/* ================================================================== */

#define MAX_BLOCKS 128
#define MAX_INSTRS 256

typedef enum
{
    IR_ASSIGN,     /* x = y op z   */
    IR_COPY,       /* x = y        */
    IR_LOAD_CONST, /* x = #imm     */
    IR_JUMP,       /* goto label   */
    IR_BRANCH,     /* if x goto L1 else L2 */
    IR_CALL,       /* x = f(args)  */
    IR_RETURN,     /* return x     */
    IR_PHI,        /* x = phi(y,z) -- SSA */
    IR_NOP
} IROp;

typedef struct
{
    IROp op;
    char dst[64];
    char src1[64];
    char src2[64];
    char op_str[8]; /* "+", "-", "*", "/" */
    int imm;
    char label[64]; /* Sprungziel */
    bool is_const;  /* fuer CPP */
    int const_val;
} IRInstr;

typedef struct
{
    char name[64];
    int id;
    IRInstr instrs[MAX_INSTRS];
    int ninstr;
    int succs[4]; /* Nachfolger im CFG */
    int nsuccs;
    bool reachable;      /* wird von DCE gesetzt */
    bool is_loop_header; /* wird von SEP gesetzt */
    int dom_parent;      /* Dominanzbaum (Lengauer-Tarjan) */
    int idom;
} BasicBlock;

typedef struct
{
    BasicBlock blocks[MAX_BLOCKS];
    int nblocks;
    int entry;         /* Index des Eintrittsblocks */
    Graph cfg_graph;   /* Adjazenzmatrix-Darstellung */
    Graph dom_graph;   /* Dominatorgraph */
    Graph kombi_graph; /* kombinierter Graph fuer SEP */
} CFG;

void cfg_build_graphs(CFG *cfg);
void dce_eliminate(CFG *cfg);  /* Reduktion auf SI (DCE) */
void sep_find_loops(CFG *cfg); /* Reduktion auf SI (SEP) */

/* ================================================================== */
/* 5. SSA-FORM + KONSTANTENPROPAGATION  (CPP)                         */
/* ================================================================== */

#define MAX_SSA_VARS 256

typedef struct
{
    char name[64]; /* x_1, x_2, ... */
    int def_block;
    bool is_const;
    int const_val;
    int inputs[8]; /* SSA-Phi-Eingaben (Indices) */
    int ninputs;
} SSAVar;

typedef struct
{
    SSAVar vars[MAX_SSA_VARS];
    int nvars;
    Graph ssa_graph;             /* SSA-Abhaengigkeitsgraph */
    Graph const_graph;           /* G_K: induzierter Teilgraph der Konstanten */
    bool is_const[MAX_SSA_VARS]; /* nach CPP */
    int const_val[MAX_SSA_VARS];
} SSAForm;

void ssa_transform(CFG *cfg, SSAForm *ssa);
void cpp_propagate(SSAForm *ssa); /* Reduktion auf SI (CPP) */

/* ================================================================== */
/* 6. REGISTERALLOKATION  (RAP)                                       */
/* ================================================================== */

#define MAX_VARS 256
#define MAX_REGS 32

typedef struct
{
    int var_to_reg[MAX_VARS]; /* -1 = stack (spill) */
    int nregs_used;
} RegAlloc;

void rap_allocate(const SSAForm *ssa, int k_regs, RegAlloc *out);

/* ================================================================== */
/* 7. STATISCHE INITIALISIERUNG  (IRP)                                */
/* ================================================================== */

#define MAX_STATIC_OBJS 64

typedef struct
{
    char name[64];
    int deps[MAX_STATIC_OBJS];
    int ndeps;
    int init_order; /* Ergebnis nach IRP */
} StaticObj;

typedef struct
{
    StaticObj objs[MAX_STATIC_OBJS];
    int count;
    int order[MAX_STATIC_OBJS]; /* azyklische Reihenfolge */
    int order_count;
    bool has_siof; /* Static Initialization Order Fiasco? */
} InitSystem;

void irp_resolve(InitSystem *is); /* Reduktion auf SI (IRP) */

/* ================================================================== */
/* 8. CODEGENERATOR (x86-64 AT&T-Syntax)                              */
/* ================================================================== */

typedef struct
{
    char text[65536];
    int pos;
} CodeBuffer;

void codegen_function(const CFG *cfg,
                      const SSAForm *ssa,
                      const RegAlloc *ra,
                      const char *func_name,
                      CodeBuffer *out);

/* ================================================================== */
/* 9. OBJEKTDATEI (.o-Format, vereinfacht)                            */
/* ================================================================== */

#define MAX_SYMBOLS 128
#define MAX_OBJ_CODE 65536

typedef enum
{
    SYM_DEFINED,
    SYM_UNDEFINED
} SymKind;

typedef struct
{
    char name[64];
    SymKind kind;
    int offset; /* Offset im Textabschnitt */
} Symbol;

typedef struct
{
    char name[64]; /* Dateiname (z.B. "main.o") */
    Symbol syms[MAX_SYMBOLS];
    int nsyms;
    uint8_t code[MAX_OBJ_CODE];
    int code_len;
    char asm_text[65536]; /* lesbare ASM-Darstellung */
} ObjFile;

/* ================================================================== */
/* 10. COMPILER-HAUPTFUNKTION                                         */
/* ================================================================== */

typedef struct
{
    TypeSystem types;
    InitSystem statics;
    CFG cfg;
    SSAForm ssa;
    RegAlloc regalloc;
    CodeBuffer codebuf;
    ObjFile obj;
    char errors[4096];
    bool ok;
} CompileUnit;

bool compile(const char *source, const char *filename, ObjFile *out);

#endif /* COMPILER_H */
