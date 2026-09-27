"""
py2 Präprozessor
================
Wandelt .py2-Dateien in ausführbares Python um.

Regeln:
    a(i, j) = expr    →    a.__setcall__(i, j, expr)       (Array/Liste)
    a(i, j)           →    a.__getcall__(i, j)             (Lesezugriff)
    a(i, j) OP= expr  →    a.__setcall__(i, j, a(i,j) OP expr)  (Verbund)
    (a, b, c)         →    [a, b, c]                       (Tupel → Liste)

Es gibt in py2 keine unveränderbaren Listen (Tupel). Runde Klammern ohne
vorangehendes Bezeichner-Zeichen werden stets in Listen umgewandelt.

Aufruf:
    python3 py2_preprocessor.py skript.py2          # gibt Python-Code aus
    python3 py2_preprocessor.py skript.py2 --run    # direkt ausführen
    python3 py2_preprocessor.py skript.py2 --out skript.py  # in Datei schreiben
"""

import re
import sys
import os
import textwrap

# -----------------------------------------------------------------------
# Reguläre Ausdrücke
# -----------------------------------------------------------------------

# Erkennt:  NAME(ARGS) = EXPR
# Gruppe 1: Name der Array-Variable
# Gruppe 2: Argumente (i, j)
# Gruppe 3: rechte Seite
ASSIGN_RE = re.compile(
    r'^(\s*)'                        # führende Leerzeichen (Einrückung)
    r'([A-Za-z_]\w*)'               # Variablenname
    r'\(([^)]+)\)'                   # (args)
    r'\s*=(?!=)'                     # = aber nicht ==
    r'\s*(.+)$'                      # rechte Seite
)

# Erkennt Lesezugriff:  NAME(i, j)  →  NAME.__getcall__(i, j)
# Nur wenn es kein regulärer Funktionsaufruf ist (wird durch Kontext geregelt)
# [^()]+ statt [^)]+ damit verschachtelte Aufrufe (z. B. round(b(k), 6)) korrekt
# von innen nach außen aufgelöst werden.
READ_RE = re.compile(
    r'\b([A-Za-z_]\w*)\(([^()]+)\)'
)

# Tupel-Literale:  (a, b, c)  →  [a, b, c]
# Bedingung: kein Wortzeichen direkt vor '(' (kein Funktionsaufruf),
# und mindestens ein Komma im Inhalt.
TUPLE_RE = re.compile(r'(?<!\w)\(([^()]*,[^()]*)\)')

# Erkennt: NAME(ARGS) OP= EXPR  (Verbundzuweisung, z. B. b(j) -= ...)
COMPOUND_ASSIGN_RE = re.compile(
    r'^(\s*)'
    r'([A-Za-z_]\w*)'
    r'\(([^)]+)\)'
    r'\s*(//|\*\*|>>|<<|[+\-*/%&|^])='
    r'\s*(.+)$'
)

# -----------------------------------------------------------------------
# Bekannte Array-Variablen (werden beim Scannen gesammelt)
# -----------------------------------------------------------------------

def collect_arrays(lines: list[str]) -> set[str]:
    """
    Sammelt alle Variablennamen, die mit Array1D(...) oder Array2D(...) initialisiert werden.
    Beispiel:  a = Array2D(5, 3)  →  'a' wird als Array erkannt.
    """
    arrays = set()
    init_re = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*Array(?:1D|2D)\s*\(')
    for line in lines:
        m = init_re.match(line)
        if m:
            arrays.add(m.group(1))
    return arrays


# -----------------------------------------------------------------------
# Zeilenweise Transformation
# -----------------------------------------------------------------------

def transform_line(line: str, arrays: set[str]) -> str:
    """
    Transformiert eine einzelne Zeile:
    1. Prüft ob es eine Zuweisung  arr(i,j) = expr  ist  → __setcall__
    2. Ersetzt alle Lesezugriffe   arr(i,j)          → __getcall__
    """

    # Kommentare und leere Zeilen unverändert
    stripped = line.strip()
    if stripped.startswith('#') or stripped == '':
        return line

    # 1. Verbundzuweisung:  a(i,j) OP= expr
    mc = COMPOUND_ASSIGN_RE.match(line)
    if mc:
        indent  = mc.group(1)
        varname = mc.group(2)
        args    = mc.group(3)
        op      = mc.group(4)
        rhs     = mc.group(5)
        if varname in arrays:
            rhs_transformed = transform_reads(rhs, arrays)
            getter = f"{varname}.__getcall__({args})"
            return f"{indent}{varname}.__setcall__({args}, {getter} {op} ({rhs_transformed}))\n"

    # 2. Einfache Zuweisung:  a(i, j) = expr
    m = ASSIGN_RE.match(line)
    if m:
        indent  = m.group(1)
        varname = m.group(2)
        args    = m.group(3)
        rhs     = m.group(4)

        if varname in arrays:
            # RHS: Lesezugriffe transformieren
            rhs_transformed = transform_reads(rhs, arrays)
            return f"{indent}{varname}.__setcall__({args}, {rhs_transformed})\n"

    # 3. Nur Lesezugriffe transformieren
    return transform_reads(line, arrays)


def transform_reads(text: str, arrays: set[str]) -> str:
    """
    Ersetzt  arr(i, j)  durch  arr.__getcall__(i, j)
    für alle bekannten Array-Variablen, danach
    Tupel-Literale  (a, b, c)  durch  [a, b, c].
    """
    def replacer(m):
        varname = m.group(1)
        args    = m.group(2)
        if varname in arrays:
            return f"{varname}.__getcall__({args})"
        return m.group(0)   # unbekannte Funktion: unverändert

    result = READ_RE.sub(replacer, text)
    # Nach Array-Transforms: verbleibende Tupel-Literale → Listen
    result = TUPLE_RE.sub(lambda m: f"[{m.group(1)}]", result)
    return result


# -----------------------------------------------------------------------
# Haupttransformation
# -----------------------------------------------------------------------

RUNTIME_IMPORT = '''\
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from array2d_core import Array1D, Array2D
'''

def preprocess(source: str) -> str:
    lines = source.splitlines(keepends=True)

    # Bereits vorhandene array2d_core Imports entfernen (idempotent)
    lines = [l for l in lines if 'array2d_core' not in l]

    arrays = collect_arrays(lines)

    result = [RUNTIME_IMPORT]
    for line in lines:
        result.append(transform_line(line, arrays))

    return ''.join(result)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    src_file = args[0]
    run_mode = '--run' in args
    out_file = None
    if '--out' in args:
        idx = args.index('--out')
        out_file = args[idx + 1]

    with open(src_file, 'r') as f:
        source = f.read()

    transformed = preprocess(source)

    if run_mode:
        # Temporäre Datei im selben Verzeichnis erstellen und ausführen
        tmp = src_file.replace('.py2', '_generated.py')
        with open(tmp, 'w') as f:
            f.write(transformed)
        os.system(f"python3 {tmp}")
    elif out_file:
        with open(out_file, 'w') as f:
            f.write(transformed)
        print(f"Geschrieben: {out_file}")
    else:
        print(transformed)


if __name__ == '__main__':
    main()
