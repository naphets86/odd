# py2 – Python-Erweiterung für 2D-Array-Zuweisung

Dieses Projekt stellt eine kleine Präprozessor-Pipeline bereit, die eine eigene Syntax (`.py2`) in ausführbares Python 3 übersetzt. Das ermöglicht die intuitive Schreibweise `a(i,j) = expr` für 2D-Array-Zuweisungen – die in nativem Python syntaktisch ungültig wäre.

## Motivation

In numerischen Algorithmen (z. B. LU-Zerlegung) ist die Schreibweise

```
a(i, j) = a(i, j) - a(i, k) * a(k, j)
```

deutlich lesbarer als der Python-Äquivalent mit explizitem `__setitem__`-Aufruf. Der Präprozessor übernimmt die Übersetzung automatisch.

## Projektstruktur

| Datei | Beschreibung |
|---|---|
| `array2d_core.py` | `Array2D`-Klasse mit `__getcall__`/`__setcall__`-Protokoll |
| `py2_preprocessor.py` | Präprozessor: wandelt `.py2` → Python 3 |
| `crout.py2` | Beispiel: Crout-LU-Zerlegung für ein Tridiagonalsystem |

## Syntax

In `.py2`-Dateien gelten folgende Umwandlungsregeln:

| py2-Syntax | Generiertes Python | Beschreibung |
|---|---|---|
| `a(i, j) = expr` | `a.__setcall__(i, j, expr)` | Schreibzugriff |
| `a(i, j) OP= expr` | `a.__setcall__(i, j, a(i,j) OP expr)` | Verbundzuweisung |
| `a(i, j)` (lesen) | `a.__getcall__(i, j)` | Lesezugriff |
| `(a, b, c)` | `[a, b, c]` | Tupel → Liste |
| alles andere | unverändert | |

### Design-Prinzip: keine Tupel

In py2 sind runde Klammern für den Element-Zugriff auf `Array1D`- und `Array2D`-Objekte reserviert. Daher existieren **keine Tupel**. Alle Sequenzen sind veränderbare Listen:

```python
# py2                      # generiertes Python
x = (1, 2, 3)              x = [1, 2, 3]
return (a, b)              return [a, b]
```

Funktionsaufrufe wie `range(1, n)` oder `round(x, 6)` werden nicht angetastet, da ihnen ein Bezeichner direkt voransteht.

## Verwendung

```bash
# Generierten Python-Code anzeigen
python3 py2_preprocessor.py crout.py2

# Direkt ausführen
python3 py2_preprocessor.py crout.py2 --run

# In eine .py-Datei schreiben
python3 py2_preprocessor.py crout.py2 --out crout.py
```

## Beispiel ausführen

`crout.py2` enthält eine Crout-LU-Zerlegung für ein tridiagonales Gleichungssystem sowie einen eingebauten Test:

```bash
python3 py2_preprocessor.py crout.py2 --run
```

Erwartete Ausgabe:

```
Matrix vor LU:
Array2D(5x3):
  [  2.0000  -1.0000   0.0000]
  [ -1.0000   2.0000  -1.0000]
  [ -1.0000   2.0000   0.0000]
  ...

Loesung: [1.0, 1.0, 1.0]
Erwartet: [1.0, 1.0, 1.0]
```

## Anforderungen

- Python 3.10 oder neuer (wegen `list[str]`-Typannotationen)
- Keine externen Abhängigkeiten
