# Der Tiefpassfilter und die e-Funktion

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-148%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-96%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

Eine ausführliche Untersuchung der Differentialgleichung und der Reihenglied-Problematik

## Installation

Installation mit Hilfe der `pyproject.toml`.

```python
pip install -e ".[dev]"
```

## Tests

Tests werden mit `pytest` mit Code Coverage ausgeführt. Der Code Coverage Report wird generiert nach doc/coverage/.

```python
pytest
```

## Experimente

Jedes Experiment
1. berechnet seine Daten,
2. speichert sie als JSON   (<outdir>/expNN_<name>.json),
3. zeichnet einen matplotlib-Plot und speichert ihn als PDF (<outdir>/expNN_<name>.pdf).

Der Plot wird aus den *aus der JSON-Datei zurückgelesenen* Daten erzeugt,
d. h. PDF und JSON sind garantiert konsistent.
Hier die Tabelle in Markdown:

| Nr. | Name | Beschreibung |
|----:|------|--------------|
| 1 | `leibniz_basis` | Leibniz-Reihe, O(1/N)-Konvergenz, asymmetrische Zerlegung |
| 2 | `beschleunigung` | Euler / Mitteln / Shanks / Formeln der Arbeit im Vergleich |
| 3 | `dirichlet_beta` | $β(s)$ auf $ℝ$, Funktionalgleichung, Sonderwerte ($β$ ist ganz) |
| 4 | `catalan` | Darstellungen der Catalan-Konstante G |
| 5 | `parametrisierte_leibniz` | $L(t;λ)$ = $arctan(exp(-λt))$, Integrodifferentialgleichung |
| 6 | `e_funktion_taylor` | Ableitung der Taylor-Reihe (Beobachtung 49): "ein Glied geht verloren" |
| 7 | `rc_tiefpass` | Bode-Diagramm, Sprung- und Sinusantwort des RC-Filters |
| 8 | `rlc_daempfung` | RLC-Sprungantworten, Überschwingen vs. Dämpfung ζ |
| 9 | `formel_audit` | Numerische Prüfung der Formeln in Arbeit und Modulen |

```python
python src/experiments.py
```