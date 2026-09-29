# gen - Biologische Netzwerkanalyse mit dem Subgraph Algorithmus

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-95%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

Der Subgraph Algorithmus für die Analyse biologischer Netzwerke

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

## Demo

Ausführliche Demo zur Demonstration der Arbeitsweise:

```python
python src/demo.py
```

Beispiel-Ausgabe der Ausführung:

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║      GEN - Biologische Netzwerkanalyse mit Subgraph Algorithmus    ║
║                                                                    ║
║      Demo: Metabolische Pfade & Protein-Interaktions-Netzwerke     ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

🔧 Initialisiere Analyzer und lade Netzwerke...
✓ Netzwerk hinzugefügt: Glycolysis (7 Knoten)
✓ Netzwerk hinzugefügt: Partial_Glycolysis (4 Knoten)
✓ Netzwerk hinzugefügt: Krebs_Cycle_Fragment (4 Knoten)
✓ Netzwerk hinzugefügt: DNA_Damage_Response (5 Knoten)
✓ Netzwerk hinzugefügt: Pentose_Phosphate (4 Knoten)
✓ Netzwerk hinzugefügt: Alternative_Glucose (4 Knoten)
✓ 6 biologische Netzwerke geladen


======================================================================
  1. NETZWERK-EIGENSCHAFTEN
======================================================================

Glycolysis
   Knoten: 7
   Kanten: 7
   Beispiel-Kanten: [('Glucose', 'G6P'), ('G6P', 'F6P'), ('F6P', 'FBP')]
                   ... und 4 weitere

Partial_Glycolysis
   Knoten: 4
   Kanten: 3
   Beispiel-Kanten: [('Glucose', 'G6P'), ('G6P', 'F6P'), ('F6P', 'FBP')]

Krebs_Cycle_Fragment
   Knoten: 4
   Kanten: 3
   Beispiel-Kanten: [('Citrate', 'Isocitrate'), ('Isocitrate', 'α-Ketoglutarate'), ('α-Ketoglutarate', 'Succinyl-CoA')]

DNA_Damage_Response
   Knoten: 5
   Kanten: 6
   Beispiel-Kanten: [('p53', 'MDM2'), ('MDM2', 'p53'), ('ATM', 'p53')]
                   ... und 3 weitere

Pentose_Phosphate
   Knoten: 4
   Kanten: 3
   Beispiel-Kanten: [('G6P', '6PG'), ('6PG', 'Ru5P'), ('Ru5P', 'R5P')]

Alternative_Glucose
   Knoten: 4
   Kanten: 3
   Beispiel-Kanten: [('Glucose', 'G6P'), ('G6P', '6PG'), ('6PG', 'Ru5P')]


======================================================================
  2. PFAD-VERGLEICH
======================================================================


--- Vergleich: Vollständige vs. Teilweise Glykolyse ---

...
```