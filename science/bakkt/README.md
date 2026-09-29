# Bakterien in Mensch und Natur: Konsequenzen, Bewusstsein und natürliche Ordnungssysteme

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-104%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-70%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

Die vorliegende Arbeit und dieses Projekt behandelt eine fundamentale biologische und philosophische Frage, deren Bedeutung für das Verständnis natürlicher Ordnungssysteme bislang unterschätzt wurde. Sie befasst sich mit Organismen, die das Leben auf der Erde überhaupt erst ermöglichten, die in jedem lebenden System vorhanden sind, deren Konsequenzen aber oft unbeachtet bleiben: den Bakterien.

## Vollständige Python-Implementierung aller Konzepte und Algorithmen

Dieses Modul implementiert alle theoretischen und algorithmischen Konzepte
aus der wissenschaftlichen Arbeit "Bakterien in Mensch und Natur":
- Lokale Informationssysteme LIS(∞)
- Konsequenzfolge und Determinismus
- Hilfsfunktionen und Kategorisierung
- Der Subgraph Algorithmus
- Populationsdynamik und Netzwerk-Analyse
- Formale Tests aller Konzepte mit pytest

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

Umfassende experimentelle Validierung aller Konzepte mit:
- Detaillierte Experiment-Durchführung
- Matplotlib-Visualisierungen für alle Ergebnisse
- JSON-Export der Experimentdaten
- Statistische Analysen
- Performance-Messungen
- Reproduzierbare Ergebnisse

```python
python src/experiments.py
```

## Demo

Ausführliche Demo zur Demonstration der Arbeitsweise:

```
python src/bakkt_implementation.py

================================================================================
DEMO 1: E. coli Chemotaxis
================================================================================
Ohne Glukose: Aktion = BacterialAction.REPRODUCE
Mit Glukose: Aktion = BacterialAction.MOVE_FORWARD
Mit Nährstoffen + optimaler Temperatur: Aktion = BacterialAction.MOVE_FORWARD

================================================================================
DEMO 2: Symbiotische Hilfsfunktionen
================================================================================

Rhizobium:
  → Hilft plant           : metabolic    (Magnitude: 0.80)
  → Hilft human           : neutral      (Magnitude: 0.00)
  → Hilft animal          : neutral      (Magnitude: 0.00)
  → Hilft organic_matter  : neutral      (Magnitude: 0.00)

E. coli:
  → Hilft plant           : neutral      (Magnitude: 0.00)
  → Hilft human           : metabolic    (Magnitude: 0.50)
  → Hilft animal          : neutral      (Magnitude: 0.00)
  → Hilft organic_matter  : metabolic    (Magnitude: 0.90)

Cyanobacteria:
  → Hilft plant           : metabolic    (Magnitude: 0.70)
  → Hilft human           : metabolic    (Magnitude: 0.60)
  → Hilft animal          : neutral      (Magnitude: 0.00)
  → Hilft organic_matter  : neutral      (Magnitude: 0.00)

================================================================================
DEMO 3: Subgraph Algorithmus
================================================================================

Netzwerk: 12 Knoten, 28 Kanten

1. DFS - Zusammenhängende Komponenten:
   Komponente 0: 12 Knoten

2. Greedy Dense Subgraphs:
   Dichter Subgraph 0: 12 Knoten

3. Subgraph-Eigenschaften:
   Subgraph 0:
      Knoten: 12
      Kanten: 28
      Dichte: 0.424
      Durchmesser: 3

================================================================================
DEMO 4: Populationsdynamik
================================================================================
Startp opulation: {'Species_1': 5, 'Species_2': 3}

Populations-Entwicklung:
  t=1: {'Species_1': 5, 'Species_2': 3}
  t=3: {'Species_1': 5, 'Species_2': 3}
  t=5: {'Species_1': 5, 'Species_2': 3}
  t=7: {'Species_1': 5, 'Species_2': 3}
  t=9: {'Species_1': 5, 'Species_2': 3}

================================================================================
DEMO 5: Bewusstsein und Wille (Satz 7.1)
================================================================================

Szenario: Zwei Organismen mit unterschiedlicher Wahrnehmungsreichweite

Bakterium:
  Wahrnehmungsreichweite: 10.0 μm
  Verarbeitungskapazität: 0.20
  Willens-Stärke: 0.002

Mensch:
  Wahrnehmungsreichweite: 1000.0 μm
  Verarbeitungskapazität: 0.90
  Willens-Stärke: 0.900

Theorem: Ein Mensch hat stärkeren Willen WEIL er mehr wahrnehmen kann.
Der Wille ist nicht ''frei'', sondern eine Funktion der Wahrnehmung.

--- Entscheidungsszenario ---
Bakterium wählt: reproduce (Werte: {'reproduce': 0.9, 'survive': 0.8, 'help_neighbor': 0.1})
Mensch wählt: help_neighbor (Werte: {'reproduce': 0.3, 'survive': 0.7, 'help_neighbor': 0.9})

Begründung: Die größere Wahrnehmungsreichweite des Menschen
führt zu anderen Wertpräferenzen und somit anderen Entscheidungen.

================================================================================
ALLE DEMOS ABGESCHLOSSEN
================================================================================

Zum Ausführen der pytest-Tests:
  pytest bakkt_implementation.py -v
  pytest bakkt_implementation.py -v --tb=short
================================================================================
```