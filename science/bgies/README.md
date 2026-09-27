# Die Wechselwirkung zwischen Bakterien und DNA

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-38%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-94%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

Die Wechselwirkung zwischen Bakterien und DNA: Formale Analyse lokaler Informationssysteme und ihre genetischen Konsequenzen

## Bacteria-DNA Wechselwirkung: Formale Modellierung und Implementierung

Dieses Modul implementiert alle Konzepte aus der wissenschaftlichen Arbeit:
- Bacterial-Genomic Information Exchange System (BGIES)
- Modifikationsmechanismen (HGT, Plasmide, Mutagenese, Epigenetik, Rekombination, StressResp)
- Genomische Einflussmetrik
- Funktionaler Raum
- Populationsdynamik
- Mikrobiom-Netzwerke

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

Experiments Modul: Umfangreiche Experimente zur Bacteria-DNA Wechselwirkung

Experimente durchführen:
1. Genomische Modifikationen über Zeit
2. Populationsdynamik und Equilibrium
3. Mikrobiom-Netzwerk-Komplexität
4. Einflussmetrik-Komponenten-Analyse
5. Informationskaskaden
6. Co-Evolution von Wirt und Mikrobiom
7. Stress-Response-Dynamik
8. Funktionale Komplementarität

Für jedes Experiment:
- Daten sammeln
- Plots mit matplotlib erstellen
- Ergebnisse in JSON speichern

```python
python src/experiments.py
```