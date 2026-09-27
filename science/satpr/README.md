# satpr - Wahrscheinlichkeitsverteilung und Laufzeitanalyse des Subgraph-SAT-Solvers

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-167%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-85.13%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

satpr implementiert eine umfassende mathematische Analyse der Wahrscheinlichkeitsverteilung bei SAT-Problemlösern. Das Projekt demonstriert, wie die Laufzeit eines **Subgraph-SAT-Solvers** sich mit der **Wahrscheinlichkeitsverschiebung** über drei charakteristische Phasen entwickelt:

- **Phase 1:** Logarithmische Laufzeit O(log n)
- **Phase 2:** Übergansphase mit superlinearer Konvergenz
- **Phase 3:** Asymptotische Annäherung an Worst-Case O(n³)

Das Projekt verbindet theoretische Komplexitätsanalyse mit praktischen Simulationen und bietet werkzeuge zur Validierung dieser theoretischen Vorhersagen.

## Inhalt

### Mathematische Kernmodule

- **SAT-Solver** (`sat_solver.py`)
  - Subgraph-SAT-Solver mit formaler Analyse
  - Kombinationspyramide-Datenstruktur
  - Worst-Case-Komplexität: O(n³)

- **Wahrscheinlichkeitsverschiebungs-Theorie** (`probability_shift.py`)
  - Analyse der drei Phasen
  - Bimodale Verteilungen in Phase 1
  - Konvergenzrate zu Worst-Case

- **Logarithmisches Belegungsverfahren (LBV)** (`lbv_solver.py`)
  - O(m log m) kombinierte Laufzeit mit Subgraph-Solver
  - Iterative Halbierungsstrategie
  - Erfolgswahrscheinlichkeit-Theorie

- **Metaverteilungs-Theorie** (`metadistribution.py`)
  - Wahrscheinlichkeitsverteilungen von Wahrscheinlichkeitsverteilungen
  - Entropie-Analyse
  - Shift-Trajektorien durch Verteilungsraum

- **Statistische Analysen** (`statistics.py`)
  - Phase-Übergänge erkennen
  - Konvergenz-Analyse
  - Entropie-Metriken (Shannon, Rényi)
  - Statistische Tests (KS, χ², Mann-Whitney)

### Simulationen

- **Monte-Carlo Simulator** (`simulation.py`)
  - Validierung theoretischer Vorhersagen
  - Phase-spezifische Simulationen
  - Laufzeitvergleiche

- **LBV Simulator**
  - Erfolgsrate-Analyse
  - Binomial-Verteilungs-Validierung
  - Wiederholtes LBV

### Test & Coverage

- Integration mit pytest und pytest-cov

## Installation

### Voraussetzungen

- Python 3.9+
- pip (oder conda)

### Setup

```bash
# 1. Repository klonen
git clone https://github.com/naphets29/satpr.git
cd satpr

# 2. Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv

# Aktivieren:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Abhängigkeiten installieren
pip install -e .

# 4. Entwicklungs-Abhängigkeiten (optional, für Tests)
pip install -e ".[dev]"
```

## Verwendung

### Beispiel 

Beispiel-Skript demonstriert die Verwendung aller SATPR-Module:
1. SAT-Solver Grundlagen
2. Wahrscheinlichkeitsverschiebungs-Analyse
3. Logarithmisches Belegungsverfahren
4. Simulationen
5. Statistische Analysen

  
```python
python .\src\examples.py
```

### Grundlegende SAT-Formel lösen

```python
from satpr import SATFormula, Clause, SubgraphSATSolver

# Erstelle einfache Formel: (x1 OR x2) AND (NOT x1 OR x3)
clauses = [
    Clause([1, 2]),      # x1 OR x2
    Clause([-1, 3])      # NOT x1 OR x3
]
formula = SATFormula(clauses, n_vars=3)

# Löse mit Solver
solver = SubgraphSATSolver(formula)
satisfiable, assignment, ops = solver.solve()

print(f"Erfüllbar: {satisfiable}")
print(f"Belegung: {assignment}")
print(f"Operationen: {ops}")
```

### Wahrscheinlichkeitsverschiebungs-Analyse

```python
from satpr import ProbabilityShiftAnalyzer

analyzer = ProbabilityShiftAnalyzer(n=5)

# Bestimme Phase
phase = analyzer.get_phase(k=10)

# Erwartete Laufzeit
runtime = analyzer.expected_runtime(k=10)

# Bimodale Verteilung
dist = analyzer.bimodal_distribution(k=1)
print(f"Modus 1: {dist.mode1}, Modus 2: {dist.mode2}")
```

### LBV-Algorithmus

```python
from satpr import LogarithmicAssignmentProcedure

lbv = LogarithmicAssignmentProcedure(n_variables=8)

# Iterationsanzahl
k = lbv.iteration_count()  # K = ⌈log₂ 8⌉ + 2 = 5

# Erfolgswahrscheinlichkeit
prob = lbv.expected_success_probability(success_prob_per_iteration=1/3)

# Löse Formel
satisfiable, assignment, iterations = lbv.solve(formula)
```

### Simulationen

```python
from satpr import MonteCarloSimulator, LBVSimulator

# Monte-Carlo Simulation
mc_sim = MonteCarloSimulator(n_variables=4, seed=42)
results = mc_sim.run_full_simulation(n_samples_per_phase=5)
stats = mc_sim.get_statistics()

# LBV Simulation
lbv_sim = LBVSimulator(n_variables=5, seed=42)
success_data = lbv_sim.simulate_lbv_success_rate(n_formulas=100)
print(f"Erfolgsrate: {success_data['success_rate']:.2%}")
```

### Beispiel-Demonstrationen

```bash
# Führe alle Beispiele aus
python examples.py

# Oder in Python
python -c "from examples import *; example_1_basic_sat()"
```

## Tests ausführen

### Alle Tests mit Coverage

```bash
pytest
```

Dies generiert automatisch:
- Terminal-Output mit Zeilen-Coverage
- HTML-Report: `doc/coverage/index.html`
- JSON-Report: `doc/coverage/coverage.json`

### Spezifische Tests

```bash
# Nur SAT-Solver Tests
pytest tests/test_sat_solver.py -v

# Nur schnelle Tests
pytest -m "not slow"

# Mit detailliertem Output
pytest -vv --tb=long
```

### Coverage-Schwelle prüfen

```bash
pytest --cov=satpr --cov-fail-under=85
```

## Verwendungsbeispiele

### 1. SAT-Problem lösen

```python
from satpr import SATFormula

# Erstelle zufällige 3-SAT Formel
formula = SATFormula.random_formula(n_vars=10, n_clauses=30, k_literals=3)

# Löse sie
from satpr import SubgraphSATSolver
solver = SubgraphSATSolver(formula)
satisfiable, assignment, ops = solver.solve()
```

### 2. Laufzeit-Vorhersagen validieren

```python
from satpr import MonteCarloSimulator

sim = MonteCarloSimulator(n_variables=5, seed=42)
results = sim.run_full_simulation(n_samples_per_phase=10)
validation = sim.validate_phase3_cubic_convergence()

print(f"Phase 3 konvergiert zu O(n³): {validation['valid']}")
```

### 3. LBV-Effektivität testen

```python
from satpr import LBVSimulator

lbv_sim = LBVSimulator(n_variables=6, seed=42)
result = lbv_sim.simulate_lbv_success_rate(n_formulas=100)

print(f"Erfolgsrate: {result['success_rate']:.2%}")
print(f"Erwartete Iterationen: {result['theoretical_iterations']}")
```

### 4. Entropie-Analyse

```python
from satpr import EntropyAnalysis

probs = [0.5, 0.3, 0.2]
entropy = EntropyAnalysis.shannon_entropy(probs)
renyi2 = EntropyAnalysis.renyi_entropy(probs, alpha=2.0)

print(f"Shannon: {entropy:.3f} bits, Rényi(α=2): {renyi2:.3f} bits")
```