# Strukturen in der Algebra

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-51%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-91.10%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

**Schwerpunkterkennung und effiziente Lösungsmethoden**

Ein Python-Projekt zur Implementierung von Algorithmen für algebraische Strukturen, Schwerpunkt-Erkennung und Multiplikationslängen-Analyse.


## Inhaltsverzeichnis

- [Features](#features)
- [Installation](#installation)
- [Schnelleinstieg](#schnelleinstieg)
- [Beispiele ausführen](#beispiele-ausführen)
- [API-Dokumentation](#api-dokumentation)


## Features

### 1. Vektor-Schwerpunkt-Erkennung
Findet das Element mit der maximalen Magnitude in einem Vektor.

### 2. Matrix-Schwerpunkt-Erkennung
- Extrahiert Diagonalelemente
- Berechnet den **DSI (Diagonalisierungs-Schwerpunkt-Index)** [0, 1]
- Misst, wie "diagonal" eine Matrix ist

### 3. Multiplikationslängen (ML-1 bis ML-5)
Analysiert die Struktur von Matrix-Multiplikationen:
- **ML-1:** Direkte Multiplikation
- **ML-2:** Schwerpunkt-Rebalancierung
- **ML-3:** Strukturelle Rotation
- **ML-4:** Zweite Rebalancierung
- **ML-5:** Indirekte Effekte

### 4. Diagonalisierung
- **SVD-basierte Methode:** Schwerpunkt-offensive Diagonalisierung
- **QR-Algorithmus:** Mit Schwerpunkt-Vorkonditionierung für Eigenwerte

### 5. Iterative Verfeinerung
Verfeinert Matrizen schrittweise gegen ihre Diagonale mit exponentieller Konvergenz.

### 6. Lineare Gleichungssysteme
Löst `Ax = b` mit optimierter Schwerpunkt-Vorkonditionierung.

### 7. Strukturelle Unsicherheit
Misst Shannon-Entropie und Spektrallücke zur Konvergenzanalyse.


## Installation

### Voraussetzungen
- Python 3.8+
- pip

### Schritt 1: Repository klonen oder Dateien herunterladen
```bash
cd science/algebstrct
```

### Schritt 2: Virtual Environment
```bash
# Erstellen:
python -m venv venv

# Aktivieren:
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac
```

### Schritt 3: Dependencies installieren
```bash
pip install -e ".[dev]"
```

## Schnelleinstieg

### 1. Alle Tests ausführen
```bash
pytest
```

**Erwartet:** [PASS] 51 passed

### 2. Beispiele ausführen
```bash
python src/examples.py
```

**Output:** 9 praktische Beispiele mit Erklärungen

### 3. Spezifischen Test ausführen
```bash
# Nur einen Test:
pytest tests/test_algorithms.py::TestVectorCentroidFinder::test_simple_vector -v

# Mit Coverage:
pytest tests/test_algorithms.py --cov=src --cov-report=html
```

## Beispiele ausführen

### Option 1: Alle Beispiele auf einmal
```bash
python src/examples.py
```

Output: ~200 Zeilen mit allen 9 Beispielen + Vergleich

### Option 2: Einzelne Beispiele importieren
```python
from src.examples import (
    beispiel_1_vektor_schwerpunkt,
    beispiel_3_multiplikationslängen,
    beispiel_6_qr_centroid
)

# Einzelne Beispiele aufrufen
beispiel_1_vektor_schwerpunkt()
beispiel_3_multiplikationslängen()
```

### Option 3: In eigenem Skript verwenden
```python
import numpy as np
from src.algorithms import QRCentroid, MatrixCentroidFinder

# Eigene Matrix erstellen
A = np.array([[2.0, 0.5], [0.5, 3.0]])

# Eigenwerte berechnen
eigenvals, eigenvecs, iterations = QRCentroid.qr_centroid(A)
print(f"Eigenwerte: {eigenvals}")

# Schwerpunkt analysieren
diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
print(f"DSI: {dsi:.4f}")
```


## API-Dokumentation

### VectorCentroidFinder

```python
from src.algorithms import VectorCentroidFinder

v = np.array([1.0, 5.0, 2.0])
idx, val = VectorCentroidFinder.find_vector_centroid(v)
# idx = 1, val = 5.0
```

**Komplexität:** O(n) Zeit, O(1) Speicher


### MatrixCentroidFinder

```python
A = np.array([[4.0, 0.1], [0.1, 3.0]])
diag_vector, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
# diag_vector = [4.0, 3.0]
# dsi = 0.9975... (99.75% diagonal!)
```

**Komplexität:** O(n²) Zeit, O(n) Speicher


### MatrixLengthsMultiplication

```python
A = np.eye(2)
B = np.eye(2)

result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
# result = {1: M1, 2: M2, 3: M3, 4: M4, 5: M5}

for k in range(1, 6):
    print(f"M_{k}:\n{result[k]}\n")
```

**Komplexität:** O(n³) für Matrix-Multiplikation, O(n⁴) für Rotation


### QRCentroid

```python
A = np.array([[2.0, 0.5], [0.5, 3.0]])
eigenvals, eigenvecs, iterations = QRCentroid.qr_centroid(A)

print(f"Eigenwerte: {eigenvals}")
print(f"Konvergiert in {iterations} Iterationen")
```

**Komplexität:** O(n³) pro Iteration, typisch O(n³·log(ε⁻¹))


### CentroidRefinement

```python
A = np.random.rand(3, 3)
A = (A + A.T) / 2  # Symmetrisch machen

A_refined, convergence = CentroidRefinement.refine_to_diagonal(
    A, 
    alpha=0.5,           # Mischungsparameter
    max_iterations=100,
    tolerance=1e-8
)

print(f"Konvergiert in {len(convergence)} Schritten")
print(f"Finaler Fehler: {convergence[-1]:.2e}")
```

**Konvergenz:** Exponentiell mit Rate α


### CentroidPreconditioning

```python
A = np.diag([2.0, 3.0, 4.0]) + 0.1 * np.ones((3, 3))
b = np.array([2.1, 3.1, 4.1])

x, residuals = CentroidPreconditioning.solve_preconditioned(A, b)
print(f"Lösung: {x}")
print(f"Residual-Norm: {residuals[-1]:.2e}")
```

**Vorkonditionierung:** Diagonal aus Matrixdiagonale


### StructuralUncertainty

```python
A = np.random.rand(3, 3)

entropy = StructuralUncertainty.compute_entropy(A)
gap = StructuralUncertainty.compute_spectral_gap(A)

print(f"Entropie: {entropy:.4f}")
print(f"Spektrallücke: {gap:.4f}")
```