# Der Tiefpassfilter und die e-Funktion

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-483%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-90%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

Eine ausführliche Untersuchung der Differentialgleichung und der Reihenglied-Problematik

## Installation

Installation mit Hilfe der `pyproject.toml`.

```python
pip install -e ".[dev]"
```


## Python-Module

| Modul | Beschreibung |
|-------|--------------|
| [`__init__.py`](src/__init__.py) | Initialisiert das Python-Paket |
| [`central_triangles.py`](src/central_triangles.py) | Geometrie der Zentraldreiecke und regulären Vielecke |
| [`circuit_cascade.py`](src/circuit_cascade.py) | Schaltungstypen, ungerade Harmonische und RLC-Kaskaden |
| [`circuit_uncertainty.py`](src/circuit_uncertainty.py) | Unsicherheitsindex, Toleranzen und Stabilität von RLC-Schaltungen |
| [`convergence_acceleration.py`](src/convergence_acceleration.py) | Beschleunigung der Leibniz-Reihe mit Euler, Richardson und Shanks |
| [`dirichlet_beta.py`](src/dirichlet_beta.py) | Dirichlet-Beta-Funktion und Catalan-Konstante |
| [`economic_dynamics.py`](src/economic_dynamics.py) | Leibniz- und Tiefpassmodelle für Wirtschaftsdynamiken |
| [`experiments.py`](src/experiments.py) | Numerische Experimente mit JSON-Ausgaben und PDF-Plots |
| [`leibniz_analysis.py`](src/leibniz_analysis.py) | Parametrisierte Leibniz-Funktion und asymmetrische Reihenzerlegung |
| [`lowpass_filter.py`](src/lowpass_filter.py) | Exponentialfunktion sowie RC-, RL- und RLC-Filter |
| [`main.py`](src/main.py) | Hauptmodul zur Integration der mathematischen Analysen |
| [`odd.py`](src/odd.py) | Ungerader Anteil der e-Reihe und daraus abgeleitete Funktionen |
| [`resonance_window.py`](src/resonance_window.py) | Goldener Schnitt, harmonisches Resonanzfenster und Zonenmodelle |
| [`special_functions.py`](src/special_functions.py) | Polylogarithmen, elliptische Integrale und weitere Spezialfunktionen |

## Experimente

Jedes Experiment
1. berechnet seine Daten,
2. speichert sie als JSON   (<outdir>/expNN_<name>.json),
3. zeichnet einen matplotlib-Plot und speichert ihn als PDF (<outdir>/expNN_<name>.pdf).

Der Plot wird aus den *aus der JSON-Datei zurückgelesenen* Daten erzeugt,
d. h. PDF und JSON sind garantiert konsistent.

| Nr. | Name | Experiment |
|----:|------|------------|
| 1 | `leibniz_basis` | Leibniz-Reihe: O(1/N)-Konvergenz und asymmetrische Zerlegung |
| 2 | `beschleunigung` | Beschleunigung der Leibniz-Reihe: Verfahren im Vergleich |
| 3 | `dirichlet_beta` | Dirichlet-Betafunktion β(s): Fortsetzung, Nullstellen und Sonderwerte |
| 4 | `catalan` | Catalan-Konstante G: Integral- und Reihendarstellungen |
| 5 | `parametrisierte_leibniz` | Parametrisierte Leibniz-Funktion L(t; λ) = arctan(exp(-λt)) und ihre Differentialgleichung |
| 6 | `e_funktion_taylor` | Beobachtung 49: Die Ableitung der Taylor-Reihe verschiebt um ein Glied |
| 7 | `rc_tiefpass` | RC-Tiefpass: Frequenzgang, Sprung- und Sinusantwort |
| 8 | `rlc_daempfung` | RLC-Kreis: Sprungantworten und Überschwingen im Vergleich zur Dämpfung ζ |
| 9 | `formel_audit` | Formel-Audit: Numerische Prüfung der Aussagen in Arbeit und Modulen |

```python
python src/experiments.py
```

## Tests

Tests werden mit `pytest` mit Code Coverage ausgeführt. Der Code Coverage Report wird generiert nach doc/coverage/.

```python
pytest
```

## Hauptmodul

Integration aller mathematischen Analysen

Kombiniert alle Module für:
1. Neue Leibniz Ergebnisse (Dirichlet Beta, Catalan, etc.)
2. Beschleunigungsmethoden für π/4 Konvergenz
3. Low-Pass Filter Analyse mit e-Funktion
4. Spezielle mathematische Funktionen

Verwendung:
    from main import MathematicalAnalysis
    analysis = MathematicalAnalysis()
    analysis.run_comprehensive_analysis()

```python
python src/main.py
```

```
======================================================================
=                                                                    =
=  UMFASSENDE MATHEMATISCHE ANALYSE                                  =
=                                                                    =
======================================================================

Basierend auf:
  1. Neue Leibniz Ergebnisse (Dirichlet Beta, Catalan-Konstante)
  2. Beschleunigungsmethoden für π/4 Konvergenz
  3. Low-Pass Filter und e-Funktion
  4. Spezielle Funktionen und Transzendenz

======================================================================
DIRICHLET BETA-FUNKTION ANALYSE
======================================================================

Spezielle Werte:
  β(1) (Leibniz): 0.7853731634
  β(2) (Catalan): 0.9159655929
  β(3): 0.9689461462
  β(4): 0.9889445518
  β(5): 0.9961578281

Pole (erste 5):
  Pol bei s = -1, Residuum = 0.0000000000e+00
  Pol bei s = -3, Residuum = 0.0000000000e+00
  Pol bei s = -5, Residuum = 0.0000000000e+00
  Pol bei s = -7, Residuum = 0.0000000000e+00
  Pol bei s = -9, Residuum = 0.0000000000e+00

======================================================================
CATALAN-KONSTANTE ANALYSE
======================================================================

Referenzwert: 0.915965594177219

Verschiedene Darstellungen:
  Series         : 0.915965592927218 (Fehler: 1.25e-09)
  Arctan         : 0.915965594177219 (Fehler: 0.00e+00)
  Logarithm      : 0.915965594177219 (Fehler: 3.33e-16)
  Sine           : 0.915965594177219 (Fehler: 0.00e+00)

======================================================================
KONVERGENZ-BESCHLEUNIGUNGSMETHODEN
======================================================================

Vergleich: Naive vs. Euler vs. Richardson vs. Shanks
    n |  Naive Error |  Euler Error |   Richardson
-------------------------------------------------------
   10 | 2.268079e-02 | 2.536210e-04 | 3.521065e-06
   50 | 4.901490e-03 | 2.220446e-16 | 8.878181e-09
  100 | 2.475187e-03 | 2.220446e-16 | 5.887507e-10
  500 | 4.990015e-04 | 2.220446e-16 | 9.876544e-13

======================================================================
EXPONENTIALFUNKTION UND REPRODUZIERBARKEIT
======================================================================

Eigenschaften: f(t) = e^(st) mit s = 1.0
  t    |   e^(st)   | s*e^(st)   |  f'(t)    |  Fehler
------------------------------------------------------------
   0.5 |     1.648721 |     1.648721 |     1.648721 |   0.00e+00
   1.0 |     2.718282 |     2.718282 |     2.718282 |   0.00e+00
   2.0 |     7.389056 |     7.389056 |     7.389056 |   0.00e+00

======================================================================
RC-TIEFPASSFILTER ANALYSE
======================================================================

Filter-Parameter: τ = 1.0
  Grenzfrequenz f_c = 0.159155 Hz
  Kreisgrenzfrequenz ω_c = 1.000000 rad/s
  Einschwingzeit (95%): 3.000000 s

Frequenzgang bei verschiedenen Frequenzen:
 ω [rad/s] |    |H(jω)| |   ∠H(jω) [°] |  Atten. [dB]
--------------------------------------------------
       0.1 |   0.995037 |        -5.71 |        -0.04
       1.0 |   0.707107 |       -45.00 |        -3.01
      10.0 |   0.099504 |       -84.29 |       -20.04

Sprungantwort s(t) = 1 - e^(-t/τ):
   t/τ |       s(t)
--------------------
   0.0 |   0.000000
   1.0 |   0.632121
   3.0 |   0.950213
   5.0 |   0.993262

======================================================================
SPEZIELLE FUNKTIONEN
======================================================================

Polylogarithmen Li_2(x):
  Li_2(0.5) = 0.5822405265
  Li_2(0.8) = 1.0747946000

Elliptische Integrale:
  m = 0.1: K(m) = 1.612441, E(m) = 1.530758
  m = 0.5: K(m) = 1.854075, E(m) = 1.350644
  m = 0.9: K(m) = 2.578092, E(m) = 1.104775

Integraldarstellungen von π/4:
  Via Sinus-Integral:    0.7853981634
  Via Arcsin-Integral:   0.7853981634
  Via Geometrie:         0.7853981634
  Theoretischer Wert:    0.7853981634

======================================================================
TRANSZENDENZASPEKTE
======================================================================

Lindemann-Weierstrass Theorem:

        Lindemann-Weierstrass Theorem impliziert:
        1. π ist transzendent
        2. e ist transzendent
        3. π/4 ist transzendent (Vielfaches von π)
        4. Catalan-Konstante G ist vermutlich transzendent


Irrationalitätsmaß von π:
  Bekannte obere Schranke: μ(π) ≤ 7.6063
  Offen: μ(π) = 2?

======================================================================
ZUSAMMENFASSUNG
======================================================================

✓ Dirichlet Beta-Funktion: analysiert
✓ Catalan-Konstante: mehrere Darstellungen bestätigt
✓ Konvergenz-Beschleunigung: Euler > Richardson > Naive
✓ Exponentialfunktion: Reproduzierbarkeit bestätigt
✓ RC-Filter: Frequenzgang und Sprungantwort analysiert
✓ Spezielle Funktionen: Polylog, Elliptische Integrale
✓ Transzendenzaspekte: Lindemann-Weierstrass angewendet

======================================================================
Alle Analysen abgeschlossen. Details in self.results.
======================================================================


======================================================================
DETAILLIERTE KONVERGENZ-DEMONSTRATION
======================================================================

Konvergenz zu π/4 ≈ 0.785398...

Vergleich nach N Termen:
    N |           Naive |           Euler |      Richardson
-------------------------------------------------------
   10 |    0.8080789524 |    0.7851445424 |    0.7853946423
   20 |    0.7972961956 |    0.7853979819 |    0.7853978722
   50 |    0.7902996532 |    0.7853981634 |    0.7853981545
  100 |    0.7878733503 |    0.7853981634 |    0.7853981628

Ziel: π/4 = 0.785398163397448

======================================================================
PROGRAMM ERFOLGREICH ABGESCHLOSSEN
======================================================================
```