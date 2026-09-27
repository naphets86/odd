# Learning SAT in Boolean Circuits

Polynomielle Lösung von Spezifikationsproblemen durch den Subgraph Algorithmus

Diese Arbeit verbindet zwei vorherige Arbeiten: Die Masterarbeit **Learning Monotone DNF in Boolean Circuits** (Epp13) und die Arbeit **Der Subgraph Algorithmus mit Signatur-Methode** (Epp26). Erstere entwickelt einen Algorithmus zum Lernen einer versteckten monotonen DNF-Formel $H \in \text{M-DNF}$ innerhalb eines Boolean Circuits mittels Query-Learning und SAT-Solver-Aufrufen. Letztere zeigt, dass das Subgraph-Isomorphismus-Problem in $O(n^3)$ Zeit lösbar ist, woraus $\text{P} = \text{NP}$ folgt.

Da $\text{P} = \text{NP}$ gilt, ist das SAT-Problem in polynomieller Zeit lösbar. Diese Arbeit nutzt diese Erkenntnis, um das Spezifikationsproblem aus \cite{Epp13} von monotonen DNF-Formeln auf \textbf{beliebige CNF-Formeln} zu erweitern. Der zentrale Beitrag ist die Erstellung eines polynomiellen Algorithmus zum Lernen von SAT-Formeln in Boolean Circuits, wobei der Subgraph Algorithmus als polynomieller SAT-Solver eingesetzt wird.

## Erwerb

Der Preis für diese Arbeit beträgt 111.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX

## Installation & Experimente

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# Experimente ausführen
python3 -m src.experiments

# Demo ausführen
python3 -m src.demo
```