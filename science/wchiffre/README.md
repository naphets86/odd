# Ein modulares kryptografisches Verfahren mit Dummy-Fenster-Fragmentierung und inverser Schluesselstruktur durch $23 \cdot 7 \equiv 1 \pmod{32}$

Wir präsentieren die **Fenster-Chiffre** (FC), ein neuartiges kryptografisches Protokoll, das drei unabhängige Sicherheitsmechanismen vereint:  

(i) eine RSA-analoge modulare Exponentiation mit dem inversen Schluesselpaar $(e,d) = (23,\,7)$ modulo $n = 32$ bzw. einem echten RSA-Modul,  

(ii) eine informationstheoretisch sichere Dummy-Fragmentierung, bei der jedes $k$-te Paket ein inhaltsloses Attrapper-Paket (Dummy) ist, dessen Position durch einen geheimen Offset~$s$ festgelegt wird, und  

(iii) eine schluesselgesteuerte Fensterauswahl, die ohne Kenntnis von~$s$ keine Unterscheidung echter von unechten Paketen erlaubt.

Wir beweisen die Korrektheit, Konfidenzialität und Ununterscheidbarkeit des Verfahrens formal und analysieren seine Komplexität sowie seinen Sicherheitsgrad. Der Ausblick zeigt Erweiterungen auf gitterbasierte post-quanten-sichere Varianten sowie adaptive Dummy-Strategien.

## Installation & Test
```bash
# Abhängigkeiten installieren
pip install ".[dev]"

# Tests mit Coverage ausführen
pytest
```

## Erwerb

Der Preis für diese Arbeit beträgt 9.999.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX
