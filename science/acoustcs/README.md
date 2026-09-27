# Acoustics Hearing


[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-120%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-87.31%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

Biologisch inspiriertes Modell fuer auditive Raumorientierung, feline
Hoerphysiologie, Cochlea-Filterung und digitale A/D-Wandlung. Die
Implementierung begleitet die wissenschaftliche Arbeit
`science/acoustcs/science/acoustcs.tex` und bildet die wesentlichen Inhalte
aus den Kapiteln meiner wissenschaftlichen Arbeit ab.

## Projektstruktur

```text
src/acoustcs/
	audio_converter.py       A/D-Wandlung und Quantisierung
	cochlear_model.py        Basilarmembran und Cochlea-Mechanik
	feline_auditory.py       Katzen-, Hunde- und Menschenmodell
	filterbank.py            Gammatone-nahe Cochlea-Filterbank
	spatial_localization.py  ITD, ILD und Raumlokalisierung
tests/                       Automatisierte Tests
science/acoustcs.tex         Wissenschaftliche Arbeit
```

## Installation

PowerShell im Projektverzeichnis:

```powershell
cd C:\Users\sepp5.AD\Git\science\science\acoustcs
.\venv\Scripts\Activate.ps1
python -m pip install ".[dev]"
```

Ohne Aktivierung der virtuellen Umgebung:

```powershell
.\venv\Scripts\python.exe -m pip install ".[dev]"
```

## Tests ausfuehren

```powershell
.\venv\Scripts\python.exe -m pytest
```

Coverage-Berichte werden entsprechend der `pyproject.toml` unter
`doc/coverage` erzeugt.

## Kurzes Beispiel

```python
import numpy as np

from acoustcs import FelineAuditory, SpatialLocalization
from acoustcs.audio_converter import AudioConverter
from acoustcs.filterbank import CochlearFilterBank

cat = FelineAuditory()
print(cat.summary())

localization = SpatialLocalization()
itd = localization.itd_woodworth_schlosberg(30.0)
print(f"ITD: {itd * 1e6:.2f} us")

filterbank = CochlearFilterBank(n_channels=32)
signal = np.zeros(9600)
cochleagram = filterbank.apply_filterbank(signal)
print(cochleagram.shape)

converter = AudioConverter(bit_depth=24)
samples = converter.quantize(signal)
print(f"Quantisierungs-SNR: {converter.quantization_snr():.2f} dB")
```

## Wissenschaftlicher Kontext

Die Kapitel meiner wissenschaftlichen Arbeit verbinden psychoakustische Raumhinweise, vergleichende
Hoerphysiologie und ein mechanisch motiviertes Cochlea-Modell. Das Python-
Paket ist als reproduzierbare numerische Begleitung gedacht: Die Formeln aus
der Arbeit werden parametrisiert, simuliert und durch Tests auf Randfaelle,
Monotonie, Skalierung und Integration geprueft.

## Erwerb

Der Preis für diese Arbeit und Software beträgt 521.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX