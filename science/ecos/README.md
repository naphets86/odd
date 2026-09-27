# Economic Analysis Module

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-239%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-98.05%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

A Python module for formal economic system analysis. The implementation combines
uncertainty, resonance windows, stability, market dynamics, entropy, harmonic
analysis, and supply-chain optimization in a single `ecos` package.

## Highlights

- Resonance-window analysis for the uncertainty index `omega`
- Productivity, predictability, growth, inequality, and stability metrics
- Low-pass filtering and periodic shock response
- Historical regime comparison and market information measures
- Fourier and Leibniz harmonic models
- Supply-chain routing optimization with SciPy
- Reproducible local, global, and Monte-Carlo experiments

## Installation

Requires Python 3.8 or newer.

```powershell
python -m pip install ".[dev]"
```

The package is imported as:

```python
from ecos.economic_analysis import resonance_analysis

analysis = resonance_analysis(0.707)
print(analysis.growth_rate)
```

## Project Structure

```text
.
├── pyproject.toml
├── README.md
├── src/
│   └── ecos/
│       ├── __init__.py
│       ├── economic_analysis.py
│       ├── 01_simple_resonance_sweep.py
│       ├── 02_local_economy_panel.py
│       ├── 03_global_economy_network.py
│       ├── 04_monte_carlo_robustness.py
│       ├── EXPERIMENTS.md
│       └── results/
└── tests/
```

## Experiments

The four reproducible experiments live in `src/ecos/` and write their data and
publication-ready PDF plots to `src/ecos/results/`.

```powershell
python src/ecos/01_simple_resonance_sweep.py
python src/ecos/02_local_economy_panel.py
python src/ecos/03_global_economy_network.py
python src/ecos/04_monte_carlo_robustness.py
```

## Tests and Coverage

Run the complete test suite from the project root:

```powershell
python -m pytest -q
```

The current suite contains 239 tests and reports 98.05% coverage for the core
module. The HTML report is generated at
[doc/coverage/index.html](doc/coverage/index.html).

Useful focused commands:

```powershell
python -m pytest tests/test_core_functions.py -q
python -m pytest tests/test_advanced_analysis.py -q
python -m pytest --cov=ecos --cov-report=term-missing
```

## Development

The project uses `pytest`, `pytest-cov`, `black`, `flake8`, and `mypy` through
the development extra. The package source is `src/ecos/economic_analysis.py`;
all tests import it through `ecos.economic_analysis`.

```powershell
black src/ecos tests
flake8 src/ecos tests
mypy src/ecos/economic_analysis.py
```

## Erwerb

Der Preis für diese Arbeit und Software beträgt 801.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX
