# Economic Analysis Module

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-297%20passed-4c1)](tests/)
[![Test Coverage](https://img.shields.io/badge/Test%20Coverage-95.93%25-brightgreen)](doc/coverage/index.html)
[![scicov](https://img.shields.io/badge/scicov-10-ff69b4)](doc/coverage/index.html)

A Python package for formal economic system analysis. It combines uncertainty,
resonance windows, stability, market dynamics, entropy, harmonic analysis,
supply-chain optimization, and a ship-freight equilibrium model. The freight
model formalizes the balance between periodic transport demand and ship capacity,
including profitability, buffer requirements, and the smallest just-worthwhile
cargo load.

## Highlights

- Resonance-window analysis for the uncertainty index `omega`
- Productivity, predictability, growth, inequality, and stability metrics
- Low-pass filtering and periodic shock response
- Historical regime comparison and market information measures
- Fourier and Leibniz harmonic models
- Supply-chain routing optimization with SciPy
- Ship-freight analysis: viable cargo loads, market equilibrium, oversupply,
  price disturbance, and buffer limits
- Reproducible local, global, and Monte-Carlo experiments

## Installation

Requires Python 3.8 or newer.

```powershell
python -m pip install ".[dev]"
```

The main analysis API is implemented in
[`src/ecos/economic_analysis.py`](src/ecos/economic_analysis.py). It provides
resonance and stability measures, market and filter analysis, supply-chain
optimization, and an integrated interface to the ship-freight model in
[`src/ecos/ship_freight.py`](src/ecos/ship_freight.py).

```python
from ecos.economic_analysis import resonance_analysis, ship_market_assessment

resonance = resonance_analysis(0.707)
freight = ship_market_assessment()

print(resonance.growth_rate)
print(freight.just_worth_load)
```

The package is installed and imported as:

```python
from ecos.economic_analysis import resonance_analysis

analysis = resonance_analysis(0.707)
print(analysis.growth_rate)
```

## Experiments

The five reproducible experiments live in `src/ecos/`. The ship-freight
experiment writes a CSV sweep, a JSON summary, and a PDF plot to
`src/ecos/results/`.

```powershell
python src/ecos/01_simple_resonance_sweep.py
python src/ecos/02_local_economy_panel.py
python src/ecos/03_global_economy_network.py
python src/ecos/04_monte_carlo_robustness.py
python src/ecos/05_ship_freight_experiment.py
```

The freight experiment is implemented in
[`src/ecos/05_ship_freight_experiment.py`](src/ecos/05_ship_freight_experiment.py).

## Tests and Coverage

Run the complete test suite from the project root:

```powershell
pytest
```

The test suite covers the core analysis and the ship-freight model. The HTML
coverage report is generated at [doc/coverage/index.html](doc/coverage/index.html).

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

Der Preis für diese Arbeit und Software beträgt 1.111.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX
