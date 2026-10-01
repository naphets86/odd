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

## Model Overview

The package turns the central idea of the accompanying work
(`science/ecos.tex`, *Band der Arbeiten zur Wirtschaft*) into code: an economic
system is characterised by a single **uncertainty index** `Ω ∈ [0, 1]`.
Low `Ω` means a strongly controlled economy, high `Ω` an unstructured,
crisis-prone one. Between the two lies a **resonance window** in which the model
yields high growth, high predictability and moderate inequality.

All metrics in `economic_analysis.py` are deterministic functions of `Ω` (plus a
few parameters). They are **model constructions and calibrations, not
estimates from data**; see [Model status](#model-status-derived-calibrated-heuristic)
and [Known limitations](#known-limitations-and-implementation-notes) before
interpreting any number as an empirical result.

### Constants

| Constant | Value | Definition |
|---|---|---|
| `GOLDEN_RATIO` | 1.6180 | φ = (1 + √5) / 2 |
| `GOLDEN_RATIO_INV` | 0.6180 | 1/φ |
| `OPTIMAL_UNCERTAINTY` = `RESONANCE_WINDOW_CENTER` | 0.7071 | sin(π/4) |
| `RESONANCE_WINDOW_MIN` | 0.6180 | 1/φ |
| `RESONANCE_WINDOW_MAX` | 0.7962 | 1/φ + 2·(sin(π/4) − 1/φ) |

The window is symmetric around `sin(π/4)` with half-width Δh = sin(π/4) − 1/φ ≈ 0.0891.

## Core Metrics

All functions validate `Ω ∈ [0, 1]` and raise `ValueError` otherwise, with the
exception of `calculate_predictability` (see limitations).

| Function | Formula | Range / values |
|---|---|---|
| `in_resonance_window(Ω)` | `1/φ ≤ Ω ≤ 1/φ + 2Δh` | `bool` |
| `calculate_productivity(Ω, λ=2)` | `P = 1/(1−Ω) − λ(Ω−Ω_c)²`, floored at machine epsilon | 0 at Ω=0; 2.60 / 3.41 / 4.89 at lower bound / centre / upper bound |
| `calculate_freedom_degree(Ω)` | `F = −ln(1−Ω)` | strictly increasing; 0.962 / 1.228 / 1.591 at lower bound / centre / upper bound |
| `calculate_predictability(Ω)` | `V = max(ε, 1 − |Ω−Ω_c| / 0.5)` | 1.0 at the centre, 0.822 at both window bounds, ε for Ω < 0.207 |
| `calculate_price_variance(Ω, γ=1)` | `σ² = γ·Ω(1−Ω)` | maximal at Ω = 0.5 (0.25γ) |
| `calculate_gini_coefficient(Ω)` | piecewise, see below | 0.391–0.409 inside the window |
| `calculate_growth_rate(Ω)` | piecewise, see below | 4.29–4.50 % inside the window |
| `resonance_analysis(Ω)` | bundles all of the above plus `stability_score = 1 − |Ω−Ω_c|` and `window_distance` | `ResonanceAnalysis` dataclass |

### Piecewise definitions

**Growth rate** `g(Ω)` (annual, as a decimal):

| Zone | Definition |
|---|---|
| `Ω < 1/φ` (over-controlled) | `g = 0.005 · Ω / (1/φ)` |
| `1/φ ≤ Ω ≤ Ω_max` (window) | `g = 0.03 · (1 + 0.4·V + 0.1·min(1, P))` |
| `Ω > Ω_max` (crisis) | `g = −0.02 · ((Ω − Ω_max)/(1 − Ω_max))²` |

**Gini coefficient** `G(Ω)`:

| Zone | Definition |
|---|---|
| `Ω < 1/φ` | `G = 0.2 + 0.4 · Ω / (1/φ)` |
| window | `G = 0.40 + 0.1 · (Ω − Ω_c)` |
| `Ω > Ω_max` | `G = 0.45 + 0.55 · (Ω − Ω_max)/(1 − Ω_max)` |

Both functions are **discontinuous at the window bounds** (growth jumps
0.5 % → 4.29 % at the lower and 4.29 % → 0 % at the upper bound; the Gini value
jumps 0.600 → 0.391 and 0.409 → 0.450). This is how the model encodes the
"window" and is visible as steps in the Experiment 1 sweep.

### Reference values across the regimes

Output of `resonance_analysis` and `interpret_regime` (growth in % p.a.):

| Ω | In window | P | F | V | Gini | Growth | `interpret_regime` |
|---|---|---|---|---|---|---|---|
| 0.10 | no | 0.37 | 0.11 | 0.00 | 0.265 | 0.08 | Extreme Control |
| 0.35 | no | 1.28 | 0.43 | 0.29 | 0.427 | 0.28 | Extreme Control |
| 0.47 | no | 1.77 | 0.64 | 0.53 | 0.504 | 0.38 | Over-controlled |
| 0.55 | no | 2.17 | 0.80 | 0.69 | 0.556 | 0.44 | Over-controlled |
| 0.618 | yes | 2.60 | 0.96 | 0.82 | 0.391 | 4.29 | Toward Optimum |
| 0.65 | yes | 2.85 | 1.05 | 0.89 | 0.394 | 4.36 | Toward Optimum |
| 0.707 | yes | 3.41 | 1.23 | 1.00 | 0.400 | 4.50 | Optimal Balance |
| 0.75 | yes | 4.00 | 1.39 | 0.91 | 0.404 | 4.40 | Toward Chaos |
| 0.796 | yes | 4.89 | 1.59 | 0.82 | 0.409 | 4.29 | Toward Chaos |
| 0.85 | no | 6.63 | 1.90 | 0.71 | 0.595 | −0.14 | Crisis Regime |
| 0.88 | no | 8.27 | 2.12 | 0.65 | 0.676 | −0.34 | Crisis Regime |
| 0.95 | no | 19.88 | 3.00 | 0.51 | 0.865 | −1.14 | Extreme Chaos |

`interpret_regime` uses the thresholds 0.4, 0.618, 0.796, 0.85 and 0.95, plus a
±0.03 band around the centre for "Optimal Balance".

### Historical regimes

`estimate_historical_omega(name)` is a **lookup table** of assumed indices (the
name is case-insensitive, spaces are mapped to underscores). It returns
`(Ω, interpretation)`; unknown names raise `ValueError` listing the keys.

| Key | Ω | In window | Modelled growth |
|---|---|---|---|
| `soviet_union` | 0.35 | no | 0.28 % |
| `east_germany` | 0.30 | no | 0.24 % |
| `north_korea` | 0.10 | no | 0.08 % |
| `switzerland` | 0.70 | yes | 4.48 % |
| `scandinavia` | 0.68 | yes | 4.43 % |
| `germany_longterm` | 0.70 | yes | 4.48 % |
| `weimar_1923` | 0.95 | no | −1.14 % |
| `argentina_2001` | 0.88 | no | −0.34 % |
| `zimbabwe_2008` | 0.98 | no | −1.63 % |
| `uk_cornlaw_before` / `_during` / `_after` | 0.72 / 0.55 / 0.70 | yes / no / yes | 4.47 % / 0.44 % / 4.48 % |

These values are inputs to the model, not measurements.

## Filter Model

Markets are modelled as a first-order low-pass filter with time constant `τ`
(`dV_out/dt + V_out/τ = V_in/τ`).

| Function | Formula |
|---|---|
| `calculate_filter_cutoff_frequency(τ)` | `f_c = 1 / (2πτ)` |
| `calculate_filter_magnitude(f, τ)` | `|H| = 1 / √(1 + (2πfτ)²)` — `f` is an ordinary frequency (cycles per time unit), the factor 2π is applied internally |
| `filter_shock_response(A, f, τ)` | returns `(A·|H|, |H|)` |
| `filter_stability_condition(T, τ)` | `T > 2πτ`, i.e. the shock frequency lies below the cutoff |

Example (Experiment 2): a periodic shock with `f = 1/12` and `τ = 2` is damped to
a factor of 0.691.

```python
from ecos.economic_analysis import filter_shock_response

damped, factor = filter_shock_response(1.0, 1 / 12, 2.0)
print(damped, factor)  # 0.6906... 0.6906...
```

Note that a first-order low-pass has no resonance: it attenuates *fast* shocks.
`filter_stability_condition` is therefore a threshold on the shock period relative
to `2πτ`, not a resonance test in the physical sense.

## Price Dynamics

`market_adjustment_dynamics(demand_shock, alpha, eta, T, t_array)` solves the
restoring price equation for the deviation `P` from the equilibrium price:

```
dP/dt = α·P(t) + η·sin(2πt/T),   α < 0,   P(0) = demand_shock
```

With `a = |α|` and `ω = 2π/T` the exact solution is

```
P(t)   = (P(0) − P_p(0))·exp(−a·t) + P_p(t)
P_p(t) = η·(a·sin(ωt) − ω·cos(ωt)) / (a² + ω²) = A·sin(ωt + φ)
A      = η / √(a² + ω²),   φ = −arctan(ω/a)
```

A `UserWarning` is issued for `alpha ≥ 0`. `ship_price_response(alpha, eta, T)`
returns the steady-state amplitude `A` through the filter representation
`(η/|α|)·|H(1/T)|` with `τ = 1/|α|`; both routes give the same value, which is
also used in the ship-freight price band.

```python
import numpy as np
from ecos.economic_analysis import market_adjustment_dynamics, ship_price_response

t = np.array([0.0, 1.0, 2.0])
print(market_adjustment_dynamics(1.0, -0.5, 0.3, 4.0, t))  # [1.  0.767 0.605]
print(ship_price_response(-0.5, 0.3, 4.0))                  # 0.1820
```

## Harmonic and Spectral Analysis

| Function | Purpose |
|---|---|
| `leibniz_series_coefficient(n)` | `(−1)ⁿ / (2n+1)` |
| `leibniz_series_sum(N)` | partial sum of `π/4 = Σ (−1)ⁿ/(2n+1)` |
| `leibniz_convergence_error(N)` | alternating-series bound `1/(2N+1)` (valid but about twice the true error: N = 100 gives bound 0.00498, actual error 0.0025) |
| `market_supply_odd_harmonics(t, S0, amplitudes, frequencies, phases=None)` | `S(t) = S₀ + Σₙ Sₙ/(2n+1)·sin((2n+1)·fₙ·t + φₙ)`; `zip` truncates silently if the lists differ in length |
| `fourier_rectangle_wave(t, amplitude, period, num_harmonics)` | `(4A/π) Σ sin((2n+1)ωt)/(2n+1)` (square wave; shows Gibbs overshoot near jumps) |

The odd-harmonic structure with `1/(2n+1)` weights is the link between the
Leibniz series, the rectangle wave and the damping of higher harmonics by the
low-pass filter, as developed in the chapters on the low-pass filter and the
Leibniz structure.

## Entropy, Information and Stability Indicators

| Function | Definition | Interpretation |
|---|---|---|
| `market_entropy(Ω)` | binary Shannon entropy `−Ω ln Ω − (1−Ω) ln(1−Ω)` | maximal (ln 2) at Ω = 0.5, *not* at the window centre |
| `mutual_information_market(Ω₁, Ω₂, ρ=0.5)` | heuristic `max(0, ½(H(Ω₁)+H(Ω₂)) − J)·ρ` with `J = −½(Ω₁+Ω₂)·ln(½(Ω₁+Ω₂))` | **coupling indicator**, not a true mutual information |
| `lyapunov_exponent_economic_system(Ω)` | `λ = −1/(1−Ω)` (−∞ for Ω = 1) | negative everywhere, i.e. a proportionality model, not a computed exponent |
| `lyapunov_stability_margin(Ω)` | `exp(λ + 0.25·Ω/(1−Ω))` | in (0, 1) and *decreasing* in Ω |

Use these as relative indicators only. In particular `λ` is more negative for
larger Ω, whereas the margin decreases; the two must not be read as one
consistent stability ordering (see limitations).

## Supply-Chain Routing

`optimize_supply_chain_routing(supply, demand, costs, ...)` minimises
`Σ cost[s,t]·x[s,t]` over allocations `x ≥ 0` subject to

- `x[s,t] ≤ supply[s,t]` for every supplier `s` and period `t`,
- `Σₛ x[s,t] ≥ demand[l,t]` for every demand row `l` and period `t`,

using SciPy's SLSQP. It returns a `SupplyChainOptimization` with `total_cost`,
`allocation` (S×T), `supply_levels`, `demand_satisfaction` and `convergence`.

```python
import numpy as np
from ecos.economic_analysis import optimize_supply_chain_routing

supply = np.array([[10.0, 10.0], [10.0, 10.0]])
demand = np.array([[5.0, 5.0]])
costs = np.array([[1.0, 1.0], [2.0, 2.0]])
result = optimize_supply_chain_routing(supply, demand, costs)
print(result.total_cost, result.convergence)  # ≈ 10.0 True
```

The optional arguments `quality`, `weights`, `min_quality` and `lambda_penalty`
are accepted for API compatibility but **currently have no effect**.

## Ship Freight

`ecos.ship_freight` (re-exported through `ship_market_assessment`,
`ship_price_response` and `interpret_ship_freight`) implements the chapter
*Schiffsfracht im wirtschaftlichen Gleichgewicht*. Main building blocks:

| Block | Functions |
|---|---|
| Demand and period gap | `demand_rate`, `gap_rate`, `period_gap`, `cumulative_gap` |
| Balance theorem `N·L = G` | `load_for_departures`, `departures_for_load`, `period_imbalance` |
| Buffer | `overflow_time_estimate`, `shortage_time_estimate`, `buffer_requirement`, `buffer_limit_sufficient`, `stock_at_calls` |
| Profitability | `unit_margin`, `optimal_load` (EOQ), `max_margin`, `viable_load_interval`, `safety_margin`, `critical_delta` |
| Equilibrium | `equilibrium_freight`, `equilibrium_oversupply`, `just_worth_load`, `tolerance_band`, `integer_departures` |
| Price disturbance | `price_amplitude`, `price_phase`, `peak_disturbance`, `mean_disturbance`, `oversupply_tolerance` |
| Cells and ports | `cell_shares`, `cell_load_limit`, `pool_load_limit` |
| Parameters | `ShipMarketParams` (frozen dataclass), `analyze`, `price_margin`, `is_price_compatible` |

`interpret_ship_freight(analysis)` classifies the outcome as `"not viable"`,
`"blocked"`, `"knife-edge"` (only the just-worthwhile load `L⁻` is admissible)
or `"tolerance band"`.

```python
from ecos.economic_analysis import ship_market_assessment, interpret_ship_freight

analysis = ship_market_assessment()          # default parameters of the chapter
print(analysis.just_worth_load)              # 68.84 kt
print(interpret_ship_freight(analysis))      # knife-edge
```

## Reproduced Results

The committed files in `src/ecos/results/` reproduce the values of the
experiment chapter (fixed seeds):

| Experiment | Key result |
|---|---|
| 1 Resonance sweep | 501 points; maximum growth 4.50 % at Ω = 0.708, productivity 3.425, predictability 0.998 |
| 2 Local panel | four regions over 60 months; shock at `f = 1/12`, `τ = 2` damped to 0.691 |
| 3 Global network | weights give Ω = 0.621 and 4.29 % growth |
| 4 Monte Carlo | 20 000 draws around Ω = 0.707 (σ = 0.12): 54.5 % in the window, mean growth 2.41 %, 90 % interval [−0.58 %, 4.48 %], 68.1 % stable filter configurations |
| 5 Ship freight | `G = 1500` kt/a, `L_E = 173.21` kt, just-worthwhile load `L⁻ = 68.84` kt, buffer limit 181.69 kt, price-compatible maximum 86.70 kt; admissible interval collapses to `L⁻` because the tolerated oversupply is 0 |

All experiments use synthetic inputs. They demonstrate the model and are
**not** empirical analyses.

## Mapping to the Paper

| Code | Location in `science/ecos.tex` |
|---|---|
| Resonance window, `P`, `F`, `V`, growth, Gini, Lyapunov | chapter *Der Tiefpassfilter und die e-Funktion*, section *Der trigonometrisch-harmonische Resonanzpunkt* |
| `F(Ω) = −ln(1−Ω)` and its monotonicity | chapter *Ebbe und Flut als Unsicherheitsstruktur*, section *Hauptsätze und Beweise* |
| Filter functions | chapter *Der Tiefpassfilter und die e-Funktion* |
| Leibniz and odd harmonics | sections *Der ungerade Anteil für Differentialgleichungen* and *Die Leibniz-Struktur in Wirtschaftsdynamiken* |
| Historical regimes | case-study plots `Plot_06`–`Plot_08` (Switzerland, Argentina, Soviet Union) |
| Ship freight | chapter *Schiffsfracht im wirtschaftlichen Gleichgewicht* |
| Experiments 1–5 | chapter *Experimentelle Auswertung der Wirtschaftsanalyse* |

## Model Status: Derived, Calibrated, Heuristic

| Status | Meaning | Items |
|---|---|---|
| Derived | follows analytically from the stated equations | `F`, filter magnitude and cutoff, price dynamics and amplitude, rectangle-wave series, Leibniz coefficients and error bound, ship-freight results |
| Calibrated | functional form and constants chosen by the author | `P`, `V`, growth, Gini, window bounds, historical `Ω` values |
| Heuristic | indicator without a formal derivation | `mutual_information_market`, `lyapunov_exponent_economic_system`, `lyapunov_stability_margin`, `filter_stability_condition` as a "stability" test |

## Erwerb

Der Preis für diese Arbeit und Software beträgt 1.111.000.000,00 EUR.

### Zahlungsinformationen

Name: Stephan Epp  
IBAN: DE24 5003 1900 0012 5603 20
BIC: BBVADEFFXXX
