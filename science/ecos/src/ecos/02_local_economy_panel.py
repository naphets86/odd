"""Experiment 2: local economy panel with a supply-chain shock."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecos.economic_analysis import (
    calculate_filter_magnitude,
    calculate_growth_rate,
    calculate_productivity,
    calculate_predictability,
    filter_shock_response,
    optimize_supply_chain_routing,
    resonance_analysis,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    months = np.arange(1, 61)
    regions = {
        "industrial": 0.59,
        "service": 0.68,
        "rural": 0.47,
        "innovation": 0.73,
    }

    records = []
    for region, baseline in regions.items():
        trend = np.linspace(0.0, 0.035, months.size)
        seasonal = 0.018 * np.sin(2.0 * np.pi * months / 12.0)
        noise = rng.normal(0.0, 0.008, months.size)
        omega_series = np.clip(baseline + trend + seasonal + noise, 0.0, 1.0)
        for month, omega in zip(months, omega_series):
            analysis = resonance_analysis(float(omega))
            records.append(
                [
                    month,
                    region,
                    omega,
                    analysis.productivity,
                    analysis.predictability,
                    analysis.growth_rate,
                    float(analysis.in_window),
                ]
            )

    output = np.array(records, dtype=object)
    np.savetxt(
        RESULTS / "02_local_panel.csv",
        output,
        delimiter=",",
        header="month,region,omega,productivity,predictability,growth_rate,in_window",
        comments="",
        fmt="%s",
    )

    shock_frequency = 1.0 / 12.0
    shock, damping = filter_shock_response(1.0, shock_frequency, tau=2.0)
    print("Experiment 2: local economy panel")
    print(f"Regions tracked: {', '.join(regions)}")
    print(f"Annualized shock frequency: {shock_frequency:.3f}")
    print(f"Filtered shock: {shock:.3f} (transfer magnitude={damping:.3f})")
    for region, baseline in regions.items():
        analysis = resonance_analysis(baseline)
        print(
            f"{region:12s} omega={baseline:.3f} "
            f"growth={analysis.growth_rate:.2%} "
            f"productivity={analysis.productivity:.2f}"
        )

    supply = np.array([[120.0] * 12, [90.0] * 12, [70.0] * 12])
    demand = np.array([[80.0] * 12, [65.0] * 12])
    costs = np.array([[1.0] * 12, [1.4] * 12, [1.9] * 12])
    routing = optimize_supply_chain_routing(supply, demand, costs)
    print(
        f"Supply-chain cost={routing.total_cost:.2f}; "
        f"converged={routing.convergence}"
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for region in regions:
        selected = output[output[:, 1] == region]
        axes[0].plot(selected[:, 0].astype(int), selected[:, 2].astype(float), label=region)
        axes[1].plot(
            selected[:, 0].astype(int),
            100 * selected[:, 5].astype(float),
            label=region,
        )
    axes[0].set_ylabel("Omega")
    axes[1].set_ylabel("Growth rate (%)")
    axes[1].set_xlabel("Month")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(RESULTS / "02_local_panel.pdf")
    plt.close(figure)


if __name__ == "__main__":
    run()