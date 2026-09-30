"""Experiment 5: ship cargo size, profitability and oversupply."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecos.ship_freight import (
    ShipMarketParams,
    analyze,
    equilibrium_oversupply,
    oversupply_tolerance,
    unit_margin,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    params = ShipMarketParams()
    result = analyze(params)

    loads = np.linspace(20.0, 520.0, 501)
    margin = np.array([unit_margin(L, params.a0, params.Cf, params.kappa) for L in loads])
    oversupply = np.array(
        [equilibrium_oversupply(m, params.eps, params.gamma) for m in margin]
    )
    tolerance = np.array(
        [
            oversupply_tolerance(
                params.G / L, params.kappa_p, params.A, params.beta, params.alpha, params.T
            )
            for L in loads
        ]
    )
    data = np.column_stack([loads, params.G / loads, margin, oversupply, tolerance])
    np.savetxt(
        RESULTS / "05_ship_freight_sweep.csv",
        data,
        delimiter=",",
        header="load,departures,unit_margin,oversupply,tolerance",
        comments="",
    )

    summary = {
        "period_gap": result.G,
        "safety_margin": result.eps,
        "eoq_load": result.L_E,
        "just_worth_load": result.just_worth_load,
        "departures_at_just_worth": result.departures_at_just_worth,
        "buffer_limit": result.buffer_limit,
        "price_compatible_max_load": result.price_compatible_max,
        "admissible_interval": list(result.admissible),
    }
    (RESULTS / "05_ship_freight_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Experiment 5: ship freight")
    print(f"Period gap G: {result.G:.1f} kt/a")
    print(f"Just-worthwhile load: {result.just_worth_load:.1f} kt")
    print(f"Price-compatible maximum load: {result.price_compatible_max:.1f} kt")
    print(f"Buffer limit: {result.buffer_limit:.1f} kt")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(loads, margin, color="#457b9d")
    axes[0].axhline(result.eps, color="#e76f51", linestyle="--")
    axes[0].axvline(result.just_worth_load, color="#2a9d8f")
    axes[0].set_xlabel("Load L per departure (kt)")
    axes[0].set_ylabel("Unit margin m(L)")
    axes[1].plot(loads, oversupply, color="#e76f51", label="oversupply")
    axes[1].plot(loads, tolerance, color="#2a9d8f", label="price tolerance")
    axes[1].set_xlabel("Load L per departure (kt)")
    axes[1].set_ylabel("kt per year")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "05_ship_freight.pdf")
    plt.close(figure)


if __name__ == "__main__":
    run()
