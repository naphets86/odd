"""Experiment 1: map the economic resonance window."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecos.economic_analysis import (
    RESONANCE_WINDOW_MAX,
    RESONANCE_WINDOW_MIN,
    calculate_growth_rate,
    calculate_predictability,
    calculate_productivity,
    resonance_analysis,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    omegas = np.linspace(0.0, 1.0, 501)
    rows = [resonance_analysis(float(omega)) for omega in omegas]

    data = np.array(
        [
            [
                row.omega,
                row.productivity,
                row.predictability,
                row.growth_rate,
                row.gini_coefficient,
                float(row.in_window),
            ]
            for row in rows
        ]
    )
    np.savetxt(
        RESULTS / "01_resonance_sweep.csv",
        data,
        delimiter=",",
        header="omega,productivity,predictability,growth_rate,gini,in_window",
        comments="",
    )

    best = rows[int(np.argmax([row.growth_rate for row in rows]))]
    print("Experiment 1: resonance sweep")
    print(f"Resonance window: {RESONANCE_WINDOW_MIN:.3f} - {RESONANCE_WINDOW_MAX:.3f}")
    print(f"Best sampled omega: {best.omega:.3f}")
    print(f"Best growth rate: {best.growth_rate:.2%}")
    print(
        "Center metrics: "
        f"productivity={calculate_productivity(best.omega):.3f}, "
        f"predictability={calculate_predictability(best.omega):.3f}"
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(omegas, data[:, 1], label="Productivity")
    axes[0].plot(omegas, data[:, 2], label="Predictability")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[1].plot(omegas, 100 * data[:, 3], label="Growth rate (%)")
    axes[1].plot(omegas, data[:, 4], label="Gini coefficient")
    axes[1].set_xlabel("Uncertainty index omega")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    for axis in axes:
        axis.axvspan(RESONANCE_WINDOW_MIN, RESONANCE_WINDOW_MAX, alpha=0.15)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "01_resonance_sweep.pdf")
    plt.close(figure)


if __name__ == "__main__":
    run()