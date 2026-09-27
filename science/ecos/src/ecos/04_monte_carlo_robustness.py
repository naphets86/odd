"""Experiment 4: Monte-Carlo robustness of the resonance-window conclusion."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecos.economic_analysis import (
    RESONANCE_WINDOW_MAX,
    RESONANCE_WINDOW_MIN,
    calculate_growth_rate,
    calculate_productivity,
    calculate_predictability,
    filter_stability_condition,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(404)
    samples = np.clip(rng.normal(loc=0.707, scale=0.12, size=20_000), 0.0, 1.0)
    growth = np.array([calculate_growth_rate(float(value)) for value in samples])
    productivity = np.array([calculate_productivity(float(value)) for value in samples])
    predictability = np.array([calculate_predictability(float(value)) for value in samples])
    in_window = (samples >= RESONANCE_WINDOW_MIN) & (samples <= RESONANCE_WINDOW_MAX)

    stability_grid = [
        filter_stability_condition(period, tau)
        for period in np.linspace(1.0, 30.0, 100)
        for tau in np.linspace(0.25, 3.0, 50)
    ]
    summary = {
        "sample_count": int(samples.size),
        "window_probability": float(np.mean(in_window)),
        "growth_mean": float(np.mean(growth)),
        "growth_p05": float(np.percentile(growth, 5)),
        "growth_p95": float(np.percentile(growth, 95)),
        "productivity_median": float(np.median(productivity)),
        "predictability_median": float(np.median(predictability)),
        "filter_stability_fraction": float(np.mean(stability_grid)),
    }
    (RESULTS / "04_monte_carlo_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Experiment 4: Monte-Carlo robustness")
    print(f"Samples: {summary['sample_count']}")
    print(f"Probability of resonance window: {summary['window_probability']:.1%}")
    print(
        f"Growth 90% interval: {summary['growth_p05']:.2%} to "
        f"{summary['growth_p95']:.2%}"
    )
    print(f"Stable filter grid fraction: {summary['filter_stability_fraction']:.1%}")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(samples, bins=50, color="#457b9d", alpha=0.85)
    axes[0].axvspan(RESONANCE_WINDOW_MIN, RESONANCE_WINDOW_MAX, color="#2a9d8f", alpha=0.25)
    axes[0].set_xlabel("Sampled omega")
    axes[0].set_ylabel("Count")
    axes[1].scatter(productivity, growth * 100, s=4, alpha=0.15, color="#e76f51")
    axes[1].set_xlabel("Productivity")
    axes[1].set_ylabel("Growth rate (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "04_monte_carlo_robustness.pdf")
    plt.close(figure)


if __name__ == "__main__":
    run()