"""Experiment 3: global regime comparison and economic connectivity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecos.economic_analysis import (
    estimate_historical_omega,
    mutual_information_market,
    optimize_supply_chain_routing,
    resonance_analysis,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run() -> None:
    RESULTS.mkdir(exist_ok=True)
    countries = ["switzerland", "scandinavia", "germany_longterm", "weimar_1923", "north_korea"]
    omegas = np.array([estimate_historical_omega(country)[0] for country in countries])
    weights = np.array([0.18, 0.20, 0.24, 0.18, 0.20])
    global_omega = float(np.average(omegas, weights=weights))
    global_analysis = resonance_analysis(global_omega)

    matrix = np.array(
        [
            [mutual_information_market(left, right, correlation=0.7 if i != j else 1.0)
             for j, right in enumerate(omegas)]
            for i, left in enumerate(omegas)
        ]
    )
    np.savetxt(
        RESULTS / "03_global_mutual_information.csv",
        matrix,
        delimiter=",",
        header=",".join(countries),
        comments="",
    )
    np.savetxt(
        RESULTS / "03_global_regimes.csv",
        np.column_stack((countries, omegas, weights)),
        delimiter=",",
        header="country,omega,weight",
        comments="",
        fmt="%s",
    )

    supply = np.array([[180.0] * 4, [150.0] * 4, [120.0] * 4, [90.0] * 4])
    demand = np.array([[100.0] * 4, [90.0] * 4, [75.0] * 4])
    costs = np.array(
        [[1.0, 1.1, 1.2, 1.3], [1.3, 1.2, 1.1, 1.0], [1.7] * 4, [2.1] * 4]
    )
    routing = optimize_supply_chain_routing(supply, demand, costs)

    print("Experiment 3: global economy network")
    for country, omega in zip(countries, omegas):
        analysis = resonance_analysis(float(omega))
        print(
            f"{country:18s} omega={omega:.2f} "
            f"growth={analysis.growth_rate:.2%} "
            f"resonance={analysis.in_window}"
        )
    print(f"Weighted global omega: {global_omega:.3f}")
    print(f"Weighted global growth: {global_analysis.growth_rate:.2%}")
    print(f"Global routing cost: {routing.total_cost:.2f}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(countries, omegas, color=["#2a9d8f", "#2a9d8f", "#2a9d8f", "#e76f51", "#264653"])
    axes[0].axhspan(0.618, 0.796, alpha=0.15, color="#2a9d8f")
    axes[0].tick_params(axis="x", rotation=55)
    axes[0].set_ylabel("Estimated uncertainty omega")
    image = axes[1].imshow(matrix, cmap="viridis")
    axes[1].set_xticks(range(len(countries)), countries, rotation=55, ha="right")
    axes[1].set_yticks(range(len(countries)), countries)
    axes[1].set_title("Mutual information")
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.tight_layout()
    figure.savefig(RESULTS / "03_global_network.pdf")
    plt.close(figure)


if __name__ == "__main__":
    run()