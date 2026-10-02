"""Experiment 5: ship cargo size, profitability and oversupply.

Part A (``run_ship_freight``): cargo size, unit margin, oversupply and price
tolerance (chapter "Schiffsfracht im wirtschaftlichen Gleichgewicht").

Part B (``run_sea_bacteria``): freshness of the cargo on the sea voyage
(chapter "Seefracht und bakterielle Entwicklung"): vehicle comparison, article
classes, freshness limit L_F of the pulse, double clamp of the voyage duration
and cooling failure. All bacterial parameters are illustrative (chosen, not
estimated), as in the chapter.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecos.ship_freight import (
    ARTICLE_CLASS_C,
    ARTICLE_CLASS_L,
    ShipMarketParams,
    TransportSegment,
    admissible_storage_age,
    age_distribution,
    analyze,
    atmosphere_exposure,
    bacterial_growth_rate,
    critical_voyage_duration,
    equilibrium_oversupply,
    evaluate_chain,
    exponential_fixed_cost,
    failure_tolerance,
    failure_work,
    fresh_and_worthwhile,
    freshness_load_limit,
    limit_duration,
    log_growth,
    min_departures_for_freshness,
    oversupply_tolerance,
    profile_work,
    sea_suitability,
    sea_temperature_threshold,
    segment_work,
    ship_failure_mean,
    ship_failure_variance,
    ship_failure_variance_limit,
    unit_margin,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# --- Part B settings (Beispiele sb-modi, sb-klassen, sb-LF, sb-ausfall) -----
DAYS_PER_YEAR = 365.0
HOLD_TEMPERATURE = 4.0  # trade holds the cargo at 4 degC
STORE_TEMPERATURE = 0.0  # port cold store
VOYAGE_DAYS_CLASSES = 22.0  # Beispiel sb-klassen
# Atmosphere of the controlled-atmosphere (CA) variant: c0, c_inf (%), k (1/d)
CA_C0, CA_C_INF, CA_LEAK = 0.0, 5.0, 0.5
# Cooling failure (Beispiel sb-ausfall)
FAIL_SET, FAIL_AMBIENT, FAIL_TAU_P = 0.0, 27.0, 0.5
FAIL_DURATIONS = (0.5, 1.0, 2.0)
# Illustrative failure probabilities for the two-stage model (chosen here)
P_SHIP, P_CONTAINER = 0.02, 0.05
CONTAINER_COUNTS = (1, 2, 5, 10, 50, 1000)


def _finite(value):
    """JSON-safe number: infinity and NaN become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def run_ship_freight() -> dict:
    """Part A: cargo size, profitability and oversupply (unchanged results)."""
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
    return summary


def _vehicle_chains(article, exposure):
    """Transport chains of Beispiel sb-modi (days, degC, degC^2)."""
    load = TransportSegment(1.5, 3.0, 2.25)
    unload = TransportSegment(0.5, 5.0, 4.0)

    def voyage(theta, with_atmosphere):
        return TransportSegment(
            19.0, theta, 0.16, exposure=exposure if with_atmosphere else None
        )

    return {
        "sea_standard": [load, voyage(0.0, False), unload],
        "sea_ca": [load, voyage(0.0, True), unload],
        "sea_ca_subcooled": [load, voyage(-1.5, True), unload],
        "road": [TransportSegment(3.0, 2.0, 1.0)],
        "air": [
            TransportSegment(0.5, 4.0, 4.0),
            TransportSegment(0.5, 12.0, 16.0),
            TransportSegment(0.5, 3.0, 0.64),
            TransportSegment(0.5, 4.0, 4.0),
        ],
    }


def run_sea_bacteria() -> dict:
    """Part B: freshness of the cargo on the voyage."""
    RESULTS.mkdir(exist_ok=True)
    article = ARTICLE_CLASS_C
    params = ShipMarketParams()
    econ = analyze(params)
    omega_s = article.omega_s
    mu_hold = bacterial_growth_rate(HOLD_TEMPERATURE, 0.0, article)
    mu_store = bacterial_growth_rate(STORE_TEMPERATURE, 0.0, article)

    # --- 1. Vehicle comparison (Beispiel sb-modi) ---------------------------
    exposure = atmosphere_exposure(19.0, CA_C0, CA_C_INF, CA_LEAK, article.kappa)
    chains = _vehicle_chains(article, exposure)
    chain_rows = {}
    for name, segments in chains.items():
        r = evaluate_chain(segments, article, mu_hold)
        chain_rows[name] = {
            "duration_d": r.duration,
            "omega": r.omega,
            "log_increase": r.log_increase,
            "growth_factor": r.growth_factor,
            "omega_over_omega_s": r.budget_ratio,
            "fresh": r.fresh,
            "residual_life_d": r.residual_life,
        }
    threshold = sea_temperature_threshold(5.0, 2.0, 20.0, article.theta_min)

    # --- 2. Article classes (Beispiel sb-klassen) ---------------------------
    class_rows = {}
    frozen = TransportSegment(VOYAGE_DAYS_CLASSES, -18.0)
    for label, art, seg in (
        ("C", ARTICLE_CLASS_C, TransportSegment(VOYAGE_DAYS_CLASSES, 2.0, 0.25)),
        ("L", ARTICLE_CLASS_L, TransportSegment(VOYAGE_DAYS_CLASSES, 2.0, 0.25)),
        ("F", ARTICLE_CLASS_C, frozen),  # class F: C cargo kept below theta_min
    ):
        omega = profile_work([seg], art)
        mu_voyage = bacterial_growth_rate(seg.theta_bar, 0.0, art)
        class_rows[label] = {
            "omega_s": art.omega_s,
            "omega_voyage": omega,
            "log_increase": log_growth(omega, art.h0),
            "sigma": _finite(sea_suitability(art.omega_s, mu_voyage, seg.length)),
            "limit_duration_d": _finite(
                limit_duration(art.omega_s, 0.0, 0.0, mu_voyage)
            ),
        }

    # --- 3. Pulse age and freshness limit (Beispiel sb-LF) ------------------
    just_worth = econ.just_worth_load
    departures = econ.G / just_worth
    u_bar = econ.G / DAYS_PER_YEAR  # kt per day
    ages = age_distribution(econ.G / departures, u_bar)
    unload_work = segment_work(0.5, 5.0, 4.0, article)
    retail_work = segment_work(1.0, HOLD_TEMPERATURE, 0.0, article)
    omega_after = unload_work + retail_work
    freshness_rows = {}
    for name in ("sea_ca", "sea_ca_subcooled"):
        omega_before = profile_work(chains[name][:2], article)
        a_ok = admissible_storage_age(omega_s, omega_before, omega_after, mu_store)
        l_f = freshness_load_limit(u_bar, a_ok)
        n_f = min_departures_for_freshness(DAYS_PER_YEAR, a_ok, u_bar)
        freshness_rows[name] = {
            "omega_before": omega_before,
            "omega_after": omega_after,
            "a_ok_d": a_ok,
            "min_departures_per_year": n_f,
            "freshness_load_limit_kt": l_f,
            "fresh_and_worthwhile": fresh_and_worthwhile(l_f, just_worth),
        }
    # Class L (tex value a_ok = 277.9 d): voyage 22 d at 2 degC, store at
    # 2 degC, minimum retail period 1 d at 4 degC.
    mu_store_l = bacterial_growth_rate(2.0, 0.0, ARTICLE_CLASS_L)
    omega_voyage_l = profile_work(
        [TransportSegment(VOYAGE_DAYS_CLASSES, 2.0, 0.25)], ARTICLE_CLASS_L
    )
    omega_retail_l = segment_work(1.0, HOLD_TEMPERATURE, 0.0, ARTICLE_CLASS_L)
    a_ok_l = admissible_storage_age(
        ARTICLE_CLASS_L.omega_s, omega_voyage_l, omega_retail_l, mu_store_l
    )
    l_f_l = freshness_load_limit(u_bar, a_ok_l)
    freshness_rows["class_L"] = {
        "a_ok_d": a_ok_l,
        "freshness_load_limit_kt": l_f_l,
        "fresh_and_worthwhile": fresh_and_worthwhile(l_f_l, just_worth),
    }

    # --- 4. Double clamp of the voyage duration (Satz sb-klemme) ------------
    fixed_c = (
        segment_work(1.5, 3.0, 2.25, article) + omega_after
    )  # loading + unloading + retail
    mu_sea_c = (
        segment_work(19.0, -1.5, 0.16, article, exposure=exposure) / 19.0
    )  # CA, subcooled
    fixed_l = (
        segment_work(1.5, 3.0, 2.25, ARTICLE_CLASS_L)
        + segment_work(0.5, 5.0, 4.0, ARTICLE_CLASS_L)
        + omega_retail_l
    )
    mu_sea_l = omega_voyage_l / VOYAGE_DAYS_CLASSES
    clamp_rows = {
        "class_C_ca_subcooled_days": critical_voyage_duration(
            params, mu_sea_c, mu_store, omega_s, fixed_c
        ),
        "class_L_days": critical_voyage_duration(
            params, mu_sea_l, mu_store_l, ARTICLE_CLASS_L.omega_s, fixed_l
        ),
    }

    # --- 5. Cooling failure and correlated risk (Beispiele sb-ausfall) ------
    failure_rows = []
    for d in FAIL_DURATIONS:
        work = failure_work(d, FAIL_SET, FAIL_AMBIENT, FAIL_TAU_P, article)
        steady = bacterial_growth_rate(FAIL_SET, 0.0, article) * d
        failure_rows.append(
            {
                "duration_d": d,
                "omega_fail": work,
                "omega_steady": steady,
                "share_of_omega_s": work / omega_s,
            }
        )
    tolerance_d = failure_tolerance(omega_s, FAIL_SET, FAIL_AMBIENT, FAIL_TAU_P, article)
    warm_load_cost = exponential_fixed_cost(FAIL_SET, 12.0, 0.5, article)
    risk_rows = [
        {"n": n, "variance": ship_failure_variance(P_SHIP, P_CONTAINER, n)}
        for n in CONTAINER_COUNTS
    ]

    summary = {
        "article_C": {"omega_s": omega_s, "mu_hold_4C": mu_hold},
        "ca_exposure_19d": exposure,
        "sea_air_threshold_degC": threshold,
        "chains": chain_rows,
        "classes": class_rows,
        "pulse": {
            "departures_per_year": departures,
            "arrival_interval_d": ages.arrival_interval,
            "mean_store_age_d": ages.mean,
            "just_worth_load_kt": just_worth,
        },
        "freshness": freshness_rows,
        "critical_voyage_duration": {k: _finite(v) for k, v in clamp_rows.items()},
        "failure": {
            "durations": failure_rows,
            "tolerance_duration_d": tolerance_d,
            "warm_load_fixed_cost": warm_load_cost,
        },
        "risk": {
            "p_ship": P_SHIP,
            "p_container": P_CONTAINER,
            "mean": ship_failure_mean(P_SHIP, P_CONTAINER),
            "variance_by_containers": risk_rows,
            "variance_limit": ship_failure_variance_limit(P_SHIP, P_CONTAINER),
        },
    }
    (RESULTS / "05_sea_bacteria_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savetxt(
        RESULTS / "05_sea_bacteria_chains.csv",
        np.array(
            [
                [
                    row["duration_d"],
                    row["omega"],
                    row["log_increase"],
                    row["growth_factor"],
                    row["omega_over_omega_s"],
                    row["residual_life_d"],
                ]
                for row in chain_rows.values()
            ]
        ),
        delimiter=",",
        header="duration_d,omega,log_increase,growth_factor,omega_over_omega_s,residual_life_d",
        comments="",
    )

    print("Experiment 5b: sea freight and bacterial development")
    print(f"Shelf-life budget Omega_s (class C): {omega_s:.3f}")
    for name, row in chain_rows.items():
        print(
            f"  {name:<18} Omega={row['omega']:6.2f}  "
            f"Omega/Omega_s={row['omega_over_omega_s']:.2f}  "
            f"R={row['residual_life_d']:.2f} d"
        )
    print(f"Sea/air threshold (20 d vs 2 d): {threshold:.2f} degC")
    print(f"Mean age in port store: {ages.mean:.2f} d")
    for name, row in freshness_rows.items():
        print(f"  {name:<18} L_F={row['freshness_load_limit_kt']:.1f} kt")
    print(f"Failure tolerance D*: {tolerance_d:.2f} d")

    # --- Figure -------------------------------------------------------------
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    names = list(chain_rows)
    ratios = [chain_rows[n]["omega_over_omega_s"] for n in names]
    colors = ["#e76f51" if r > 1.0 else "#2a9d8f" for r in ratios]
    ax.bar(range(len(names)), ratios, color=colors)
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel(r"$\Omega/\Omega_s$")

    ax = axes[0, 1]
    temps = np.linspace(article.theta_min, 5.0, 201)
    omega_sea = np.array(
        [bacterial_growth_rate(t, 0.0, article) * 20.0 for t in temps]
    )
    omega_air = bacterial_growth_rate(5.0, 0.0, article) * 2.0
    ax.plot(temps, omega_sea, color="#457b9d", label="sea, 20 d")
    ax.axhline(omega_air, color="#e76f51", linestyle="--", label="air, 2 d at 5 degC")
    ax.axvline(threshold, color="#2a9d8f")
    ax.set_xlabel("sea temperature (degC)")
    ax.set_ylabel(r"$\Omega$")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    tau_grid = np.linspace(0.0, 60.0, 241)
    a_grid = (omega_s - fixed_c - mu_sea_c * tau_grid) / mu_store
    l_f_grid = u_bar * a_grid
    ax.plot(tau_grid, l_f_grid, color="#457b9d", label=r"$L_F(\tau_S)$, class C (CA, subcooled)")
    ax.axhline(just_worth, color="#e76f51", linestyle="--", label=r"$L^-_\varepsilon$ (default horizon)")
    ax.set_xlabel(r"voyage duration $\tau_S$ (d)")
    ax.set_ylabel("kt")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    d_grid = np.linspace(0.0, 2.0, 201)
    w_grid = np.array(
        [
            failure_work(d, FAIL_SET, FAIL_AMBIENT, FAIL_TAU_P, article) if d > 0 else 0.0
            for d in d_grid
        ]
    )
    ax.plot(d_grid, w_grid, color="#457b9d", label=r"$\Omega_{fail}(D)$")
    ax.axhline(omega_s, color="#e76f51", linestyle="--", label=r"$\Omega_s$")
    ax.axvline(tolerance_d, color="#2a9d8f")
    ax.set_xlabel("failure duration D (d)")
    ax.set_ylabel(r"$\Omega$")
    ax.legend(fontsize=8)

    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS / "05_sea_bacteria.pdf")
    plt.close(figure)
    return summary


def run() -> None:
    run_ship_freight()
    run_sea_bacteria()


if __name__ == "__main__":
    run()
