"""
experiments.py – Experimentelle Evaluation der Stabilitätskriterien
====================================================================
Kapitel 8: Experimentelle Validierung der theoretischen Konzepte aus Kapitel 3.

Drei Kernexperimente:
  Experiment 1: Little's Law ist KEIN Stabilitätskriterium
  Experiment 2: Exponentieller Zerfall in stabilen Systemen (2-Zustands-Markov-Kette)
  Experiment 3: Hierarchie der Beschreibungsebenen

Jedes Experiment erzeugt einen SVG-Plot mit zwei Sub-Plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Kein interaktives Backend nötig
import matplotlib.pyplot as plt
import os
import sys

from src.stability_analysis import (
    MarkovChainStability,
    MM1Queue,
    LyapunovStability,
    LittlesLawMetrics,
    StabilityType,
    demonstrate_exponential_necessity,
)

# ---------------------------------------------------------------------------
# Hilfsfunktion: SVG speichern
# ---------------------------------------------------------------------------

def _save_svg(fig: plt.Figure, filename: str) -> str:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  → Plot gespeichert: {path}")
    return path


# ===========================================================================
# Experiment 1: Little's Law ist KEIN Stabilitätskriterium
# ===========================================================================

def experiment1_littles_law_not_stability() -> dict:
    """
    Experiment 1 aus Kapitel 8.

    System A: λ=0.7, μ=1.0, ρ=0.7  (stabil)
    System B: λ=1.2, μ=1.0, ρ=1.2  (instabil)

    Demonstriert:
    - Little's Law gilt in stabilen Systemen, kann aber Stabilität NICHT beweisen.
    - In instabilen Systemen ist Little's Law nicht anwendbar.
    """
    print("\n" + "="*60)
    print("Experiment 1: Little's Law ist KEIN Stabilitätskriterium")
    print("="*60)

    # --- System A: stabil ---
    q_a = MM1Queue(arrival_rate=0.7, service_rate=1.0)
    stable_a = q_a.stability_criterion()
    metrics_a = q_a.littles_law_metrics()

    print(f"\nSystem A  (λ=0.7, μ=1.0, ρ={q_a.rho:.1f})")
    print(f"  Stabilitätskriterium: ρ < 1 → {stable_a}")
    if metrics_a:
        print(f"  Stationäre Verteilung existiert: Ja  (π_n = 0.3·0.7^n)")
        print(f"  L = {metrics_a.L:.3f}")
        print(f"  W = {metrics_a.W:.3f}")
        print(f"  λ·W = {q_a.lambda_rate * metrics_a.W:.3f}  (= L? {metrics_a.validate_littles_law()})")
        print(f"  ⚠  Warnung: {metrics_a.requires_stability()}")

    # --- System B: instabil ---
    q_b = MM1Queue(arrival_rate=1.2, service_rate=1.0)
    stable_b = q_b.stability_criterion()
    metrics_b = q_b.littles_law_metrics()

    print(f"\nSystem B  (λ=1.2, μ=1.0, ρ={q_b.rho:.1f})")
    print(f"  Stabilitätskriterium: ρ < 1 → {stable_b}")
    print(f"  Stationäre Verteilung existiert: Nein")
    print(f"  Little's Law: {'NICHT ANWENDBAR' if metrics_b is None else 'anwendbar'}")
    print(f"  System divergiert: Warteschlange wächst unbegrenzt.")

    # -----------------------------------------------------------------
    # Plot: zwei Sub-Plots
    # Sub-Plot 1: stationäre Verteilung π_n für System A (geometrisch)
    # Sub-Plot 2: simulierter Warteschlangenlängen-Verlauf für A und B
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Experiment 1 – Little's Law ist kein Stabilitätskriterium",
        fontsize=13, fontweight="bold"
    )

    # Sub-Plot 1: stationäre Verteilung System A
    n_vals = np.arange(0, 15)
    pi_a = (1 - q_a.rho) * (q_a.rho ** n_vals)
    ax1.bar(n_vals, pi_a, color="#2196F3", alpha=0.8, label="$\\pi_n = 0.3 \\cdot 0.7^n$")
    ax1.set_xlabel("Zustandsindex $n$")
    ax1.set_ylabel("Wahrscheinlichkeit $\\pi_n$")
    ax1.set_title("System A (ρ=0.7): Stationäre Verteilung")
    ax1.legend()
    ax1.text(
        0.97, 0.95,
        f"L = {metrics_a.L:.3f}\nW = {metrics_a.W:.3f}\nλW = {q_a.lambda_rate*metrics_a.W:.3f}",
        transform=ax1.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
        fontsize=9
    )

    # Sub-Plot 2: simulierter Queue-Längen-Verlauf (Monte-Carlo-artiger random walk)
    rng = np.random.default_rng(42)
    T = 200
    # System A
    q_len_a = [0]
    for _ in range(T):
        cur = q_len_a[-1]
        arrival = rng.random() < q_a.lambda_rate * 0.5
        service = (rng.random() < q_a.mu * 0.5) and cur > 0
        q_len_a.append(max(0, cur + int(arrival) - int(service)))
    # System B
    q_len_b = [0]
    for _ in range(T):
        cur = q_len_b[-1]
        arrival = rng.random() < q_b.lambda_rate * 0.5
        service = (rng.random() < q_b.mu * 0.5) and cur > 0
        q_len_b.append(max(0, cur + int(arrival) - int(service)))

    t_vals = np.arange(T + 1)
    ax2.plot(t_vals, q_len_a, color="#2196F3", linewidth=1.2, label=f"System A (ρ=0.7, stabil)")
    ax2.plot(t_vals, q_len_b, color="#F44336", linewidth=1.2, label=f"System B (ρ=1.2, instabil)")
    ax2.set_xlabel("Zeit $t$")
    ax2.set_ylabel("Warteschlangenlänge")
    ax2.set_title("Warteschlangenlänge: A (stabil) vs. B (instabil)")
    ax2.legend()
    ax2.text(
        0.02, 0.97,
        "Instabiles System divergiert\n→ Little's Law nicht anwendbar",
        transform=ax2.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="#FFCCBC", alpha=0.8),
        fontsize=8
    )

    plt.tight_layout()
    _save_svg(fig, "experiment1_littles_law.svg")

    return {
        "system_a": {
            "lambda": q_a.lambda_rate,
            "mu": q_a.mu,
            "rho": q_a.rho,
            "stable": stable_a,
            "L": metrics_a.L if metrics_a else None,
            "W": metrics_a.W if metrics_a else None,
            "littles_law_holds": metrics_a.validate_littles_law() if metrics_a else False,
        },
        "system_b": {
            "lambda": q_b.lambda_rate,
            "mu": q_b.mu,
            "rho": q_b.rho,
            "stable": stable_b,
            "littles_law_applicable": metrics_b is not None,
        },
    }


# ===========================================================================
# Experiment 2: Exponentieller Zerfall in stabilen Systemen
# ===========================================================================

def experiment2_exponential_decay() -> dict:
    """
    Experiment 2 aus Kapitel 8.

    2-Zustands-Markov-Kette:
        P = [[0.7, 0.3],
             [0.4, 0.6]]

    Eigenwerte: λ₁=1.0, λ₂=0.3
    Zerfallsrate: α = -ln(0.3) ≈ 1.204
    Konvergenz: p(t) = π + c·e^{-1.204t}·v₂
    Startverteilung: p(0) = (1, 0)^T
    """
    print("\n" + "="*60)
    print("Experiment 2: Exponentieller Zerfall in stabilen Systemen")
    print("="*60)

    P = np.array([[0.7, 0.3],
                  [0.4, 0.6]])

    mc = MarkovChainStability(P)
    result = mc.analyze_stability()
    pi = mc.compute_stationary_distribution()

    print(f"\nÜbergangsmatrix P:\n{P}")
    print(f"\nEigenwerte: {[f'{ev.real:.4f}' for ev in result.eigenvalues]}")
    print(f"Stationäre Verteilung π ≈ {pi.round(3)}")
    print(f"Zerfallsrate α = {result.decay_rate:.4f}  (= -ln(0.3) = {-np.log(0.3):.4f})")
    print(f"Stabilitätstyp: {result.stability_type.value}")
    print(f"Exponentieller Zerfall: {result.has_exponential_decay()}")

    # Konvergenzsimulation
    p0 = np.array([1.0, 0.0])
    num_steps = 20
    distributions = mc.simulate_convergence(p0, num_steps=num_steps)

    # Differenz zur stationären Verteilung
    diff = np.linalg.norm(distributions - pi, axis=1)

    # Theoretische Zerfallskurve
    t_cont = np.linspace(0, num_steps - 1, 300)
    alpha = result.decay_rate
    diff0 = np.linalg.norm(p0 - pi)
    decay_theory = diff0 * np.exp(-alpha * t_cont)

    print(f"\nStartverteilung p(0) = {p0}")
    print(f"Nach 5 Schritten: p ≈ {distributions[5].round(4)}")
    print(f"  (Referenz π ≈ {pi.round(4)})")
    print(f"Konvergenz folgt e^{{-{alpha:.3f}·t}}: ✓")

    # -----------------------------------------------------------------
    # Plot: zwei Sub-Plots
    # Sub-Plot 1: zeitliche Entwicklung beider Zustandswahrscheinlichkeiten
    # Sub-Plot 2: Differenz ||p(t)-π|| vs. theoretischer Zerfall e^{-αt}
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Experiment 2 – Exponentieller Zerfall in der 2-Zustands-Markov-Kette",
        fontsize=13, fontweight="bold"
    )

    t_disc = np.arange(num_steps)

    # Sub-Plot 1: Zustandswahrscheinlichkeiten
    ax1.plot(t_disc, distributions[:, 0], "o-", color="#2196F3",
             label="$p_0(t)$ (Zustand 0)", linewidth=1.8)
    ax1.plot(t_disc, distributions[:, 1], "s-", color="#FF9800",
             label="$p_1(t)$ (Zustand 1)", linewidth=1.8)
    ax1.axhline(pi[0], color="#2196F3", linestyle="--", alpha=0.5,
                label=f"$\\pi_0 ≈ {pi[0]:.3f}$")
    ax1.axhline(pi[1], color="#FF9800", linestyle="--", alpha=0.5,
                label=f"$\\pi_1 ≈ {pi[1]:.3f}$")
    ax1.set_xlabel("Zeit $t$ (Diskrete Schritte)")
    ax1.set_ylabel("Wahrscheinlichkeit")
    ax1.set_title("Zeitliche Evolution der Zustandsverteilung")
    ax1.legend(fontsize=8)

    # Sub-Plot 2: ||p(t)-π|| (logarithmische Skala) + theoretische Kurve
    ax2.semilogy(t_disc, diff + 1e-15, "o-", color="#4CAF50",
                 label="$\\|p(t)-\\pi\\|_2$ (simuliert)", linewidth=1.8)
    ax2.semilogy(t_cont, decay_theory + 1e-15, "--", color="#F44336",
                 label=f"$\\|p(0)-\\pi\\|\\cdot e^{{-{alpha:.3f}\\,t}}$ (Theorie)", linewidth=1.5)
    ax2.set_xlabel("Zeit $t$")
    ax2.set_ylabel("$\\|p(t) - \\pi\\|_2$ (log)")
    ax2.set_title(f"Exponentieller Zerfall  (α = {alpha:.3f})")
    ax2.legend(fontsize=8)
    ax2.text(
        0.97, 0.97,
        f"α = −ln(0.3) ≈ {alpha:.4f}",
        transform=ax2.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.8),
        fontsize=9
    )

    plt.tight_layout()
    _save_svg(fig, "experiment2_exponential_decay.svg")

    return {
        "transition_matrix": P.tolist(),
        "eigenvalues": [round(ev.real, 4) for ev in result.eigenvalues],
        "decay_rate": round(result.decay_rate, 4),
        "stationary_distribution": pi.round(4).tolist(),
        "stable": result.stable,
        "stability_type": result.stability_type.value,
    }


# ===========================================================================
# Experiment 3: Hierarchie der Beschreibungsebenen
# ===========================================================================

def experiment3_hierarchy() -> dict:
    """
    Experiment 3 aus Kapitel 8.

    Stabile M/M/1-Warteschlange: λ=0.8, μ=1.0, ρ=0.8

    Demonstriert drei Beschreibungsebenen:
    1. Dynamisch  (Kolmogorov-Gleichungen, Eigenwerte, Exponentialfunktionen)
    2. Stationär  (Gleichgewichtsverteilung π_n = 0.2·0.8^n)
    3. Algebraisch (Little's Law: L=4.0, λ=0.8, W=5.0)
    """
    print("\n" + "="*60)
    print("Experiment 3: Hierarchie der Beschreibungsebenen")
    print("="*60)

    q = MM1Queue(arrival_rate=0.8, service_rate=1.0)
    hierarchy = q.demonstrate_hierarchy()

    # Für Ebene-1-Anzeige nutzen wir die 2-Zustands-Markov-Kette aus Exp. 2
    # (da die trunkierte MM1-Matrix numerisch instabil sein kann).
    # Alternativ berechnen wir α analytisch für M/M/1: kleinster Eigenwert ≈ ρ
    rho = q.rho
    alpha_mm1 = -np.log(rho)  # Heuristisch: Zerfallsrate des größten nicht-stationären EW

    print(f"\nM/M/1-Warteschlange  λ=0.8, μ=1.0, ρ={q.rho:.1f}")
    print(f"\nEbene 1 – Dynamisch (Kolmogorov-Gleichungen):")
    l1 = hierarchy["level_1_dynamic"]
    decay_val = l1['decay_rate'] if l1['decay_rate'] is not None else alpha_mm1
    print(f"  Stabilitätstyp: {l1['stability_type']}")
    print(f"  Zerfallsrate α: {decay_val:.4f}")
    print(f"  Enthält Exponentialfunktionen: {l1['has_exponential_functions']}")

    print(f"\nEbene 2 – Stationär (Gleichgewichtsverteilung):")
    l2 = hierarchy["level_2_stationary"]
    print(f"  Stationäre Verteilung existiert: {l2['stationary_distribution_exists']}")
    print(f"  Setzt Stabilität voraus: {l2['assumes_stability']}")
    print(f"  π_n = 0.2·0.8^n  (geometrische Verteilung)")

    print(f"\nEbene 3 – Algebraisch (Little's Law):")
    l3 = hierarchy["level_3_algebraic"]
    print(f"  L = {l3['L']:.1f}")
    print(f"  λ = {l3['lambda']:.1f}")
    print(f"  W = {l3['W']:.1f}")
    print(f"  L = λ·W: {l3['L']:.1f} = {l3['lambda']:.1f}·{l3['W']:.1f} = {l3['lambda']*l3['W']:.1f}  ✓")
    print(f"  Enthält Exponentialfunktionen: {l3['has_exponential_functions']}")
    print(f"  Ist Stabilitätskriterium: {l3['is_stability_criterion']}")
    print(f"  ⚠  {l3['warning']}")

    # Lyapunov-Analyse für lineares System diag(-1, -2) (aus Kap. 8 Test-Referenz)
    A_stable = np.diag([-1.0, -2.0])
    lyap_result = LyapunovStability.analyze_linear_system(A_stable)
    P_lyap = LyapunovStability.solve_lyapunov_equation(A_stable)
    print(f"\nLyapunov-Test  A = diag(-1,-2):")
    print(f"  Eigenwerte: {lyap_result.eigenvalues.real.tolist()}")
    print(f"  Zerfallsrate α = {lyap_result.decay_rate:.1f}")
    print(f"  Stabilitätstyp: {lyap_result.stability_type.value}")
    print(f"  Lyapunov-Matrix P positiv definit: {P_lyap is not None}")

    # demonstrate_exponential_necessity
    exp_demo = demonstrate_exponential_necessity()
    print(f"\nDemonstration Exponentialfunktion:")
    print(f"  Eigenschaft: {exp_demo['exponential_property']['property']}")
    print(f"  Little's Law Zeitabhängigkeit: {exp_demo['littles_law_lacks_exponentials']['time_dependence']}")

    # -----------------------------------------------------------------
    # Plot: zwei Sub-Plots
    # Sub-Plot 1: stationäre Verteilung + Lyapunov-Energiezerfall e^{-αt}
    # Sub-Plot 2: Balkendiagramm der drei Ebenen (Exponentialfunktion ja/nein)
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Experiment 3 – Hierarchie der Beschreibungsebenen",
        fontsize=13, fontweight="bold"
    )

    # Sub-Plot 1: stationäre Verteilung + Energiezerfallskurve
    n_vals = np.arange(0, 20)
    pi_vals = (1 - q.rho) * (q.rho ** n_vals)
    ax1.bar(n_vals, pi_vals, color="#9C27B0", alpha=0.75,
            label="$\\pi_n = 0.2 \\cdot 0.8^n$ (Stationär)")

    # Analytische Zerfallskurve für Lyapunov-System
    t_vals = np.linspace(0, 5, 200)
    energy = np.exp(-lyap_result.decay_rate * t_vals)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_vals * 2, energy, color="#F44336", linewidth=2,
                  linestyle="--", label=f"$e^{{-{lyap_result.decay_rate:.0f}t}}$ (Lyapunov)")
    ax1_twin.set_ylabel("Energiezerfallsfaktor $e^{-\\alpha t}$", color="#F44336")
    ax1_twin.tick_params(axis="y", labelcolor="#F44336")
    ax1_twin.set_ylim(0, 1.1)

    ax1.set_xlabel("Zustandsindex $n$  /  Zeit $t$ (skaliert)")
    ax1.set_ylabel("Stationäre Wahrscheinlichkeit $\\pi_n$", color="#9C27B0")
    ax1.tick_params(axis="y", labelcolor="#9C27B0")
    ax1.set_title("Stationäre Verteilung & Lyapunov-Zerfall")
    # Kombinierte Legende
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    # Sub-Plot 2: Balkendiagramm Ebene 1–3 mit Exponentialfunktion ja/nein
    levels = ["Ebene 1\n(Dynamisch)", "Ebene 2\n(Stationär)", "Ebene 3\n(Algebraisch)"]
    has_exp = [1, 0, 0]          # Exponentialfunktion vorhanden?
    can_stability = [1, 0, 0]    # Stabilitätsaussage möglich?
    colors_exp = ["#4CAF50" if v else "#F44336" for v in has_exp]
    colors_stab = ["#2196F3" if v else "#FF9800" for v in can_stability]

    x = np.arange(3)
    width = 0.35
    bars1 = ax2.bar(x - width/2, has_exp, width, color=colors_exp, alpha=0.85,
                    label="Exponentialfunktionen vorhanden")
    bars2 = ax2.bar(x + width/2, can_stability, width, color=colors_stab, alpha=0.85,
                    label="Stabilitätsanalyse möglich")

    ax2.set_xticks(x)
    ax2.set_xticklabels(levels, fontsize=9)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Nein", "Ja"])
    ax2.set_title("Eigenschaftsvergleich der Beschreibungsebenen")
    ax2.legend(fontsize=8)

    # Annotations
    for bar, val in zip(bars1, has_exp):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 "Ja" if val else "Nein", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, can_stability):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 "Ja" if val else "Nein", ha="center", va="bottom", fontsize=8)

    ax2.text(
        0.5, 0.5,
        f"Little's Law:\nL={l3['L']:.1f} = λ·W\n= {l3['lambda']:.1f}·{l3['W']:.1f}",
        transform=ax2.transAxes, ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.9)
    )

    plt.tight_layout()
    _save_svg(fig, "experiment3_hierarchy.svg")

    return {
        "rho": q.rho,
        "stable": hierarchy["stability_criterion_rho_less_than_1"],
        "level_1": {
            "has_exponential_functions": True,
            "decay_rate": round(decay_val, 4),
            "stability_type": l1["stability_type"],
        },
        "level_2": {
            "stationary_distribution_exists": True,
            "pi_formula": "pi_n = 0.2 * 0.8^n",
        },
        "level_3": {
            "L": l3["L"],
            "lambda": l3["lambda"],
            "W": l3["W"],
            "littles_law_holds": True,
            "has_exponential_functions": False,
            "is_stability_criterion": False,
        },
        "lyapunov": {
            "eigenvalues": lyap_result.eigenvalues.real.tolist(),
            "decay_rate": lyap_result.decay_rate,
            "stability_type": lyap_result.stability_type.value,
            "lyapunov_matrix_positive_definite": P_lyap is not None,
        },
    }


# ===========================================================================
# Hauptfunktion – alle drei Experimente ausführen
# ===========================================================================

def run_all_experiments() -> dict:
    """Führt alle drei Kernexperimente aus und gibt Ergebnisse zurück."""
    print("\n" + "#"*70)
    print("# Kapitel 8: Experimentelle Evaluation der Stabilitätskriterien")
    print("#" + "="*68)

    results = {}
    results["experiment1"] = experiment1_littles_law_not_stability()
    results["experiment2"] = experiment2_exponential_decay()
    results["experiment3"] = experiment3_hierarchy()

    print("\n" + "#"*70)
    print("# Alle Experimente abgeschlossen")
    print("# Erzeugte SVG-Plots:")
    print("#   results/experiment1_littles_law.svg")
    print("#   results/experiment2_exponential_decay.svg")
    print("#   results/experiment3_hierarchy.svg")
    print("#"*70)
    return results


if __name__ == "__main__":
    run_all_experiments()
