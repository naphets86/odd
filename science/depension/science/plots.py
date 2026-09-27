#!/usr/bin/env python3
"""
Depension – Abbildungen für die wissenschaftliche Arbeit
Autor: Stephan Epp
Datum: 30. März 2026

Alle Plots werden als PDF im Verzeichnis ~/Git/depension/science/ gespeichert.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ─── Globale Stileinstellungen ───────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":      False,
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

MAINBLUE  = "#19468C"
ACCENTRED = "#B4321E"
DARKGREEN = "#1E6432"
ORANGE    = "#C86400"
PURPLE    = "#6B2D8B"
GRAY      = "#888888"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 1: Das Konzept der Depension                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_depension_concept():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Depension: Drei Erscheinungsformen der Abhängigkeitsmodellierung",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── Spalte 1: Statische Depension (Lineare Regression) ──
    ax = axes[0]
    rng = np.random.default_rng(42)
    x_d = np.linspace(0, 10, 50)
    y_d = 2.5 * x_d + 1.2 + rng.normal(0, 3, 50)
    ax.scatter(x_d, y_d, color=MAINBLUE, s=20, alpha=0.7, label="Beobachtungen")
    m, b = np.polyfit(x_d, y_d, 1)
    ax.plot(x_d, m * x_d + b, color=ACCENTRED, lw=2,
            label=f"$\\hat{{y}} = {m:.2f}x + {b:.2f}$")
    ax.set_title("Statische Depension\n(Lineare Regression)")
    ax.set_xlabel("$x$ (unabh. Variable)")
    ax.set_ylabel("$y$ (abh. Variable)")
    ax.legend()
    ax.text(0.05, 0.92, "DP-statisch", transform=ax.transAxes,
            fontsize=9, color=MAINBLUE, fontweight="bold")

    # ── Spalte 2: Stochastische Depension (Logistische Regression) ──
    ax = axes[1]
    x_l = np.linspace(-6, 6, 200)
    sigma = 1 / (1 + np.exp(-x_l))
    ax.plot(x_l, sigma, color=MAINBLUE, lw=2.5, label="Sigmoid $\\sigma(z)$")
    x_pts = rng.uniform(-5, 5, 60)
    y_pts = (1 / (1 + np.exp(-x_pts + rng.normal(0, 0.5, 60))) > 0.5).astype(float)
    ax.scatter(x_pts, y_pts + rng.normal(0, 0.03, 60),
               color=ACCENTRED, s=15, alpha=0.5, label="Klassen {0,1}")
    ax.axhline(0.5, color=GRAY, ls="--", lw=1, alpha=0.7)
    ax.set_title("Stochastische Depension\n(Logistische Regression)")
    ax.set_xlabel("$z = w^T x + b$")
    ax.set_ylabel("$P(Y=1|x)$")
    ax.legend()
    ax.text(0.05, 0.92, "DP-stochastisch", transform=ax.transAxes,
            fontsize=9, color=MAINBLUE, fontweight="bold")

    # ── Spalte 3: Temporale Depension (Dynamisches System KS-I) ──
    ax = axes[2]
    A = np.array([[-0.5, 0.8], [-0.8, -0.5]])
    t = np.linspace(0, 8, 400)
    colors_traj = [MAINBLUE, ACCENTRED, DARKGREEN, ORANGE]
    x0s = [(2, 1), (-2, 1.5), (1, -2), (-1.5, -1)]
    for i, x0 in enumerate(x0s):
        traj = np.array([expm(A * ti) @ np.array(x0) for ti in t])
        ax.plot(traj[:, 0], traj[:, 1], color=colors_traj[i], lw=1.5, alpha=0.8)
        ax.plot(*traj[0], "o", color=colors_traj[i], ms=6)
        ax.plot(*traj[-1], "s", color=colors_traj[i], ms=5)
    ax.plot(0, 0, "k*", ms=10, label="Gleichgewicht")
    ax.set_title("Temporale Depension\n(Dynamisches System, KS-I)")
    ax.set_xlabel("$x_1(t)$")
    ax.set_ylabel("$x_2(t)$")
    ax.legend()
    ax.text(0.05, 0.92, "DP-temporal", transform=ax.transAxes,
            fontsize=9, color=MAINBLUE, fontweight="bold")

    plt.tight_layout()
    plt.savefig("fig_depension_concept.pdf")
    plt.close()
    print("fig_depension_concept.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 2: Depensionsklassen DP-I bis DP-IV im Eigenwertraum         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_depension_klassen():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Links: Eigenwertraum ──
    ax = axes[0]
    ax.axvline(0, color="black", lw=1.2)
    ax.axhline(0, color="black", lw=1.2)
    ax.fill_betweenx([-4, 4], -5, 0, alpha=0.08, color=MAINBLUE, label="Re(λ) < 0")
    ax.fill_betweenx([-4, 4],  0, 5, alpha=0.08, color=ACCENTRED, label="Re(λ) > 0")

    # DP-I: alle negativ
    lam_1 = [(-1.5, 1.2), (-2.2, -0.8), (-0.8, 2.5), (-1.8, 0)]
    for l in lam_1:
        ax.plot(*l, "o", color=MAINBLUE, ms=10, markeredgecolor="navy", zorder=5)

    # DP-II: rein imaginär und negativ
    lam_2 = [(0, 2.0), (0, -2.0), (-0.5, 1.0), (-0.5, -1.0)]
    for l in lam_2:
        ax.plot(*l, "^", color=DARKGREEN, ms=10, markeredgecolor="darkgreen", zorder=5)

    # DP-III: alle positiv
    lam_3 = [(1.0, 1.5), (2.0, -0.5), (0.5, -2.0)]
    for l in lam_3:
        ax.plot(*l, "s", color=ACCENTRED, ms=10, markeredgecolor="darkred", zorder=5)

    # DP-IV: gemischt
    lam_4 = [(-1.0, 0.5), (1.5, 1.0), (-0.3, -1.5), (2.5, -2.0)]
    for l in lam_4:
        ax.plot(*l, "D", color=ORANGE, ms=10, markeredgecolor="darkorange", zorder=5)

    ax.set_xlim(-4, 4); ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel("Re$(\\lambda)$"); ax.set_ylabel("Im$(\\lambda)$")
    ax.set_title("Depensionsklassen im Eigenwertraum $\\mathbb{C}$")

    legend_handles = [
        mpatches.Patch(color=MAINBLUE,   alpha=0.7, label="DP-I: alle Re(λ) < 0 (konvergent)"),
        mpatches.Patch(color=DARKGREEN,  alpha=0.7, label="DP-II: max Re(λ) = 0 (grenzstabil)"),
        mpatches.Patch(color=ACCENTRED,  alpha=0.7, label="DP-III: alle Re(λ) > 0 (divergent)"),
        mpatches.Patch(color=ORANGE,     alpha=0.7, label="DP-IV: gemischte Vorzeichen"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # ── Rechts: Zeitverläufe aller vier Klassen ──
    ax2 = axes[1]
    t = np.linspace(0, 5, 500)
    x0 = np.array([1.0, 0.5])

    # DP-I
    A1 = np.array([[-1.0, 0.5], [-0.5, -1.0]])
    traj1 = np.array([np.linalg.norm(expm(A1 * ti) @ x0) for ti in t])
    ax2.plot(t, traj1, color=MAINBLUE, lw=2, label="DP-I: $\\|x(t)\\| \\to 0$")

    # DP-II
    A2 = np.array([[0, 1.5], [-1.5, 0]])
    traj2 = np.array([np.linalg.norm(expm(A2 * ti) @ x0) for ti in t])
    ax2.plot(t, traj2, color=DARKGREEN, lw=2, label="DP-II: $\\|x(t)\\|$ beschränkt")

    # DP-III
    A3 = np.array([[0.5, 0.3], [0.3, 0.5]])
    traj3 = np.array([np.linalg.norm(expm(A3 * ti) @ x0) for ti in t])
    ax2.plot(t, traj3, color=ACCENTRED, lw=2, label="DP-III: $\\|x(t)\\| \\to \\infty$")

    # DP-IV
    A4 = np.array([[-1.0, 0], [0, 0.3]])
    traj4 = np.array([np.linalg.norm(expm(A4 * ti) @ x0) for ti in t])
    ax2.plot(t, traj4, color=ORANGE, lw=2, ls="--", label="DP-IV: gemischt")

    ax2.axhline(1.0, color=GRAY, ls=":", lw=1)
    ax2.set_xlabel("Zeit $t$"); ax2.set_ylabel("$\\|x(t)\\|_2$")
    ax2.set_title("Normentwicklung der Zustandsvektoren")
    ax2.legend(); ax2.set_ylim(0, 8)

    plt.tight_layout()
    plt.savefig("fig_depension_klassen.pdf")
    plt.close()
    print("fig_depension_klassen.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 3: Gradientenfluss als dynamisches System                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_gradient_dynamics():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Gradientenfluss als DP-I-System: Konvergenz zur optimalen Depension",
                 fontweight="bold")

    # ── Verlustlandschaft (2D) ──
    ax = axes[0]
    w1 = np.linspace(-3, 3, 200)
    w2 = np.linspace(-3, 3, 200)
    W1, W2 = np.meshgrid(w1, w2)
    # Konvexe quadratische Verlustfunktion (Modell für Kreuzentropie)
    L = 0.8 * W1**2 + 1.5 * W2**2 + 0.4 * W1 * W2 + 0.5

    cf = ax.contourf(W1, W2, L, levels=20, cmap="Blues_r", alpha=0.8)
    ax.contour(W1, W2, L, levels=20, colors="white", linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="$\\mathcal{L}(w)$", shrink=0.8)

    # Gradientenabstieg-Bahn
    def grad_L(w):
        return np.array([2*0.8*w[0] + 0.4*w[1],
                         2*1.5*w[1] + 0.4*w[0]])

    eta = 0.15
    traj = [np.array([-2.5, 2.5])]
    for _ in range(60):
        traj.append(traj[-1] - eta * grad_L(traj[-1]))
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1], "o-", color=ACCENTRED, ms=3, lw=1.5,
            label="GD-Bahn")
    ax.plot(*traj[0], "^", color=ACCENTRED, ms=10, label="Start")
    ax.plot(0, 0, "*", color="gold", ms=14, markeredgecolor="black", label="Minimum $w^*$")
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("Verlustlandschaft $\\mathcal{L}(w)$\nund GD-Trajektorie")
    ax.legend(fontsize=8)

    # ── Eigenwerte der Hesse-Matrix ──
    ax2 = axes[1]
    H = np.array([[2*0.8, 0.4], [0.4, 2*1.5]])
    eigvals = np.linalg.eigvalsh(H)
    lambdas_reg = [eigvals]
    reg_vals = np.linspace(0, 3, 50)
    spectra = [np.linalg.eigvalsh(H + r * np.eye(2)) for r in reg_vals]
    min_eig = [s[0] for s in spectra]
    max_eig = [s[1] for s in spectra]

    ax2.plot(reg_vals, min_eig, color=MAINBLUE, lw=2, label="$\\lambda_{\\min}(H + \\lambda I)$")
    ax2.plot(reg_vals, max_eig, color=ACCENTRED, lw=2, label="$\\lambda_{\\max}(H + \\lambda I)$")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.fill_between(reg_vals, 0, min_eig, alpha=0.15, color=MAINBLUE,
                     label="KS-I / DP-I Zone")
    ax2.set_xlabel("Regularisierung $\\lambda$")
    ax2.set_ylabel("Eigenwert")
    ax2.set_title("Regularisierung verschiebt\nEigenwertre in DP-I")
    ax2.legend(fontsize=8)

    # ── Konvergenzrate vs. Konditionszahl ──
    ax3 = axes[2]
    kappa_vals = np.linspace(1, 50, 100)
    # Theoretische Konvergenzrate: (kappa-1)/(kappa+1)
    rho = (kappa_vals - 1) / (kappa_vals + 1)
    # Anzahl der Schritte bis eps = 1e-6
    eps = 1e-6
    steps = np.log(eps) / np.log(rho + 1e-15)
    ax3.semilogy(kappa_vals, np.abs(steps), color=MAINBLUE, lw=2)
    ax3.set_xlabel("Konditionszahl $\\kappa(H)$")
    ax3.set_ylabel("Schritte bis $\\varepsilon = 10^{-6}$")
    ax3.set_title("Konvergenzrate des GD\nals Funktion der Konditionszahl")
    ax3.fill_between(kappa_vals, 0, np.abs(steps), alpha=0.1, color=MAINBLUE)

    plt.tight_layout()
    plt.savefig("fig_gradient_dynamics.pdf")
    plt.close()
    print("fig_gradient_dynamics.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 4: Depensions-Index als neues Gütemass                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_depension_index():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Depensions-Index (DI): Vereinheitlichtes Gütemaß",
                 fontweight="bold")

    # ── Links: DI für Lineare Regression (= R²) ──
    ax = axes[0]
    np.random.seed(7)
    x = np.linspace(0, 10, 80)
    noises = [0.5, 2.0, 5.0]
    colors = [MAINBLUE, DARKGREEN, ACCENTRED]
    for noise, col in zip(noises, colors):
        y = 2 * x + 1 + np.random.normal(0, noise, len(x))
        m, b = np.polyfit(x, y, 1)
        y_pred = m * x + b
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot
        ax.scatter(x, y, s=10, color=col, alpha=0.4)
        ax.plot(x, y_pred, color=col, lw=2, label=f"$\\sigma={noise}$, DI=$R^2$={r2:.2f}")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Statische Depension:\nDI = $R^2$")
    ax.legend(fontsize=8)

    # ── Mitte: DI für Dynamisches System ──
    ax2 = axes[1]
    t = np.linspace(0, 10, 500)
    x0 = np.array([2.0, 1.0])

    def depension_index_dynamic(A, x0, t_end=10):
        ts = np.linspace(0, t_end, 300)
        norms = np.array([np.linalg.norm(expm(A * ti) @ x0) for ti in ts])
        alpha_star = max(np.real(np.linalg.eigvals(A)))
        max_eig_abs = max(np.abs(np.linalg.eigvals(A)))
        if max_eig_abs == 0:
            return 0.0
        DI = 1 - (alpha_star / max_eig_abs) if max_eig_abs > 0 else 1.0
        return np.clip(DI, 0, 1)

    # Systeme mit variierender Stabilität
    alphas = np.linspace(-2, 2, 50)
    DIs = []
    for alpha in alphas:
        A = np.array([[alpha, 0.5], [-0.5, alpha - 0.3]])
        DI = depension_index_dynamic(A, x0)
        DIs.append(DI)

    ax2.plot(alphas, DIs, color=MAINBLUE, lw=2.5)
    ax2.axvline(0, color=GRAY, ls="--", lw=1)
    ax2.axhline(0.5, color=GRAY, ls=":", lw=1)
    ax2.fill_between(alphas, 0, DIs, alpha=0.15, color=MAINBLUE)
    ax2.set_xlabel("Spektralabszisse $\\alpha^*$")
    ax2.set_ylabel("Depensions-Index DI")
    ax2.set_title("Temporale Depension:\nDI als Stabilitätsmass")
    ax2.annotate("DP-I\n(DI→1)", xy=(-1.5, 0.9), fontsize=9, color=MAINBLUE)
    ax2.annotate("DP-III\n(DI→0)", xy=(0.8, 0.1), fontsize=9, color=ACCENTRED)

    # ── Rechts: DI für logistische Regression ──
    ax3 = axes[2]
    np.random.seed(42)
    n_pts = 200
    x_log = np.random.randn(n_pts, 2)
    sep_strengths = [0.5, 1.5, 3.0, 5.0]
    dis_log = []
    accs = []
    for sep in sep_strengths:
        y_log = (x_log[:, 0] * sep + x_log[:, 1] * 0.5 + np.random.randn(n_pts) > 0).astype(int)
        # Logistische Regression via mini-Newton
        w = np.zeros(3)
        X_aug = np.column_stack([np.ones(n_pts), x_log])
        for _ in range(100):
            z = X_aug @ w
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            grad = X_aug.T @ (p - y_log) / n_pts
            S = np.diag(p * (1 - p))
            H = X_aug.T @ S @ X_aug / n_pts + 1e-3 * np.eye(3)
            w -= np.linalg.solve(H, grad)
        z = X_aug @ w
        p = 1 / (1 + np.exp(-z))
        acc = np.mean((p > 0.5) == y_log)
        # DI für logistische Regression = Pseudo-R² (McFadden)
        ll_model = np.sum(y_log * np.log(p + 1e-15) + (1-y_log) * np.log(1-p+1e-15))
        p0 = y_log.mean()
        ll_null = n_pts * (p0*np.log(p0+1e-15) + (1-p0)*np.log(1-p0+1e-15))
        di = 1 - ll_model / (ll_null + 1e-15)
        dis_log.append(np.clip(di, 0, 1))
        accs.append(acc)

    ax3.bar(range(len(sep_strengths)), dis_log, color=MAINBLUE, alpha=0.7, label="DI (McFadden $R^2$)")
    ax3.plot(range(len(sep_strengths)), accs, "o-", color=ACCENTRED, lw=2, ms=8, label="Genauigkeit")
    ax3.set_xticks(range(len(sep_strengths)))
    ax3.set_xticklabels([f"sep={s}" for s in sep_strengths])
    ax3.set_ylabel("Wert")
    ax3.set_title("Stochastische Depension:\nDI (McFadden) vs. Genauigkeit")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("fig_depension_index.pdf")
    plt.close()
    print("fig_depension_index.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 5: VIF im Depensions-Framework                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_vif_depension():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("VIF und Multikollinearität in der Depensionstheorie",
                 fontweight="bold")

    # ── Korrelationsmatrix und VIF ──
    ax = axes[0]
    d = 8
    rng = np.random.default_rng(99)
    # Zufällige Korrelationsmatrix mit einigen hohen Korrelationen
    A_raw = rng.standard_normal((200, d))
    # Füge Korrelationen ein
    A_raw[:, 2] = 0.9 * A_raw[:, 0] + 0.1 * rng.standard_normal(200)
    A_raw[:, 5] = 0.8 * A_raw[:, 3] + 0.2 * rng.standard_normal(200)
    A_raw[:, 7] = 0.7 * A_raw[:, 1] + 0.3 * rng.standard_normal(200)

    corr = np.corrcoef(A_raw.T)
    im = ax.imshow(np.abs(corr), cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="|Korrelation|", shrink=0.85)
    ax.set_xticks(range(d))
    ax.set_xticklabels([f"$x_{i+1}$" for i in range(d)])
    ax.set_yticks(range(d))
    ax.set_yticklabels([f"$x_{i+1}$" for i in range(d)])
    ax.set_title("Korrelationsmatrix der\nPrädiktorvariablen")
    # Annotiere Werte
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{corr[i,j]:.1f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr[i,j]) < 0.7 else "white")

    # ── VIF-Werte ──
    ax2 = axes[1]
    # Berechne VIF
    from numpy.linalg import lstsq
    vifs = []
    for j in range(d):
        X_j = np.delete(A_raw, j, axis=1)
        x_j = A_raw[:, j]
        coeffs, _, _, _ = lstsq(X_j, x_j, rcond=None)
        x_j_hat = X_j @ coeffs
        ss_res = np.sum((x_j - x_j_hat)**2)
        ss_tot = np.sum((x_j - x_j.mean())**2)
        R2 = 1 - ss_res / (ss_tot + 1e-15)
        vif = 1 / (1 - R2 + 1e-15)
        vifs.append(vif)

    colors_vif = []
    for v in vifs:
        if v < 5:
            colors_vif.append(MAINBLUE)
        elif v < 10:
            colors_vif.append(ORANGE)
        else:
            colors_vif.append(ACCENTRED)

    bars = ax2.bar(range(d), vifs, color=colors_vif, alpha=0.8, edgecolor="white")
    ax2.axhline(5,  color=ORANGE,   ls="--", lw=1.5, label="VIF=5 (Warnschwelle)")
    ax2.axhline(10, color=ACCENTRED, ls="--", lw=1.5, label="VIF=10 (Kritisch)")
    ax2.set_xticks(range(d))
    ax2.set_xticklabels([f"$x_{i+1}$" for i in range(d)])
    ax2.set_ylabel("VIF$_j = 1/(1-R_j^2)$")
    ax2.set_title("Varianzinflationsfaktoren\njeder Prädiktorvariable")
    ax2.legend(fontsize=8)
    for i, v in enumerate(vifs):
        ax2.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    # ── VIF und Eigenwertverlauf der Präzisionsmatrix ──
    ax3 = axes[2]
    # Precision matrix = inverse of covariance
    Sigma = np.cov(A_raw.T)
    try:
        Theta = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Theta = np.linalg.pinv(Sigma)
    eigvals_theta = np.sort(np.real(np.linalg.eigvals(Theta)))[::-1]

    ax3.bar(range(len(eigvals_theta)), eigvals_theta,
            color=[MAINBLUE if e > 0 else ACCENTRED for e in eigvals_theta],
            alpha=0.8, edgecolor="white")
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("Eigenwert-Index")
    ax3.set_ylabel("Eigenwert der Präzisionsmatrix $\\Theta$")
    ax3.set_title("Präzisionsmatrix-Spektrum\n(Depensions-Spektrum)")
    ax3.text(0.05, 0.9,
             f"Alle $\\lambda_k > 0$: DP-I\n(Kovarianzmatrix pos. definit)",
             transform=ax3.transAxes, fontsize=8, color=MAINBLUE,
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig("fig_vif_depension.pdf")
    plt.close()
    print("fig_vif_depension.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 6: Regulierungspfad und Klassenwechsel                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_regularisierung_klasse():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Regularisierung als Depensionsklassen-Steuerung",
                 fontweight="bold")

    # ── Links: Koeffizientenpfade L1/L2 ──
    ax = axes[0]
    rng = np.random.default_rng(12)
    d = 6; n = 150
    X = rng.standard_normal((n, d))
    # Starke Multikollinearität: Spalte 1 ≈ Spalte 0
    X[:, 1] = 0.85 * X[:, 0] + 0.15 * rng.standard_normal((n))
    w_true = np.array([2.0, 0.0, -1.5, 0.8, 0.0, 1.2])
    y = X @ w_true + 0.5 * rng.standard_normal((n))

    # Ridge-Pfad
    lambdas = np.logspace(-3, 2, 100)
    coefs_ridge = []
    for lam in lambdas:
        H = X.T @ X / n + lam * np.eye(d)
        g = X.T @ y / n
        coefs_ridge.append(np.linalg.solve(H, g))
    coefs_ridge = np.array(coefs_ridge)

    for j in range(d):
        ax.semilogx(lambdas, coefs_ridge[:, j],
                    lw=1.8, label=f"$w_{j+1}$",
                    color=plt.cm.tab10(j / d))
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Regularisierungsparameter $\\lambda$")
    ax.set_ylabel("Koeffizient $w_j$")
    ax.set_title("Ridge-Regularisierungspfad\n(L2-Depensions-Dämpfung)")
    ax.legend(fontsize=7, ncol=2)

    # ── Rechts: Spektral-Verschiebung durch λ ──
    ax2 = axes[1]
    H_base = X.T @ X / n  # unregularisierter Hessian
    eigvals_base = np.sort(np.real(np.linalg.eigvals(H_base)))

    lambdas_fine = np.linspace(0, 2, 80)
    min_eigs, max_eigs, cond_nums = [], [], []
    for lam in lambdas_fine:
        H_reg = H_base + lam * np.eye(d)
        ev = np.sort(np.real(np.linalg.eigvals(H_reg)))
        min_eigs.append(ev[0])
        max_eigs.append(ev[-1])
        cond_nums.append(ev[-1] / (ev[0] + 1e-15))

    ax2.plot(lambdas_fine, min_eigs, color=MAINBLUE, lw=2,
             label="$\\lambda_{\\min}(H + \\lambda I)$")
    ax2.plot(lambdas_fine, max_eigs, color=ACCENTRED, lw=2,
             label="$\\lambda_{\\max}(H + \\lambda I)$")
    ax2_r = ax2.twinx()
    ax2_r.plot(lambdas_fine, cond_nums, color=DARKGREEN, lw=1.5, ls="--",
               label="$\\kappa(H + \\lambda I)$")
    ax2_r.set_ylabel("Konditionszahl $\\kappa$", color=DARKGREEN)

    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    lam_crit = max(0, -eigvals_base[0])
    if lam_crit > 0 and lam_crit < lambdas_fine[-1]:
        ax2.axvline(lam_crit, color=ORANGE, ls="--", lw=1.5,
                    label=f"$\\lambda_{{\\min}}^{{\\rm crit}}={lam_crit:.3f}$")

    ax2.set_xlabel("Regularisierung $\\lambda$")
    ax2.set_ylabel("Eigenwert")
    ax2.set_title("Klassen-Übergang DP-IV → DP-I\ndurch L2-Regularisierung")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_regularisierung_klasse.pdf")
    plt.close()
    print("fig_regularisierung_klasse.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 7: Hierarchische und temporale Depension                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_hierarchische_depension():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Hierarchische Depension: Nilpotenzstruktur und Feedforward-Ausbreitung",
                 fontweight="bold")

    # ── Links: Nilpotente Systemmatrix (Block-Dreiecks) ──
    ax = axes[0]
    h = 4  # Ebenen
    n_per = 3
    n = h * n_per

    A_nilp = np.zeros((n, n))
    for k in range(h - 1):
        block = np.random.randn(n_per, n_per) * 0.5
        A_nilp[k*n_per:(k+1)*n_per, (k+1)*n_per:(k+2)*n_per] = block

    im = ax.imshow(np.abs(A_nilp), cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax, shrink=0.8, label="|Eintrag|")
    ax.set_title(f"Nilpotente Depensionsmatrix\n($h={h}$ Ebenen, $A^{h}=0$)")
    for k in range(h):
        rect = plt.Rectangle((k*n_per-0.5, k*n_per-0.5), n_per, n_per,
                              linewidth=2, edgecolor=ACCENTRED, facecolor="none")
        ax.add_patch(rect)
    ax.set_xlabel("Spaltenindex $j$ (Sender)")
    ax.set_ylabel("Zeilenindex $i$ (Empfänger)")
    # Labels
    for k in range(h):
        ax.text(k*n_per + n_per/2 - 0.5, n - 0.3,
                f"$L_{k+1}$", ha="center", fontsize=9, color=ACCENTRED)

    # ── Mitte: Potenzen A^k → 0 ──
    ax2 = axes[1]
    powers = range(1, h + 2)
    norms_A = []
    Ak = np.eye(n)
    for k in powers:
        Ak = Ak @ A_nilp
        norms_A.append(np.linalg.norm(Ak, "fro"))

    ax2.bar(list(powers), norms_A, color=[MAINBLUE] * (h) + [ACCENTRED],
            alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Potenz $k$")
    ax2.set_ylabel("$\\|A^k\\|_F$")
    ax2.set_title(f"Nilpotenz: $\\|A^k\\|_F \\to 0$\nfür $k \\geq h={h}$")
    ax2.annotate(f"$A^{h}=0$", xy=(h, 0), xytext=(h-0.5, max(norms_A)*0.3),
                 arrowprops=dict(arrowstyle="->"), fontsize=10, color=ACCENTRED)

    # ── Rechts: Zustandsausbreitung durch Ebenen ──
    ax3 = axes[2]
    x0 = np.zeros(n); x0[:n_per] = 1.0   # Eingabe nur in Ebene 1
    t_steps = list(range(h + 1))
    state_norms_per_layer = []
    x_curr = x0.copy()
    A_disc = np.zeros_like(A_nilp)  # Diskrete Version mit I + A/10
    for k in range(h - 1):
        A_disc[k*n_per:(k+1)*n_per, (k+1)*n_per:(k+2)*n_per] = \
            A_nilp[k*n_per:(k+1)*n_per, (k+1)*n_per:(k+2)*n_per]

    for t in t_steps:
        layer_norms = []
        xs = np.linalg.matrix_power(np.eye(n) + A_disc, t) @ x0
        for k in range(h):
            layer_norms.append(np.linalg.norm(xs[k*n_per:(k+1)*n_per]))
        state_norms_per_layer.append(layer_norms)

    state_norms_per_layer = np.array(state_norms_per_layer)
    for k in range(h):
        ax3.plot(t_steps, state_norms_per_layer[:, k],
                 "o-", lw=2, ms=7, color=plt.cm.tab10(k / h),
                 label=f"Ebene $L_{k+1}$")
    ax3.set_xlabel("Zeitschritt $t$")
    ax3.set_ylabel("$\\|x_{L_k}(t)\\|$")
    ax3.set_title("Depensions-Ausbreitung\ndurch Hierarchie-Ebenen")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_hierarchische_depension.pdf")
    plt.close()
    print("fig_hierarchische_depension.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 8: Informationsgeometrie der Depension                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_information_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Informationsgeometrie der Depension: Fisher-Information als Riemannsche Metrik",
                 fontweight="bold")

    # ── Links: Fisher-Information der logistischen Regression ──
    ax = axes[0]
    z = np.linspace(-5, 5, 300)
    sigma = 1 / (1 + np.exp(-z))
    fisher = sigma * (1 - sigma)  # Fischer-Information = σ'(z) = σ(1-σ)
    ax.plot(z, sigma,  color=MAINBLUE, lw=2.5, label="$\\sigma(z)$")
    ax.plot(z, fisher, color=ACCENTRED, lw=2.5, label="$\\mathcal{I}(z) = \\sigma(1-\\sigma)$")
    ax.fill_between(z, 0, fisher, alpha=0.15, color=ACCENTRED)
    ax.axvline(0, color=GRAY, ls="--", lw=1)
    ax.set_xlabel("log-Odds $z$")
    ax.set_ylabel("Wert")
    ax.set_title("Fisher-Information der Bernoulli-\nVerteilung (Depensions-Krümmung)")
    ax.legend()
    ax.annotate("Max. Krümmung\nbei $z=0$", xy=(0, 0.25), xytext=(1.5, 0.3),
                arrowprops=dict(arrowstyle="->"), fontsize=9)

    # ── Mitte: Log-Partition-Funktion und ihre Ableitungen ──
    ax2 = axes[1]
    eta = np.linspace(-4, 4, 300)
    A_eta = np.log(1 + np.exp(eta))           # Log-Partition
    A_prime = np.exp(eta) / (1 + np.exp(eta)) # Erwartungswert = σ(η)
    A_pp    = A_prime * (1 - A_prime)          # Varianz = Fisher-Info

    ax2.plot(eta, A_eta,   color=MAINBLUE,  lw=2, label="$A(\\eta) = \\ln(1+e^\\eta)$")
    ax2.plot(eta, A_prime, color=DARKGREEN, lw=2, label="$A'(\\eta) = \\mathbb{E}[Y]$")
    ax2.plot(eta, A_pp,    color=ACCENTRED, lw=2, label="$A''(\\eta) = \\mathrm{Var}[Y]$")
    ax2.set_xlabel("Kanonischer Parameter $\\eta$")
    ax2.set_title("Log-Partition-Funktion\nund Momente der Depension")
    ax2.legend(fontsize=8)

    # ── Rechts: Krümmungsfeld im Parameterraum ──
    ax3 = axes[2]
    w1 = np.linspace(-3, 3, 40)
    w2 = np.linspace(-3, 3, 40)
    W1, W2 = np.meshgrid(w1, w2)

    # Simuliere Fisher-Information-Matrix-Determinante (∝ Gausskrümmung)
    # Für logistische Regression: FIM ≈ X^T diag(σ(Xw)(1-σ(Xw))) X
    # Näherung: det(FIM) ≈ σ(w1²+w2²) * (1 - σ(w1²+w2²))
    z_val = W1**2 + W2**2
    s = 1 / (1 + np.exp(-0.3 * z_val))
    det_FIM = s * (1 - s)

    cf = ax3.contourf(W1, W2, det_FIM, levels=15, cmap="plasma")
    plt.colorbar(cf, ax=ax3, label=r"$\sqrt{\det(\mathcal{F})}$", shrink=0.85)
    ax3.contour(W1, W2, det_FIM, levels=15, colors="white", linewidths=0.3, alpha=0.4)
    ax3.set_xlabel("$w_1$"); ax3.set_ylabel("$w_2$")
    ax3.set_title("Krümmungsfeld der Depensions-\nMannigfaltigkeit $\\mathcal{M}$")
    ax3.plot(0, 0, "w*", ms=12, label="Optimum $w^*$")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("fig_information_geometry.pdf")
    plt.close()
    print("fig_information_geometry.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 9: Sparse Depension – Broadcasting-Strukturen                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_sparse_depension():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Sparse Depension: Strukturelle Heterogenität und Ausbreitungseffizienz",
                 fontweight="bold")

    # ── Links: Out-Degree-Verteilung (Powerlaw vs. Normal) ──
    ax = axes[0]
    n = 200
    rng = np.random.default_rng(55)

    # Gleichmäßige Verteilung:
    outdeg_unif = rng.integers(3, 15, n)
    # Power-Law (Broadcasting):
    outdeg_power = (rng.pareto(1.5, n) * 5 + 1).astype(int)
    outdeg_power = np.clip(outdeg_power, 1, n-1)

    bins = np.arange(0, 60, 2)
    ax.hist(outdeg_unif,  bins=bins, alpha=0.6, color=MAINBLUE,
            label="Uniform (symmetrisch)")
    ax.hist(outdeg_power, bins=bins, alpha=0.6, color=ACCENTRED,
            label="Power-Law (Broadcasting)")
    ax.set_xlabel("Out-Degree $d_{\\rm out}(j)$")
    ax.set_ylabel("Häufigkeit")
    ax.set_title("Out-Degree-Verteilung:\nSymmetrisch vs. Broadcasting")
    ax.legend()

    # ── Mitte: Beschleunigung durch spaltenbasierte Berechnung ──
    ax2 = axes[1]
    sparsity_vals = np.linspace(0.0, 0.95, 50)
    speedup_cols = 1 / (1 - sparsity_vals + 1e-6)
    # mit Broadcasting-Struktur (Hub-Effekt):
    speedup_hub = speedup_cols * (1 + 0.2 * sparsity_vals**2)

    ax2.plot(sparsity_vals * 100, speedup_cols, color=MAINBLUE, lw=2,
             label="Spaltenbasiert (uniform sparse)")
    ax2.plot(sparsity_vals * 100, speedup_hub, color=ACCENTRED, lw=2, ls="--",
             label="Spaltenbasiert + Hub-Effekt")
    ax2.fill_between(sparsity_vals * 100, 1, speedup_cols,
                     alpha=0.1, color=MAINBLUE)
    ax2.axhline(1, color=GRAY, ls=":", lw=1)
    ax2.set_xlabel("Aktivierungs-Sparsity (%)")
    ax2.set_ylabel("Beschleunigungsfaktor")
    ax2.set_title("Rechenbeschleunigung durch\nSparse Depension")
    ax2.legend()
    ax2.set_ylim(0, 25)

    # ── Rechts: Strukturelle Identifikation via Depensions-Tiefe ──
    ax3 = axes[2]
    depths = [1, 2, 3, 4, 5, 6]
    n_deps_direct = [n * (0.3 ** d) * 100 for d in depths]   # direkte Abhängigkeiten
    n_deps_indirect = [n * (0.15 ** d) * 100 for d in depths] # indirekte

    ax3.bar([d - 0.2 for d in depths], n_deps_direct,
            width=0.35, color=MAINBLUE, alpha=0.8, label="Direkte Depension (Tiefe $k$)")
    ax3.bar([d + 0.2 for d in depths], n_deps_indirect,
            width=0.35, color=ACCENTRED, alpha=0.8, label="Indirekte Depension ($A^k$-Eintrag)")
    ax3.set_xlabel("Depensions-Tiefe $k$ (Potenz $A^k$)")
    ax3.set_ylabel("Aktive Abhängigkeiten")
    ax3.set_title("Depensions-Tiefe:\nDirekte vs. indirekte Abhängigkeit")
    ax3.legend(fontsize=8)
    ax3.set_xticks(depths)

    plt.tight_layout()
    plt.savefig("fig_sparse_depension.pdf")
    plt.close()
    print("fig_sparse_depension.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 10: Unified Framework – alle Konzepte vereint                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_unified_framework():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Unified Depension Framework: Synthese aus logist. Regression, Systemtheorie und Strukturanalyse",
        fontweight="bold", fontsize=12
    )

    # ── (0,0) Depensions-Spektrum der 3 Modellklassen ──
    ax = axes[0, 0]
    models = ["Lineare\nReg.", "Logist.\nReg.", "Dynamisch\nKS-I", "Dynamisch\nKS-II",
              "Dynamisch\nKS-III", "Dynamisch\nKS-IV"]
    # Spektralabszisse alpha* (normiert)
    alpha_stars = [-np.inf, -1.5, -1.0, 0.0, 1.0, 0.5]
    colors_m = [MAINBLUE, MAINBLUE, DARKGREEN, DARKGREEN, ACCENTRED, ORANGE]
    bar_vals = [2.5, 1.5, 1.0, 0.0, 1.0, 0.5]  # für Balkenplot

    bars = ax.bar(range(len(models)), bar_vals, color=colors_m, alpha=0.8, edgecolor="white")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=7)
    ax.set_ylabel("$|\\alpha^*|$ (Konvergenzstärke)")
    ax.set_title("Depensions-Stabilität\naller Modellklassen")
    ax.text(0.5, 0.9, "DP-I / endlich", transform=ax.transAxes,
            ha="center", fontsize=9, color=DARKGREEN)
    ax.text(4.0, 1.1, "DP-III", fontsize=8, color=ACCENTRED)

    # ── (0,1) Sigmoid vs. Matrixexponential ──
    ax = axes[0, 1]
    z = np.linspace(-4, 4, 200)
    sigma = 1 / (1 + np.exp(-z))
    ax.plot(z, sigma, color=MAINBLUE, lw=2.5, label="$\\sigma(z)$  (statisch)")

    # Matrixexponential-Kurven für verschiedene λ
    ts = np.linspace(0, 4, 200)
    for lam, col, label in [(-1, DARKGREEN, "$e^{-t}$ (KS-I)"),
                             (0,  GRAY,     "$e^{0}=1$ (KS-II)"),
                             (0.5, ACCENTRED, "$e^{0.5t}$ (KS-III)")]:
        ax.plot(ts, np.exp(lam * ts), color=col, lw=2, ls="--", label=label)

    ax.set_xlabel("Argument")
    ax.set_ylabel("Ausgabe")
    ax.set_title("Sigmoid vs. Matrixexponential:\nBeide sind Depensions-Operatoren")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.1, 2.5)

    # ── (0,2) Hesse-Matrix vs. Systemmatrix-Eigenwerte ──
    ax = axes[0, 2]
    rng2 = np.random.default_rng(33)
    n_samp = 100
    d_feat = 5
    X = rng2.standard_normal((n_samp, d_feat))
    w = rng2.standard_normal((d_feat))
    z_log = X @ w
    p_log = 1 / (1 + np.exp(-z_log))
    S = np.diag(p_log * (1 - p_log))
    H = X.T @ S @ X / n_samp
    ev_H = np.sort(np.real(np.linalg.eigvals(H)))[::-1]

    # Systemmatrix für KS-I: A = -H
    A_sys = -H
    ev_A = np.sort(np.real(np.linalg.eigvals(A_sys)))

    ax.barh(range(d_feat), ev_H,  left=0, color=MAINBLUE,  alpha=0.7,
            label="Eigenwerte $H$ (Hesse)")
    ax.barh(range(d_feat), ev_A,  left=0, color=ACCENTRED, alpha=0.7,
            label="Eigenwerte $-H$ (System)")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Eigenwert")
    ax.set_ylabel("Index")
    ax.set_title("Hesse-Matrix = neg. Systemmatrix\n($H \\leftrightarrow -A$, DP-I)")
    ax.legend(fontsize=8)

    # ── (1,0) Vergleich R² / McFadden-R² / DI ──
    ax = axes[1, 0]
    metrics = ["$R^2$\n(lin.)", "McFadden-$R^2$\n(log.)", "DI (KS-I)", "DI (KS-II)", "DI (KS-III)"]
    values  = [0.82, 0.71, 0.95, 0.50, 0.10]
    colors_bar = [MAINBLUE, DARKGREEN, MAINBLUE, ORANGE, ACCENTRED]
    ax.bar(range(len(metrics)), values, color=colors_bar, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("Depensions-Index DI ∈ [0, 1]")
    ax.set_title("DI: Vereinheitlichtes Gütemaß\nüber alle Depensions-Klassen")
    ax.set_ylim(0, 1.15)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    # ── (1,1) Depensions-Tiefe vs. Modell-Komplexität ──
    ax = axes[1, 1]
    complexity = np.array([1, 2, 3, 5, 10, 20, 50])  # Modellparameter * 10
    di_train = 1 - 0.4 / np.sqrt(complexity)
    di_test  = 1 - 0.4 / np.sqrt(complexity) - 0.05 * np.log(1 + complexity / 5)

    ax.semilogx(complexity, di_train, "o-", color=MAINBLUE, lw=2, label="DI (Training)")
    ax.semilogx(complexity, di_test,  "s--", color=ACCENTRED, lw=2, label="DI (Test)")
    ax.fill_between(complexity, di_test, di_train, alpha=0.15, color=GRAY)
    ax.set_xlabel("Modell-Komplexität (log-Skala)")
    ax.set_ylabel("Depensions-Index DI")
    ax.set_title("Bias-Varianz-Tradeoff\nim Depensions-Framework")
    ax.legend()
    ax.set_ylim(0.3, 1.05)

    # ── (1,2) Vollständige Klassifikations-Tabelle ──
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Klasse", "Bedingung", "Beispiel", "Endlich"],
        ["DP-I",   r"alle Re(λ) < 0", "Stab. System, L2-Reg.", "Ja"],
        ["DP-II",  r"max Re(λ) = 0", "Grenzstab., kein L2", "Ja"],
        ["DP-III", r"alle Re(λ) > 0", "Explosives System", "Nein"],
        ["DP-IV",  r"gemischte Re(λ)", "Sattelp., kein Reg.", "Nein"],
    ]
    col_colors = [[MAINBLUE + "33"] * 4,
                  [MAINBLUE + "18"] * 4,
                  [DARKGREEN + "18"] * 4,
                  [ACCENTRED + "18"] * 4,
                  [ORANGE + "18"] * 4]
    t = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc="center", loc="center",
                 colColours=["#19468C22"] * 4)
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.2, 1.7)
    ax.set_title("Depensionsklassen-Tabelle\n(Vollständige Übersicht)", fontsize=10)

    plt.tight_layout()
    plt.savefig("fig_unified_framework.pdf")
    plt.close()
    print("fig_unified_framework.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 11: Depensions-Operator und Phasenportraet                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_phasenportrait_klassen():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Phasenporträts der vier Depensionsklassen DP-I bis DP-IV",
                 fontweight="bold")

    configs = [
        (np.array([[-1.0,  0.6], [-0.6, -0.8]]),
         "DP-I: Asymptotisch stabile Depension\n$\\alpha^* < 0$, Zustandsvektor endlich",
         MAINBLUE),
        (np.array([[0.0,  1.5], [-1.5,  0.0]]),
         "DP-II: Grenzstabile Depension\n$\\alpha^* = 0$, permanente Oszillation",
         DARKGREEN),
        (np.array([[0.8,  0.4], [0.4,  0.6]]),
         "DP-III: Instabile Depension\n$\\alpha^* > 0$, Divergenz",
         ACCENTRED),
        (np.array([[-0.8,  0.3], [0.3,  0.6]]),
         "DP-IV: Hyperbolische Depension\n$\\lambda_1 < 0 < \\lambda_2$ (Sattel)",
         ORANGE),
    ]

    for idx, (A, title, col) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        t = np.linspace(0, 6, 500)

        # Trajektorien aus verschiedenen Anfangsbedingungen
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        r = 2.0
        for ang in angles:
            x0 = np.array([r * np.cos(ang), r * np.sin(ang)])
            traj = np.array([expm(A * ti) @ x0 for ti in t])
            max_norm = max(np.linalg.norm(traj, axis=1))
            if max_norm > 30:
                traj[np.linalg.norm(traj, axis=1) > 15] = np.nan
            ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.2, alpha=0.7)
            ax.annotate("", xy=traj[min(25, len(traj)-1)],
                        xytext=traj[min(20, len(traj)-2)],
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

        ax.plot(0, 0, "k.", ms=10)
        ev = np.linalg.eigvals(A)
        info = f"$\\lambda_{{1,2}} \\approx {np.real(ev[0]):.2f} \\pm {np.abs(np.imag(ev[0])):.2f}i$"
        ax.text(0.05, 0.05, info, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)

    plt.tight_layout()
    plt.savefig("fig_phasenportrait_klassen.pdf")
    plt.close()
    print("fig_phasenportrait_klassen.pdf  ✓")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Abbildung 12: Depensions-Mannigfaltigkeit                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fig_depension_manifold():
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Depensions-Mannigfaltigkeit: Parameterraum und Krümmung",
                 fontweight="bold")

    # ── Links: 2D Schnitt durch Mannigfaltigkeit ──
    ax1 = fig.add_subplot(131)
    w1 = np.linspace(-3, 3, 150)
    w2 = np.linspace(-3, 3, 150)
    W1, W2 = np.meshgrid(w1, w2)

    # Simulierte Kreuzentropie-Verlustfläche
    np.random.seed(0)
    n_s = 80
    X_s = np.random.randn(n_s, 2)
    y_s = (X_s[:, 0] + X_s[:, 1] > 0).astype(float)

    L_surf = np.zeros_like(W1)
    for i in range(len(w1)):
        for j in range(len(w2)):
            z_s = X_s @ np.array([W1[j, i], W2[j, i]])
            p_s = 1 / (1 + np.exp(-z_s))
            p_s = np.clip(p_s, 1e-7, 1-1e-7)
            L_surf[j, i] = -np.mean(y_s*np.log(p_s) + (1-y_s)*np.log(1-p_s))

    cf = ax1.contourf(W1, W2, L_surf, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax1, label="$\\mathcal{L}(w)$", shrink=0.8)
    ax1.contour(W1, W2, L_surf, levels=20, colors="white", linewidths=0.3, alpha=0.4)
    # Gradient-Vektorfeld
    skip = 10
    dL_w1 = np.gradient(L_surf, w1, axis=1)
    dL_w2 = np.gradient(L_surf, w2, axis=0)
    ax1.quiver(W1[::skip, ::skip], W2[::skip, ::skip],
               -dL_w1[::skip, ::skip], -dL_w2[::skip, ::skip],
               color="white", alpha=0.5, scale=8)
    ax1.set_xlabel("$w_1$"); ax1.set_ylabel("$w_2$")
    ax1.set_title("Verlust-Mannigfaltigkeit\nmit Gradientenfeld")

    # ── Mitte: Geodätik auf Mannigfaltigkeit ──
    ax2 = fig.add_subplot(132)
    # Newton vs. Gradient Descent Pfade
    def loss_and_grad(w, X, y):
        z_v = X @ w
        p_v = 1 / (1 + np.exp(-z_v))
        p_v = np.clip(p_v, 1e-7, 1-1e-7)
        L_v = -np.mean(y*np.log(p_v) + (1-y)*np.log(1-p_v))
        g_v = X.T @ (p_v - y) / len(y)
        return L_v, g_v, p_v

    # GD-Pfad
    w_gd = np.array([-2.5, 2.5])
    path_gd = [w_gd.copy()]
    lr = 0.3
    for _ in range(40):
        _, g_v, _ = loss_and_grad(w_gd, X_s, y_s)
        w_gd -= lr * g_v
        path_gd.append(w_gd.copy())
    path_gd = np.array(path_gd)

    # Newton-Pfad
    w_nt = np.array([-2.5, 2.5])
    path_nt = [w_nt.copy()]
    for _ in range(10):
        _, g_v, p_v = loss_and_grad(w_nt, X_s, y_s)
        S_v = np.diag(p_v * (1-p_v))
        H_v = X_s.T @ S_v @ X_s / len(y_s) + 1e-4*np.eye(2)
        w_nt -= np.linalg.solve(H_v, g_v)
        path_nt.append(w_nt.copy())
    path_nt = np.array(path_nt)

    cf2 = ax2.contourf(W1, W2, L_surf, levels=20, cmap="viridis", alpha=0.7)
    ax2.plot(path_gd[:, 0], path_gd[:, 1], "o-", color="white", ms=3, lw=1.5,
             label=f"GD (40 Schr.)")
    ax2.plot(path_nt[:, 0], path_nt[:, 1], "s-", color="yellow", ms=5, lw=2,
             label=f"Newton (10 Schr.)")
    ax2.plot(*path_gd[0], "^", color="white", ms=10)
    ax2.set_xlabel("$w_1$"); ax2.set_ylabel("$w_2$")
    ax2.set_title("Geodätiken auf $\\mathcal{M}$:\nGD vs. Newton")
    ax2.legend(fontsize=8)

    # ── Rechts: Konvergenzverläufe ──
    ax3 = fig.add_subplot(133)
    losses_gd, losses_nt = [], []
    w_gd2 = np.array([-2.5, 2.5])
    w_nt2 = np.array([-2.5, 2.5])
    for k in range(50):
        Lv, gv, pv = loss_and_grad(w_gd2, X_s, y_s)
        losses_gd.append(Lv)
        w_gd2 -= 0.3 * gv

    for k in range(15):
        Lv, gv, pv = loss_and_grad(w_nt2, X_s, y_s)
        losses_nt.append(Lv)
        Sv = np.diag(pv * (1-pv))
        Hv = X_s.T @ Sv @ X_s / len(y_s) + 1e-4*np.eye(2)
        w_nt2 -= np.linalg.solve(Hv, gv)

    ax3.semilogy(range(len(losses_gd)), losses_gd, color=MAINBLUE, lw=2,
                 label="GD: lineare Konvergenz")
    ax3.semilogy(range(len(losses_nt)), losses_nt, "s-", color=ACCENTRED, lw=2.5, ms=7,
                 label="Newton: quadratische Konvergenz")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Verlust $\\mathcal{L}$ (log)")
    ax3.set_title("Konvergenz auf\nDepensions-Mannigfaltigkeit")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_depension_manifold.pdf")
    plt.close()
    print("fig_depension_manifold.pdf  ✓")


# ════ Alle Plots generieren ══════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generiere Abbildungen für depension.tex …\n")

    fig_depension_concept()
    fig_depension_klassen()
    fig_gradient_dynamics()
    fig_depension_index()
    fig_vif_depension()
    fig_regularisierung_klasse()
    fig_hierarchische_depension()
    fig_information_geometry()
    fig_sparse_depension()
    fig_unified_framework()
    fig_phasenportrait_klassen()
    fig_depension_manifold()

    print("\nAlle Abbildungen erfolgreich gespeichert.")
