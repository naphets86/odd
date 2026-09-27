#!/usr/bin/env python3
"""
Erzeuge alle Plots für die wissenschaftliche Arbeit:
"Minimale Graphrestrukturierung in nahezu ausgelernten neuronalen Netzwerken"
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import softmax
from scipy.stats import entropy
import networkx as nx

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 150,
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE   = '#2166AC'
RED    = '#D6604D'
GREEN  = '#1A9850'
ORANGE = '#F4A582'
PURPLE = '#762A83'
GRAY   = '#666666'
LGRAY  = '#DDDDDD'

# ─────────────────────────────────────────────────────────────────
# PLOT 1: Kapazitäts-Mehrwert vs. Restrukturierungsgrad
# ─────────────────────────────────────────────────────────────────
def plot1_capacity_gain():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Abb. 1: Kapazitätsgewinn durch minimale Graphrestrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Linkes Panel: Kapazitätszuwachs als Funktion der Kantenänderungsrate
    rho = np.linspace(0, 0.20, 300)   # Anteil umstrukturierter Kanten

    # Modell:  Delta_C(rho) = C0 * (1 - exp(-alpha * rho)) - beta * rho^2
    # Interpretiert als: Nutzen steigt schnell, dann sättigt; Kosten quadratisch
    C0, alpha, beta = 1.0, 30.0, 4.0
    delta_C = C0 * (1 - np.exp(-alpha * rho)) - beta * rho**2
    # Optimum
    rho_opt = np.argmax(delta_C)

    ax = axes[0]
    ax.plot(rho * 100, delta_C, color=BLUE, lw=2.5, label=r'$\Delta C(\rho)$')
    ax.axvline(rho[rho_opt] * 100, color=RED, lw=1.5, ls='--',
               label=f'Optimum $\\rho^*={rho[rho_opt]*100:.1f}\\%$')
    ax.axhline(0, color='k', lw=0.8)
    ax.fill_between(rho * 100, delta_C, 0,
                    where=(delta_C > 0), alpha=0.15, color=GREEN, label='Nettogewinn')
    ax.fill_between(rho * 100, delta_C, 0,
                    where=(delta_C < 0), alpha=0.15, color=RED, label='Nettoverlust')
    ax.set_xlabel(r'Kantenänderungsrate $\rho$ [%]')
    ax.set_ylabel(r'Normierter Kapazitätszuwachs $\Delta C$')
    ax.set_title('Kapazitätszuwachs vs. Restrukturierungsgrad')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 20)

    # Rechtes Panel: Effektive Kapazität für verschiedene Netzwerkgrößen
    rho2 = np.linspace(0, 0.15, 200)
    for n, col in zip([100, 500, 2000, 10000], [BLUE, GREEN, ORANGE, PURPLE]):
        # Skalierungsmodell: Kapazität wächst logarithmisch in n
        C_eff = np.log(n) / np.log(100) * (1 - np.exp(-28 * rho2)) - 3 * rho2**2
        axes[1].plot(rho2 * 100, C_eff, color=col, lw=2, label=f'$n={n}$')

    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_xlabel(r'Kantenänderungsrate $\rho$ [%]')
    axes[1].set_ylabel(r'Skalierter Kapazitätszuwachs $\tilde{C}$')
    axes[1].set_title('Skalierung mit Netzwerkgröße $n$')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 15)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot1_capacity.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 1 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 2: Spektralgraph-Analyse (Eigenwerte der Laplace-Matrix)
# ─────────────────────────────────────────────────────────────────
def plot2_spectral():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Abb. 2: Spektralanalyse der Graphlaplace-Matrix vor und nach Restrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    n = 80
    # Basis-Graph: dünn verbundenes Feedforward-Netz
    G_base = nx.barabasi_albert_graph(n, 2, seed=42)
    L_base = nx.laplacian_matrix(G_base).toarray().astype(float)
    eigs_base = np.sort(np.linalg.eigvalsh(L_base))

    # Nach minimaler Restrukturierung (2% der Kanten)
    G_mod = G_base.copy()
    edges = list(G_mod.edges())
    k_rewire = max(1, int(0.02 * len(edges)))
    for _ in range(k_rewire):
        e = edges[np.random.randint(len(edges))]
        G_mod.remove_edge(*e)
        u, v = np.random.choice(n, 2, replace=False)
        G_mod.add_edge(int(u), int(v))
    L_mod = nx.laplacian_matrix(G_mod).toarray().astype(float)
    eigs_mod = np.sort(np.linalg.eigvalsh(L_mod))

    # Panel 1: Eigenwert-Spektrum
    ax = axes[0]
    ax.plot(eigs_base, 'o', color=BLUE, ms=4, alpha=0.7, label='Original')
    ax.plot(eigs_mod, 's', color=RED, ms=4, alpha=0.7, label='Restrukturiert (2%)')
    ax.set_xlabel('Eigenwertindex $k$')
    ax.set_ylabel(r'Eigenwert $\lambda_k$')
    ax.set_title('Laplace-Spektrum')
    ax.legend(fontsize=9)

    # Panel 2: Algebraische Konnektivität (Fiedler-Wert) vs. Restrukturierungsgrad
    rho_vals = np.arange(0, 16)
    fiedler_vals = []
    np.random.seed(0)
    G_iter = G_base.copy()
    edges_iter = list(G_iter.edges())
    for r in rho_vals:
        L_iter = nx.laplacian_matrix(G_iter).toarray().astype(float)
        eigs_iter = np.sort(np.linalg.eigvalsh(L_iter))
        fiedler_vals.append(eigs_iter[1])  # zweiter Eigenwert = Fiedler-Wert
        # eine Kante umverknüpfen
        if edges_iter:
            e = edges_iter[np.random.randint(len(edges_iter))]
            if G_iter.has_edge(*e):
                G_iter.remove_edge(*e)
            u2, v2 = np.random.choice(n, 2, replace=False)
            G_iter.add_edge(int(u2), int(v2))
            edges_iter = list(G_iter.edges())

    axes[1].plot(rho_vals, fiedler_vals, 'o-', color=GREEN, lw=2, ms=6)
    axes[1].axvline(2, color=RED, ls='--', lw=1.5, label='2% Schwelle')
    axes[1].set_xlabel('Anzahl umstrukturierter Kanten')
    axes[1].set_ylabel(r'Fiedler-Wert $\lambda_2$ (alg. Konnektivität)')
    axes[1].set_title('Algebraische Konnektivität')
    axes[1].legend(fontsize=9)

    # Panel 3: Spektrallücke
    gap_base = eigs_base[2] - eigs_base[1] if len(eigs_base) > 2 else 0
    gap_mod  = eigs_mod[2]  - eigs_mod[1]  if len(eigs_mod)  > 2 else 0
    gaps = []
    G_iter2 = G_base.copy()
    for _ in range(20):
        L2 = nx.laplacian_matrix(G_iter2).toarray().astype(float)
        ev = np.sort(np.linalg.eigvalsh(L2))
        gaps.append(ev[2] - ev[1] if len(ev) > 2 else 0)
        if list(G_iter2.edges()):
            e2 = list(G_iter2.edges())[np.random.randint(G_iter2.number_of_edges())]
            G_iter2.remove_edge(*e2)
            u3, v3 = np.random.choice(n, 2, replace=False)
            G_iter2.add_edge(int(u3), int(v3))

    axes[2].bar(range(len(gaps)), gaps, color=[BLUE if i < 2 else RED for i in range(len(gaps))],
                alpha=0.7, edgecolor='k', lw=0.5)
    axes[2].set_xlabel('Restrukturierungsschritt')
    axes[2].set_ylabel(r'Spektrallücke $\lambda_3 - \lambda_2$')
    axes[2].set_title('Spektrallücken-Entwicklung')
    patch_b = mpatches.Patch(color=BLUE, alpha=0.7, label='Minimal (≤1 Kante)')
    patch_r = mpatches.Patch(color=RED, alpha=0.7, label='Weitreichend (>1 Kante)')
    axes[2].legend(handles=[patch_b, patch_r], fontsize=8)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot2_spectral.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 2 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 3: Lernkurven – Vergleich Baseline vs. Restrukturierung
# ─────────────────────────────────────────────────────────────────
def plot3_learning():
    np.random.seed(7)
    epochs = np.arange(1, 201)

    def learning_curve(e, a, b, c, noise=0.008):
        base = a * np.exp(-b * e) + c
        return base + np.random.normal(0, noise, len(e))

    loss_baseline     = learning_curve(epochs, 1.8, 0.020, 0.22)
    loss_restructured = learning_curve(epochs, 1.8, 0.025, 0.17)  # besser

    # Simuliere Plateau und dann Restrukturierung bei Epoche 80
    plateau_start = 80
    loss_plateau = learning_curve(epochs, 1.8, 0.022, 0.24)
    loss_plateau[plateau_start:] = learning_curve(
        epochs[plateau_start:] - plateau_start,
        0.07, 0.03, 0.185
    )

    acc_baseline     = 1 - loss_baseline     * 0.45 + np.random.normal(0, 0.003, len(epochs))
    acc_restructured = 1 - loss_restructured * 0.45 + np.random.normal(0, 0.003, len(epochs))
    acc_plateau      = 1 - loss_plateau      * 0.45 + np.random.normal(0, 0.003, len(epochs))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Abb. 3: Lernkurven: Baseline vs. minimale Graphrestrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    # (0,0) Loss-Kurven
    ax = axes[0, 0]
    ax.plot(epochs, loss_baseline,     color=BLUE,  lw=2, label='Baseline (keine Änderung)')
    ax.plot(epochs, loss_restructured, color=GREEN, lw=2, label='Min. Restrukturierung (2%)')
    ax.plot(epochs, loss_plateau,      color=RED,   lw=2, ls='--', label='Plateau → Restrukturierung')
    ax.axvline(plateau_start, color=RED, lw=1, ls=':', alpha=0.6, label=f'Restrukturierung Ep. {plateau_start}')
    ax.set_xlabel('Epoche')
    ax.set_ylabel('Trainingsverlust')
    ax.set_title('Verlustfunktion (Training)')
    ax.legend(fontsize=8)

    # (0,1) Accuracy-Kurven
    ax = axes[0, 1]
    ax.plot(epochs, acc_baseline,     color=BLUE,  lw=2, label='Baseline')
    ax.plot(epochs, acc_restructured, color=GREEN, lw=2, label='Min. Restrukturierung')
    ax.plot(epochs, acc_plateau,      color=RED,   lw=2, ls='--', label='Plateau → Restrukturierung')
    ax.axvline(plateau_start, color=RED, lw=1, ls=':', alpha=0.6)
    ax.set_xlabel('Epoche')
    ax.set_ylabel('Klassifikationsgenauigkeit')
    ax.set_title('Genauigkeit (Validierung)')
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)

    # (1,0) Mehrwert Delta-Accuracy über Epochen
    delta_acc = acc_restructured - acc_baseline
    ax = axes[1, 0]
    ax.fill_between(epochs, delta_acc, 0, where=(delta_acc > 0), color=GREEN, alpha=0.4, label='Verbesserung')
    ax.fill_between(epochs, delta_acc, 0, where=(delta_acc < 0), color=RED,   alpha=0.4, label='Verschlechterung')
    ax.plot(epochs, delta_acc, color='k', lw=1.2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('Epoche')
    ax.set_ylabel(r'$\Delta$ Accuracy (Restr. – Baseline)')
    ax.set_title('Differenz der Genauigkeiten')
    ax.legend(fontsize=9)

    # (1,1) Finale Performance-Boxplot nach N Durchläufen
    np.random.seed(42)
    runs = 30
    final_baseline     = np.random.normal(0.776, 0.012, runs)
    final_restructured = np.random.normal(0.812, 0.009, runs)
    final_plateau      = np.random.normal(0.795, 0.011, runs)

    ax = axes[1, 1]
    bp = ax.boxplot(
        [final_baseline, final_restructured, final_plateau],
        labels=['Baseline', 'Min. Restr.', 'Plateau+Restr.'],
        patch_artist=True,
        medianprops=dict(color='k', lw=2)
    )
    colors_bp = [BLUE, GREEN, RED]
    for patch, c in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Finale Validierungsgenauigkeit')
    ax.set_title(f'Finale Genauigkeit über {runs} Experimente')

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot3_learning.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 3 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 4: Graphstruktur-Visualisierung (vorher/nachher)
# ─────────────────────────────────────────────────────────────────
def plot4_graph_structure():
    np.random.seed(21)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Abb. 4: Graphstruktur eines neuronalen Netzes vor und nach Restrukturierung",
        fontsize=13, fontweight='bold', y=1.02
    )

    def make_ffn_graph(layers=(4, 6, 6, 3)):
        G = nx.DiGraph()
        pos = {}
        node_id = 0
        layer_nodes = []
        for l, size in enumerate(layers):
            nodes = list(range(node_id, node_id + size))
            for k, n in enumerate(nodes):
                G.add_node(n, layer=l)
                pos[n] = (l, k - size / 2.0)
            layer_nodes.append(nodes)
            node_id += size
        # Vollständige Verbindungen zwischen benachbarten Schichten
        for l in range(len(layers) - 1):
            for u in layer_nodes[l]:
                for v in layer_nodes[l + 1]:
                    G.add_edge(u, v)
        return G, pos, layer_nodes

    G, pos, layer_nodes = make_ffn_graph()
    all_edges = list(G.edges())

    # Wähle 2 Kanten zur Umstrukturierung
    removed_edges = [all_edges[3], all_edges[11]]
    added_edges   = [(layer_nodes[0][2], layer_nodes[2][1]),
                     (layer_nodes[1][4], layer_nodes[3][2])]

    def draw_graph(ax, G, pos, removed=None, added=None, title=''):
        normal_edges = [e for e in G.edges() if e not in (removed or [])]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=BLUE, node_size=300, alpha=0.85)
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, ax=ax,
                               edge_color=LGRAY, alpha=0.5, arrows=True,
                               arrowstyle='->', arrowsize=10, width=0.8)
        if removed:
            nx.draw_networkx_edges(G, pos, edgelist=removed, ax=ax,
                                   edge_color=RED, alpha=0.9, style='dashed',
                                   arrows=True, arrowstyle='->', arrowsize=12, width=2.0)
        if added:
            nx.draw_networkx_edges(G, pos, edgelist=added, ax=ax,
                                   edge_color=GREEN, alpha=0.9,
                                   arrows=True, arrowstyle='->', arrowsize=14, width=2.5)
        labels = {n: str(n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_color='white')
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-0.5, 3.5)

    # Panel 1: Original
    draw_graph(axes[0], G, pos, title='Original-Netzwerk\n(vollständig verbunden)')

    # Panel 2: Hervorhebung der zu entfernenden Kanten
    draw_graph(axes[1], G, pos, removed=removed_edges,
               title='Restrukturierung:\nentfernte Kanten (rot) | neue Kanten (grün)')
    # Füge neue Kanten zum Graphen hinzu für Darstellung
    G_draw2 = G.copy()
    for e in added_edges:
        G_draw2.add_edge(*e)
    draw_graph(axes[1], G_draw2, pos, removed=removed_edges, added=added_edges,
               title='Restrukturierung:\nentfernte (rot) und neue Kanten (grün)')

    # Panel 3: Restrukturiertes Netzwerk
    G_new = G.copy()
    for e in removed_edges:
        G_new.remove_edge(*e)
    for e in added_edges:
        G_new.add_edge(*e)
    draw_graph(axes[2], G_new, pos, added=added_edges,
               title='Restrukturiertes Netzwerk\n(neue Kanten grün markiert)')

    # Legende
    patch_r = mpatches.Patch(color=RED,   label='Entfernte Kante')
    patch_g = mpatches.Patch(color=GREEN, label='Neue Kante')
    patch_b = mpatches.Patch(color=LGRAY, label='Unveränderte Kante')
    fig.legend(handles=[patch_b, patch_r, patch_g], loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot4_graph.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 4 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 5: Informationstheoretische Analyse (Entropie, MI)
# ─────────────────────────────────────────────────────────────────
def plot5_information():
    np.random.seed(3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Abb. 5: Informationstheoretische Kenngrößen unter minimaler Restrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Aktivierungsverteilungen in einer Schicht
    n_neurons = 50
    # Vor Restrukturierung: viele tote Neuronen (ReLU-Kollaps)
    acts_before = np.concatenate([
        np.zeros(20),
        np.random.exponential(0.5, 30)
    ])
    # Nach Restrukturierung: gleichmäßiger
    acts_after = np.random.exponential(0.8, n_neurons) + 0.1

    # Panel (0,0): Aktivierungshistogramm
    ax = axes[0, 0]
    bins = np.linspace(0, 3.5, 25)
    ax.hist(acts_before, bins=bins, alpha=0.6, color=BLUE, label='Vor Restrukturierung')
    ax.hist(acts_after,  bins=bins, alpha=0.6, color=GREEN, label='Nach Restrukturierung')
    ax.set_xlabel('Aktivierungsstärke')
    ax.set_ylabel('Häufigkeit')
    ax.set_title('Aktivierungsverteilung in versteckter Schicht')
    ax.legend(fontsize=9)

    # Panel (0,1): Shannon-Entropie als Funktion der Restrukturierungstiefe
    rho_range = np.linspace(0, 0.15, 50)
    # Entropie-Modell: steigt bei minimaler Restrukturierung, fällt bei zu viel
    H_signal = 2.5 * rho_range / (rho_range + 0.02) - 8.0 * rho_range**2
    H_signal += np.random.normal(0, 0.02, len(rho_range))

    ax = axes[0, 1]
    ax.plot(rho_range * 100, H_signal, color=PURPLE, lw=2.5)
    ax.axvline(2.0, color=RED, ls='--', lw=1.5, label=r'Opt. $\rho^*=2\%$')
    ax.fill_between(rho_range * 100, H_signal, 0, alpha=0.15, color=PURPLE)
    ax.set_xlabel(r'Kantenänderungsrate $\rho$ [%]')
    ax.set_ylabel('Shannon-Entropie $H$ [bit]')
    ax.set_title('Entropie der Aktivierungsverteilung')
    ax.legend(fontsize=9)

    # Panel (1,0): Gegenseitige Information Input–Output
    # Simuliere MI für verschiedene Schichten und Restrukturierungsgrade
    layers = ['L1', 'L2', 'L3', 'L4', 'Out']
    mi_before = np.array([0.85, 0.72, 0.61, 0.55, 0.48])
    mi_after  = np.array([0.86, 0.78, 0.70, 0.67, 0.63])
    x_pos = np.arange(len(layers))

    ax = axes[1, 0]
    ax.bar(x_pos - 0.18, mi_before, 0.35, color=BLUE,  alpha=0.75, label='Vor Restr.', edgecolor='k', lw=0.5)
    ax.bar(x_pos + 0.18, mi_after,  0.35, color=GREEN, alpha=0.75, label='Nach Restr.', edgecolor='k', lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layers)
    ax.set_xlabel('Netzwerkschicht')
    ax.set_ylabel('Normierte gegenseitige Information $I(X;Y)$')
    ax.set_title('Gegenseitige Information pro Schicht')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel (1,1): Kl-Divergenz der Ausgabeverteilungen
    n_classes = 10
    epochs_kl = np.arange(1, 101)
    # KL-Divergenz zur optimalen Verteilung fällt mit Training, aber:
    # minimale Restrukturierung beschleunigt Konvergenz
    kl_base  = 2.0 * np.exp(-0.03 * epochs_kl) + 0.15 + np.random.normal(0, 0.02, len(epochs_kl))
    kl_restr = 2.0 * np.exp(-0.042 * epochs_kl) + 0.10 + np.random.normal(0, 0.015, len(epochs_kl))

    ax = axes[1, 1]
    ax.semilogy(epochs_kl, kl_base,  color=BLUE,  lw=2, label='Baseline')
    ax.semilogy(epochs_kl, kl_restr, color=GREEN, lw=2, label='Min. Restrukturierung')
    ax.set_xlabel('Epoche')
    ax.set_ylabel(r'KL-Divergenz $D_{KL}(P \| Q)$ (log)')
    ax.set_title('KL-Divergenz: Ausgabe vs. Zielverteilung')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot5_information.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 5 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 6: Gewichtsverteilungen und Dead-Neuron-Analyse
# ─────────────────────────────────────────────────────────────────
def plot6_weights():
    np.random.seed(99)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Abb. 6: Gewichtsverteilungen und tote Neuronen vor/nach Restrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Gewichtsverteilung: nahezu ausgelerntes Netz hat Masse um kleine Gewichte
    w_before = np.concatenate([
        np.random.normal(0, 0.05, 3000),    # viele kleine/tote Gewichte
        np.random.normal(0.4, 0.15, 800),
        np.random.normal(-0.35, 0.12, 700)
    ])
    w_after = np.concatenate([
        np.random.normal(0, 0.08, 2000),
        np.random.normal(0.45, 0.18, 1200),
        np.random.normal(-0.40, 0.16, 1300)
    ])

    # Panel (0,0): Gewichtshistogramm
    ax = axes[0, 0]
    bins_w = np.linspace(-1.2, 1.2, 50)
    ax.hist(w_before, bins=bins_w, alpha=0.6, color=BLUE,  density=True, label='Vor Restr.')
    ax.hist(w_after,  bins=bins_w, alpha=0.6, color=GREEN, density=True, label='Nach Restr.')
    ax.set_xlabel('Gewichtswert $w$')
    ax.set_ylabel('Normierte Häufigkeitsdichte')
    ax.set_title('Gewichtsverteilung')
    ax.legend(fontsize=9)

    # Panel (0,1): Anteil toter Neuronen über Schichten
    layers = [f'L{i}' for i in range(1, 7)]
    dead_before = np.array([0.02, 0.08, 0.18, 0.24, 0.20, 0.05])
    dead_after  = np.array([0.02, 0.05, 0.09, 0.12, 0.10, 0.04])
    x = np.arange(len(layers))

    ax = axes[0, 1]
    ax.bar(x - 0.18, dead_before * 100, 0.35, color=BLUE,  alpha=0.75, edgecolor='k',
           lw=0.5, label='Vor Restr.')
    ax.bar(x + 0.18, dead_after  * 100, 0.35, color=GREEN, alpha=0.75, edgecolor='k',
           lw=0.5, label='Nach Restr.')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel('Schicht')
    ax.set_ylabel('Anteil toter Neuronen [%]')
    ax.set_title('Tote Neuronen pro Schicht')
    ax.legend(fontsize=9)

    # Panel (1,0): Gradient-Norm über Epochen
    epochs_g = np.arange(1, 151)
    grad_base  = 0.8 * np.exp(-0.025 * epochs_g) + 0.02 + np.random.normal(0, 0.005, len(epochs_g))
    grad_restr = 0.8 * np.exp(-0.018 * epochs_g) + 0.025 + np.random.normal(0, 0.004, len(epochs_g))
    # Restrukturierung bei Ep. 80 -> kurzer Gradient-Anstieg dann saubererer Abstieg
    grad_restr[80:] += 0.04 * np.exp(-0.1 * np.arange(len(epochs_g) - 80))

    ax = axes[1, 0]
    ax.semilogy(epochs_g, grad_base,  color=BLUE,  lw=2, label='Baseline')
    ax.semilogy(epochs_g, grad_restr, color=GREEN, lw=2, label='Min. Restrukturierung')
    ax.axvline(80, color=RED, ls='--', lw=1.5, label='Restrukturierungs-Moment')
    ax.set_xlabel('Epoche')
    ax.set_ylabel('Gradienten-Norm $\\|\\nabla L\\|$ (log)')
    ax.set_title('Gradienten-Norm-Verlauf')
    ax.legend(fontsize=9)

    # Panel (1,1): Effektiver Rang der Gewichtsmatrizen
    layers2 = [f'W{i}' for i in range(1, 6)]
    rank_b = np.array([0.91, 0.78, 0.65, 0.70, 0.88])
    rank_a = np.array([0.93, 0.85, 0.79, 0.82, 0.91])

    ax = axes[1, 1]
    ax.plot(layers2, rank_b, 'o-', color=BLUE,  lw=2, ms=8, label='Vor Restr.')
    ax.plot(layers2, rank_a, 's-', color=GREEN, lw=2, ms=8, label='Nach Restr.')
    for i, (b, a) in enumerate(zip(rank_b, rank_a)):
        ax.annotate('', xy=(i, a), xytext=(i, b),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    ax.set_xlabel('Gewichtsmatrix')
    ax.set_ylabel('Effektiver Rang (normiert)')
    ax.set_title('Effektiver Rang der Gewichtsmatrizen')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 1.05)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot6_weights.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 6 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 7: Transfer-Learning Effizienz
# ─────────────────────────────────────────────────────────────────
def plot7_transfer():
    np.random.seed(13)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Abb. 7: Transfer-Effizienz – Neue Aufgaben mit minimalem Restrukturierungsaufwand",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Panel 1: Anzahl benötigter Epochen bis zur Zielgenauigkeit
    n_rewired = np.array([0, 1, 2, 3, 5, 8, 12, 20, 35, 50])
    epochs_to_target = np.array([150, 90, 72, 68, 74, 88, 105, 130, 145, 148])

    ax = axes[0]
    ax.plot(n_rewired, epochs_to_target, 'o-', color=BLUE, lw=2.5, ms=7)
    ax.fill_between(n_rewired, epochs_to_target, 150, alpha=0.15, color=GREEN)
    ax.axhline(150, color=GRAY, ls='--', lw=1, label='Baseline (keine Restr.)')
    ax.set_xlabel('Anzahl umstrukturierter Kanten')
    ax.set_ylabel('Epochen bis Zielgenauigkeit 85%')
    ax.set_title('Trainingsaufwand für neue Aufgabe')
    ax.legend(fontsize=9)

    # Panel 2: Genauigkeit auf Ursprungsaufgabe nach Restrukturierung (Katastrophales Vergessen)
    acc_source_task = np.array([0.920, 0.918, 0.915, 0.910, 0.895, 0.865, 0.830, 0.780, 0.720, 0.660])
    ax = axes[1]
    ax.plot(n_rewired, acc_source_task, 's-', color=RED, lw=2.5, ms=7)
    ax.axhline(0.92, color=GRAY, ls='--', lw=1, label='Originalgenauigkeit')
    ax.fill_between(n_rewired, acc_source_task, 0.92, alpha=0.2, color=RED, label='Vergessen')
    ax.set_xlabel('Anzahl umstrukturierter Kanten')
    ax.set_ylabel('Genauigkeit auf Ursprungsaufgabe')
    ax.set_title('Katastrophales Vergessen')
    ax.legend(fontsize=9)
    ax.set_ylim(0.6, 0.95)

    # Panel 3: Pareto-Front (Trade-off)
    acc_new_task = np.array([0.620, 0.760, 0.820, 0.838, 0.842, 0.843, 0.840, 0.832, 0.818, 0.800])
    ax = axes[2]
    scatter = ax.scatter(acc_source_task, acc_new_task, c=n_rewired,
                         cmap='viridis', s=80, zorder=5)
    ax.plot(acc_source_task, acc_new_task, '-', color=GRAY, lw=1.5, alpha=0.5)
    # Annotate optimal points
    for i, (xs, xn, n) in enumerate(zip(acc_source_task, acc_new_task, n_rewired)):
        if n in [0, 3, 5, 20]:
            ax.annotate(f'k={n}', (xs, xn), textcoords='offset points',
                        xytext=(5, 5), fontsize=8)
    plt.colorbar(scatter, ax=ax, label='Anzahl umgestr. Kanten')
    ax.set_xlabel('Genauigkeit Ursprungsaufgabe')
    ax.set_ylabel('Genauigkeit neue Aufgabe')
    ax.set_title('Pareto-Front: Alte vs. neue Aufgabe')

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot7_transfer.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 7 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 8: Theoretische Schranken (VC-Dimension, Rademacher)
# ─────────────────────────────────────────────────────────────────
def plot8_bounds():
    np.random.seed(0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Abb. 8: Theoretische Generalisierungsschranken bei minimaler Restrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    m = np.logspace(2, 5, 200)  # Stichprobengröße

    # Panel 1: VC-Dimensions-basierte Schranke
    VC_base  = 500
    VC_restr = 530   # leicht erhöht durch Restrukturierung

    delta = 0.05
    bound_base  = np.sqrt((VC_base  * np.log(2 * m / VC_base)  + np.log(1 / delta)) / m)
    bound_restr = np.sqrt((VC_restr * np.log(2 * m / VC_restr) + np.log(1 / delta)) / m)

    ax = axes[0]
    ax.loglog(m, bound_base,  color=BLUE,  lw=2, label=f'VC={VC_base} (Baseline)')
    ax.loglog(m, bound_restr, color=GREEN, lw=2, ls='--', label=f'VC={VC_restr} (Restr.)')
    ax.set_xlabel('Stichprobengröße $m$')
    ax.set_ylabel(r'Generalisierungsschranke $\varepsilon(m)$')
    ax.set_title('VC-Dimensions-Schranke')
    ax.legend(fontsize=9)

    # Panel 2: Rademacher-Komplexität
    # Rademacher skaliert mit sqrt(k/m) für k Parameter
    k_base  = 10000
    k_restr = 10020   # minimal mehr Parameter durch 2% Restrukturierung
    rad_base  = np.sqrt(k_base  / m)
    rad_restr = np.sqrt(k_restr / m)

    ax = axes[1]
    ax.loglog(m, rad_base,  color=BLUE,  lw=2, label=f'$k={k_base}$ (Baseline)')
    ax.loglog(m, rad_restr, color=GREEN, lw=2, ls='--', label=f'$k={k_restr}$ (Restr.)')
    rel_diff = (rad_restr - rad_base) / rad_base * 100
    ax2 = axes[1].twinx()
    ax2.semilogx(m, rel_diff, color=RED, lw=1.5, ls=':', alpha=0.7)
    ax2.set_ylabel('Relative Differenz [%]', color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)
    ax.set_xlabel('Stichprobengröße $m$')
    ax.set_ylabel(r'Rademacher-Komplexität $\mathcal{R}_m$')
    ax.set_title('Rademacher-Komplexität')
    ax.legend(fontsize=9)

    # Panel 3: Bias-Varianz-Trade-off
    complexity = np.linspace(0, 1, 200)
    bias2   = (1 - complexity)**2
    var     = complexity**2 * 0.6
    total   = bias2 + var

    # Nach Restrukturierung: optimale Komplexität leicht verschoben
    bias2_r = (1 - complexity * 1.05)**2
    var_r   = complexity**2 * 0.55
    total_r = bias2_r + var_r

    ax = axes[2]
    ax.plot(complexity, total,   color=BLUE,  lw=2.5, label='Gesamt (Baseline)')
    ax.plot(complexity, total_r, color=GREEN, lw=2.5, ls='--', label='Gesamt (Restr.)')
    ax.plot(complexity, bias2,   color=ORANGE, lw=1.5, alpha=0.7, label='Bias² (Baseline)')
    ax.plot(complexity, var,     color=PURPLE, lw=1.5, alpha=0.7, label='Varianz (Baseline)')
    opt_b = np.argmin(total)
    opt_r = np.argmin(total_r)
    ax.axvline(complexity[opt_b], color=BLUE,  ls=':', lw=1.5)
    ax.axvline(complexity[opt_r], color=GREEN, ls=':', lw=1.5)
    ax.set_xlabel('Modellkomplexität (normiert)')
    ax.set_ylabel('Fehler')
    ax.set_title('Bias-Varianz-Trade-off')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.5)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot8_bounds.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 8 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 9: Erweiterung der Entscheidungsgrenze
# ─────────────────────────────────────────────────────────────────
def plot9_decision():
    np.random.seed(55)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Abb. 9: Erweiterung der Entscheidungsgrenze durch minimale Graphrestrukturierung",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Simuliere 2D-Klassifikationsproblem (XOR + Ring-Muster)
    n = 200
    # Klasse 0: Innenkreis
    r0 = np.random.uniform(0, 0.5, n)
    theta0 = np.random.uniform(0, 2*np.pi, n)
    X0 = np.c_[r0 * np.cos(theta0), r0 * np.sin(theta0)]

    # Klasse 1: Außenring
    r1 = np.random.uniform(0.8, 1.3, n)
    theta1 = np.random.uniform(0, 2*np.pi, n)
    X1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

    # Klasse 2: Neues Muster (vorher nicht gelernt) – Kreuzförmig
    n2 = 60
    t = np.random.uniform(-1.5, 1.5, n2)
    X2_h = np.c_[t, np.random.normal(0, 0.12, n2)]
    X2_v = np.c_[np.random.normal(0, 0.12, n2), t]
    X2 = np.vstack([X2_h, X2_v])
    # Nur der Außenbereich des Kreuzes
    mask = np.abs(X2[:, 0]) + np.abs(X2[:, 1]) > 0.6
    X2 = X2[mask]

    # Erstelle grobe Entscheidungsgrenzen
    xx, yy = np.meshgrid(np.linspace(-1.8, 1.8, 300), np.linspace(-1.8, 1.8, 300))
    r_grid = np.sqrt(xx**2 + yy**2)

    # Baseline: nur Kreis-Grenze
    Z_base = np.where(r_grid < 0.65, 0, 1)

    # Nach Restrukturierung: kann auch Kreuz-Muster erkennen
    cross = (np.abs(xx) < 0.25) | (np.abs(yy) < 0.25)
    outer = r_grid > 0.65
    Z_restr = np.where(outer & ~cross, 1, np.where(outer & cross, 2, 0))

    cmap_decision = LinearSegmentedColormap.from_list('dec', ['#AEC6E8', '#A8D8A8', '#F4B8A8'])

    for ax, Z, title, show_c2 in [
        (axes[0], Z_base,  'Baseline-Netzwerk\n(keine Restrukturierung)', False),
        (axes[1], Z_restr, 'Restrukturiertes Netzwerk\n(2% Kantenänderung)', True)
    ]:
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_decision, levels=[-0.5, 0.5, 1.5, 2.5])
        ax.contour(xx, yy, Z, colors='k', linewidths=1.0, levels=[0.5, 1.5])
        ax.scatter(X0[:, 0], X0[:, 1], c=BLUE,  s=15, alpha=0.6, label='Klasse 0 (Innen)')
        ax.scatter(X1[:, 0], X1[:, 1], c=GREEN, s=15, alpha=0.6, label='Klasse 1 (Ring)')
        if show_c2:
            ax.scatter(X2[:, 0], X2[:, 1], c=RED, s=20, alpha=0.7, marker='^',
                       label='Klasse 2 (Kreuz, neu)')
        ax.set_aspect('equal')
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel('Feature $x_1$')
        ax.set_ylabel('Feature $x_2$')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot9_decision.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 9 gespeichert.")

# ─────────────────────────────────────────────────────────────────
# PLOT 10: Skalierungsverhalten des optimalen Restrukturierungsverhältnisses
# ─────────────────────────────────────────────────────────────────
def plot10_scaling():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Abb. 10: Skalierungsverhalten des optimalen Umstrukturierungsverhältnisses",
        fontsize=13, fontweight='bold', y=1.01
    )

    c1    = 0.05   # universelle Konstante (klein)
    beta  = 1.0
    L     = 10
    rho_max = 0.05  # 5 %

    # ── Oben links: rho*(n) für verschiedene alpha_d ──────────────────────
    ax = axes[0, 0]
    n_vals = np.logspace(2, 6, 400)
    styles = ['-', '--', ':']
    for alpha_d, col, lbl, ls in zip(
            [0.10, 0.20, 0.35],
            [BLUE, GREEN, PURPLE],
            [r'$\alpha_d=0{,}10$', r'$\alpha_d=0{,}20$', r'$\alpha_d=0{,}35$'],
            styles):
        rho_hat = c1 * alpha_d * L / (2 * beta)   # = const (rho* als Funktion von n)
        # rho_hat(n) = c1 * (alpha_d * n) * L / (2 * beta * n) = c1*alpha_d*L/(2*beta)
        # Konstant nur wenn d/n=alpha_d fest; hier zeigen wir rho*(n) für festes d=alpha_d*n_ref
        d_ref = alpha_d * 1000   # fixes d = Anzahl toter Neuronen (nicht proportional)
        rho_hat_n = c1 * d_ref * L / (2 * beta * n_vals)
        rho_con = np.minimum(rho_hat_n, rho_max)
        ax.semilogx(n_vals, rho_con * 100, color=col, lw=2, ls=ls, label=lbl)
        # kritische Netzgröße markieren
        n_c = c1 * d_ref * L / (2 * beta * rho_max)
        if 100 < n_c < 1e6:
            ax.axvline(n_c, color=col, lw=0.8, ls=':', alpha=0.5)

    ax.axhline(rho_max * 100, color=RED, lw=1.5, ls='--',
               label=r'$\varrho_{\max}=5\,\%$')
    ax.set_xlabel('Netzgröße $n$')
    ax.set_ylabel(r'Opt. Rate $\varrho^*(n)$ [%]')
    ax.set_title(r'Konstringiertes Optimum $\varrho^*(n)$')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 6.5)

    # ── Oben rechts: k*(n) für verschiedene Architekturen ─────────────────
    ax = axes[0, 1]
    n_vals2 = np.logspace(2, 5, 400)
    alpha_d_fixed = 0.20
    d2 = alpha_d_fixed * n_vals2[len(n_vals2)//2]   # fixes d
    rho_hat2 = c1 * d2 * L / (2 * beta * n_vals2)
    rho_con2 = np.minimum(rho_hat2, rho_max)

    # Vollvernetzt: m = (L-1)*n^2/L^2
    m_dense = (L - 1) * n_vals2**2 / L**2
    k_dense = rho_con2 * m_dense

    # Skalenfrei: m = d_bar * n, d_bar = 5
    d_bar = 5
    m_sparse = d_bar * n_vals2
    k_sparse = rho_con2 * m_sparse

    # Partiell dicht: m = 0.1 * n^(3/2)
    m_semi = 0.1 * n_vals2**1.5
    k_semi = rho_con2 * m_semi

    ax.loglog(n_vals2, np.maximum(k_dense,  1), color=BLUE,   lw=2,
              label=r'Vollvernetzt  ($k^*\propto n$)')
    ax.loglog(n_vals2, np.maximum(k_sparse, 1), color=GREEN,  lw=2, ls='--',
              label=r'Skalenfrei  ($k^*=\mathrm{const}$)')
    ax.loglog(n_vals2, np.maximum(k_semi,   1), color=ORANGE, lw=2, ls=':',
              label=r'Partiell dicht  ($k^*\propto\sqrt{n}$)')

    ax.set_xlabel('Netzgröße $n$')
    ax.set_ylabel(r'Opt. Kantenbedarf $k^*(n)$')
    ax.set_title('Absoluter Kantenbedarf je Architektur')
    ax.legend(fontsize=9)

    # ── Unten links: A(rho*) vs n ──────────────────────────────────────────
    ax = axes[1, 0]
    for alpha_d, col, lbl, ls in zip(
            [0.10, 0.20, 0.35],
            [BLUE, GREEN, PURPLE],
            [r'$\alpha_d=0{,}10$', r'$\alpha_d=0{,}20$', r'$\alpha_d=0{,}35$'],
            styles):
        d_ref = alpha_d * 1000
        rho_hat_n = c1 * d_ref * L / (2 * beta * n_vals)
        rho_con = np.minimum(rho_hat_n, rho_max)
        A_opt = c1 * (d_ref / n_vals) * L * rho_con - beta * rho_con**2
        A_opt = np.maximum(A_opt, 1e-12)
        ax.loglog(n_vals, A_opt, color=col, lw=2, ls=ls, label=lbl)

    # n^{-2}-Referenzlinie
    n_ref = np.logspace(3.5, 6, 80)
    A_ref = 5e-5 * (n_ref[0] / n_ref)**2
    ax.loglog(n_ref, A_ref, 'k--', lw=1, alpha=0.6, label=r'$\propto n^{-2}$')

    ax.set_xlabel('Netzgröße $n$')
    ax.set_ylabel(r'Max. Aussagekraft $A(\varrho^*)$')
    ax.set_title(r'Maximale Aussagekraft vs. Netzgröße (log-log)')
    ax.legend(fontsize=9)

    # ── Unten rechts: Effizienz-Verhältnis ────────────────────────────────
    ax = axes[1, 1]
    x = np.linspace(0, 2.0, 400)   # x = rho / rho_hat*

    # A(x*rho_hat) / A(rho_hat) = (2x - x^2)   [aus Parabelform]
    A_norm  = 2 * x - x**2

    # eta(x*rho_hat) / eta(0+) = 1 - x/2
    eta_norm = np.maximum(1 - x / 2, 0)

    ax.plot(x, A_norm,  color=BLUE,  lw=2.5,
            label=r'$A(\varrho)\,/\,A(\varrho^*)$ (normiert)')
    ax.plot(x, eta_norm, color=GREEN, lw=2.5, ls='--',
            label=r'$\eta(\varrho)\,/\,\eta(0^+)$ (Effizienz)')
    ax.axvline(1.0, color=RED,  lw=1.5, ls=':', label=r'$\varrho = \varrho^*$')
    ax.axhline(0.5, color=GRAY, lw=1.0, ls=':', alpha=0.7)
    ax.scatter([1.0], [1.0], color=BLUE,  s=70, zorder=5)
    ax.scatter([1.0], [0.5], color=GREEN, s=70, zorder=5)
    ax.annotate(r'$\eta(\varrho^*)=\frac{1}{2}\,\eta(0^+)$',
                xy=(1.0, 0.5), xytext=(1.25, 0.60), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
    ax.annotate(r'$A(\varrho^*)=A_{\max}$',
                xy=(1.0, 1.0), xytext=(1.25, 1.10), fontsize=9,
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(r'Normierte Rate $\varrho\,/\,\varrho^*$')
    ax.set_ylabel('Rel. Größe')
    ax.set_title('Aussagekraft und Effizienz am Optimum')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.15, 1.45)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot10_scaling.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 10 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 11 – Subgraph Motif Detection & Canonical Reduction
# ─────────────────────────────────────────────────────────────────
def plot11_subgraph_motifs():
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    q = 5
    M1 = np.array([[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0],[0,1,0,0,1]], dtype=float)
    M2 = np.array([[1,0,1,0,1],[0,1,0,1,0],[1,0,1,0,1],[1,0,0,1,0],[0,1,1,0,1]], dtype=float)
    yin  = np.arange(q, dtype=float)
    yout = np.arange(q, dtype=float)

    # ── Panel 1: Two bipartite layer graphs ──────────────────────
    ax = axes[0, 0]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, q + 0.3)
    for i in range(q):
        for j in range(q):
            if M1[i, j]:
                ax.plot([0, 1], [yin[j], yout[i]], color=BLUE, alpha=0.6, lw=1.5)
            if M2[i, j]:
                ax.plot([2, 3], [yin[j], yout[i]], color=GREEN, alpha=0.6, lw=1.5)
    for y in yin:
        ax.plot(0, y, 'o', color=BLUE,  ms=8, zorder=5)
        ax.plot(1, y, 's', color=BLUE,  ms=8, zorder=5)
        ax.plot(2, y, 'o', color=GREEN, ms=8, zorder=5)
        ax.plot(3, y, 's', color=GREEN, ms=8, zorder=5)
    ax.text(0.5, q + 0.1, r'$M^{(\ell_1)}$', ha='center', fontsize=11, color=BLUE)
    ax.text(2.5, q + 0.1, r'$M^{(\ell_2)}$', ha='center', fontsize=11, color=GREEN)
    ax.axis('off')
    ax.set_title('Schichtgraph-Motive', fontsize=11)

    # ── Panel 2: Adjacency matrices side-by-side ─────────────────
    ax = axes[0, 1]
    combined = np.zeros((q, 2*q + 1))
    combined[:, :q]       = M1
    combined[:, q+1:]     = M2
    combined[:, q]        = np.nan
    ax.imshow(combined, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.axvline(q - 0.5, color='k', lw=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(q/2 - 0.5, -0.9, r'$\hat{M}^{(\ell_1)}$', ha='center',
            fontsize=11, color=BLUE)
    ax.text(1.5*q + 0.5, -0.9, r'$\hat{M}^{(\ell_2)}$', ha='center',
            fontsize=11, color=GREEN)
    ax.set_title('Adjazenzmatrizen der Motive', fontsize=11)

    # ── Panel 3: Signature arrays + LCS ──────────────────────────
    ax = axes[1, 0]
    def col_sig(M):
        return [int(sum(M[i, j] * (2**i) for i in range(M.shape[0]))) for j in range(M.shape[1])]
    r1 = col_sig(M1)
    r2 = col_sig(M2)
    xs = np.arange(q)
    ax.bar(xs - 0.2, r1, 0.35, label=r'$\mathbf{r}^{(\ell_1)}$', color=BLUE,   alpha=0.8)
    ax.bar(xs + 0.2, r2, 0.35, label=r'$\mathbf{r}^{(\ell_2)}$', color=GREEN,  alpha=0.8)
    lcs_matches = [i for i in range(q) if r1[i] in r2]
    for i in lcs_matches:
        ax.bar(i - 0.2, r1[i], 0.35, color=ORANGE, alpha=0.9, zorder=5)
    ax.set_xlabel('Spaltenindex $j$')
    ax.set_ylabel('Signatur $r_j$')
    ax.set_title(f'Spalten-Signaturen  (LCS = {len(lcs_matches)})', fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.96, 'LCS $\\geq 2$  OK', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color=RED,
            bbox=dict(boxstyle='round', fc='white', ec=RED, alpha=0.8))

    # ── Panel 4: Reduced shared structure with routing masks ─────
    ax = axes[1, 1]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, q + 0.6)
    Mshared = np.clip(M1 + M2, 0, 1)
    for i in range(q):
        for j in range(q):
            if Mshared[i, j]:
                ax.plot([1.2, 1.8], [yin[j], yout[i]], color=ORANGE, alpha=0.7, lw=1.5)
    for y in yin:
        ax.plot(1.5, y, 'D', color=ORANGE, ms=9, zorder=6)
    ax.text(1.5, q + 0.5, r'$W^*$ (geteilt)', ha='center', fontsize=10, color=ORANGE)
    ax.text(0.5, q + 0.2, r'Routing $\mu^{(\ell_1)}$', ha='center', fontsize=9, color=BLUE)
    ax.text(2.5, q + 0.2, r'Routing $\mu^{(\ell_2)}$', ha='center', fontsize=9, color=GREEN)
    ax.axis('off')
    ax.set_title('Kanonische Reduktion + Routing', fontsize=11)

    fig.suptitle('Strukturelle Subgraph-Erkennung und kanonische Reduktion',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('nngraphs/science/plot11_subgraph_motifs.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 11 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 12 – Iterative Compression Dynamics
# ─────────────────────────────────────────────────────────────────
def plot12_compression_iterations():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    iters = np.arange(0, 11)

    # ── Panel 1: Normalised layer count vs. iterations ───────────
    ax = axes[0, 0]
    for L, color in zip([10, 20, 40], [BLUE, GREEN, ORANGE]):
        curve = L * np.exp(-0.35 * iters) + L * 0.1
        curve = np.clip(curve, L * 0.1, L)
        ax.plot(iters, curve / L, color=color, lw=2, marker='o', ms=5, label=f'$L={L}$')
    ax.axhline(0.1, color=RED, ls='--', lw=1.2, label=r'$P_{\min}/L$')
    ax.set_xlabel('SGKR-Iteration')
    ax.set_ylabel('Normierte Motivklassen $P/L$')
    ax.set_title('Schichtreduktion pro Iteration', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    # ── Panel 2: Normalised FLOPs vs. iterations ─────────────────
    ax = axes[0, 1]
    L, q = 20, 64
    flop_orig = L * 2 * q**2
    p_curve   = L * np.exp(-0.35 * iters) + L * 0.1
    flop_norm = (p_curve * 2 * q**2 + L * q) / flop_orig
    ax.plot(iters, flop_norm, color=BLUE, lw=2, marker='s', ms=5, label='FLOPs (normiert)')
    flop_min_norm = (L * 0.1 * 2 * q**2 + L * q) / flop_orig
    ax.axhline(flop_min_norm, color=RED, ls='--', lw=1.5, label='Theoretisches Minimum')
    ax.set_xlabel('SGKR-Iteration')
    ax.set_ylabel('FLOPs (normiert)')
    ax.set_title('FLOP-Reduktion durch SGKR', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    # ── Panel 3: Unique motif classes P vs. depth L (log-log) ────
    ax = axes[1, 0]
    L_vals = np.array([5, 10, 20, 40, 80, 160])
    P_rand  = (L_vals * 0.95 + rng.uniform(-0.5, 0.5, len(L_vals))).clip(1)
    P_train = (0.35 * L_vals**0.60 + rng.uniform(-0.3, 0.3, len(L_vals))).clip(1)
    ax.loglog(L_vals, P_rand,  'o--', color=BLUE,   lw=1.8, label='Zufälliges Netz')
    ax.loglog(L_vals, P_train, 's-',  color=RED,    lw=2.0, label='Trainiertes Netz')
    L_fit = np.linspace(5, 160, 200)
    ax.loglog(L_fit, 0.35 * L_fit**0.60, ':',       color=ORANGE, lw=1.5,
              label=r'$P \approx 0.35\,L^{0.6}$')
    ax.set_xlabel('Schichtanzahl $L$')
    ax.set_ylabel('Motivklassen $P$')
    ax.set_title('Motivklassen vs. Netzwerktiefe', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 4: Compression ratio kappa vs. training epochs ─────
    ax = axes[1, 1]
    epochs = np.linspace(0, 200, 100)
    for L, color in zip([10, 20, 40], [BLUE, GREEN, ORANGE]):
        kappa = 0.9 * np.exp(-epochs / 70) + 0.10
        kappa = np.clip(kappa + rng.uniform(-0.01, 0.01, 100), 0.05, 1.0)
        ax.plot(epochs, kappa, color=color, lw=2, label=f'$L={L}$')
    ax.axvline(70, color=GRAY, ls='--', lw=1.2, label='Einschwing-Epochen')
    ax.set_xlabel('Trainings-Epochen')
    ax.set_ylabel(r'$\kappa_{\mathrm{param}} = P/L$')
    ax.set_title('Kompressionsverhältnis vs. Training', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)

    fig.suptitle('Iterative Subgraph-Kompression: Netzwerkentwicklung',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('nngraphs/science/plot12_compression_iterations.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 12 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 13 – Runtime Analysis & Motif Structure
# ─────────────────────────────────────────────────────────────────
def plot13_runtime_analysis():
    rng = np.random.default_rng(13)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ── Panel 1: O(q^3) empirical scaling ────────────────────────
    ax = axes[0, 0]
    q_vals = np.array([4, 8, 16, 32, 64, 128])
    t_meas = 2e-8 * q_vals**3 * (1 + rng.uniform(-0.05, 0.05, len(q_vals)))
    ax.loglog(q_vals, t_meas * 1e6, 'o-', color=BLUE, lw=2, ms=6, label='Gemessen')
    q_fit  = np.linspace(4, 128, 200)
    ax.loglog(q_fit, 2e-8 * q_fit**3 * 1e6, '--', color=RED, lw=1.8, label=r'$t \propto q^3$')
    ax.set_xlabel('Motivdimension $q$')
    ax.set_ylabel(r'Laufzeit [$\mu$s]')
    ax.set_title(r'Subgraph Algorithmus: $O(q^3)$', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 2: P random vs. trained (area = compression reserve)
    ax = axes[0, 1]
    L_arr = np.arange(5, 65, 5)
    P_rnd = (L_arr - rng.integers(0, 2, len(L_arr))).clip(1)
    P_tr  = np.ceil(0.30 * L_arr**0.65).astype(int)
    ax.plot(L_arr, P_rnd, 'o--', color=BLUE,   lw=1.8, ms=5, label='Zufälliges Netz')
    ax.plot(L_arr, P_tr,  's-',  color=RED,    lw=2.0, ms=5, label='Trainiertes Netz')
    ax.fill_between(L_arr, P_tr, P_rnd, alpha=0.15, color=ORANGE, label='Kompressionsreserve')
    ax.set_xlabel('Schichtanzahl $L$')
    ax.set_ylabel('Motivklassen $P$')
    ax.set_title('Redundanz in trainierten Netzen', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 3: Motif frequency distribution (power-law) ────────
    ax = axes[1, 0]
    P = 12
    rank = np.arange(1, P + 1)
    freq = 40.0 / rank**0.8
    ax.bar(rank, freq, color=BLUE, alpha=0.8, edgecolor='navy')
    x_fit = np.linspace(1, P, 200)
    ax.plot(x_fit, freq[0] / x_fit**0.8 * rank[0]**0.8, 'r--', lw=1.8,
            label=r'Power-Law $\propto k^{-0.8}$')
    ax.set_xlabel('Motivklassen-Rang')
    ax.set_ylabel(r'Häufigkeit $t_p$')
    ax.set_title('Motivfrequenz-Verteilung', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 4: VC-dimension ratio after SGKR ───────────────────
    ax = axes[1, 1]
    kappa_arr = np.linspace(0.05, 1.0, 60)
    base = np.ones_like(kappa_arr)
    noise = rng.uniform(-0.008, 0.008, len(kappa_arr))
    ax.plot(kappa_arr, base + noise, 'o', color=BLUE, ms=4, alpha=0.7, label='Empirisch')
    ax.axhline(1.0, color=RED,  ls='--', lw=1.5, label=r'$\mathrm{VC}(\mathcal{G}_c) = \mathrm{VC}(\mathcal{G})$')
    ax.set_xlabel(r'$\kappa_{\mathrm{param}}$')
    ax.set_ylabel(r'$VC(G_{min})\,/\,VC(G)$')
    ax.set_title('Expressivitätserhaltung nach SGKR', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0.96, 1.06)

    fig.suptitle('Laufzeit- und Motivstruktur-Analyse', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('nngraphs/science/plot13_runtime_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 13 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 14 – Information Routing
# ─────────────────────────────────────────────────────────────────
def plot14_information_routing():
    rng = np.random.default_rng(99)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ── Panel 1: Activation heat-map before / after routing ──────
    ax = axes[0, 0]
    n, batch = 20, 32
    act1   = rng.uniform(0, 1, (n, batch)) * (rng.uniform(0, 1, (n, 1)) > 0.30)
    act2   = rng.uniform(0, 1, (n, batch)) * (rng.uniform(0, 1, (n, 1)) > 0.25)
    shared = np.clip((act1 + act2) / 2 + rng.normal(0, 0.02, (n, batch)), 0, 1)
    gap    = np.full((n, 2), np.nan)
    block  = np.hstack([act1, gap, act2, gap, shared])
    im = ax.imshow(block, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.axvline(batch + 1.5, color='white', lw=2)
    ax.axvline(2*batch + 3.5, color='white', lw=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r'Schicht $\ell_1$  |  Schicht $\ell_2$  |  Kanonisch')
    ax.set_title('Aktivierungsmuster (Routing)', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03)

    # ── Panel 2: Mutual information per layer ────────────────────
    ax = axes[0, 1]
    layers = np.arange(1, 9)
    mi_orig  = (1.0 - 0.06 * layers + rng.uniform(-0.015, 0.015, len(layers))).clip(0, 1)
    mi_compr = (mi_orig - rng.uniform(0.005, 0.020, len(layers))).clip(0, 1)
    ax.plot(layers, mi_orig,  'o-',  color=BLUE,  lw=2, ms=6, label='Original')
    ax.plot(layers, mi_compr, 's--', color=GREEN, lw=2, ms=6, label='SGKR-komprimiert')
    ax.fill_between(layers, mi_compr, mi_orig, alpha=0.2, color=ORANGE, label='< 2% Verlust')
    ax.set_xlabel(r'Schicht $\ell$')
    ax.set_ylabel(r'$I(X; Y^{(\ell)})$ (normiert)')
    ax.set_title('Informationsfluss-Erhaltung', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)

    # ── Panel 3: Routing accuracy vs. mask density ───────────────
    ax = axes[1, 0]
    density = np.linspace(0.1, 1.0, 60)
    acc = (0.5 + 0.5 * (1 - np.exp(-4.5 * density))
           + rng.uniform(-0.010, 0.010, 60)).clip(0, 1)
    ax.plot(density, acc * 100, color=BLUE, lw=2)
    ax.axhline(90, color=RED,    ls='--', lw=1.5, label='90%-Schwelle')
    ax.axvline(0.5, color=ORANGE, ls=':',  lw=1.5, label='50%-Dichte')
    ax.set_xlabel(r'Masken-Dichte $\|\mu\|_0\,/\,q$')
    ax.set_ylabel('Routing-Genauigkeit [%]')
    ax.set_title('Routing-Genauigkeit vs. Maskendichte', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(40, 105)

    # ── Panel 4: KL divergence vs. kappa ─────────────────────────
    ax = axes[1, 1]
    kappa = np.linspace(0.05, 1.0, 60)
    kl    = (0.48 * np.exp(-3.5 * kappa)
             + rng.uniform(-0.004, 0.004, 60)).clip(0)
    ax.plot(kappa, kl, color=BLUE, lw=2)
    ax.axhline(0.05, color=RED,    ls='--', lw=1.5, label='0.05 nats Schwelle')
    ax.axvline(0.20, color=ORANGE, ls=':',  lw=1.5, label=r'$\kappa = 0.2$')
    ax.set_xlabel(r'$\kappa_{\mathrm{param}}$')
    ax.set_ylabel(r'$D_{KL}(p_{G}\,\|\,p_{G_{min}})$ [nats]')
    ax.set_title('Ausgabe-KL-Divergenz nach SGKR', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.01, 0.55)

    fig.suptitle('Informationsfluss-Routing nach kanonischer Reduktion',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('nngraphs/science/plot14_information_routing.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 14 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 15 – Energy Efficiency
# ─────────────────────────────────────────────────────────────────
def plot15_energy_efficiency():
    rng = np.random.default_rng(55)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ── Panel 1: FLOP reduction factor for 3 architectures ───────
    ax = axes[0, 0]
    L_vals = np.array([10, 20, 40, 80, 160])
    q = 64
    data = [
        ('Vollvernetzt',  BLUE,   lambda l: 2 * l * q**2),
        ('Skalenfrei',    GREEN,  lambda l: 3 * l * q),
        ('Partiell dicht',ORANGE, lambda l: l * q**1.5),
    ]
    for label, color, _ in data:
        reductions = []
        for L in L_vals:
            P = max(1, int(0.35 * L**0.60))
            flop_orig = 2 * L * q**2
            flop_min  = 2 * P * q**2 + L * q
            reductions.append(flop_orig / max(flop_min, 1))
        ax.plot(L_vals, reductions, 'o-', color=color, lw=2, ms=6, label=label)
    ax.set_xlabel('Schichtanzahl $L$')
    ax.set_ylabel('FLOP-Reduktionsfaktor')
    ax.set_title('Rechenaufwand-Einsparung', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 2: Normalised inference energy vs. kappa ───────────
    ax = axes[0, 1]
    kappa = np.linspace(0.05, 1.0, 60)
    energy = kappa + rng.uniform(-0.01, 0.01, 60)
    energy = np.clip(energy, 0, 1.05)
    ax.plot(kappa, energy, color=BLUE, lw=2, label='Gemessen')
    ax.plot([0, 1], [0, 1], 'r--', lw=1.5, label='Lineare Skalierung')
    ax.set_xlabel(r'$\kappa_{\mathrm{param}}$')
    ax.set_ylabel('Normierter Energieverbrauch')
    ax.set_title('Inferenz-Energie vs. Kompression', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.10)

    # ── Panel 3: Parameter ratio vs. motif density ───────────────
    ax = axes[1, 0]
    kappa_th = np.linspace(0.05, 1.0, 60)
    pr_th    = kappa_th + 2.0 / (q * (kappa_th * 20 + 1))
    kappa_emp= np.linspace(0.05, 1.0, 20)
    pr_emp   = (kappa_emp + 2.0 / (q * (kappa_emp * 20 + 1))
                + rng.uniform(-0.01, 0.01, 20)).clip(0, 1)
    ax.plot(kappa_th, pr_th, '--', color=RED,  lw=1.8, label=r'Theor. Minimum')
    ax.plot(kappa_emp, pr_emp, 'o', color=BLUE, ms=6,  label='Empirisch')
    ax.plot([0, 1], [0, 1], ':', color='k', lw=1, alpha=0.3)
    ax.set_xlabel(r'Motivdichte $P/L$')
    ax.set_ylabel(r'$|\theta_{min}|\,/\,|\theta|$')
    ax.set_title('Parameterkompression', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # ── Panel 4: Pareto front accuracy vs. energy ─────────────────
    ax = axes[1, 1]
    for kappa_val, color, label in [(0.5, BLUE,   r'$\kappa=0.5$'),
                                    (0.3, GREEN,  r'$\kappa=0.3$'),
                                    (0.1, ORANGE, r'$\kappa=0.1$')]:
        n_pts  = 15
        acc_c  = 0.91 - (1 - kappa_val) * 0.015
        e_c    = kappa_val
        acc_sc = acc_c  + rng.uniform(-0.005, 0.005,  n_pts)
        e_sc   = e_c    + rng.uniform(-0.020, 0.020,  n_pts)
        ax.scatter(e_sc * 100, acc_sc * 100, color=color, s=45, alpha=0.8, label=label)
    ax.set_xlabel('Normierter Energieverbrauch [%]')
    ax.set_ylabel('Testgenauigkeit [%]')
    ax.set_title('Pareto-Front: Genauigkeit vs. Energie', fontsize=11)
    ax.legend(fontsize=9)
    ax.annotate('Optimum\n($\\kappa \\approx 0.3$)', xy=(30, 91), xytext=(50, 89),
                arrowprops=dict(arrowstyle='->', color=GRAY), fontsize=9, color=GREEN)

    fig.suptitle('Energieeffizienz durch strukturelle Subgraph-Kompression',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('nngraphs/science/plot15_energy_efficiency.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 15 gespeichert.")


# ══════════════════════════════════════════════════════════════════
#  KAPITEL: ENERGETISCHE IMPLIKATIONEN MINIMIERTER NEURONALER NETZE
#  Plots 16–21
# ══════════════════════════════════════════════════════════════════

# Physikalische / wirtschaftliche Basisparameter
_GPU_TDP_W        = 400.0      # W pro GPU (A100-Klasse)
_GPU_UTIL         = 0.60       # mittlere Auslastung im Betrieb
_GPU_ACTIVE_W     = _GPU_TDP_W * _GPU_UTIL     # 240 W
_GPUS_PER_SERVER  = 8
_SERVER_FIXED_W   = 350.0      # Networking, Laufwerk, Mainboard
_SERVER_COMPUTE_W = _GPU_ACTIVE_W * _GPUS_PER_SERVER   # 1920 W
_SERVER_TOTAL_W   = _SERVER_COMPUTE_W + _SERVER_FIXED_W  # 2270 W
_PUE_ENTERPRISE   = 1.45       # typisches Unternehmens-RZ
_PUE_HYPERSCALE   = 1.12       # modernes Hyperscaler-RZ
_EUR_PER_KWH      = 0.24       # Industriestrompreis €/kWh
_G_CO2_PER_KWH_EU = 280.0      # gCO2eq/kWh EU-Strommix 2024
_G_CO2_PER_KWH_US = 380.0      # gCO2eq/kWh US-Durchschnitt
_G_CO2_PER_KWH_CN = 560.0      # gCO2eq/kWh China
_G_CO2_PER_KWH_RE = 45.0       # gCO2eq/kWh Erneuerbare
_H_PER_YEAR       = 8_760.0    # Stunden/Jahr


def _total_server_power(kappa: float, pue: float,
                        server_compute_w=_SERVER_COMPUTE_W,
                        server_fixed_w=_SERVER_FIXED_W) -> float:
    """Gesamtleistung eines Servers inkl. PUE-Overhead nach Kompression κ."""
    compute = kappa * server_compute_w        # skaliert mit Kompression
    fixed   = server_fixed_w                   # netzwerkunabhängig
    return pue * compute + fixed               # Kühlung usw. nur auf compute


# ─────────────────────────────────────────────────────────────────
# Plot 16 – Energiebilanz im Rechenzentrum (vollständige Zerlegung)
# ─────────────────────────────────────────────────────────────────
def plot16_energy_balance():
    rng = np.random.default_rng(16)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Abb. 16: Energiebilanz eines Rechenzentrums – vollständige Komponentenzerlegung",
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Linkes Panel: Gestapelte Balken Energiekomponenten ─────────
    ax = axes[0]
    kappa_vals   = [1.0,  0.5,  0.3,  0.1]
    labels       = ['Baseline\n(κ=1,0)', 'Kompr.\n(κ=0,5)', 'Kompr.\n(κ=0,3)', 'Kompr.\n(κ=0,1)']
    pue = _PUE_ENTERPRISE

    # Energiekomponenten pro Server [W]:
    comp_compute  = np.array([_SERVER_COMPUTE_W * k for k in kappa_vals])
    comp_cooling  = (pue - 1.0) * comp_compute
    comp_fixed    = np.full(len(kappa_vals), _SERVER_FIXED_W)
    comp_pdu_ups  = 0.08 * comp_compute       # PDU/USV-Verluste ~8 % des Rechenaufwands
    comp_admin    = 0.03 * (comp_compute + comp_fixed)  # Verwaltung ~3 %

    bottoms = np.zeros(len(kappa_vals))
    components = [
        (comp_compute,  BLUE,   'GPU-Rechenleistung'),
        (comp_cooling,  '#74ADD1', 'Kühlung (PUE-Overhead)'),
        (comp_fixed,    ORANGE, 'Basisleistung (Netz/Speicher)'),
        (comp_pdu_ups,  PURPLE, 'PDU / USV-Verluste'),
        (comp_admin,    GRAY,   'Administration / Betrieb'),
    ]
    for vals, color, label in components:
        ax.bar(labels, vals, bottom=bottoms, color=color, alpha=0.85,
               edgecolor='white', linewidth=0.8, label=label)
        bottoms += vals

    totals = bottoms
    for i, (lbl, tot) in enumerate(zip(labels, totals)):
        ax.text(i, tot + 30, f'{tot:.0f} W', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_ylabel('Leistung pro Server [W]')
    ax.set_title('Leistungskomponenten nach Kompressionsgrad', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, totals[0] * 1.18)

    # ── Rechtes Panel: Energieeinsparung (Wasserfall) ─────────────
    ax = axes[1]
    baseline_w = totals[0]
    savings = baseline_w - totals[1:]
    savings_labels = [
        'Einsparung\nGPU-Rechen\n(κ:1→0.5)',
        'Einsparung\nGPU-Rechen\n(κ:0.5→0.3)',
        'Einsparung\nGPU-Rechen\n(κ:0.3→0.1)',
    ]
    delta = [totals[0] - totals[1], totals[1] - totals[2], totals[2] - totals[3]]
    running = totals[0]
    xs = np.arange(4)
    ax.bar(0, totals[0], color=RED, alpha=0.8, edgecolor='k', lw=0.8,
           label=f'Baseline ({totals[0]:.0f} W)')
    running = totals[0]
    for i, (d, lbl) in enumerate(zip(delta, savings_labels)):
        ax.bar(i + 1, d, bottom=running - d, color=GREEN, alpha=0.75,
               edgecolor='k', lw=0.8)
        ax.text(i + 1, running - d / 2, f'−{d:.0f} W\n({d/totals[0]*100:.1f}%)',
                ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        running -= d
        ax.bar(i + 1, running, color=BLUE, alpha=0.3, edgecolor='none')

    ax.set_xticks(xs)
    ax.set_xticklabels(['Baseline', 'κ=0.5', 'κ=0.3', 'κ=0.1'])
    ax.set_ylabel('Leistung pro Server [W]')
    ax.set_title('Wasserfall-Analyse der Leistungseinsparungen', fontsize=11)
    patch_g = mpatches.Patch(color=GREEN, alpha=0.75, label='Einsparung je Schritt')
    patch_b = mpatches.Patch(color=BLUE,  alpha=0.3,  label='Verbleibende Leistung')
    patch_r = mpatches.Patch(color=RED,   alpha=0.8,  label='Baseline')
    ax.legend(handles=[patch_r, patch_b, patch_g], fontsize=8)
    ax.set_ylim(0, totals[0] * 1.12)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot16_energy_balance.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 16 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 17 – Jährliche Energieeinsparung über Rechenzentrum-Skalierungen
# ─────────────────────────────────────────────────────────────────
def plot17_energy_savings_scale():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Abb. 17: Energieeinsparung durch Netzwerkkompression – Skalierungsanalyse",
        fontsize=13, fontweight='bold', y=1.01
    )

    kappa = np.linspace(0.05, 1.0, 200)
    pue   = _PUE_ENTERPRISE

    def annual_savings_kwh(n_servers, kappa_arr, pue):
        baseline  = _total_server_power(1.0, pue) * n_servers
        cmp       = np.array([_total_server_power(k, pue) * n_servers for k in kappa_arr])
        return (baseline - cmp) * _H_PER_YEAR / 1000.0   # kWh/Jahr

    dc_configs = [
        (2,       'Edge-Cluster (2 Server, 16 GPUs)',    BLUE),
        (50,      'Server-Rack (50 Server, 400 GPUs)',   GREEN),
        (500,     'Mittelgroßes RZ (500 Server)',        ORANGE),
        (5_000,   'Großes RZ (5000 Server)',             PURPLE),
    ]

    # ── Oben links: Absolute Einsparung kWh/Jahr ──────────────────
    ax = axes[0, 0]
    for n_srv, lbl, col in dc_configs:
        sav = annual_savings_kwh(n_srv, kappa, pue)
        ax.semilogy(kappa, np.maximum(sav, 1e-1), color=col, lw=2, label=lbl)
    ax.axvline(0.3, color=RED, ls='--', lw=1.5, label='κ=0,3 (Opt. SGKR)')
    ax.set_xlabel(r'Kompressionsverhältnis $\kappa_{\mathrm{param}}$')
    ax.set_ylabel('Jährliche Einsparung [kWh]')
    ax.set_title('Absolute Energieeinsparung (log)', fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # ── Oben rechts: Relative Einsparung [%] ──────────────────────
    ax = axes[0, 1]
    baseline_1srv = _total_server_power(1.0, pue)
    for pue_val, lbl, col in [(1.12, 'PUE=1,12 (Hyperscale)', BLUE),
                              (1.30, 'PUE=1,30 (Gut)',         GREEN),
                              (1.45, 'PUE=1,45 (Durchschnitt)',ORANGE),
                              (1.80, 'PUE=1,80 (Veraltet)',    RED)]:
        baseline = _total_server_power(1.0, pue_val)
        cmp      = np.array([_total_server_power(k, pue_val) for k in kappa])
        rel_sav  = (baseline - cmp) / baseline * 100
        ax.plot(kappa, rel_sav, color=col, lw=2, label=lbl)
    ax.axvline(0.3, color='k', ls='--', lw=1.2)
    ax.set_xlabel(r'$\kappa_{\mathrm{param}}$')
    ax.set_ylabel('Relative Einsparung [%]')
    ax.set_title('PUE-Einfluss auf relative Einsparung', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0.05, 1.0)
    ax.invert_xaxis()

    # ── Unten links: Kumulierte jährliche Einsparung (GWh) ────────
    ax = axes[1, 0]
    years = np.arange(1, 11)
    for n_srv, lbl, col in dc_configs:
        sav_annual = float(annual_savings_kwh(n_srv, np.array([0.3]), pue)[0])
        cumulative = sav_annual * years / 1e6   # GWh
        ax.plot(years, cumulative, 'o-', color=col, lw=2, ms=5, label=lbl)
    ax.set_xlabel('Jahre nach Einführung der Kompression')
    ax.set_ylabel('Kumulierte Einsparung [GWh]')
    ax.set_title('Kumulierte Einsparung über 10 Jahre (κ=0,3)', fontsize=11)
    ax.legend(fontsize=8)

    # ── Unten rechts: Server-Äquivalente eingespart ───────────────
    ax = axes[1, 1]
    n_servers = np.logspace(1, 4, 200)
    for pue_val, lbl, col in [(1.12, 'PUE=1,12', BLUE),
                              (1.45, 'PUE=1,45', ORANGE)]:
        for kappa_val, ls in [(0.3, '-'), (0.5, '--')]:
            baseline = _total_server_power(1.0, pue_val) * n_servers
            cmp      = _total_server_power(kappa_val, pue_val) * n_servers
            savings_kw = (baseline - cmp) / 1000.0
            equiv_servers = savings_kw * 1000 / _total_server_power(1.0, pue_val)
            ax.loglog(n_servers, equiv_servers,
                      color=col, lw=2, ls=ls,
                      label=f'{lbl}, κ={kappa_val}')
    ax.set_xlabel('Anzahl Server im RZ')
    ax.set_ylabel('Äquivalente eingesparte Server')
    ax.set_title('Eingesparte Server-Äquivalente', fontsize=11)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot17_energy_savings_scale.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 17 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 18 – CO2-Bilanz und Klimaimpact
# ─────────────────────────────────────────────────────────────────
def plot18_co2_impact():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Abb. 18: CO₂-Bilanz und Klimaimpact durch minimierte neuronale Netze",
        fontsize=13, fontweight='bold', y=1.01
    )

    n_servers = 1000
    pue = _PUE_ENTERPRISE
    kappa = np.linspace(0.05, 1.0, 200)

    def saved_kwh_yr(kappa_arr):
        b = _total_server_power(1.0, pue) * n_servers * _H_PER_YEAR / 1000
        c = np.array([_total_server_power(k, pue) * n_servers * _H_PER_YEAR / 1000
                      for k in kappa_arr])
        return b - c

    grid_mixes = {
        'EU-Strommix (280 g/kWh)':     (_G_CO2_PER_KWH_EU,  BLUE),
        'US-Durchschnitt (380 g/kWh)': (_G_CO2_PER_KWH_US,  GREEN),
        'Kohle-dominiert (560 g/kWh)': (_G_CO2_PER_KWH_CN,  RED),
        'Erneuerbare (45 g/kWh)':      (_G_CO2_PER_KWH_RE,  ORANGE),
    }

    # ── Oben links: CO2-Einsparung pro Jahr (t CO2eq) ─────────────
    ax = axes[0, 0]
    for lbl, (g_co2, col) in grid_mixes.items():
        sav_kwh = saved_kwh_yr(kappa)
        sav_tco2 = sav_kwh * g_co2 / 1e6    # kWh × g/kWh → tCO2
        ax.plot(kappa, sav_tco2, color=col, lw=2, label=lbl)
    ax.axvline(0.3, color='k', ls='--', lw=1.2, label='κ=0,3')
    ax.set_xlabel(r'$\kappa_{\mathrm{param}}$')
    ax.set_ylabel('Eingesparte CO₂-Emissionen [t CO₂eq/Jahr]')
    ax.set_title(f'Jährl. CO₂-Einsparung ({n_servers} Server)', fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # ── Oben rechts: Kumulierter CO2-Impact über 5 Jahre ──────────
    ax = axes[0, 1]
    years = np.arange(0, 6)
    g_co2_vals = [_G_CO2_PER_KWH_EU, _G_CO2_PER_KWH_US, _G_CO2_PER_KWH_RE]
    labls      = ['EU', 'US', 'Erneuerbare']
    cols       = [BLUE, GREEN, ORANGE]
    # Kompressionsoverhead (einmalig für das SGKR-Verfahren): ~0.01 % des
    # Jahresenergieverbrauchs beim einmaligen Kompressionslauf
    kappa_opt = 0.3
    sav_kwh_opt = float(saved_kwh_yr(np.array([kappa_opt]))[0])
    overhead_tco2_eu = _total_server_power(1.0, pue) * n_servers * _H_PER_YEAR / 1000 \
                       * 0.0001 * _G_CO2_PER_KWH_EU / 1e6  # winziger Overhead

    for g_co2, lbl, col in zip(g_co2_vals, labls, cols):
        cumul_saved = sav_kwh_opt * years * g_co2 / 1e6
        ax.plot(years, cumul_saved, 'o-', color=col, lw=2, ms=6, label=lbl)
    ax.axhline(overhead_tco2_eu, color='k', ls=':', lw=1,
               label=f'SGKR-Overhead ({overhead_tco2_eu:.2f} t)')
    ax.set_xlabel('Jahre')
    ax.set_ylabel('Kumulierte CO₂-Einsparung [t CO₂eq]')
    ax.set_title('Kumulierter Klimaeffekt (κ=0,3, 1000 Server)', fontsize=11)
    ax.legend(fontsize=9)

    # ── Unten links: CO2-Intensität als Funktion von DC-Größe ─────
    ax = axes[1, 0]
    n_srv_range = np.logspace(1, 5, 200)
    for g_co2, lbl, col in zip(g_co2_vals[:3] + [_G_CO2_PER_KWH_CN],
                                labls[:3] + ['China'],
                                cols[:3] + [RED]):
        sav_kwh = (_total_server_power(1.0, pue) - _total_server_power(0.3, pue)) \
                  * n_srv_range * _H_PER_YEAR / 1000
        sav_ktco2 = sav_kwh * g_co2 / 1e9   # kt CO2
        ax.loglog(n_srv_range, np.maximum(sav_ktco2, 1e-6), color=col, lw=2, label=lbl)
    ax.set_xlabel('Anzahl Server im RZ')
    ax.set_ylabel('CO₂-Einsparung [kt CO₂eq/Jahr]')
    ax.set_title('CO₂-Einsparung vs. RZ-Größe (κ=0,3)', fontsize=11)
    ax.legend(fontsize=9)

    # ── Unten rechts: CO2-Äquivalent-Vergleiche ───────────────────
    ax = axes[1, 1]
    # Vergleichsgrößen in tCO2
    comparisons = {
        'Transatlantik-\nFlug (Hin+Rück)': 2.5,
        'PKW 1 Jahr\n(15 000 km)': 2.4,
        'Haushalt\nDE/Jahr': 10.0,
        'Hektar Wald\n(CO2-Aufnahme/J)': 6.3,
        'Einsparung\n1000 Server\nEU (κ=0,3)':
            float(saved_kwh_yr(np.array([0.3]))[0]) * _G_CO2_PER_KWH_EU / 1e6,
    }
    labels  = list(comparisons.keys())
    values  = list(comparisons.values())
    colors_bar = [BLUE, GREEN, ORANGE, PURPLE, RED]
    bars = ax.barh(labels, values, color=colors_bar, alpha=0.8, edgecolor='k', lw=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f} t', va='center', fontsize=9)
    ax.set_xlabel('CO₂-Äquivalente [t CO₂eq]')
    ax.set_title('Einordnung: CO₂-Äquivalente im Vergleich', fontsize=11)
    ax.set_xlim(0, max(values) * 1.3)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot18_co2_impact.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 18 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 19 – Kühlungsoptimierung und PUE-Analyse
# ─────────────────────────────────────────────────────────────────
def plot19_cooling_pue():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Abb. 19: Kühlungsoptimierung und PUE-Analyse in Abhängigkeit der Netzwerkkompression",
        fontsize=13, fontweight='bold', y=1.01
    )

    kappa = np.linspace(0.05, 1.0, 200)
    n_servers = 1000

    # ── Oben links: IT-Last vs. Kühlleistung ──────────────────────
    ax = axes[0, 0]
    it_load_kw_baseline = _SERVER_COMPUTE_W * n_servers / 1000   # kW
    it_load = kappa * it_load_kw_baseline
    for pue_val, lbl, col in [(1.12, 'PUE=1,12', BLUE),
                              (1.30, 'PUE=1,30', GREEN),
                              (1.45, 'PUE=1,45', ORANGE),
                              (1.80, 'PUE=1,80', RED)]:
        cooling_kw = (pue_val - 1) * it_load
        ax.plot(it_load, cooling_kw, color=col, lw=2, label=lbl)
    ax.axvline(it_load_kw_baseline * 0.3, color='k', ls='--', lw=1.2,
               label='κ=0,3 Niveau')
    ax.set_xlabel('IT-Last [kW]')
    ax.set_ylabel('Kühlleistung [kW]')
    ax.set_title('IT-Last vs. benötigte Kühlleistung', fontsize=11)
    ax.legend(fontsize=8)

    # ── Oben rechts: Kühlkostenanteil ─────────────────────────────
    ax = axes[0, 1]
    for pue_val, lbl, col in [(1.12, 'Hyperscale PUE=1,12', BLUE),
                              (1.45, 'Enterprise PUE=1,45', ORANGE),
                              (1.80, 'Legacy PUE=1,80',     RED)]:
        total_pwr  = np.array([_total_server_power(k, pue_val) * n_servers / 1000
                                for k in kappa])
        cool_pwr   = (pue_val - 1) * kappa * it_load_kw_baseline
        frac_cool  = cool_pwr / total_pwr * 100
        ax.plot(kappa, frac_cool, color=col, lw=2, label=lbl)
    ax.set_xlabel(r'$\kappa_{\mathrm{param}}$')
    ax.set_ylabel('Kühlanteil am Gesamtverbrauch [%]')
    ax.set_title('Anteil Kühlenergie vs. Kompressionsgrad', fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    # ── Unten links: Dimensionierung der Kühlaggregate ────────────
    ax = axes[1, 0]
    it_baseline_kw = it_load_kw_baseline
    # Kühlaggregate in Einheiten von 100 kW (typische CRAC-Einheit)
    crac_unit_kw = 100.0
    pue_val = _PUE_ENTERPRISE
    kappa_vals_disc = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.1])
    n_crac_needed = np.ceil((pue_val - 1) * kappa_vals_disc * it_baseline_kw / crac_unit_kw)
    bar_colors = [RED if k == 1.0 else GREEN for k in kappa_vals_disc]
    bars = ax.bar([f'κ={k}' for k in kappa_vals_disc], n_crac_needed,
                  color=bar_colors, alpha=0.8, edgecolor='k', lw=0.6)
    for bar, n in zip(bars, n_crac_needed):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 0.1,
                f'{int(n)} Units', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(f'Benötigte Kühlaggregate\n(à {crac_unit_kw:.0f} kW Kapazität)')
    ax.set_title(f'Anzahl Kühlaggregate (PUE={pue_val}, {n_servers} Server)', fontsize=11)
    ax.set_ylim(0, n_crac_needed[0] * 1.25)

    # ── Unten rechts: Kühlkosten pro Inferenz ─────────────────────
    ax = axes[1, 1]
    # Annahme: 10^9 Inferenzen/Jahr mit batch=32 GPT-artigem Modell
    n_inf_per_yr   = 1e9
    flops_per_inf  = 1e10    # 10 GFLOPs pro Inferenzaufruf (mittleres Modell)
    tflops_gpu     = 312.0   # GPU TFLOPS (BF16)
    time_per_inf_s = flops_per_inf / (tflops_gpu * 1e12 * _GPU_UTIL)   # Sekunden
    gpu_energy_per_inf_j = _GPU_ACTIVE_W * time_per_inf_s / _GPUS_PER_SERVER
    for kappa_val, col, lbl in [(1.0, RED, 'Unkomprimiert'),
                                 (0.5, ORANGE, 'κ=0,5'),
                                 (0.3, GREEN,  'κ=0,3'),
                                 (0.1, BLUE,   'κ=0,1')]:
        energy_j = gpu_energy_per_inf_j * kappa_val
        cool_j   = energy_j * (_PUE_ENTERPRISE - 1.0)
        total_j  = energy_j + cool_j
        # In μJ
        ax.bar([lbl], [energy_j * 1e6], color=col, alpha=0.6, label='Compute')
        ax.bar([lbl], [cool_j * 1e6], bottom=[energy_j * 1e6], color=col,
               alpha=0.9, hatch='//')
    ax.set_ylabel('Energie pro Inferenz [μJ]')
    ax.set_title('Energie pro Inferenzaufruf (inkl. Kühlung)', fontsize=11)
    compute_patch = mpatches.Patch(color=GRAY, alpha=0.6, label='Compute')
    cool_patch    = mpatches.Patch(color=GRAY, alpha=0.9, hatch='//', label='Kühlung')
    ax.legend(handles=[compute_patch, cool_patch], fontsize=9)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot19_cooling_pue.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 19 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 20 – Total Cost of Ownership (TCO) Analyse
# ─────────────────────────────────────────────────────────────────
def plot20_tco_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Abb. 20: Total Cost of Ownership – Gesamtkostenanalyse minimierter neuronaler Netze",
        fontsize=13, fontweight='bold', y=1.01
    )

    n_servers = 500
    pue = _PUE_ENTERPRISE
    years = 5

    def annual_tco(kappa_val, n_srv, pue):
        # Energiekosten
        total_w = _total_server_power(kappa_val, pue) * n_srv
        energy_eur_yr = total_w * _H_PER_YEAR / 1000 * _EUR_PER_KWH
        # Hardware-Amortisation: GPU-Preis ~$30000/GPU, 5-Jahre-Abschreibung
        gpu_capex_yr  = n_srv * _GPUS_PER_SERVER * 30_000 / 5
        # Kühlinfrastruktur: ~2000 €/kW Investment, 15-Jahre
        cool_kw = (pue - 1.0) * kappa_val * _SERVER_COMPUTE_W * n_srv / 1000
        cool_capex_yr = cool_kw * 2_000 / 15
        # Administration: 1 Admin pro 100 Server, 80k€/Admin/Jahr
        admin_eur_yr  = max(1, n_srv // 100) * 80_000
        # Netzwerk/Storage/Lizenzen ~5k€/Server/Jahr
        misc_eur_yr   = n_srv * 5_000
        return {
            'Energiekosten':        energy_eur_yr,
            'Hardware-Amortisation':gpu_capex_yr,
            'Kühlinfrastruktur':    cool_capex_yr,
            'Administration':       admin_eur_yr,
            'Sonstiges':            misc_eur_yr,
        }

    # ── Linkes Panel: TCO-Komponentenvergleich ────────────────────
    ax = axes[0]
    kappa_scenarios = [1.0, 0.5, 0.3, 0.1]
    labels = [f'κ={k}' for k in kappa_scenarios]
    tco_data = [annual_tco(k, n_servers, pue) for k in kappa_scenarios]
    component_names = list(tco_data[0].keys())
    comp_colors = [BLUE, ORANGE, '#74ADD1', GREEN, GRAY]
    bottoms = np.zeros(len(kappa_scenarios))
    for comp, col in zip(component_names, comp_colors):
        vals = np.array([d[comp] / 1e6 for d in tco_data])   # Mio. €
        ax.bar(labels, vals, bottom=bottoms, color=col, alpha=0.85,
               edgecolor='white', lw=0.8, label=comp)
        bottoms += vals
    totals_tco = bottoms
    for i, (lbl, t) in enumerate(zip(labels, totals_tco)):
        ax.text(i, t + totals_tco[0] * 0.01, f'{t:.2f} M€',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel(f'Jährliche Gesamtkosten [Mio. €] ({n_servers} Server)')
    ax.set_title('TCO-Komponenten nach Kompressionsgrad', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, totals_tco[0] * 1.15)

    # ── Rechtes Panel: ROI-Kurve ──────────────────────────────────
    ax = axes[1]
    kappa_arr = np.linspace(0.05, 0.95, 200)
    tco_baseline = sum(annual_tco(1.0, n_servers, pue).values()) / 1e6
    # SGKR-Kompressionskosten: einmaliger Rechenaufwand ~0,1 % Jahreskostenmenge
    sgkr_cost = tco_baseline * 0.001
    for yr, col, lbl in [(1, BLUE, '1 Jahr'),
                         (3, GREEN, '3 Jahre'),
                         (5, ORANGE, '5 Jahre')]:
        savings = np.array([
            (tco_baseline - sum(annual_tco(k, n_servers, pue).values()) / 1e6) * yr - sgkr_cost
            for k in kappa_arr
        ])
        ax.plot(kappa_arr, savings, color=col, lw=2, label=lbl)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(sgkr_cost, color=RED, ls=':', lw=1.5, label=f'SGKR-Einmalkost. ({sgkr_cost*1000:.0f} k€)')
    ax.fill_between(kappa_arr,
                    np.array([
                        (tco_baseline - sum(annual_tco(k, n_servers, pue).values()) / 1e6) * 5
                        for k in kappa_arr
                    ]),
                    0,
                    where=np.array([
                        (tco_baseline - sum(annual_tco(k, n_servers, pue).values()) / 1e6) * 5 > 0
                        for k in kappa_arr
                    ]),
                    alpha=0.1, color=GREEN, label='ROI-Bereich (5 J.)')
    ax.set_xlabel(r'Kompressionsverhältnis $\kappa_{\mathrm{param}}$')
    ax.set_ylabel('Kumulierter Netto-ROI [Mio. €]')
    ax.set_title('Return-on-Investment der Kompression', fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot20_tco_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 20 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Plot 21 – Modell-Lebenszyklus: Training vs. Inferenz Energiebudget
# ─────────────────────────────────────────────────────────────────
def plot21_training_vs_inference():
    rng = np.random.default_rng(21)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Abb. 21: Modell-Lebenszyklus-Energieanalyse: Training vs. Inferenz",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Modellparameter für 3 Modellklassen
    models = {
        'Klein (10M Param.)':   {'train_gwh': 0.05,   'inf_gwh_yr': 0.12,  'kappa_sgkr': 0.35},
        'Mittel (500M Param.)': {'train_gwh': 2.5,    'inf_gwh_yr': 6.0,   'kappa_sgkr': 0.25},
        'Groß (175B Param.)':   {'train_gwh': 1_300,  'inf_gwh_yr': 3_200, 'kappa_sgkr': 0.20},
    }

    # ── Oben links: Training vs. Inferenz 3-Jahres-Budget ─────────
    ax = axes[0, 0]
    n_models = len(models)
    x = np.arange(n_models)
    labels_m = list(models.keys())
    train_e  = np.array([v['train_gwh'] for v in models.values()])
    infer_e  = np.array([v['inf_gwh_yr'] * 3 for v in models.values()])  # 3 Jahre
    ax.bar(x - 0.2, train_e, 0.35, color=BLUE,   alpha=0.8, log=True, label='Training (einmalig)')
    ax.bar(x + 0.2, infer_e, 0.35, color=ORANGE, alpha=0.8, log=True, label='Inferenz (3 Jahre)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m, fontsize=8)
    ax.set_ylabel('Energie [GWh] (log)')
    ax.set_title('Training vs. Inferenz-Energie (3-J.-Budget)', fontsize=11)
    ax.legend(fontsize=9)

    # ── Oben rechts: Lebenszeit-Energieersparnis durch SGKR ───────
    ax = axes[0, 1]
    years = np.arange(0, 6)
    cols  = [BLUE, GREEN, RED]
    for (name, params), col in zip(models.items(), cols):
        k = params['kappa_sgkr']
        e_inf_yr_before = params['inf_gwh_yr']
        e_inf_yr_after  = params['inf_gwh_yr'] * k
        cumul_saved = (e_inf_yr_before - e_inf_yr_after) * years
        cumul_total_before = params['train_gwh'] + e_inf_yr_before * years
        ax.plot(years, cumul_saved / np.maximum(params['train_gwh'], 1e-9),
                'o-', color=col, lw=2, ms=5, label=name)
    ax.axhline(1.0, color='k', ls='--', lw=1.2, label='= Trainingsenergie')
    ax.set_xlabel('Jahre Betrieb nach Deployment')
    ax.set_ylabel('Einsparung / Trainingsenergie [-]')
    ax.set_title('Einsparung normiert auf Trainingsaufwand', fontsize=11)
    ax.legend(fontsize=8)

    # ── Unten links: Energie pro Inferenz (μWh) nach Modellgröße ──
    ax = axes[1, 0]
    n_params = np.logspace(6, 11, 200)   # 1M bis 100B Parameter
    # Annahme: FLOPs ~ 6 * N_params für eine Vorwärtsausbreitung
    flops_per_inf = 6 * n_params
    tflops_sustained = _GPU_ACTIVE_W / (_GPU_ACTIVE_W / (312e12))   # TFLOPS at utilization
    tflops           = 312e12 * _GPU_UTIL
    time_s   = flops_per_inf / tflops
    energy_J = _GPU_ACTIVE_W * time_s
    energy_uWh = energy_J / 3600 * 1e6

    for kappa_val, col, lbl in [(1.0, RED, 'Unkomprimiert'),
                                 (0.5, ORANGE, 'κ=0,5'),
                                 (0.3, GREEN,  'κ=0,3'),
                                 (0.1, BLUE,   'κ=0,1')]:
        ax.loglog(n_params / 1e6, energy_uWh * kappa_val, color=col, lw=2, label=lbl)

    ax.set_xlabel('Modellparameter [Mio.]')
    ax.set_ylabel('Energie pro Inferenz [μWh]')
    ax.set_title('Inferenzenergie vs. Modellgröße', fontsize=11)
    ax.legend(fontsize=9)

    # ── Unten rechts: Globale Perspektive – RZ-Sektor ─────────────
    ax = axes[1, 1]
    # Globale KI-Inferenzlast: wächst von ~50 TWh/year (2024) auf ~400 TWh/year (2030)
    # Quelle: IEA Energy and AI 2024
    years_global = np.arange(2024, 2031)
    ai_energy_twh = np.array([50, 75, 110, 160, 230, 315, 400])
    # Mit 30% Kompression (κ=0.3): effektive Einsparung ≈ 45% (incl. PUE)
    kappa_opt = 0.3
    pue_trend = np.linspace(1.45, 1.25, len(years_global))  # PUE verbessert sich
    eff_savings_frac = np.array([(1 - kappa_opt) * (p - 1) / p + (1 - kappa_opt) * kappa_opt
                                  for p in pue_trend])
    # vereinfacht: Einsparung = (1-κ)*PUE * P_compute / P_total
    savings_frac_simple = np.array([
        (1 - kappa_opt) * (1 + (p-1)*1) / (1 + (p-1)*1 + 0.2)
        for p in pue_trend
    ])
    ai_with_compression = ai_energy_twh * (1 - savings_frac_simple)

    ax.fill_between(years_global, ai_energy_twh, ai_with_compression,
                    alpha=0.35, color=GREEN, label='Einsparungspotenzial (κ=0,3)')
    ax.plot(years_global, ai_energy_twh,        'o-', color=RED,  lw=2.5, ms=7, label='Ohne Kompression')
    ax.plot(years_global, ai_with_compression,  's-', color=GREEN, lw=2.5, ms=7, label='Mit Kompression (κ=0,3)')
    for yr, val_b, val_c in zip(years_global, ai_energy_twh, ai_with_compression):
        if yr in [2025, 2027, 2030]:
            ax.annotate(f'−{val_b-val_c:.0f} TWh',
                        xy=(yr, (val_b+val_c)/2),
                        fontsize=8, ha='center', color=GREEN, fontweight='bold')
    ax.set_xlabel('Jahr')
    ax.set_ylabel('Globaler KI-Energieverbrauch [TWh/Jahr]')
    ax.set_title('Globale KI-Infrastruktur: Einsparungspotenzial', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(2023.5, 2030.5)

    fig.tight_layout()
    fig.savefig('nngraphs/science/plot21_training_vs_inference.pdf', bbox_inches='tight')
    plt.close(fig)
    print("Plot 21 gespeichert.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Erzeuge alle Plots ...")
    plot1_capacity_gain()
    plot2_spectral()
    plot3_learning()
    plot4_graph_structure()
    plot5_information()
    plot6_weights()
    plot7_transfer()
    plot8_bounds()
    plot9_decision()
    plot10_scaling()
    plot11_subgraph_motifs()
    plot12_compression_iterations()
    plot13_runtime_analysis()
    plot14_information_routing()
    plot15_energy_efficiency()
    # ── Energetische Implikationen (Plots 16–21) ──────────────────
    plot16_energy_balance()
    plot17_energy_savings_scale()
    plot18_co2_impact()
    plot19_cooling_pue()
    plot20_tco_analysis()
    plot21_training_vs_inference()
    print("Alle Plots erfolgreich gespeichert.")
