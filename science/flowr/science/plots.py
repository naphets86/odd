import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.patches import FancyArrowPatch, Circle, Ellipse, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

FIGDIR = 'figures"
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ─────────────────────────────────────────────
# Fig 1: Duftstoff-Spektrum verschiedener Blüten (Balkendiagramm)
# ─────────────────────────────────────────────
def fig_duft_spektrum():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Flüchtige organische Verbindungen (VOCs) ausgewählter Blütenpflanzen",
                 fontsize=14, fontweight='bold', y=1.01)

    plants = {
        "Rose (Rosa centifolia)":      {"Geraniol": 38, "Citronellol": 25, "Nerol": 12,
                                         "Linalool": 10, "2-Phenylethanol": 8, "Eugenol": 7},
        "Lavendel (Lavandula ang.)":   {"Linalool": 42, "Linalylacetat": 28, "Camphor": 9,
                                         "β-Ocimen": 8, "1,8-Cineol": 7, "Terpinen-4-ol": 6},
        "Apfelblüte (Malus dom.)":     {"Benzylalkohol": 30, "α-Farnesen": 22, "Hexanal": 15,
                                         "2-Hexenal": 13, "Methanol": 12, "Ethylbutyrat": 8},
        "Raps (Brassica napus)":       {"p-Anisaldehyd": 35, "Benzaldehyd": 20, "Linalool": 18,
                                         "Methylbenzoat": 14, "Phenylacetaldehyd": 8, "Eugenol": 5},
        "Kirschblüte (Prunus av.)":    {"Benzaldehyd": 40, "Linalool": 22, "Limonen": 14,
                                         "Nonanal": 10, "Decanal": 8, "α-Pinen": 6},
        "Linde (Tilia cordata)":       {"Linalool": 45, "Lilial": 20, "Farnesol": 12,
                                         "α-Terpineol": 10, "Nerol": 8, "Geranylacetat": 5},
    }
    colors_list = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']

    for ax, (plant, compounds) in zip(axes.flat, plants.items()):
        names = list(compounds.keys())
        values = list(compounds.values())
        bars = ax.barh(names, values, color=colors_list[:len(names)], edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val}%', va='center', fontsize=9, fontweight='bold')
        ax.set_xlim(0, 55)
        ax.set_title(plant, fontsize=10, fontweight='bold')
        ax.set_xlabel("Relativer Anteil (%)")
        ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig01_duft_spektrum.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 1 done")

# ─────────────────────────────────────────────
# Fig 2: Duftstoff-Überlagerungsmodell (Heatmap + Kontur)
# ─────────────────────────────────────────────
def fig_duft_ueberlagerung():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Räumliche Überlagerung von Duftstofffeldern in einer Wiese",
                 fontsize=14, fontweight='bold')

    x = np.linspace(0, 20, 300)
    y = np.linspace(0, 15, 300)
    X, Y = np.meshgrid(x, y)

    # Quellen: (x, y, Stärke, Breite)
    quellen = [(4, 4, 1.0, 2.5), (10, 7, 0.8, 3.0), (16, 4, 0.9, 2.0),
               (7, 12, 0.6, 2.8), (13, 11, 0.7, 2.2)]
    namen = ["Rose", "Lavendel", "Apfelblüte", "Raps", "Kirsche"]
    farben = ['#E91E63','#9C27B0','#FF9800','#FFEB3B','#F44336']

    Z_total = np.zeros_like(X)
    Z_layers = []
    for (px, py, stk, breite) in quellen:
        Z = stk * np.exp(-((X-px)**2 + (Y-py)**2) / (2*breite**2))
        Z_layers.append(Z)
        Z_total += Z

    # Subplot 1: Einzelschichten
    ax = axes[0]
    for i, (Z, farbe, name) in enumerate(zip(Z_layers, farben, namen)):
        cs = ax.contourf(X, Y, Z, levels=[0.05, 0.2, 0.5, 0.8],
                         colors=[farbe+'33', farbe+'88', farbe+'BB', farbe+'FF'], alpha=0.5)
    for (px, py, _, _), farbe, name in zip(quellen, farben, namen):
        ax.plot(px, py, 'o', color=farbe, markersize=12, zorder=5)
        ax.annotate(name, (px, py), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_title("Einzelne Duftstofffelder", fontweight='bold')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_xlim(0,20); ax.set_ylim(0,15)

    # Subplot 2: Überlagerung
    ax = axes[1]
    im = ax.contourf(X, Y, Z_total, levels=20, cmap='YlOrRd')
    ax.contour(X, Y, Z_total, levels=8, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(im, ax=ax, label='Konzentration (rel.)')
    for (px, py, _, _), farbe in zip(quellen, farben):
        ax.plot(px, py, 'o', color=farbe, markersize=10, markeredgecolor='white', zorder=5)
    ax.set_title("Summiertes Duftstofffeld", fontweight='bold')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # Subplot 3: Windverwehung
    ax = axes[2]
    # Wind in +x-Richtung
    Z_wind = np.zeros_like(X)
    for (px, py, stk, breite) in quellen:
        shift = (X - px - 1.5)
        Z_w = stk * np.exp(-(shift**2 / (2*(breite*1.8)**2)) - (Y-py)**2/(2*breite**2))
        Z_w *= (X >= px).astype(float) * 0.7 + 0.3
        Z_wind += Z_w
    im2 = ax.contourf(X, Y, Z_wind, levels=20, cmap='PuBu')
    plt.colorbar(im2, ax=ax, label='Konzentration (rel.)')
    ax.quiver(np.linspace(1,19,8), np.ones(8)*1, np.ones(8)*2, np.zeros(8),
              color='navy', alpha=0.5, scale=30)
    ax.set_title("Windverwehte Duftwolken", fontweight='bold')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig02_duft_ueberlagerung.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 2 done")

# ─────────────────────────────────────────────
# Fig 3: Insekten-Orientierungsmodell (Vektoren)
# ─────────────────────────────────────────────
def fig_insekten_orientierung():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Duftstoff-gesteuertes Navigationsverhalten von Honigbienen",
                 fontsize=14, fontweight='bold')

    np.random.seed(42)

    # Subplot 1: Trajektorien mehrerer Bienen
    ax = axes[0]
    ax.set_xlim(0, 20); ax.set_ylim(0, 20)
    ax.set_facecolor('#f0f8e8')
    ax.set_title("Flugbahnen zur Blütenquelle", fontweight='bold')

    # Blütenquelle
    qx, qy = 14, 14
    circle = Circle((qx, qy), 1.5, color='#FFD700', alpha=0.4, zorder=2)
    ax.add_patch(circle)
    ax.plot(qx, qy, '*', color='#FF8C00', markersize=20, zorder=3)
    ax.annotate("Blüte", (qx, qy), xytext=(qx+1.5, qy+1), fontsize=10, fontweight='bold')

    colors_bees = ['#2196F3','#4CAF50','#E91E63','#FF9800','#9C27B0','#00BCD4']
    for i in range(6):
        sx = np.random.uniform(1, 5)
        sy = np.random.uniform(1, 8)
        # Zufällige Trajektorie mit Drift zur Quelle
        n_steps = 40
        traj_x = [sx]
        traj_y = [sy]
        for s in range(n_steps):
            dx = (qx - traj_x[-1]) / (n_steps - s) + np.random.normal(0, 0.6)
            dy = (qy - traj_y[-1]) / (n_steps - s) + np.random.normal(0, 0.6)
            traj_x.append(traj_x[-1] + dx * 0.8)
            traj_y.append(traj_y[-1] + dy * 0.8)
        ax.plot(traj_x, traj_y, '-', color=colors_bees[i], alpha=0.7, linewidth=1.5)
        ax.plot(traj_x[0], traj_y[0], 'o', color=colors_bees[i], markersize=8)
        ax.annotate(f"B{i+1}", (traj_x[0], traj_y[0]), fontsize=8)

    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # Subplot 2: Konzentrationsgradient-Folgen
    ax = axes[1]
    ax.set_title("Chemotaxis entlang des Duftstoffgradienten", fontweight='bold')
    x = np.linspace(0, 20, 200)
    y = np.linspace(0, 20, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X-14)**2 + (Y-14)**2) / 18)
    im = ax.contourf(X, Y, Z, levels=15, cmap='YlOrRd', alpha=0.8)
    plt.colorbar(im, ax=ax, label='Duftstoffkonzentration (rel.)')
    # Gradient-Vektoren
    step = 20
    gx, gy = np.gradient(Z)
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              gy[::step, ::step], gx[::step, ::step],
              color='navy', alpha=0.6, scale=15, width=0.003)
    ax.plot(14, 14, '*', color='#FF8C00', markersize=20, zorder=5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig03_insekten_orientierung.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 3 done")

# ─────────────────────────────────────────────
# Fig 4: Samenklassen – Übersichtsplot
# ─────────────────────────────────────────────
def fig_samenklassen():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor('#fafaf8')
    fig.patch.set_facecolor('#fafaf8')
    ax.set_title("Klassifikation der Samenkörner nach morphologischen und biochemischen Merkmalen",
                 fontsize=14, fontweight='bold', pad=15)

    klassen = [
        {"name": "Klasse I\nNährstoffreiche\nOelsamen", "color": "#FF8F00", "x": 1.5, "y": 7,
         "beispiele": "Raps, Sonnenblume\nSesam, Leinsamen",
         "merkmale": "Fettgehalt >30%\nProtein 15-25%\nVitamin E"},
        {"name": "Klasse II\nStärkehaltige\nGetreidesamen", "color": "#8BC34A", "x": 5, "y": 7,
         "beispiele": "Weizen, Mais\nGerste, Hafer",
         "merkmale": "Stärke 60-75%\nGluten-Protein\nB-Vitamine"},
        {"name": "Klasse III\nEiweissreiche\nHülsenfrüchte", "color": "#2196F3", "x": 8.5, "y": 7,
         "beispiele": "Erbse, Bohne\nLinse, Soja",
         "merkmale": "Protein 20-40%\nLecithine\nIsoflavone"},
        {"name": "Klasse IV\nZuckerhaltige\nObstsamen", "color": "#E91E63", "x": 12, "y": 7,
         "beispiele": "Apfel, Kirsche\nBirne, Pflaume",
         "merkmale": "Glycoside\nAmygdalin\nFlüchtige VOCs"},
        {"name": "Klasse V\nAromatische\nGewürzsamen", "color": "#9C27B0", "x": 1.5, "y": 3.5,
         "beispiele": "Fenchel, Kümmel\nKoriander, Anis",
         "merkmale": "Ätherische Öle\nFlavanoide\nTerpene"},
        {"name": "Klasse VI\nAlkaloide-\nhaltige Samen", "color": "#F44336", "x": 5, "y": 3.5,
         "beispiele": "Mohn, Lupine\nStechapfel",
         "merkmale": "Alkaloide\nGlycoalkaloide\nDefensin-Proteine"},
        {"name": "Klasse VII\nWasserspeichernde\nSukkulenten-Samen", "color": "#00BCD4", "x": 8.5, "y": 3.5,
         "beispiele": "Kaktus, Aloe\nEcheveria",
         "merkmale": "Mucilage\nPolysaccharide\nSilica-Strukturen"},
        {"name": "Klasse VIII\nWindverbreitete\nLeichtsamen", "color": "#607D8B", "x": 12, "y": 3.5,
         "beispiele": "Löwenzahn, Pappel\nBirke, Ahorn",
         "merkmale": "Pappus-Fasern\nFlügelstrukturen\nMiniatur-Lipide"},
    ]

    for kl in klassen:
        box = FancyBboxPatch((kl['x']-1.3, kl['y']-1.3), 2.5, 2.5,
                             boxstyle="round,pad=0.1", facecolor=kl['color']+'22',
                             edgecolor=kl['color'], linewidth=2.5)
        ax.add_patch(box)
        ax.text(kl['x'], kl['y']+0.6, kl['name'], ha='center', va='center',
                fontsize=9, fontweight='bold', color=kl['color'])
        ax.text(kl['x'], kl['y']-0.25, kl['beispiele'], ha='center', va='center',
                fontsize=8, color='#333333', style='italic')
        ax.text(kl['x'], kl['y']-1.0, kl['merkmale'], ha='center', va='center',
                fontsize=7.5, color='#555555')

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig04_samenklassen.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 4 done")

# ─────────────────────────────────────────────
# Fig 5: Biochemische Zusammensetzung der Samenklassen (Radar)
# ─────────────────────────────────────────────
def fig_samen_biochemie():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(polar=True))
    fig.suptitle("Biochemisches Profil der acht Samenklassen (Radardiagramme)",
                 fontsize=14, fontweight='bold')

    kategorien = ['Fette', 'Proteine', 'Stärke', 'Zucker', 'Terpene', 'Alkaloide', 'Wasser']
    N = len(kategorien)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    klassen_daten = [
        ("Kl. I Ölsamen",    '#FF8F00', [90, 60, 10, 20, 30, 5, 20]),
        ("Kl. II Getreide",  '#8BC34A', [10, 50, 95, 30, 10, 5, 15]),
        ("Kl. III Hülsenf.", '#2196F3', [25, 90, 40, 20, 15, 10, 30]),
        ("Kl. IV Obstsamen", '#E91E63', [30, 40, 30, 80, 60, 15, 50]),
        ("Kl. V Gewürze",    '#9C27B0', [40, 35, 20, 30, 95, 20, 25]),
        ("Kl. VI Alkaloide", '#F44336', [20, 50, 25, 20, 40, 90, 15]),
        ("Kl. VII Sukkul.",  '#00BCD4', [15, 25, 60, 70, 20, 5, 80]),
        ("Kl. VIII Wind",    '#607D8B', [35, 30, 15, 20, 25, 5, 10]),
    ]

    for ax, (name, color, werte) in zip(axes.flat, klassen_daten):
        werte_plot = werte + werte[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles, werte_plot, 'o-', color=color, linewidth=2)
        ax.fill(angles, werte_plot, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(kategorien, size=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], size=6)
        ax.set_title(name, fontsize=10, fontweight='bold', color=color, pad=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig05_samen_biochemie.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 5 done")

# ─────────────────────────────────────────────
# Fig 6: Samen → Wachstum → Blüte → Duft (Informationsfluss)
# ─────────────────────────────────────────────
def fig_informationsfluss():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("Informationsfluss: Vom Samenkorn zur Duftstofforientierung",
                 fontsize=14, fontweight='bold', pad=10)

    stufen = [
        (1.5, 3, "SAMENKORN\n(Genetische Info)", '#8D6E63',
         "DNA-Codierung\nVOC-Biosynthese-\ngene\nRegulationsproteine"),
        (4.5, 3, "KEIMLING\n(Epigenetik)", '#66BB6A',
         "Genexpression\nHormonkaskaden\nUmweltadaption"),
        (7.5, 3, "PFLANZE\n(Morphogenese)", '#26A69A',
         "Blattform, Farbe\nWuchsform\nDrüsenentwicklung"),
        (10.5, 3, "BLÜTE\n(Signal-Emission)", '#EF5350',
         "Terpen-Synthesen\nVOC-Emission\nFarbmuster UV"),
        (13.5, 3, "INSEKT\n(Navigation)", '#7E57C2',
         "Chemo-Rezeptoren\nKognitive Karten\nMemory-Effekte"),
    ]

    for (x, y, titel, farbe, details) in stufen:
        box = FancyBboxPatch((x-1.2, y-1.5), 2.4, 3.0,
                             boxstyle="round,pad=0.15", facecolor=farbe+'30',
                             edgecolor=farbe, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x, y+0.9, titel, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=farbe)
        ax.text(x, y-0.3, details, ha='center', va='center', fontsize=7.5,
                color='#333333', linespacing=1.4)

    # Pfeile
    pfeile_x = [2.8, 5.8, 8.8, 11.8]
    for px in pfeile_x:
        ax.annotate('', xy=(px+0.1, 3), xytext=(px-0.1, 3),
                    arrowprops=dict(arrowstyle='->', color='#424242',
                                   lw=2.5, mutation_scale=20))

    # Zeitachse
    ax.annotate('', xy=(15.5, 0.4), xytext=(0.5, 0.4),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.5))
    ax.text(8, 0.15, "Evolutionärer Informationsfluss (Jahrmillionen → Einzelsaison)",
            ha='center', fontsize=9, color='#616161', style='italic')

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig06_informationsfluss.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 6 done")

# ─────────────────────────────────────────────
# Fig 7: VOC-Emissionskinetik über die Tageszeit
# ─────────────────────────────────────────────
def fig_tagesrhythmus():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Zirkadiane Rhythmik der Duftstoffemission bei Blütenpflanzen",
                 fontsize=14, fontweight='bold')

    stunden = np.linspace(0, 24, 500)

    pflanzen = {
        "Rose": {'peak': 10, 'breite': 3, 'color': '#E91E63', 'max': 1.0},
        "Lavendel": {'peak': 14, 'breite': 4, 'color': '#9C27B0', 'max': 0.85},
        "Apfelblüte": {'peak': 11, 'breite': 2.5, 'color': '#FF9800', 'max': 0.75},
        "Nachtviole": {'peak': 22, 'breite': 3.5, 'color': '#3F51B5', 'max': 0.95},
        "Linde": {'peak': 13, 'breite': 5, 'color': '#4CAF50', 'max': 0.9},
    }

    ax = axes[0]
    ax.set_facecolor('#f8f8f0')
    # Tageszeit-Hintergrund
    ax.axvspan(0, 6, alpha=0.1, color='navy', label='Nacht')
    ax.axvspan(6, 20, alpha=0.05, color='yellow')
    ax.axvspan(20, 24, alpha=0.1, color='navy')

    for name, p in pflanzen.items():
        emission = p['max'] * np.exp(-(stunden - p['peak'])**2 / (2*p['breite']**2))
        # Nacht-Variante für Nachtviole
        if p['peak'] > 18:
            emission += p['max'] * np.exp(-(stunden - (p['peak']-24))**2 / (2*p['breite']**2))
        ax.plot(stunden, emission, color=p['color'], linewidth=2.5, label=name)

    ax.set_xlabel("Tageszeit [h]")
    ax.set_ylabel("Relative VOC-Emission")
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax.set_title("Tagesgänge der Emissionen", fontweight='bold')
    ax.legend(fontsize=9)

    # Subplot 2: Temperaturabhängigkeit
    ax = axes[1]
    temperaturen = np.linspace(5, 40, 200)
    pflanzen_temp = {
        "Rose": (25, 0.06, '#E91E63'),
        "Lavendel": (28, 0.05, '#9C27B0'),
        "Raps": (18, 0.08, '#CDDC39'),
        "Sonnenblume": (30, 0.05, '#FF9800'),
    }
    for name, (t_opt, k, color) in pflanzen_temp.items():
        emission = np.exp(-k * (temperaturen - t_opt)**2)
        emission *= (temperaturen > 8).astype(float)
        ax.plot(temperaturen, emission, color=color, linewidth=2.5, label=name)

    ax.set_xlabel("Temperatur [°C]")
    ax.set_ylabel("Relative VOC-Emissionsrate")
    ax.set_title("Temperaturabhängigkeit der VOC-Emission", fontweight='bold')
    ax.axvline(15, color='blue', linestyle='--', alpha=0.4, label='Kühle Grenze')
    ax.axvline(35, color='red', linestyle='--', alpha=0.4, label='Hitzestress')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig07_tagesrhythmus.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 7 done")

# ─────────────────────────────────────────────
# Fig 8: Lipid-Protein-Stärke-Verteilung in Samenklassen (Stacked Bar)
# ─────────────────────────────────────────────
def fig_samen_zusammensetzung():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Makronährstoff-Zusammensetzung der Samenklassen (Trockengewicht)",
                 fontsize=14, fontweight='bold')

    klassen = ['Kl.I\nÖlsamen', 'Kl.II\nGetreide', 'Kl.III\nHülsenf.', 'Kl.IV\nObstsam.',
               'Kl.V\nGewürze', 'Kl.VI\nAlkaloide', 'Kl.VII\nSukkul.', 'Kl.VIII\nWind']

    fette    = [45, 4, 6, 12, 18, 8, 5, 20]
    proteine = [25, 13, 28, 18, 15, 20, 10, 15]
    staerke  = [8, 65, 45, 20, 15, 25, 30, 10]
    zucker   = [5, 5, 8, 35, 10, 8, 30, 8]
    sonstige = [17, 13, 13, 15, 42, 39, 25, 47]

    x = np.arange(len(klassen))
    width = 0.55
    ax = axes[0]
    b1 = ax.bar(x, fette, width, label='Fette/Öle', color='#FF8F00')
    b2 = ax.bar(x, proteine, width, bottom=fette, label='Proteine', color='#2196F3')
    b3 = ax.bar(x, staerke, width, bottom=np.array(fette)+np.array(proteine),
                label='Stärke', color='#8BC34A')
    b4 = ax.bar(x, zucker, width, bottom=np.array(fette)+np.array(proteine)+np.array(staerke),
                label='Zucker', color='#E91E63')
    b5 = ax.bar(x, sonstige, width,
                bottom=np.array(fette)+np.array(proteine)+np.array(staerke)+np.array(zucker),
                label='Sonstige (Mineral., VOCs, Alkaloide)', color='#9E9E9E')
    ax.set_xticks(x); ax.set_xticklabels(klassen, fontsize=9)
    ax.set_ylabel("Anteil (%)"); ax.set_title("Makronährstoff-Verteilung", fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 115)

    # Subplot 2: Spezifische Duftstoffe im Samen
    ax = axes[1]
    samen_vocs = {
        'Apfelkern\n(Malus)':     [45, 20, 15, 10, 10],
        'Kirschkern\n(Prunus)':   [55, 18, 12, 8, 7],
        'Fenchelsamen\n(Foenic.)': [12, 60, 15, 8, 5],
        'Rapssamen\n(Brassica)':  [20, 10, 40, 20, 10],
        'Lavendel\n(Lavandula)':  [15, 55, 10, 12, 8],
    }
    voc_gruppen = ['Terpene', 'Aromat. Verb.', 'Fettsäure-Deriv.', 'Aldehyde', 'Ester']
    voc_colors = ['#FF5722','#9C27B0','#2196F3','#4CAF50','#FF9800']

    x2 = np.arange(len(samen_vocs))
    bottoms = np.zeros(len(samen_vocs))
    for i, (gruppe, color) in enumerate(zip(voc_gruppen, voc_colors)):
        werte = [v[i] for v in samen_vocs.values()]
        ax.bar(x2, werte, 0.5, bottom=bottoms, label=gruppe, color=color)
        bottoms += np.array(werte)
    ax.set_xticks(x2)
    ax.set_xticklabels(list(samen_vocs.keys()), fontsize=9)
    ax.set_ylabel("Relativer Anteil VOC-Fraktionen (%)")
    ax.set_title("VOC-Klassen in ausgewählten Samenkernen", fontweight='bold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig08_samen_zusammensetzung.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 8 done")

# ─────────────────────────────────────────────
# Fig 9: Duftstoffrezeptoren bei Insekten (Scatter)
# ─────────────────────────────────────────────
def fig_rezeptoren():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Olfaktorische Rezeptorsysteme bei Bestäubungsinsekten",
                 fontsize=14, fontweight='bold')

    insekten_rezeptoren = {
        'Honigbiene\n(Apis mellifera)': (170, '#FF9800'),
        'Hummel\n(Bombus terr.)': (160, '#CDDC39'),
        'Schmetterling\n(Danaus)': (140, '#E91E63'),
        'Fliege\n(Drosophila)': (60, '#2196F3'),
        'Nachtfalter\n(Manduca)': (120, '#9C27B0'),
        'Schwebfliege\n(Episyr.)': (90, '#00BCD4'),
    }

    # Subplot 1: Anzahl OR-Gene
    ax = axes[0]
    namen = list(insekten_rezeptoren.keys())
    rezeptoren_n = [v[0] for v in insekten_rezeptoren.values()]
    farben = [v[1] for v in insekten_rezeptoren.values()]
    bars = ax.barh(namen, rezeptoren_n, color=farben, edgecolor='white')
    for bar, val in zip(bars, rezeptoren_n):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.set_xlabel("Anzahl Olfaktorischer Rezeptor-Gene")
    ax.set_title("OR-Gen-Anzahl", fontweight='bold')
    ax.set_xlim(0, 200)

    # Subplot 2: Antennensensitivität
    ax = axes[1]
    frequenzen = np.linspace(200, 600, 300)
    sensitivitaeten = {
        'Linalool (Biene)': (380, 30, '#FF9800'),
        'Geraniol (Biene)': (310, 25, '#FF5722'),
        'Benzaldehyd (Schmetterl.)': (520, 40, '#E91E63'),
        'p-Anisaldehyd (Hummel)': (420, 35, '#CDDC39'),
        'cis-3-Hexenol (allg.)': (260, 20, '#4CAF50'),
    }
    for name, (peak, breite, color) in sensitivitaeten.items():
        sens = np.exp(-(frequenzen - peak)**2 / (2*breite**2))
        ax.plot(frequenzen, sens, color=color, linewidth=2, label=name)
    ax.set_xlabel("Molekulargewicht [g/mol]")
    ax.set_ylabel("Relative Antennensensitivität")
    ax.set_title("Ligand-Spezifität", fontweight='bold')
    ax.legend(fontsize=7.5)

    # Subplot 3: Gehirn-Aktivierungsmuster (Glomeruli)
    ax = axes[2]
    ax.set_aspect('equal')
    np.random.seed(7)
    n_glom = 60
    positions = np.random.rand(n_glom, 2) * 10
    # Aktivierung durch Rose-Duft
    aktivierung_rose = np.exp(-((positions[:, 0]-3)**2+(positions[:, 1]-4)**2)/4)
    # Aktivierung durch Lavendel
    aktivierung_lavend = np.exp(-((positions[:, 0]-7)**2+(positions[:, 1]-6)**2)/5)

    sc = ax.scatter(positions[:, 0], positions[:, 1],
                   c=aktivierung_rose + aktivierung_lavend*0.7,
                   s=200, cmap='hot', edgecolors='grey', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='Glomeruli-Aktivierung')
    ax.set_title("Antennenlobus-Muster\n(Biene, Rose+Lavendel)", fontweight='bold')
    ax.set_xlabel("Strukturelle Position"); ax.set_ylabel("Dorsoventralachse")
    # Markiere spezifische Glomeruli
    ax.annotate('T1-28\n(Linalool)', xy=(3.2, 4.5), fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig09_rezeptoren.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 9 done")

# ─────────────────────────────────────────────
# Fig 10: Biosynthesewege der Terpene aus Samen-Vorstufen
# ─────────────────────────────────────────────
def fig_biosynthese():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Terpen-Biosynthesewege: Vom Samen-Acetyl-CoA zu Blüten-Duftstoffen",
                 fontsize=13, fontweight='bold')

    knoten = {
        'Acetyl-CoA\n(Samen-DNA)': (1, 4, '#8D6E63'),
        'Mevalonat\n(MVA-Weg)': (3, 6, '#FF8F00'),
        'MEP/DXP\n(Plastiden)': (3, 2, '#66BB6A'),
        'IPP/DMAPP\n(C5-Einheiten)': (5.5, 4, '#2196F3'),
        'GPP (C10)\nGeranylpyrophosphat': (7.5, 6, '#9C27B0'),
        'FPP (C15)\nFarnesylpyrophosphat': (7.5, 2, '#E91E63'),
        'Monoterpene\n(C10-VOCs)': (10.5, 7, '#AB47BC'),
        'Sesquiter.\n(C15-VOCs)': (10.5, 4.5, '#EF5350'),
        'Diterpene\n(C20)': (10.5, 2.5, '#26A69A'),
        'Linalool\nGeraniol\nLimonen': (13, 7, '#CE93D8'),
        'Farnesol\nα-Farnesen\nNerolidol': (13, 4.5, '#EF9A9A'),
        'Taxol\nGibberelline\nAba': (13, 2.5, '#80DEEA'),
    }

    pfeil_verbindungen = [
        ('Acetyl-CoA\n(Samen-DNA)', 'Mevalonat\n(MVA-Weg)'),
        ('Acetyl-CoA\n(Samen-DNA)', 'MEP/DXP\n(Plastiden)'),
        ('Mevalonat\n(MVA-Weg)', 'IPP/DMAPP\n(C5-Einheiten)'),
        ('MEP/DXP\n(Plastiden)', 'IPP/DMAPP\n(C5-Einheiten)'),
        ('IPP/DMAPP\n(C5-Einheiten)', 'GPP (C10)\nGeranylpyrophosphat'),
        ('IPP/DMAPP\n(C5-Einheiten)', 'FPP (C15)\nFarnesylpyrophosphat'),
        ('GPP (C10)\nGeranylpyrophosphat', 'Monoterpene\n(C10-VOCs)'),
        ('FPP (C15)\nFarnesylpyrophosphat', 'Sesquiter.\n(C15-VOCs)'),
        ('FPP (C15)\nFarnesylpyrophosphat', 'Diterpene\n(C20)'),
        ('Monoterpene\n(C10-VOCs)', 'Linalool\nGeraniol\nLimonen'),
        ('Sesquiter.\n(C15-VOCs)', 'Farnesol\nα-Farnesen\nNerolidol'),
        ('Diterpene\n(C20)', 'Taxol\nGibberelline\nAba'),
    ]

    for start, end in pfeil_verbindungen:
        x1, y1, _ = knoten[start]
        x2, y2, _ = knoten[end]
        ax.annotate('', xy=(x2-0.6, y2), xytext=(x1+0.6, y1),
                    arrowprops=dict(arrowstyle='->', color='#616161', lw=1.8, mutation_scale=15))

    for name, (x, y, color) in knoten.items():
        box = FancyBboxPatch((x-0.8, y-0.65), 1.6, 1.3,
                             boxstyle="round,pad=0.1", facecolor=color+'33',
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color=color)

    ax.text(1, 7.5, "Im Samen codiert →", fontsize=10, color='#8D6E63', fontweight='bold', style='italic')

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig10_biosynthese.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 10 done")

# ─────────────────────────────────────────────
# Fig 11: Bestäubungseffektivität vs. Duftstoff-Diversität
# ─────────────────────────────────────────────
def fig_bestaeubung_effektivitaet():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Zusammenhang zwischen Duftstoff-Diversität und Bestäubungserfolg",
                 fontsize=14, fontweight='bold')

    np.random.seed(12)
    n_pflanzen = 80
    voc_diversitaet = np.random.exponential(8, n_pflanzen) + 2
    bestaeubung = 0.6 * voc_diversitaet + np.random.normal(0, 5, n_pflanzen)
    bestaeubung = np.clip(bestaeubung, 0, 100)

    ax = axes[0]
    farben_scatter = cm.viridis(voc_diversitaet / voc_diversitaet.max())
    ax.scatter(voc_diversitaet, bestaeubung, c=farben_scatter, s=60, alpha=0.7, edgecolors='grey', lw=0.5)

    # Regressionslinie
    z = np.polyfit(voc_diversitaet, bestaeubung, 1)
    p = np.poly1d(z)
    x_line = np.linspace(voc_diversitaet.min(), voc_diversitaet.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2.5, label=f'Regression (r²={np.corrcoef(voc_diversitaet, bestaeubung)[0,1]**2:.2f})')
    ax.set_xlabel("VOC-Diversität (Anzahl Komponenten)")
    ax.set_ylabel("Bestäubungserfolg (%)")
    ax.set_title("VOC-Diversität vs. Bestäubungsrate", fontweight='bold')
    ax.legend()

    # Subplot 2: Insektenbesuch nach Duftstoffklasse
    ax = axes[1]
    kategorien_duft = ['Mono-\nterpene', 'Sesqui-\nterpene', 'Aromat.\nVerbind.', 'Fettsäure-\nDerivate', 'Gemisch\nmultimodal']
    insekten = {
        'Honigbiene': [30, 25, 15, 10, 45],
        'Hummel': [25, 20, 20, 15, 40],
        'Schmetterling': [15, 10, 35, 8, 30],
        'Schwebfliege': [10, 15, 25, 20, 35],
    }
    x = np.arange(len(kategorien_duft))
    breite = 0.18
    farben_ins = ['#FF9800','#CDDC39','#E91E63','#00BCD4']
    for i, (ins, werte) in enumerate(insekten.items()):
        ax.bar(x + i*breite, werte, breite, label=ins, color=farben_ins[i], alpha=0.85)
    ax.set_xticks(x + 1.5*breite)
    ax.set_xticklabels(kategorien_duft, fontsize=9)
    ax.set_ylabel("Relative Besuchshäufigkeit (%)")
    ax.set_title("Insektenbesuche nach Duftstofftyp", fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig11_bestaeubung.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 11 done")

# ─────────────────────────────────────────────
# Fig 12: Samenkeimung und VOC-Entwicklung
# ─────────────────────────────────────────────
def fig_keimung_voc():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("VOC-Entwicklung während Keimung und Wachstum (Apfelbaum, Malus domestica)",
                 fontsize=13, fontweight='bold')

    tage = np.linspace(0, 365, 500)

    # Phase 1: Keimung (0-14 d), Phase 2: Keimling (14-60 d), Phase 3: Jungpflanze (60-180 d), Phase 4: Blüte (180-250 d)
    def sigmoid(x, x0, k): return 1 / (1 + np.exp(-k*(x-x0)))

    terpene = 0.1 + 0.9 * sigmoid(tage, 190, 0.08) * np.exp(-((tage-220)**2)/1800) + 0.05*sigmoid(tage, 350, 0.1)
    alkohole = 0.3*sigmoid(tage, 20, 0.5) + 0.5*sigmoid(tage, 185, 0.12)*np.exp(-((tage-215)**2)/2000) + 0.2*sigmoid(tage, 330, 0.2)
    aldehyde = 0.2*sigmoid(tage, 10, 0.8) + 0.4*sigmoid(tage, 180, 0.15)*np.exp(-((tage-210)**2)/1500)
    ester = 0.05 + 0.85 * sigmoid(tage, 200, 0.1) * np.exp(-((tage-225)**2)/1600) + 0.6*sigmoid(tage, 300, 0.3)

    ax = axes[0]
    ax.fill_between(tage, 0, terpene, alpha=0.3, color='#9C27B0', label='Terpene')
    ax.fill_between(tage, 0, alkohole, alpha=0.3, color='#4CAF50', label='Terpenalkohole')
    ax.fill_between(tage, 0, aldehyde, alpha=0.3, color='#FF9800', label='Aldehyde')
    ax.fill_between(tage, 0, ester, alpha=0.3, color='#E91E63', label='Ester')
    ax.plot(tage, terpene, '#9C27B0', lw=2); ax.plot(tage, alkohole, '#4CAF50', lw=2)
    ax.plot(tage, aldehyde, '#FF9800', lw=2); ax.plot(tage, ester, '#E91E63', lw=2)

    ax.axvspan(0, 14, alpha=0.08, color='brown'); ax.text(7, 1.05, 'Keimung', fontsize=7.5, ha='center')
    ax.axvspan(14, 60, alpha=0.08, color='green'); ax.text(37, 1.05, 'Keimling', fontsize=7.5, ha='center')
    ax.axvspan(60, 180, alpha=0.06, color='lime'); ax.text(120, 1.05, 'Wachstum', fontsize=7.5, ha='center')
    ax.axvspan(180, 260, alpha=0.1, color='gold'); ax.text(220, 1.05, 'Blüte', fontsize=7.5, ha='center', color='#FF8C00', fontweight='bold')
    ax.axvspan(260, 365, alpha=0.06, color='orange'); ax.text(312, 1.05, 'Frucht', fontsize=7.5, ha='center')

    ax.set_xlabel("Tag des Jahres"); ax.set_ylabel("Relative VOC-Emissionsrate")
    ax.set_title("Saisonale VOC-Entwicklung", fontweight='bold')
    ax.legend(fontsize=8.5); ax.set_ylim(0, 1.15)

    # Subplot 2: Sameninhalt → Blüte (pie charts)
    ax = axes[1]
    ax.axis('off')

    # Miniatur-Pie-Charts
    pies_data = [
        ([30, 25, 20, 15, 10], ['#8D6E63','#66BB6A','#2196F3','#FF9800','#9E9E9E'],
         "Samen\n(metabolisch)", (0.15, 0.65)),
        ([10, 30, 15, 20, 25], ['#8D6E63','#66BB6A','#2196F3','#FF9800','#9E9E9E'],
         "Keimling\n(aktivierend)", (0.45, 0.65)),
        ([5, 15, 10, 50, 20], ['#8D6E63','#66BB6A','#2196F3','#FF9800','#9E9E9E'],
         "Blüte\n(VOC-Emission)", (0.75, 0.65)),
    ]
    labels_pie = ['Terpenvorstufen', 'Aminosäuren', 'Kohlenhydrate', 'VOC-Vorstufen', 'Sonstiges']
    for data, colors, titel, pos in pies_data:
        inset = ax.inset_axes([pos[0]-0.18, 0.0, 0.35, 0.9])
        inset.pie(data, colors=colors, autopct='%1.0f%%', pctdistance=0.75,
                  textprops={'fontsize': 7})
        inset.set_title(titel, fontsize=9, fontweight='bold', pad=3)

    ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(
        ['#8D6E63','#66BB6A','#2196F3','#FF9800','#9E9E9E'], labels_pie)],
        loc='lower center', fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig12_keimung_voc.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 12 done")

# ─────────────────────────────────────────────
# Fig 13: Vergleich UV-Muster + Duftstoff-Synergismus
# ─────────────────────────────────────────────
def fig_uv_duft_synergismus():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Multimodales Orientierungssystem: UV-Muster und Duftstoff-Synergismus",
                 fontsize=13, fontweight='bold')

    # Subplot 1: UV-Reflexionsmuster simuliert
    ax = axes[0]
    r = np.linspace(0, 1, 200)
    theta = np.linspace(0, 2*np.pi, 200)
    R, T = np.meshgrid(r, theta)
    X_uv = R * np.cos(T)
    Y_uv = R * np.sin(T)
    # Blütenblatt-Muster
    n_petals = 6
    UV = np.sin(n_petals * T / 2)**2 * np.exp(-R*2) + np.exp(-R**2 * 8)
    ax.contourf(X_uv, Y_uv, UV, levels=20, cmap='YlOrRd')
    ax.set_title("UV-Reflexionsmuster\n(simulierte Blüte, Bienen-Sicht)", fontweight='bold')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')
    # Nektar-Guide
    for i in range(n_petals):
        angle = i * 2*np.pi/n_petals
        ax.plot([0, 0.5*np.cos(angle)], [0, 0.5*np.sin(angle)], 'w--', lw=1.5, alpha=0.6)

    # Subplot 2: Synergismus-Heatmap (UV x VOC)
    ax = axes[1]
    uv_intensitaet = np.linspace(0, 1, 50)
    voc_konzentration = np.linspace(0, 1, 50)
    UV2, VOC = np.meshgrid(uv_intensitaet, voc_konzentration)
    # Synergismus: überproportionale Wirkung bei Kombination
    synergismus = UV2 * VOC + 0.3 * UV2**2 * VOC + 0.3 * UV2 * VOC**2
    im = ax.contourf(uv_intensitaet, voc_konzentration, synergismus, levels=20, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Besuchswahrscheinlichkeit')
    ax.contour(uv_intensitaet, voc_konzentration, synergismus, levels=5, colors='black', alpha=0.3, lw=0.5)
    ax.set_xlabel("UV-Reflexionsintensität (rel.)")
    ax.set_ylabel("VOC-Konzentration (rel.)")
    ax.set_title("Synergismus UV × Duftstoff", fontweight='bold')
    ax.plot([0, 1], [0, 1], 'w--', lw=1.5, label='Lineare Referenz', alpha=0.7)
    ax.legend(fontsize=8)

    # Subplot 3: Sinnesmodalitäten-Gewichtung
    ax = axes[2]
    entfernungen = np.linspace(0.1, 100, 300)
    gewicht_duft = 0.9 * np.exp(-entfernungen / 40) + 0.1
    gewicht_farbe = 0.85 * np.exp(-entfernungen / 10) + 0.05
    gewicht_uv = 0.8 * np.exp(-entfernungen / 8) + 0.03
    gewicht_vibrat = 0.6 * np.exp(-entfernungen / 3) + 0.02

    ax.fill_between(entfernungen, 0, gewicht_duft, alpha=0.3, color='#9C27B0')
    ax.fill_between(entfernungen, 0, gewicht_farbe, alpha=0.3, color='#FF9800')
    ax.fill_between(entfernungen, 0, gewicht_uv, alpha=0.3, color='#2196F3')
    ax.fill_between(entfernungen, 0, gewicht_vibrat, alpha=0.3, color='#4CAF50')
    ax.plot(entfernungen, gewicht_duft, '#9C27B0', lw=2.5, label='Duftstoff (Olfaktion)')
    ax.plot(entfernungen, gewicht_farbe, '#FF9800', lw=2.5, label='Farbe (Sicht)')
    ax.plot(entfernungen, gewicht_uv, '#2196F3', lw=2.5, label='UV-Muster')
    ax.plot(entfernungen, gewicht_vibrat, '#4CAF50', lw=2.5, label='Vibration (Nahfeld)')
    ax.set_xscale('log')
    ax.set_xlabel("Entfernung zur Blüte [m]")
    ax.set_ylabel("Relative Signalgewichtung")
    ax.set_title("Sinnesmodalitäten vs. Entfernung", fontweight='bold')
    ax.legend(fontsize=8.5)
    ax.axvline(1, color='grey', lw=1, linestyle=':', alpha=0.6)
    ax.text(1.2, 0.5, '1m', color='grey', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig13_uv_duft_synergismus.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 13 done")

# ─────────────────────────────────────────────
# Fig 14: Genetische Samenkorn-Information (DNA → Terpen)
# ─────────────────────────────────────────────
def fig_genetik_samen():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Genetische Informationstiefe im Samenkorn: VOC-kodierende Genfamilien",
                 fontsize=13, fontweight='bold')

    # Subplot 1: Genfamilien-Größen
    ax = axes[0]
    genfamilien = {
        'TPS (Terpensynthasen)': 32,
        'CYP450 (Hydroxylasen)': 45,
        'BAHD (Acetyltransfer.)': 18,
        'OMT (O-Methyltransfer.)': 24,
        'BEAT (Alkoholacetyltransf.)': 12,
        'LOX (Lipoxygenase)': 15,
        'ADH (Alkoholdehydrogen.)': 20,
        'ALDH (Aldehyddehydrogen.)': 17,
        'FaQR (Reduktasen)': 8,
        'SAMT (Methyltransfer.)': 9,
    }
    namen_g = list(genfamilien.keys())
    werte_g = list(genfamilien.values())
    colors_g = cm.tab10(np.linspace(0, 1, len(namen_g)))
    bars = ax.barh(namen_g, werte_g, color=colors_g, edgecolor='white')
    for bar, val in zip(bars, werte_g):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.set_xlabel("Anzahl Gen-Kopien (Arabidopsis-Referenzgenom)")
    ax.set_title("VOC-kodierende Genfamilien", fontweight='bold')
    ax.set_xlim(0, 55)

    # Subplot 2: Expressionsmuster in Geweben
    ax = axes[1]
    gewebe = ['Samen\n(ruhend)', 'Samen\n(keimend)', 'Wurzel', 'Blatt', 'Stängel', 'Blüte', 'Frucht']
    genfam_sel = ['TPS', 'CYP450', 'LOX', 'BEAT', 'SAMT']
    # Heatmap-Daten (relative Expression 0-100)
    expression = np.array([
        [5,  45, 10, 30, 15, 100, 60],   # TPS
        [10, 40, 25, 35, 20, 85,  70],   # CYP450
        [20, 50, 30, 40, 25, 70,  90],   # LOX
        [2,  20, 5,  15, 8,  95,  40],   # BEAT
        [3,  15, 8,  20, 10, 80,  35],   # SAMT
    ])
    im = ax.imshow(expression, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Relative Genexpression (%)')
    ax.set_xticks(range(len(gewebe))); ax.set_xticklabels(gewebe, fontsize=9)
    ax.set_yticks(range(len(genfam_sel))); ax.set_yticklabels(genfam_sel, fontsize=10, fontweight='bold')
    ax.set_title("Gewebespezifische Genexpression", fontweight='bold')
    for i in range(len(genfam_sel)):
        for j in range(len(gewebe)):
            ax.text(j, i, str(expression[i,j]), ha='center', va='center',
                    fontsize=8, color='black' if expression[i,j] < 60 else 'white')

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig14_genetik_samen.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 14 done")

# ─────────────────────────────────────────────
# Fig 15: Vorkommen wichtiger VOCs in Blütenpflanzen (Netzwerk-ähnlich)
# ─────────────────────────────────────────────
def fig_voc_vorkommen():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_facecolor('#fafbfe')
    fig.patch.set_facecolor('#fafbfe')
    ax.set_title("Vorkommen von VOC-Schlüsselverbindungen in Blütenpflanzen\n(Netzwerkdarstellung)",
                 fontsize=13, fontweight='bold')

    # Zentrale VOCs
    zentrale_vocs = [
        ("Linalool",     0.0,  0.0,  '#9C27B0', 0.20),
        ("Geraniol",     0.7,  0.5,  '#E91E63', 0.16),
        ("Benzaldehyd",  0.0,  0.9,  '#2196F3', 0.15),
        ("α-Pinen",     -0.7,  0.5,  '#4CAF50', 0.14),
        ("Linalylac.",   0.0, -0.9,  '#FF9800', 0.15),
        ("p-Anisaldeh.", -0.9,  0.0,  '#00BCD4', 0.13),
        ("Farnesol",     0.85, -0.5,  '#795548', 0.12),
        ("Eugenol",     -0.85, -0.5,  '#F44336', 0.13),
    ]

    # Pflanzen ringsrum
    pflanzen_ring = [
        ("Rose",       1.3,  0.1,  '#E91E63'),
        ("Lavendel",   0.9,  1.1,  '#9C27B0'),
        ("Apfel",      0.0,  1.4,  '#FF9800'),
        ("Raps",      -1.0,  1.1,  '#CDDC39'),
        ("Linde",     -1.3,  0.1,  '#4CAF50'),
        ("Kirsche",   -1.1, -0.9,  '#F44336'),
        ("Sonnenbl.", -0.0, -1.4,  '#FFC107'),
        ("Fenchel",    1.1, -0.9,  '#00BCD4'),
    ]

    # Verbindungen Pflanze → VOC
    verbindungen = {
        "Rose":       ["Linalool", "Geraniol", "Eugenol"],
        "Lavendel":   ["Linalool", "Linalylac."],
        "Apfel":      ["α-Pinen", "Benzaldehyd", "Farnesol"],
        "Raps":       ["p-Anisaldeh.", "Benzaldehyd", "Linalool"],
        "Linde":      ["Linalool", "Farnesol", "Linalylac."],
        "Kirsche":    ["Benzaldehyd", "Linalool"],
        "Sonnenbl.":  ["α-Pinen", "Linalool", "Farnesol"],
        "Fenchel":    ["Linalool", "p-Anisaldeh."],
    }

    voc_pos = {v[0]: (v[1], v[2]) for v in zentrale_vocs}
    pfl_pos = {p[0]: (p[1], p[2]) for p in pflanzen_ring}

    for pfl, vocs_list in verbindungen.items():
        for voc in vocs_list:
            px, py = pfl_pos[pfl]
            vx, vy = voc_pos[voc]
            ax.plot([px, vx], [py, vy], '-', color='#BDBDBD', lw=1.2, alpha=0.5, zorder=1)

    for name, x, y, color, size in zentrale_vocs:
        circle = Circle((x, y), size, color=color, alpha=0.35, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', color=color, zorder=4)

    for name, x, y, color in pflanzen_ring:
        circle = Circle((x, y), 0.11, color=color, alpha=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y+0.16, name, ha='center', va='bottom', fontsize=8.5, color=color, fontweight='bold', zorder=4)

    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig15_voc_vorkommen.pdf", bbox_inches='tight')
    plt.close()
    print("Fig 15 done")


if __name__ == "__main__":
    print("Generating figures...")
    fig_duft_spektrum()
    fig_duft_ueberlagerung()
    fig_insekten_orientierung()
    fig_samenklassen()
    fig_samen_biochemie()
    fig_informationsfluss()
    fig_tagesrhythmus()
    fig_samen_zusammensetzung()
    fig_rezeptoren()
    fig_biosynthese()
    fig_bestaeubung_effektivitaet()
    fig_keimung_voc()
    fig_uv_duft_synergismus()
    fig_genetik_samen()
    fig_voc_vorkommen()
    print("All figures generated!")
