#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depression als Graph-Modellierung: Visualisierungen
Generiert alle wissenschaftlichen Plots für die LaTeX-Arbeit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import warnings
warnings.filterwarnings('ignore')

# Matplotlib-Konfiguration für hochwertige PDFs
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# Farben
COLOR_NORMAL = '#1a468c'  # Dunkles Blau
COLOR_DEPRESSED = '#b43214'  # Tiefes Rot
COLOR_INTERMEDIATE = '#d4a574'  # Braun
COLOR_HIGHLIGHT = '#1e6432'  # Dunkelgrün

print("Starte Visualisierungs-Generierung...")

# ============================================================================
# PLOT 1: Erreichbarkeitsinzidenz - Normal vs. Depressiv
# ============================================================================
print("Generiere Plot 1: Erreichbarkeitsinzidenz...")

np.random.seed(42)

def generate_graph(n_nodes=1000, edge_prob=0.15, mean_weight=0.8, std_weight=0.15):
    """Generiert einen zufälligen gewichteten Graphen"""
    # Adjacency matrix mit probabilistischen Kanten
    adj = np.random.rand(n_nodes, n_nodes) < edge_prob
    np.fill_diagonal(adj, 0)  # Keine Selbst-Schleifen
    
    # Kantengewichte
    weights = np.random.normal(mean_weight, std_weight, (n_nodes, n_nodes))
    weights = np.clip(weights, 0.1, 1.0)  # Clip zu [0.1, 1.0]
    
    # Nur gewichtete Kanten
    weighted_adj = adj.astype(float) * weights
    
    return weighted_adj

def compute_reachability(adj, threshold=0.3, max_depth=10):
    """Berechnet Erreichbarkeitsmengen für jeden Knoten"""
    n = adj.shape[0]
    reachability = np.zeros(n)
    
    for start_node in range(n):
        # BFS mit Gewicht-Schwellwert
        visited = set([start_node])
        queue = [(start_node, 1.0)]  # (node, accumulated_weight)
        
        while queue:
            node, weight = queue.pop(0)
            
            # Nachbarn mit signifikantem Gewicht
            neighbors = np.where(adj[node] >= threshold)[0]
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_weight = weight * adj[node, neighbor]
                    if new_weight >= threshold:
                        visited.add(neighbor)
                        queue.append((neighbor, new_weight))
        
        reachability[start_node] = len(visited) / n
    
    return reachability

# Generiere Graphen
print("  - Generiere normalen Graph...")
G_normal = generate_graph(n_nodes=1000, edge_prob=0.15, mean_weight=0.8)
reach_normal = compute_reachability(G_normal, threshold=0.3)
ri_normal = np.mean(reach_normal)

print("  - Generiere depressiven Graph...")
G_depressed = generate_graph(n_nodes=1000, edge_prob=0.09, mean_weight=0.5)
reach_depressed = compute_reachability(G_depressed, threshold=0.5)
ri_depressed = np.mean(reach_depressed)

print(f"  - RI Normal: {ri_normal:.3f}, RI Depressed: {ri_depressed:.3f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogramme der Erreichbarkeit
bins = np.linspace(0, 1, 30)

ax1.hist(reach_normal, bins=bins, alpha=0.7, color=COLOR_NORMAL, label='Normal', edgecolor='black', linewidth=0.5)
ax1.axvline(ri_normal, color=COLOR_NORMAL, linestyle='--', linewidth=2, label=f'Mittelwert = {ri_normal:.3f}')
ax1.set_xlabel('Erreichbarkeitsinzidenz (Anteil erreichbarer Knoten)', fontsize=11)
ax1.set_ylabel('Häufigkeit (Anzahl Knoten)', fontsize=11)
ax1.set_title('Normales Gehirn', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle=':')
ax1.set_ylim(0, 150)

ax2.hist(reach_depressed, bins=bins, alpha=0.7, color=COLOR_DEPRESSED, label='Depressiv', edgecolor='black', linewidth=0.5)
ax2.axvline(ri_depressed, color=COLOR_DEPRESSED, linestyle='--', linewidth=2, label=f'Mittelwert = {ri_depressed:.3f}')
ax2.set_xlabel('Erreichbarkeitsinzidenz (Anteil erreichbarer Knoten)', fontsize=11)
ax2.set_ylabel('Häufigkeit (Anzahl Knoten)', fontsize=11)
ax2.set_title('Depressives Gehirn', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle=':')
ax2.set_ylim(0, 150)

plt.tight_layout()
plt.savefig('/home/claude/plot_01_reachability.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_01_reachability.pdf")
plt.close()

# ============================================================================
# PLOT 2: Konnektivitäts-Vergleich und Komponenten-Analyse
# ============================================================================
print("Generiere Plot 2: Konnektivitäts-Analyse...")

def get_giant_component_info(adj, threshold=0.1):
    """Findet die Giant Component und Komponenten-Informationen"""
    # Binarisiere (Kanten > threshold)
    adj_binary = (adj > threshold).astype(int)
    
    # Verwendet sparse representation für Effizienz
    adj_sparse = csr_matrix(adj_binary)
    n_components, labels = connected_components(adj_sparse, directed=False)
    
    # Größen der Komponenten
    component_sizes = np.bincount(labels)
    component_sizes.sort()
    
    largest_component_size = component_sizes[-1]
    largest_component_fraction = largest_component_size / len(adj)
    
    return n_components, largest_component_fraction, component_sizes

n_comp_normal, giant_normal, comp_normal = get_giant_component_info(G_normal)
n_comp_depressed, giant_depressed, comp_depressed = get_giant_component_info(G_depressed)

print(f"  - Normal: {n_comp_normal} Komponenten, Giant = {giant_normal*100:.1f}%")
print(f"  - Depressed: {n_comp_depressed} Komponenten, Giant = {giant_depressed*100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 2a: Giant Component Vergleich
ax = axes[0, 0]
categories = ['Normal', 'Depressiv']
giant_fractions = [giant_normal, giant_depressed]
other_fractions = [1 - giant_normal, 1 - giant_depressed]

x_pos = np.arange(len(categories))
width = 0.5

ax.bar(x_pos, giant_fractions, width, label='Giant Component', color=COLOR_NORMAL, edgecolor='black', linewidth=1)
ax.bar(x_pos, other_fractions, width, bottom=giant_fractions, label='Kleine Komponenten', color=COLOR_HIGHLIGHT, alpha=0.6, edgecolor='black', linewidth=1)

ax.set_ylabel('Anteil der Knoten', fontsize=11)
ax.set_title('Giant Component Anteil', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle=':')

# Plot 2b: Anzahl Komponenten
ax = axes[0, 1]
n_comps = [n_comp_normal, n_comp_depressed]
colors_comp = [COLOR_NORMAL, COLOR_DEPRESSED]
bars = ax.bar(categories, n_comps, color=colors_comp, edgecolor='black', linewidth=1, width=0.5)

for bar, n_comp in zip(bars, n_comps):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(n_comp)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Anzahl stark zusammenhängender Komponenten', fontsize=11)
ax.set_title('Graph-Fragmentierung', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(n_comps) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle=':')

# Plot 2c: Komponenten-Größen (Normal)
ax = axes[1, 0]
comp_normal_sorted = np.sort(comp_normal)[::-1]
ax.bar(range(min(50, len(comp_normal_sorted))), comp_normal_sorted[:50], color=COLOR_NORMAL, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Komponenten-Rangfolge', fontsize=11)
ax.set_ylabel('Komponenten-Größe (# Knoten)', fontsize=11)
ax.set_title('Komponenten-Verteilung (Normal)', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, linestyle=':')

# Plot 2d: Komponenten-Größen (Depressiv)
ax = axes[1, 1]
comp_depressed_sorted = np.sort(comp_depressed)[::-1]
ax.bar(range(min(50, len(comp_depressed_sorted))), comp_depressed_sorted[:50], color=COLOR_DEPRESSED, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Komponenten-Rangfolge', fontsize=11)
ax.set_ylabel('Komponenten-Größe (# Knoten)', fontsize=11)
ax.set_title('Komponenten-Verteilung (Depressiv)', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('/home/claude/plot_02_connectivity.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_02_connectivity.pdf")
plt.close()

# ============================================================================
# PLOT 3: Dynamische Aktivierungsmuster
# ============================================================================
print("Generiere Plot 3: Dynamische Aktivierungsmuster...")

def simulate_dynamics(adj, n_steps=1000, noise_level=0.1, initial_state='random'):
    """Simuliert die neurale Dynamik mit Wilson-Cowan-ähnlichem Modell"""
    n = adj.shape[0]
    
    # Initialisierung
    if initial_state == 'random':
        x = np.random.rand(n) * 0.5
    else:
        x = np.ones(n) * 0.1
    
    history = np.zeros((n_steps, n))
    
    # Sigmoid-Aktivierungsfunktion
    def sigmoid(z, gain=4):
        return 1.0 / (1.0 + np.exp(-gain * (z - 0.5)))
    
    for t in range(n_steps):
        # Speichere aktuellen Zustand
        history[t] = x
        
        # Berechne Eingaben (gewichtete Summe der aktivierten Nachbarn)
        inputs = adj @ x
        
        # Aktivierungsfunktion + Rauschen
        noise = np.random.normal(0, noise_level, n)
        x = sigmoid(inputs + noise)
        
        # Normalisierung (Energy Decay)
        x = x * 0.95 + 0.05 * np.mean(x)
    
    return history

print("  - Simuliere normales Gehirn...")
history_normal = simulate_dynamics(G_normal, n_steps=1000, noise_level=0.05)
mean_activation_normal = np.mean(history_normal, axis=1)
std_activation_normal = np.std(history_normal, axis=1)

print("  - Simuliere depressives Gehirn...")
history_depressed = simulate_dynamics(G_depressed, n_steps=1000, noise_level=0.15)
mean_activation_depressed = np.mean(history_depressed, axis=1)
std_activation_depressed = np.std(history_depressed, axis=1)

print(f"  - Mean Activation Normal: {np.mean(mean_activation_normal):.3f} ± {np.mean(std_activation_normal):.3f}")
print(f"  - Mean Activation Depressed: {np.mean(mean_activation_depressed):.3f} ± {np.mean(std_activation_depressed):.3f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Plot 3a: Zeitreihe Normal
ax = axes[0, 0]
time_window = slice(0, 300)
ax.plot(range(time_window.stop), mean_activation_normal[time_window], 
        color=COLOR_NORMAL, linewidth=1.5, label='Mittlere Aktivierung')
ax.fill_between(range(time_window.stop),
                mean_activation_normal[time_window] - std_activation_normal[time_window],
                mean_activation_normal[time_window] + std_activation_normal[time_window],
                alpha=0.3, color=COLOR_NORMAL, label='±1 Std. Dev.')
ax.set_xlabel('Zeit (Iterationen)', fontsize=11)
ax.set_ylabel('Aktivierungslevel', fontsize=11)
ax.set_title('Dynamik: Normales Gehirn', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3, linestyle=':')
ax.set_ylim(0, 1.0)

# Plot 3b: Zeitreihe Depressiv
ax = axes[0, 1]
ax.plot(range(time_window.stop), mean_activation_depressed[time_window], 
        color=COLOR_DEPRESSED, linewidth=1.5, label='Mittlere Aktivierung')
ax.fill_between(range(time_window.stop),
                mean_activation_depressed[time_window] - std_activation_depressed[time_window],
                mean_activation_depressed[time_window] + std_activation_depressed[time_window],
                alpha=0.3, color=COLOR_DEPRESSED, label='±1 Std. Dev.')
ax.set_xlabel('Zeit (Iterationen)', fontsize=11)
ax.set_ylabel('Aktivierungslevel', fontsize=11)
ax.set_title('Dynamik: Depressives Gehirn', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, linestyle=':')
ax.set_ylim(0, 1.0)

# Plot 3c: Verteilung der mittleren Aktivierungen
ax = axes[1, 0]
ax.hist(mean_activation_normal, bins=30, alpha=0.7, color=COLOR_NORMAL, 
        label=f'Normal (μ={np.mean(mean_activation_normal):.3f})', edgecolor='black', linewidth=0.5)
ax.hist(mean_activation_depressed, bins=30, alpha=0.7, color=COLOR_DEPRESSED, 
        label=f'Depressiv (μ={np.mean(mean_activation_depressed):.3f})', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Durchschnittliche Aktivierung', fontsize=11)
ax.set_ylabel('Häufigkeit', fontsize=11)
ax.set_title('Aktivierungs-Verteilung über Zeit', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle=':')

# Plot 3d: Variabilität
ax = axes[1, 1]
ax.plot(range(time_window.stop), std_activation_normal[time_window], 
        color=COLOR_NORMAL, linewidth=2, label='Normal (Rauschen=0.05)', marker='o', markersize=1)
ax.plot(range(time_window.stop), std_activation_depressed[time_window], 
        color=COLOR_DEPRESSED, linewidth=2, label='Depressiv (Rauschen=0.15)', marker='s', markersize=1)
ax.set_xlabel('Zeit (Iterationen)', fontsize=11)
ax.set_ylabel('Aktivierungs-Variabilität (Std. Dev.)', fontsize=11)
ax.set_title('Dynamische Variabilität über Zeit', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('/home/claude/plot_03_dynamics.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_03_dynamics.pdf")
plt.close()

# ============================================================================
# PLOT 4: Hub-Anfälligkeit (Betweenness-Analyse)
# ============================================================================
print("Generiere Plot 4: Hub-Anfälligkeit...")

def compute_betweenness_centrality(adj, max_pairs=1000):
    """Vereinfachte Betweenness-Berechnung (sampling-basiert für Effizienz)"""
    n = adj.shape[0]
    betweenness = np.zeros(n)
    
    # Sample random start/end pairs
    for _ in range(min(max_pairs, n)):
        start = np.random.randint(0, n)
        end = np.random.randint(0, n)
        
        if start == end:
            continue
        
        # BFS für kürzesten Pfad
        queue = [(start, [start])]
        visited = {start}
        path_found = False
        
        while queue and not path_found:
            node, path = queue.pop(0)
            
            # Nachbarn mit signifikantem Gewicht
            neighbors = np.where(adj[node] > 0.1)[0]
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    if neighbor == end:
                        for i in new_path[1:-1]:  # Exclude start and end
                            betweenness[i] += 1
                        path_found = True
                        break
                    
                    queue.append((neighbor, new_path))
    
    return betweenness / (np.max(betweenness) + 1e-10)

print("  - Berechne Betweenness für normales Gehirn...")
betweenness_normal = compute_betweenness_centrality(G_normal)
out_degree_normal = np.sum(G_normal > 0, axis=1)

print("  - Berechne Betweenness für depressives Gehirn...")
betweenness_depressed = compute_betweenness_centrality(G_depressed)
out_degree_depressed = np.sum(G_depressed > 0, axis=1)

# Kategorisiere Knoten nach Betweenness
def categorize_nodes(betweenness, labels=['Top 10% Hubs', 'Mittlere 50%', 'Bottom 10% Peripher']):
    """Kategorisiert Knoten nach Betweenness-Zentralität"""
    p90 = np.percentile(betweenness, 90)
    p10 = np.percentile(betweenness, 10)
    
    categories = {}
    for i, b in enumerate(betweenness):
        if b >= p90:
            cat = 0
        elif b <= p10:
            cat = 2
        else:
            cat = 1
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(i)
    
    return categories, [p10, p90]

cat_normal, _ = categorize_nodes(betweenness_normal)
cat_depressed, _ = categorize_nodes(betweenness_depressed)

# Berechne Aktivitätsabfälle
def compute_activity_drop(history_normal, history_depressed, node_indices):
    """Berechnet relativen Aktivitätsabfall"""
    mean_normal = np.mean(history_normal[:, node_indices])
    mean_depressed = np.mean(history_depressed[:, node_indices])
    
    drop = (mean_normal - mean_depressed) / (mean_normal + 1e-10)
    return drop * 100

drops = {}
drop_labels = []
for i, (label_idx, nodes) in enumerate([(0, 'Hub'), (1, 'Mittel'), (2, 'Peripher')]):
    if label_idx in cat_normal:
        drop = compute_activity_drop(history_normal, history_depressed, cat_normal[label_idx])
        drops[label_idx] = drop
        drop_labels.append(label_idx)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 4a: Aktivitätsabfall nach Knotenkategorie
ax = axes[0]
categories_names = ['Top 10%\nHubs', 'Mittlere\n50%', 'Bottom 10%\nPeripher']
drop_values = [drops.get(i, 0) for i in range(3)]
colors_drop = [COLOR_NORMAL, COLOR_INTERMEDIATE, COLOR_HIGHLIGHT]

bars = ax.bar(categories_names, drop_values, color=colors_drop, edgecolor='black', linewidth=1.5, width=0.6)

for bar, drop in zip(bars, drop_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{drop:.0f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Aktivitätsabfall bei Depression (%)', fontsize=11)
ax.set_title('Differentielle Hub-Verletzlichkeit', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle=':')

# Plot 4b: Betweenness vs. Aktivitätsabfall
ax = axes[1]

# Für jeden Knoten: Betweenness vs. Aktivitätsabfall
activations_normal_individual = np.mean(history_normal, axis=0)
activations_depressed_individual = np.mean(history_depressed, axis=0)
individual_drops = (activations_normal_individual - activations_depressed_individual) / (activations_normal_individual + 1e-10) * 100

# Scatter plot mit Farbcodierung nach Aktivitäts-Abfall
scatter = ax.scatter(betweenness_normal, individual_drops, 
                     c=individual_drops, cmap='RdYlBu_r', 
                     alpha=0.6, s=30, edgecolor='black', linewidth=0.3)

# Trend-Linie
z = np.polyfit(betweenness_normal, individual_drops, 2)
p = np.poly1d(z)
x_trend = np.linspace(np.min(betweenness_normal), np.max(betweenness_normal), 100)
ax.plot(x_trend, p(x_trend), color=COLOR_DEPRESSED, linewidth=2.5, linestyle='--', label='Trend (Polynom, Grad 2)')

ax.set_xlabel('Betweenness-Zentralität', fontsize=11)
ax.set_ylabel('Aktivitätsabfall (%)', fontsize=11)
ax.set_title('Hub-Zentralität vs. Vulnerabilität', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle=':')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Aktivitätsabfall (%)', fontsize=10)

plt.tight_layout()
plt.savefig('/home/claude/plot_04_hub_vulnerability.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_04_hub_vulnerability.pdf")
plt.close()

# ============================================================================
# PLOT 5: Therapeutischer Fortschritt
# ============================================================================
print("Generiere Plot 5: Therapeutischer Fortschritt...")

# Simuliere Therapie-Verlauf
def simulate_therapy_progression(weeks=24, initial_ri=0.41):
    """Simuliert Verbesserung der Erreichbarkeitsinzidenz unter Therapie"""
    
    # Verschiedene Therapie-Szenarien
    
    # Szenario 1: Unbehandelt (Plateau)
    untreated = np.ones(weeks) * initial_ri
    
    # Szenario 2: SSRI-Therapie (langsamer exponentieller Anstieg)
    ssri = initial_ri + (0.82 - initial_ri) * (1 - np.exp(-0.08 * np.arange(weeks)))
    
    # Szenario 3: Psychotherapie (schnellerer Anstieg)
    psycho = initial_ri + (0.78 - initial_ri) * (1 - np.exp(-0.12 * np.arange(weeks)))
    
    # Szenario 4: Kombinationstherapie (schnellster Anstieg)
    combo = initial_ri + (0.80 - initial_ri) * (1 - np.exp(-0.15 * np.arange(weeks)))
    
    return untreated, ssri, psycho, combo

weeks_array = np.arange(24)
untreated, ssri, psycho, combo = simulate_therapy_progression(weeks=24)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 5a: Therapie-Verlauf
ax = axes[0]
ax.plot(weeks_array, untreated, 'o-', color='gray', linewidth=2, markersize=5, label='Unbehandelt', linestyle='--')
ax.plot(weeks_array, ssri, 's-', color='#2980b9', linewidth=2.5, markersize=5, label='SSRI-Therapie')
ax.plot(weeks_array, psycho, '^-', color='#27ae60', linewidth=2.5, markersize=5, label='Psychotherapie')
ax.plot(weeks_array, combo, 'd-', color='#c0392b', linewidth=2.5, markersize=5, label='Kombinationstherapie')

# Markiere remission threshold
ax.axhline(y=0.70, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Remissions-Schwelle')
ax.axhline(y=0.41, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Depression-Baseline')

ax.set_xlabel('Wochen nach Therapie-Beginn', fontsize=11)
ax.set_ylabel('Erreichbarkeitsinzidenz (RI)', fontsize=11)
ax.set_title('Therapeutischer Fortschritt über Zeit', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 25)
ax.set_ylim(0.35, 0.85)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3, linestyle=':')

# Plot 5b: Remissions-Rate (Anteil der Patienten, die Schwelle überschreiten)
ax = axes[1]

# Simuliere Variabilität zwischen Patienten
np.random.seed(123)
n_patients = 1000

def simulate_patient_responses(base_trajectory, variability=0.05):
    """Simuliert unterschiedliche Therapie-Antworten zwischen Patienten"""
    # Jeder Patient hat leicht andere Response
    individual_responses = []
    for patient in range(n_patients):
        # Random noise und variabilität in Response-Geschwindigkeit
        noise = np.random.normal(0, variability, len(base_trajectory))
        speed_factor = np.random.normal(1.0, 0.2)
        
        # Modifiziere Trajektorie
        response = base_trajectory + noise
        individual_responses.append(response)
    
    return np.array(individual_responses)

# Remissions-Schwelle
remission_threshold = 0.70

responses_ssri = simulate_patient_responses(ssri, variability=0.04)
responses_psycho = simulate_patient_responses(psycho, variability=0.05)
responses_combo = simulate_patient_responses(combo, variability=0.03)

remission_ssri = np.sum(responses_ssri >= remission_threshold, axis=0) / n_patients * 100
remission_psycho = np.sum(responses_psycho >= remission_threshold, axis=0) / n_patients * 100
remission_combo = np.sum(responses_combo >= remission_threshold, axis=0) / n_patients * 100

ax.plot(weeks_array, remission_ssri, 's-', color='#2980b9', linewidth=2.5, markersize=5, label='SSRI')
ax.plot(weeks_array, remission_psycho, '^-', color='#27ae60', linewidth=2.5, markersize=5, label='Psychotherapie')
ax.plot(weeks_array, remission_combo, 'd-', color='#c0392b', linewidth=2.5, markersize=5, label='Kombination')

ax.fill_between(weeks_array, 0, remission_ssri, alpha=0.15, color='#2980b9')
ax.fill_between(weeks_array, 0, remission_psycho, alpha=0.15, color='#27ae60')
ax.fill_between(weeks_array, 0, remission_combo, alpha=0.15, color='#c0392b')

ax.set_xlabel('Wochen nach Therapie-Beginn', fontsize=11)
ax.set_ylabel('Remissions-Rate (%)', fontsize=11)
ax.set_title('Anteil der Patienten in Remission', fontsize=12, fontweight='bold')
ax.set_xlim(-1, 25)
ax.set_ylim(0, 100)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('/home/claude/plot_05_therapy.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_05_therapy.pdf")
plt.close()

# ============================================================================
# PLOT 6: Graph-Visualisierung (Small Networks)
# ============================================================================
print("Generiere Plot 6: Graph-Visualisierungen...")

def create_small_graph(n=50, edge_prob=0.2, seed=42):
    """Erstellt einen kleinen Graphen für Visualisierung"""
    np.random.seed(seed)
    adj = np.random.rand(n, n) < edge_prob
    np.fill_diagonal(adj, 0)
    return adj.astype(int)

small_normal = create_small_graph(n=50, edge_prob=0.25, seed=42)
small_depressed = create_small_graph(n=50, edge_prob=0.12, seed=42)

def visualize_graph(adj, title, ax, seed=42):
    """Visualisiert einen Graphen mit NetworkX"""
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    
    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=seed)
    
    # Knoten-Größe basiert auf Out-Grad
    node_sizes = np.array([G.out_degree(i) for i in range(len(G))]) * 30 + 100
    
    # Knoten-Farbe basiert auf Betweenness
    betweenness = nx.betweenness_centrality(G)
    node_colors = [betweenness[i] for i in range(len(G))]
    
    # Kanten
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                          arrows=True, arrowsize=8, alpha=0.3, 
                          width=0.5, arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Knoten
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, 
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap='viridis',
                                   vmin=0, vmax=0.15,
                                   edgecolors='black',
                                   linewidths=0.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    
    return nodes

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

nodes1 = visualize_graph(small_normal, 'Normales Netzwerk\n(Dichte=0.25)', axes[0], seed=42)
nodes2 = visualize_graph(small_depressed, 'Depressives Netzwerk\n(Dichte=0.12)', axes[1], seed=42)

# Gemeinsame Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(nodes1, cax=cbar_ax)
cbar.set_label('Betweenness-Zentralität', fontsize=10)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('/home/claude/plot_06_network_visualization.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_06_network_visualization.pdf")
plt.close()

# ============================================================================
# PLOT 7: Konnektivitäts-Matrix Heatmap
# ============================================================================
print("Generiere Plot 7: Konnektivitäts-Matrizen...")

# Cluster die Adjazenzmatrix zur besseren Visualisierung
def cluster_adjacency(adj, n_clusters=5):
    """Ordnet Knoten nach Graph-Struktur neu"""
    # Verwendet Out-Grad als einfaches Clustering-Maß
    degrees = np.sum(adj > 0, axis=1)
    order = np.argsort(degrees)
    return order

# Für kleine Graphen
n_vis = 100
small_adj_normal = G_normal[:n_vis, :n_vis]
small_adj_depressed = G_depressed[:n_vis, :n_vis]

order_normal = cluster_adjacency(small_adj_normal)
order_depressed = cluster_adjacency(small_adj_depressed)

# Reorder
reordered_normal = small_adj_normal[np.ix_(order_normal, order_normal)]
reordered_depressed = small_adj_depressed[np.ix_(order_depressed, order_depressed)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 7a: Normal
ax = axes[0]
im1 = ax.imshow(reordered_normal, cmap='Blues', aspect='auto', interpolation='nearest')
ax.set_title('Adjazenzmatrix: Normales Gehirn', fontsize=12, fontweight='bold')
ax.set_xlabel('Ziel-Knoten', fontsize=11)
ax.set_ylabel('Quell-Knoten', fontsize=11)
cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cbar1.set_label('Kantengewicht', fontsize=10)

# Plot 7b: Depressed
ax = axes[1]
im2 = ax.imshow(reordered_depressed, cmap='Reds', aspect='auto', interpolation='nearest')
ax.set_title('Adjazenzmatrix: Depressives Gehirn', fontsize=12, fontweight='bold')
ax.set_xlabel('Ziel-Knoten', fontsize=11)
ax.set_ylabel('Quell-Knoten', fontsize=11)
cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cbar2.set_label('Kantengewicht', fontsize=10)

plt.tight_layout()
plt.savefig('/home/claude/plot_07_adjacency_matrices.pdf', bbox_inches='tight', facecolor='white')
print("  ✓ Gespeichert: plot_07_adjacency_matrices.pdf")
plt.close()

print("\n" + "="*70)
print("✓ Alle Visualisierungen erfolgreich generiert!")
print("="*70)
print("\nGenerierte Dateien:")
print("  1. plot_01_reachability.pdf - Erreichbarkeitsinzidenz")
print("  2. plot_02_connectivity.pdf - Konnektivitäts-Analyse")
print("  3. plot_03_dynamics.pdf - Dynamische Aktivierungsmuster")
print("  4. plot_04_hub_vulnerability.pdf - Hub-Anfälligkeit")
print("  5. plot_05_therapy.pdf - Therapeutischer Fortschritt")
print("  6. plot_06_network_visualization.pdf - Netzwerk-Visualisierung")
print("  7. plot_07_adjacency_matrices.pdf - Adjazenzmatrizen")
print("="*70)
