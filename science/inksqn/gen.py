import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.gridspec as gridspec

# Set matplotlib parameters for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# ============= PLOT 1: Graph mit Knoten und Kanten =============
fig1, ax1 = plt.subplots(figsize=(12, 8))

G = nx.DiGraph()
nodes = ['Entscheidung', 'Handlung', 'Konsequenz 1', 'Konsequenz 2', 'Langzeitfolge']
positions = {
    'Entscheidung': (0, 3),
    'Handlung': (2, 3),
    'Konsequenz 1': (4, 4.5),
    'Konsequenz 2': (4, 1.5),
    'Langzeitfolge': (6, 3)
}

G.add_nodes_from(nodes)
edges = [
    ('Entscheidung', 'Handlung'),
    ('Handlung', 'Konsequenz 1'),
    ('Handlung', 'Konsequenz 2'),
    ('Konsequenz 1', 'Langzeitfolge'),
    ('Konsequenz 2', 'Langzeitfolge')
]
G.add_edges_from(edges)

# Draw nodes
nx.draw_networkx_nodes(G, positions, node_color='#2E86AB', node_size=3000, ax=ax1)

# Draw edges
nx.draw_networkx_edges(G, positions, arrowsize=30, arrowstyle='->', 
                       width=2.5, edge_color='#A23B72', ax=ax1,
                       connectionstyle="arc3,rad=0.1")

# Draw labels
nx.draw_networkx_labels(G, positions, font_size=10, font_weight='bold', 
                        font_color='white', ax=ax1)

ax1.set_title('Graphentheoretisches Modell: Knoten, Kanten und Kausalität', 
              fontsize=14, fontweight='bold', pad=20)
ax1.axis('off')
ax1.set_xlim(-1, 7)
ax1.set_ylim(0, 5.5)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#2E86AB', edgecolor='black', label='Knoten (Zustände/Ereignisse)'),
    mpatches.FancyArrow(0, 0, 0.5, 0, width=0.1, head_width=0.2, head_length=0.1, 
                       facecolor='#A23B72', edgecolor='black', label='Kanten (Konsequenzen)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('plot_1_graph_knoten_kanten.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============= PLOT 2: Gehirnaktivität und Konsequenzbewusstsein =============
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 2a: Gehirnregionen und Konsequenzbewusstsein
time_steps = np.linspace(0, 10, 100)

# Bewusstsein (allgemein)
consciousness = 0.6 + 0.3 * np.sin(time_steps / 2)

# Konsequenzbewusstsein (verzögert und niedriger)
consequence_awareness = 0.3 + 0.2 * np.sin(time_steps / 2 - 1.5)

# Blockade
blockade = 1 - consequence_awareness

ax2a.plot(time_steps, consciousness, linewidth=2.5, label='Allgemeines Bewusstsein', 
         color='#2E86AB', linestyle='-')
ax2a.plot(time_steps, consequence_awareness, linewidth=2.5, label='Konsequenzbewusstsein', 
         color='#F18F01', linestyle='--')
ax2a.fill_between(time_steps, consequence_awareness, consciousness, alpha=0.3, color='#C73E1D')
ax2a.set_xlabel('Zeitverlauf (willkürliche Einheiten)', fontsize=11)
ax2a.set_ylabel('Bewusstseinsniveau', fontsize=11)
ax2a.set_title('Diskrepanz: Bewusstsein vs. Konsequenzbewusstsein', fontsize=12, fontweight='bold')
ax2a.legend(loc='best', fontsize=10)
ax2a.grid(True, alpha=0.3)
ax2a.set_ylim(0, 1)

# Subplot 2b: Gehirnregionen (schematisch)
brain_regions = ['Präfrontaler\nKortex', 'Amygdala', 'Hippocampus', 'Motorischer\nKortex']
consequence_processing = [0.85, 0.45, 0.72, 0.38]
colors_brain = ['#2E86AB', '#F18F01', '#C73E1D', '#A23B72']

bars = ax2b.bar(brain_regions, consequence_processing, color=colors_brain, edgecolor='black', linewidth=1.5)
ax2b.set_ylabel('Effizienz der Konsequenzverarbeitung', fontsize=11)
ax2b.set_title('Neurale Grundlagen des Konsequenzbewusstseins', fontsize=12, fontweight='bold')
ax2b.set_ylim(0, 1)
ax2b.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax2b.text(bar.get_x() + bar.get_width()/2., height,
              f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('plot_2_gehirn_bewusstsein.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============= PLOT 3: Blockade als Funktion missverstandener Kanten =============
fig3, ax3 = plt.subplots(figsize=(11, 7))

n_kanten = np.arange(1, 11)
# Blockade = 1 - Produkt aller Verständigungen
# Annahmen: jede Kante hat 0.9 Verständigung, wenn sie nicht fehlt
blockade_level = []
for n in n_kanten:
    # Mit einer missverstandenen Kante
    blockade = 1 - (0.9 ** (n-1) * 0)  # Eine Kante ist 0
    blockade_level.append(blockade)

blockade_array = np.array([1 - 0.9**n for n in n_kanten])
perfect_chain = np.array([1 - 0.9**n for n in n_kanten])

ax3.plot(n_kanten, perfect_chain, marker='o', linewidth=2.5, markersize=8, 
         label='Mit einer missverstandenen Kante', color='#C73E1D')

# Mit allen verstandenen Kanten
perfect_understanding = 1 - (0.95 ** n_kanten)
ax3.plot(n_kanten, perfect_understanding, marker='s', linewidth=2.5, markersize=8, 
         label='Mit vollständigem Verständnis (0.95/Kante)', color='#2E86AB')

# Mit schlechterem Verständnis
poor_understanding = 1 - (0.7 ** n_kanten)
ax3.plot(n_kanten, poor_understanding, marker='^', linewidth=2.5, markersize=8,
         label='Mit schlechtem Verständnis (0.7/Kante)', color='#F18F01')

ax3.set_xlabel('Anzahl der Konsequenzketten (n)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Selbstblockade-Level', fontsize=12, fontweight='bold')
ax3.set_title('Multiplikative Natur der Selbstblockade:\nBockade = 1 - ∏ Kantenverständnis', 
              fontsize=13, fontweight='bold', pad=15)
ax3.legend(loc='best', fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1)
ax3.set_xlim(0.5, 10.5)

# Add annotation
ax3.annotate('Eine Kante mit 0% Verständnis\n= totale Blockade', 
            xy=(10, 1), xytext=(7, 0.7),
            arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=2),
            fontsize=10, color='#C73E1D', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('plot_3_blockade_multiplikativ.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============= PLOT 4: Natürliche Konsequenzen - Gravitation =============
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 4a: Fallbeschleunigung
time_fall = np.linspace(0, 3, 100)
g = 9.81  # m/s²
height = 100 - 0.5 * g * time_fall**2  # starting from 100m
velocity = -g * time_fall

ax4a.plot(time_fall, height, linewidth=2.5, color='#2E86AB', label='Höhe über Grund')
ax4a.axhline(y=0, color='#8B4513', linewidth=3, label='Boden')
ax4a.fill_between(time_fall, 0, height, alpha=0.2, color='#2E86AB')
ax4a.set_xlabel('Zeit (Sekunden)', fontsize=11)
ax4a.set_ylabel('Höhe (Meter)', fontsize=11)
ax4a.set_title('Fallgesetze: Die Unvermeidlichkeit der Konsequenz', fontsize=12, fontweight='bold')
ax4a.legend(loc='best', fontsize=10)
ax4a.grid(True, alpha=0.3)
ax4a.set_ylim(-5, 105)

# Add annotation
impact_time = np.sqrt(2 * 100 / g)
ax4a.annotate(f'Auftreffen bei t ≈ {impact_time:.1f}s\n(unvermeidlich)', 
             xy=(impact_time, 0), xytext=(impact_time-0.5, 30),
             arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=2),
             fontsize=10, color='#C73E1D', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5CC', alpha=0.8))

# Subplot 4b: Geschwindigkeit bei Aufprall
ax4b.plot(time_fall, -velocity, linewidth=2.5, color='#F18F01', label='Aufprallgeschwindigkeit')
ax4b.fill_between(time_fall, 0, -velocity, alpha=0.2, color='#F18F01')
ax4b.set_xlabel('Zeit (Sekunden)', fontsize=11)
ax4b.set_ylabel('Geschwindigkeit (m/s)', fontsize=11)
ax4b.set_title('Konsequenzintensität: Je länger der Fall, desto schwerer die Folge', 
              fontsize=12, fontweight='bold')
ax4b.legend(loc='best', fontsize=10)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_4_natuerliche_konsequenzen.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============= PLOT 5: Congruence zwischen Handlung und Konsequenz =============
fig5, ax5 = plt.subplots(figsize=(12, 7))

actions = ['Keine\nAnstrengung', 'Geringe\nAnstrengung', 'Moderate\nAnstrengung', 
           'Hohe\nAnstrengung', 'Maximalee\nAnstrengung']
x_pos = np.arange(len(actions))

# Konsequenzen
consequences_success = np.array([0.05, 0.25, 0.55, 0.85, 0.98])
consequences_failure = 1 - consequences_success

width = 0.6

bars1 = ax5.bar(x_pos, consequences_success, width, label='Erfolgswahrscheinlichkeit', 
               color='#2E86AB', edgecolor='black', linewidth=1.5)
bars2 = ax5.bar(x_pos, consequences_failure, width, bottom=consequences_success, 
               label='Misserfolgswahrscheinlichkeit', color='#C73E1D', edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Probabilität', fontsize=12, fontweight='bold')
ax5.set_xlabel('Grad der Anstrengung', fontsize=12, fontweight='bold')
ax5.set_title('Direkte Kausalität: Handlung und ihre wahrscheinliche Konsequenz', 
             fontsize=13, fontweight='bold', pad=15)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(actions)
ax5.set_ylim(0, 1)
ax5.legend(loc='upper left', fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    ax5.text(bar1.get_x() + bar1.get_width()/2., height1/2,
            f'{height1:.0%}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('plot_5_handlung_konsequenz_kausaliataet.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

# ============= PLOT 6: Konsequenzverzögerung und Bewusstseinsdefizit =============
fig6, ax6 = plt.subplots(figsize=(12, 7))

time_awareness = np.linspace(0, 20, 200)

# Zeitpunkt der Handlung
action_time = 0

# Sofortige physische Konsequenz (z.B. Schmerz beim Sturz)
immediate_consequence = np.where(time_awareness >= action_time, 1, 0)

# Verzögerte biologische Konsequenz (z.B. Verletzung heilt oder wird schlimmer)
delayed_consequence = np.where(time_awareness >= action_time + 2, 
                               0.5 + 0.3 * np.cos((time_awareness - 2) / 3), 0)
delayed_consequence = np.clip(delayed_consequence, 0, 1)

# Sehr verzögerte psychische/soziale Konsequenz
very_delayed = np.where(time_awareness >= action_time + 5,
                        0.4 + 0.2 * np.sin((time_awareness - 5) / 4), 0)
very_delayed = np.clip(very_delayed, 0, 1)

# Bewusstseinsgrad über Konsequenz
awareness_level = np.minimum(immediate_consequence * 1.0 + 
                             delayed_consequence * 0.6 + 
                             very_delayed * 0.2, 1)

ax6.plot(time_awareness, immediate_consequence, linewidth=2.5, label='Sofortige Konsequenz',
        color='#C73E1D', linestyle='-')
ax6.plot(time_awareness, delayed_consequence, linewidth=2.5, label='Verzögerte Konsequenz (2-5s)',
        color='#F18F01', linestyle='--')
ax6.plot(time_awareness, very_delayed, linewidth=2.5, label='Langfristige Konsequenz (5+s)',
        color='#A23B72', linestyle=':')
ax6.plot(time_awareness, awareness_level, linewidth=3, label='Gesamtbewusstseinsniveau',
        color='#2E86AB', linestyle='-', marker='o', markersize=4, markevery=10)

ax6.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Zeitpunkt der Handlung')

ax6.set_xlabel('Zeit nach der Handlung (Sekunden)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Konsequenzbewusstsein / Intensität', fontsize=12, fontweight='bold')
ax6.set_title('Konsequenzverzögerung: Warum das Bewusstsein scheitert', 
             fontsize=13, fontweight='bold', pad=15)
ax6.legend(loc='best', fontsize=10, ncol=2)
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0, 1.2)

# Add shaded regions for awareness gaps
ax6.axvspan(0, 2, alpha=0.1, color='yellow', label='Bewusstseins-Fenster')

plt.tight_layout()
plt.savefig('plot_6_konsequenzverzögerung.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.close()

print("Alle Plots erfolgreich generiert:")
print("✓ plot_1_graph_knoten_kanten.pdf")
print("✓ plot_2_gehirn_bewusstsein.pdf")
print("✓ plot_3_blockade_multiplikativ.pdf")
print("✓ plot_4_natuerliche_konsequenzen.pdf")
print("✓ plot_5_handlung_konsequenz_kausaliataet.pdf")
print("✓ plot_6_konsequenzverzögerung.pdf")
