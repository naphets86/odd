import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Plot 1: API-Taxonomie-Baum ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

colors = {
    'root':    '#c0392b',
    'open':    '#2980b9',
    'internal':'#27ae60',
    'partner': '#8e44ad',
    'proto':   '#e67e22',
    'leaf':    '#7f8c8d',
}

def box(ax, x, y, w, h, label, color, fontsize=9):
    rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.08", linewidth=1.2,
        edgecolor='#2c3e50', facecolor=color, alpha=0.88)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', wrap=True)

def line(ax, x1,y1,x2,y2):
    ax.plot([x1,x2],[y1,y2], color='#555', lw=1.2, zorder=0)

# root
box(ax, 1.1, 4.5, 1.6, 0.7, 'API', colors['root'], fontsize=11)

# level-1
nodes_l1 = [
    (3.2, 7.5, 'Open API',    colors['open']),
    (3.2, 4.5, 'Internal\nAPI', colors['internal']),
    (3.2, 1.5, 'Partner\nAPI', colors['partner']),
]
for x,y,lbl,c in nodes_l1:
    box(ax, x, y, 1.8, 0.65, lbl, c)
    line(ax, 1.9, 4.5, x-0.9, y)

# level-2 (protocols / sub-types)
nodes_l2 = [
    (5.8, 8.3, 'REST',       colors['proto'], 7.5),
    (5.8, 7.3, 'SOAP',       colors['proto'], 7.5),
    (5.8, 6.3, 'GraphQL',    colors['proto'], 7.5),
    (5.8, 4.9, 'B2B',        colors['proto'], 4.5),
    (5.8, 4.1, 'Frontend→\nBackend', colors['proto'], 4.5),
    (5.8, 3.3, 'Service→\nDB', colors['proto'], 4.5),
    (5.8, 2.0, 'Affiliate',  colors['proto'], 1.5),
    (5.8, 1.0, 'Data\nSharing', colors['proto'], 1.5),
]
for x,y,lbl,c,py in nodes_l2:
    box(ax, x, y, 1.7, 0.55, lbl, c, fontsize=8)
    line(ax, 4.1, py, x-0.85, y)

# level-3 (use cases) – just representative
usecases = [
    (8.6, 8.6, 'Weather\nData',    5.8, 8.3),
    (8.6, 8.1, 'Login\nSystem',    5.8, 8.3),
    (8.6, 7.6, 'Product\nFetch',   5.8, 8.3),
    (8.6, 7.1, 'Bank\nTransfer',   5.8, 7.3),
    (8.6, 6.6, 'Insurance\nClaim', 5.8, 7.3),
    (8.6, 6.1, 'Facebook\nFeed',   5.8, 6.3),
    (8.6, 5.6, 'GitHub\nStats',    5.8, 6.3),
    (8.6, 5.1, 'Payment\nSync',    5.8, 4.9),
    (8.6, 4.6, 'Token\nVerify',    5.8, 4.9),
    (8.6, 4.1, 'Login\nRequest',   5.8, 4.1),
    (8.6, 3.6, 'Live\nSearch',     5.8, 4.1),
    (8.6, 3.1, 'User\nInsert',     5.8, 3.3),
    (8.6, 2.6, 'Hotel\nBooking',   5.8, 2.0),
    (8.6, 2.1, 'Airline\nData',    5.8, 2.0),
    (8.6, 1.6, 'Health\nRecords',  5.8, 1.0),
    (8.6, 1.1, 'Finance\nData',    5.8, 1.0),
]
for x,y,lbl,sx,sy in usecases:
    box(ax, x, y, 1.55, 0.38, lbl, colors['leaf'], fontsize=7)
    line(ax, 6.65, sy, x-0.775, y)

ax.set_title('API-Taxonomie: Typen und Anwendungsfälle', fontsize=13,
             fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('plot_api_taxonomy.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 1 done")

