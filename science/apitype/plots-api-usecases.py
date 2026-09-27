import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Heatmap: API-Typ × Anwendungsfall
api_types = ['REST', 'SOAP', 'GraphQL', 'B2B', 'Frontend→\nBackend', 'Service→DB', 'Affiliate', 'DataSharing']
use_cases = ['Auth', 'Daten', 'Payment', 'Analytics', 'Integration', 'Echtzeit']

# Relevanz-Matrix (0-3)
matrix = np.array([
    [3,3,2,1,2,3],  # REST
    [2,2,3,1,2,1],  # SOAP
    [2,3,1,2,2,3],  # GraphQL
    [1,2,3,2,3,1],  # B2B
    [3,2,1,2,1,3],  # Frontend→Backend
    [2,3,1,3,2,1],  # Service→DB
    [1,2,2,3,2,1],  # Affiliate
    [2,3,1,2,3,2],  # DataSharing
])

im = axes[0].imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)
axes[0].set_xticks(range(len(use_cases)))
axes[0].set_yticks(range(len(api_types)))
axes[0].set_xticklabels(use_cases, rotation=35, ha='right', fontsize=9)
axes[0].set_yticklabels(api_types, fontsize=9)
axes[0].set_title('Relevanz: API-Typ × Anwendungsfall\n(0=gering, 3=hoch)', fontweight='bold', fontsize=10)
for i in range(len(api_types)):
    for j in range(len(use_cases)):
        axes[0].text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=10,
                    color='black' if matrix[i,j] < 2.5 else 'white')
plt.colorbar(im, ax=axes[0], label='Relevanz')

# Radar chart: Eigenschaften der API-Typen
categories = ['Leistung', 'Sicherheit', 'Flexibilität', 'Standardisierung', 'Echtzeit']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

api_profiles = {
    'REST':    [4,3,5,4,3],
    'SOAP':    [3,5,3,5,2],
    'GraphQL': [4,3,5,3,4],
    'B2B':     [3,4,3,5,3],
}
colors_r = ['#2980b9','#e74c3c','#27ae60','#e67e22']
ax = axes[1]
ax = plt.subplot(122, polar=True)
for (name, vals), col in zip(api_profiles.items(), colors_r):
    v = vals + vals[:1]
    ax.plot(angles, v, 'o-', lw=2, color=col, label=name)
    ax.fill(angles, v, alpha=0.08, color=col)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0,5)
ax.set_title('API-Eigenschaftsprofil\n(Radar)', fontweight='bold', fontsize=10, pad=15)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle('API-Anwendungsfälle und Eigenschaftsprofile', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_api_usecases.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 5 done")

