import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

n = np.arange(1, 60)

# Complexity comparison
ax = axes[0]
ax.plot(n, n**3,        label='$O(n^3)$ Subgraph', color='#2980b9', lw=2.5)
ax.plot(n, n**2*np.log(n), label='$O(n^2 \\log n)$', color='#27ae60', lw=2, ls='--')
ax.plot(n, n**2,        label='$O(n^2)$ Signatur', color='#e67e22', lw=2, ls=':')
ax.plot(n[1:], 6*n[1:]**(1.5), label='$O(n! \\cdot n^2)$ naiv (skaliert)', color='#c0392b', lw=1.5, ls='-.')
ax.set_xlabel('Anzahl API-Endpunkte $n$')
ax.set_ylabel('Operationen (normiert)')
ax.set_title('Komplexitätsvergleich', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.set_ylim(1, 1e6)
ax.grid(alpha=0.3)

# LCS matching matrix visualization
ax2 = axes[1]
sigA = [5, 18, 36, 48]
sigB = [5, 22, 36, 48, 64]
m, k = len(sigA), len(sigB)
dp = np.zeros((m+1, k+1), dtype=int)
for i in range(1, m+1):
    for j in range(1, k+1):
        if sigA[i-1] == sigB[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

im = ax2.imshow(dp, cmap='Blues', aspect='auto')
ax2.set_xticks(range(k+1))
ax2.set_yticks(range(m+1))
ax2.set_xticklabels(['-']+[str(s) for s in sigB], fontsize=8)
ax2.set_yticklabels(['-']+[str(s) for s in sigA], fontsize=8)
ax2.set_xlabel('Signaturen B (API-Netz B)')
ax2.set_ylabel('Signaturen A (API-Netz A)')
ax2.set_title('LCS-DP-Matrix: Signatur-Matching', fontsize=11, fontweight='bold')
for i in range(m+1):
    for j in range(k+1):
        ax2.text(j, i, str(dp[i,j]), ha='center', va='center', fontsize=9,
                 color='white' if dp[i,j]>2 else 'black')
plt.colorbar(im, ax=ax2, label='LCS-Länge')

fig.suptitle('Komplexitätsanalyse und LCS-Matching für API-Graphen', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_api_complexity.pdf', bbox_inches='tight', dpi=150)
plt.close()
print("Plot 3 done")

