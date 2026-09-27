import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import comb

def p_correct(n, p):
    k_max = (n-1)//2
    return sum(comb(n,k) * p**k * (1-p)**(n-k) for k in range(0, k_max+1))

p = np.linspace(0, 1, 400)
fig, ax = plt.subplots(figsize=(6,4.2))
for n in [1,3,5,7]:
    y = [p_correct(n, pi) for pi in p]
    ax.plot(p, y, label=f"n={n}")
ax.set_xlabel("Kompromittierungswahrscheinlichkeit $p$ pro Kanal")
ax.set_ylabel(r"$P_{\mathrm{korrekt}}(n,p)$")
ax.set_title("Funktionale Korrektheit bei Mehrheitsentscheidung (TMR-artig)")
ax.legend(title="Kanäle $n$")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plot1_korrektheit.pdf")
print("done")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import comb

def p_correct(n, p):
    k_max = (n-1)//2
    return sum(comb(n,k) * p**k * (1-p)**(n-k) for k in range(0, k_max+1))

def p_secure(n, p):
    # Wahrscheinlichkeit, dass KEIN Kanal kompromittiert ist (vollstaendige Integritaet)
    return (1-p)**n

p = np.linspace(0, 1, 400)
n = 5
fig, ax = plt.subplots(figsize=(6,4.2))
ax.plot(p, [p_correct(n,pi) for pi in p], label=r"$P_{\mathrm{Safety}}$ (Funktion korrekt, $n=5$)", color="#19468c")
ax.plot(p, [p_secure(n,pi) for pi in p], label=r"$P_{\mathrm{Security}}$ (kein Kanal kompromittiert, $n=5$)", color="#b4321e", linestyle="--")
ax.fill_between(p, [p_secure(n,pi) for pi in p], [p_correct(n,pi) for pi in p],
                 where=[p_correct(n,pi) >= p_secure(n,pi) for pi in p],
                 color="#1e6432", alpha=0.15, label="Inkonsistenzbereich (Safety$\\,>\\,$Security)")
ax.set_xlabel("Kompromittierungswahrscheinlichkeit $p$ pro Kanal")
ax.set_ylabel("Wahrscheinlichkeit")
ax.set_title("Divergenz von Safety- und Security-Garantie ($n=5$)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plot2_divergenz.pdf")
print("done")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import comb

def p_correct(n, p):
    k_max = (n-1)//2
    return sum(comb(n,k) * p**k * (1-p)**(n-k) for k in range(0, k_max+1))

target = 0.999999
ps = [0.05, 0.1, 0.2, 0.3]
fig, ax = plt.subplots(figsize=(6,4.2))
for p in ps:
    ns = list(range(1, 26, 2))
    ys = [p_correct(n,p) for n in ns]
    ax.plot(ns, ys, marker="o", markersize=3, label=f"$p={p}$")
ax.axhline(target, color="gray", linestyle=":", linewidth=1, label=f"Ziel $={target}$")
ax.set_xlabel("Redundanzgrad $n$ (ungerade)")
ax.set_ylabel(r"$P_{\mathrm{korrekt}}(n,p)$")
ax.set_title("Minimaler Redundanzgrad zur Erreichung eines Safety-Ziels")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plot3_redundanzgrad.pdf")
print("done")
