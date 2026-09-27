import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.figsize": (6.3, 4.2),
})

# ---------- Plot 1: HSM Funktionsblock-Architektur ----------
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.axis("off")
blocks = [
    ("HSM Core\n(TriCore Lockstep)", 0.5, 0.92, "#19468c"),
    ("TRNG", 0.12, 0.72, "#b4321e"),
    ("Sig-Engine\n(\u03c3, \u03a3)", 0.38, 0.72, "#1e6432"),
    ("Modular-ALU\n(Z_p Feldarithmetik)", 0.64, 0.72, "#1e6432"),
    ("AES-CMAC\nBoot-MAC", 0.90, 0.72, "#1e6432"),
    ("KeyStore\n(Flash, versiegelt)", 0.12, 0.50, "#3c3c46"),
    ("HSM_SignatureCipher", 0.38, 0.50, "#19468c"),
    ("HSM_ModularCipher", 0.64, 0.50, "#19468c"),
    ("HSM_SecureBoot_Verify", 0.90, 0.50, "#19468c"),
    ("HSI Interface\n(zu PFLASH/DFLASH)", 0.5, 0.28, "#b4321e"),
    ("Applikations-Kerne\n(TriCore CPU0..CPU5)", 0.5, 0.08, "#3c3c46"),
]
for label, x, y, color in blocks:
    ax.add_patch(plt.Rectangle((x-0.15, y-0.06), 0.30, 0.11, fill=True,
                                facecolor=color, edgecolor="black", alpha=0.85,
                                transform=ax.transAxes))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8.5,
            transform=ax.transAxes, fontweight="bold")
edges = [(0.5,0.86,0.12,0.78),(0.5,0.86,0.38,0.78),(0.5,0.86,0.64,0.78),(0.5,0.86,0.90,0.78),
         (0.12,0.66,0.12,0.56),(0.38,0.66,0.38,0.56),(0.64,0.66,0.64,0.56),(0.90,0.66,0.90,0.56),
         (0.5,0.44,0.5,0.34),(0.5,0.22,0.5,0.14)]
for x1,y1,x2,y2 in edges:
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.1))
ax.set_title("Architektur des HSM-Kryptographiekerns auf dem Infineon TC3x")
plt.tight_layout()
plt.savefig("plot1_hsm_architektur.pdf")
plt.close()

# ---------- Plot 2: Taktzyklen je HSM-Funktion ----------
fig, ax = plt.subplots()
funcs = ["TRNG_Read","SigCipher\nKeyGen","SigCipher\nSign","SigCipher\nVerify",
         "ModCipher\nEncrypt","ModCipher\nDecrypt","SubgraphSig\nDerive","SecureBoot\nVerify"]
cycles_n8 = [120, 2400, 3100, 3050, 980, 1010, 4200, 3300]
cycles_n16 = [120, 9100, 11800, 11650, 1850, 1900, 16800, 12700]
x = np.arange(len(funcs))
w = 0.35
ax.bar(x-w/2, cycles_n8, w, label="n = 8 (Blockgr\u00f6\u00dfe)", color="#19468c")
ax.bar(x+w/2, cycles_n16, w, label="n = 16 (Blockgr\u00f6\u00dfe)", color="#b4321e")
ax.set_xticks(x); ax.set_xticklabels(funcs, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Taktzyklen (TriCore, 300 MHz, gesch\u00e4tzt)")
ax.set_title("Gesch\u00e4tzte Taktzyklen je HSM-Funktion")
ax.legend()
plt.tight_layout()
plt.savefig("plot2_taktzyklen.pdf")
plt.close()

# ---------- Plot 3: Pipeline-Latenz der Signaturberechnung vs n ----------
fig, ax = plt.subplots()
n = np.arange(2, 33)
latency_sw = 0.045 * n**3
latency_hw = 0.006 * n**3 + 0.5*n
ax.plot(n, latency_sw, "o-", color="#b4321e", label="Software (TriCore, O(n^3))")
ax.plot(n, latency_hw, "s-", color="#1e6432", label="HSM-Hardwarepipeline (O(n^3), parallelisiert)")
ax.set_xlabel("Blockgr\u00f6\u00dfe n")
ax.set_ylabel("Latenz [\u00b5s]")
ax.set_title("Latenz der Signaturpipeline: Software vs. HSM-Hardware")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_pipeline_latenz.pdf")
plt.close()

# ---------- Plot 4: Schl\u00fcsselraum-Vergleich ----------
fig, ax = plt.subplots()
schemes = ["wchiffre\nZ_32^*", "Sig-Cipher\n(n=8,p~2^17)","Sig-Cipher\n(n=16,p~2^33)",
           "AES-128","RSA-2048","Lattice-KEM\n(lattice.tex)"]
bits = [4, 17, 33, 128, 112, 256]
colors = ["#b4321e","#1e6432","#1e6432","#3c3c46","#3c3c46","#19468c"]
ax.bar(schemes, bits, color=colors)
ax.set_ylabel("Effektive Sicherheit [Bit, log-\u00e4quivalent]")
ax.set_title("Schl\u00fcsselraum-/Sicherheitsniveau im Vergleich")
plt.xticks(rotation=20, ha="right", fontsize=8.5)
plt.tight_layout()
plt.savefig("plot4_schluesselraum.pdf")
plt.close()

# ---------- Plot 5: Energieverbrauch je Operation ----------
fig, ax = plt.subplots()
ops = ["TRNG","SigGen","SigSign","SigVerify","ModEnc","ModDec","BootVerify"]
energy_uj = [0.8, 14.2, 18.5, 18.1, 6.0, 6.1, 19.8]
ax.barh(ops, energy_uj, color="#19468c")
ax.set_xlabel("Energie pro Aufruf [\u00b5J], gesch\u00e4tzt bei 1.2 V Kernspannung")
ax.set_title("Gesch\u00e4tzter Energieverbrauch der HSM-Funktionen")
plt.tight_layout()
plt.savefig("plot5_energieverbrauch.pdf")
plt.close()

# ---------- Plot 6: Sicherheitsmarge Brute-Force ----------
fig, ax = plt.subplots()
p_bits = np.arange(8, 65, 2)
years_classical = 2**(p_bits-1) / (1e10 * 3600*24*365)
years_grover = 2**((p_bits-1)/2) / (1e9 * 3600*24*365)
ax.semilogy(p_bits, years_classical, color="#b4321e", label="Klassischer Brute-Force (10^10 Ops/s)")
ax.semilogy(p_bits, years_grover, color="#19468c", label="Grover-Suche (Quantenangriff, 10^9 Ops/s)")
ax.axhline(1, color="gray", ls="--", lw=0.8)
ax.text(10, 1.3, "1 Jahr", fontsize=8, color="gray")
ax.set_xlabel("Bitl\u00e4nge des Modulus p")
ax.set_ylabel("Erwartete Angriffsdauer [Jahre]")
ax.set_title("Sicherheitsmarge der Sig-Cipher-Parameter gegen Brute-Force")
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig("plot6_sicherheitsmarge.pdf")
plt.close()

# ---------- Plot 7: Flash-/Speicherbedarf ----------
fig, ax = plt.subplots()
labels = ["KeyStore\n(versiegelt)","Sig-Engine\nMikrocode","Modular-ALU\nMikrocode","Boot-MAC\nTabellen","TRNG\nKonfiguration","Reserve"]
sizes = [8, 12, 6, 4, 1, 5]
colors = plt.cm.Blues(np.linspace(0.4,0.9,len(labels)))
ax.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors, textprops={"fontsize":8})
ax.set_title("Aufteilung des HSM-internen Flash-Speichers (36 KB gesamt)")
plt.tight_layout()
plt.savefig("plot7_flash_aufteilung.pdf")
plt.close()

# ---------- Plot 8: Roadmap / Ausblick ----------
fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.axis("off")
milestones = [
    (0, "v1.0\nSig-Cipher +\nModCipher Core"),
    (1, "v1.1\nSecureBoot +\nKeyStore-Lock"),
    (2, "v1.2\nLattice-KEM\nIntegration"),
    (3, "v2.0\nSubgraphSig\nFirmware-Attest."),
    (4, "v2.1\nSeitenkanal-\nh\u00e4rtung (DPA/CPA)"),
    (5, "v3.0\nVollst.\nPQC-Migration"),
]
ax.plot([m[0] for m in milestones], [0]*len(milestones), color="#3c3c46", lw=2, zorder=1)
for i,(x,label) in enumerate(milestones):
    ax.scatter([x],[0], s=140, color="#19468c" if i%2==0 else "#b4321e", zorder=2)
    y_off = 0.15 if i%2==0 else -0.15
    va = "bottom" if i%2==0 else "top"
    ax.annotate(label, (x,0), xytext=(x, y_off), ha="center", va=va, fontsize=8.3,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.4, 0.4)
ax.set_title("Entwicklungs-Roadmap des HSM-Kryptographiekerns")
plt.tight_layout()
plt.savefig("plot8_roadmap.pdf")
plt.close()

print("done")

# ---------- Plot 9: Secure-Boot-Verifikationszeit vs. Flashgröße ----------
fig, ax = plt.subplots()
flash_mb = np.array([0.5, 1, 2, 4, 8, 16])
cycles_per_block = 16800  # n=16, HSM_SubgraphSig_Derive
block_bytes = 32          # n=16 -> 256 Bit
freq_hz = 300e6
boot_ms = (flash_mb * 1024 * 1024 / block_bytes) * cycles_per_block / freq_hz * 1000
ax.plot(flash_mb, boot_ms, "o-", color="#19468c", lw=2)
for x,y in zip(flash_mb, boot_ms):
    ax.annotate(f"{y:.0f} ms", (x,y), textcoords="offset points", xytext=(0,8),
                ha="center", fontsize=8)
ax.axvline(4, color="#b4321e", ls="--", lw=1)
ax.text(4.15, ax.get_ylim()[1]*0.05, "Radar-ECU\n(4 MB PFLASH)", color="#b4321e", fontsize=8.5)
ax.set_xlabel("Programm-Flash-Größe [MB]")
ax.set_ylabel("Verifikationszeit HSM_SubgraphSig_Derive [ms]")
ax.set_title("Secure-Boot-Verifikationszeit in Abhängigkeit der Flashgröße")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot9_bootzeit_flash.pdf")
plt.close()
print("plot9 done")

# ---------- Plot 10: Parallelisierungsgrad k vs. Verifikationszeit ----------
fig, ax = plt.subplots()
k = np.array([1,2,4,8,16,32,64])
T_seq_ms = 7340.0
T_k = T_seq_ms / k
ax.plot(k, T_k, "o-", color="#19468c", lw=2)
ax.axhline(500, color="#b4321e", ls="--", lw=1)
ax.text(1.2, 560, "Radar-ECU-Zeitbudget (500 ms)", color="#b4321e", fontsize=8.5)
ax.axvline(16, color="#1e6432", ls=":", lw=1)
ax.text(16.5, 4500, "k = 16\n(vorgeschlagen)", color="#1e6432", fontsize=8.5)
for x,y in zip(k, T_k):
    ax.annotate(f"{y:.0f} ms", (x,y), textcoords="offset points", xytext=(0,8),
                ha="center", fontsize=7.5)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Parallelisierungsgrad k (Anzahl Sig-Engine-Instanzen)")
ax.set_ylabel("Verifikationszeit $T_{verify}$ [ms], log-Skala")
ax.set_title("Verifikationszeit der parallelisierten Sig-Engine-Architektur")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("plot10_parallel_speedup.pdf")
plt.close()

# ---------- Plot 11: Silizium-/Flächenkosten vs. k ----------
fig, ax = plt.subplots()
k2 = np.array([1,2,4,8,16,32,64])
area_mm2 = 0.18 * k2 + 0.05*np.log2(np.maximum(k2,1))  # linear core area + small XOR-tree overhead
ax.bar([str(x) for x in k2], area_mm2, color="#3c3c46")
ax.set_xlabel("Parallelisierungsgrad k")
ax.set_ylabel("Geschätzte zusätzliche Siliziumfläche [mm²]")
ax.set_title("Geschätzter Flächenmehrbedarf der parallelen Sig-Engine-Kacheln")
plt.tight_layout()
plt.savefig("plot11_flaechenkosten.pdf")
plt.close()

print("plot10, plot11 done")
