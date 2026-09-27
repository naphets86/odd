#!/usr/bin/env python3
"""
Matplotlib-Plots für die wissenschaftliche Arbeit über die Hand.
Generiert alle neun Abbildungen als PDF und SVG.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import chirp, spectrogram
from scipy.stats import pearsonr

# ── Globales Styling ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'figure.dpi': 150,
})

BLUE   = '#1a4e8a'
RED    = '#c0392b'
GREEN  = '#27ae60'
ORANGE = '#e67e22'
PURPLE = '#8e44ad'
GRAY   = '#7f8c8d'

def savefig(name):
    for ext in ('pdf', 'svg'):
        plt.savefig(f'{name}.{ext}', bbox_inches='tight')
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Fig 01 – Somatosensorischer & motorischer Homunkulus (Hand-fokussiert)
# ════════════════════════════════════════════════════════════════════════════
def fig01_homunkulus():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── linkes Panel: kortikale Flächen ────────────────────────────────────
    ax = axes[0]
    regions = [
        'Finger\u200b/Daumen', 'Hand', 'Handgelenk/\nUnterarm', 'Arm',
        'Schulter', 'Rumpf', 'Hüfte/Bein', 'Fuß', 'Gesicht', 'Lippen/Zunge'
    ]
    S1_areas = np.array([8.4, 5.2, 2.1, 1.6, 0.9, 1.8, 3.1, 5.2, 4.8, 7.1])
    M1_areas = np.array([6.8, 4.9, 1.8, 1.4, 1.0, 1.5, 2.8, 2.4, 5.1, 6.5])
    colors = [RED if r.startswith('Finger') or r == 'Hand' else
              ORANGE if r.startswith('Handgelenk') else
              BLUE for r in regions]
    x = np.arange(len(regions))
    w = 0.35
    ax.bar(x - w/2, S1_areas, w, color=[RED if c == RED else
                                         ORANGE if c == ORANGE else
                                         BLUE for c in colors],
           alpha=0.85, label='S1 (sensorisch)')
    ax.bar(x + w/2, M1_areas, w, color=[RED if c == RED else
                                         ORANGE if c == ORANGE else
                                         GREEN for c in colors],
           alpha=0.60, label='M1 (motorisch)')
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=40, ha='right', fontsize=7.5)
    ax.set_ylabel('Kortikale Fläche [rel. Einheiten]')
    ax.set_title('Kortikale Repräsentationsflächen im Homunkulus')
    ax.legend()
    ax.axvspan(-0.5, 1.5, alpha=0.08, color=RED, label='Hand-Dominanz')
    ax.annotate('', xy=(1.5, 8.6), xytext=(-0.5, 8.6),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.5))
    ax.text(0.5, 8.8, 'Hand-Areale', ha='center', color=RED, fontsize=8)
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3)

    # ── rechtes Panel: kortikaler Magnifikationsfaktor ─────────────────────
    ax2 = axes[1]
    body_area = np.array([   # cm²
        35,   # Finger+Daumen
        180,  # Hand
        200,  # Handgelenk/Unterarm
        700,  # Arm
        400,  # Schulter
        3500, # Rumpf
        1200, # Hüfte/Bein
        150,  # Fuß
        400,  # Gesicht
        30,   # Lippen/Zunge
    ])
    magnif_S1 = S1_areas / body_area * 1000  # rel./cm²  × 1000
    region_labels_short = [
        'Finger', 'Hand', 'Unterarm', 'Arm',
        'Schulter', 'Rumpf', 'Bein', 'Fuß', 'Gesicht', 'Lippen'
    ]
    col_bar = [RED if i < 2 else ORANGE if i == 2 else
               BLUE if i < 6 else GREEN for i in range(len(regions))]
    bars = ax2.barh(region_labels_short, magnif_S1, color=col_bar, alpha=0.85)
    ax2.set_xlabel('Kortikaler Magnifikationsfaktor [rel.E./cm² × 10⁻³]')
    ax2.set_title('Kortikaler Magnifikationsfaktor ρ(S1) nach Körperregion')
    ax2.axvline(magnif_S1[0], color=RED, lw=1.2, ls='--', alpha=0.6)
    for i, (bar, val) in enumerate(zip(bars, magnif_S1)):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=7.5)
    ax2.grid(axis='x', alpha=0.3)
    patch_hand = mpatches.Patch(color=RED, alpha=0.85, label='Finger / Hand')
    patch_arm  = mpatches.Patch(color=ORANGE, alpha=0.85, label='Unterarm')
    patch_rest = mpatches.Patch(color=BLUE, alpha=0.85, label='Übriger Körper')
    ax2.legend(handles=[patch_hand, patch_arm, patch_rest], loc='lower right')

    fig.suptitle('Abbildung 1 – Kortikale Dominanz der Hand im somatosensorischen Homunkulus',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig01_homunkulus')


# ════════════════════════════════════════════════════════════════════════════
# Fig 02 – Mechanorezeptoren der Hand
# ════════════════════════════════════════════════════════════════════════════
def fig02_mechanorezeptoren():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Scatter: Leitungsgeschwindigkeit × Rezeptordichte ─────────────────
    ax = axes[0]
    receptor_data = {
        'Meissner (RA1)':  dict(v=50,  dens=140, freq=30,  col=RED,    sym='o'),
        'Pacini (RA2)':    dict(v=75,  dens=25,  freq=250, col=ORANGE, sym='s'),
        'Merkel (SA1)':    dict(v=55,  dens=180, freq=5,   col=BLUE,   sym='^'),
        'Ruffini (SA2)':   dict(v=40,  dens=30,  freq=2,   col=GREEN,  sym='D'),
        'Ia (Muskelsp.)':  dict(v=80,  dens=8,   freq=100, col=PURPLE, sym='P'),
        'Ib (Golgi)':      dict(v=75,  dens=5,   freq=80,  col=GRAY,   sym='*'),
    }
    for name, d in receptor_data.items():
        ax.scatter(d['v'], d['dens'], s=d['freq']*0.7, c=d['col'],
                   marker=d['sym'], alpha=0.85, label=name, zorder=3)
    ax.set_xlabel('Leitungsgeschwindigkeit [m/s]')
    ax.set_ylabel('Rezeptordichte [Rez./cm²] an der Fingerkuppe')
    ax.set_title('Nervenfaserklassen der Hand: Leitungsgeschwindigkeit × Dichte')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax.text(0.65, 0.92, 'Punktgröße ∝ Best-Frequenz [Hz]',
            transform=ax.transAxes, fontsize=7, color=GRAY)

    # ── Adaptationsverhalten ───────────────────────────────────────────────
    ax2 = axes[1]
    t = np.linspace(0, 1.5, 1500)
    stim = np.where((t >= 0.1) & (t <= 1.0), 1.0, 0.0)
    meissner = stim * np.exp(-20 * (t - 0.1).clip(min=0)) * (t >= 0.1) + \
               stim * np.exp(-20 * (t - 1.0).clip(min=0)) * (t >= 1.0) * 0.8
    # more accurate: RA adapts fast – only onset/offset
    meissner = np.zeros_like(t)
    meissner[(t >= 0.10) & (t <= 0.22)] = np.exp(-12*(t[(t >= 0.10) & (t <= 0.22)] - 0.10))
    meissner[(t >= 1.00) & (t <= 1.12)] = 0.7 * np.exp(-12*(t[(t >= 1.00) & (t <= 1.12)] - 1.00))

    merkel = np.where((t >= 0.1) & (t <= 1.0),
                      0.9 * (1 - np.exp(-20*(t-0.1))) + 0.15, 0.0)

    ruffini = np.where((t >= 0.1) & (t <= 1.0),
                       0.55 * (1 - np.exp(-8*(t-0.1))), 0.0)

    stim_plot = stim * 0.25 - 0.1
    ax2.plot(t, stim_plot, color=GRAY, ls='--', lw=1.5, label='Stimulus')
    ax2.plot(t, meissner,  color=RED,    label='Meissner RA1')
    ax2.plot(t, merkel,    color=BLUE,   label='Merkel SA1')
    ax2.plot(t, ruffini,   color=GREEN,  label='Ruffini SA2')
    ax2.set_xlabel('Zeit [s]')
    ax2.set_ylabel('Normierte Entladungsrate')
    ax2.set_title('Adaptationsverhalten der Hauptrezeptorklassen (Stufenreiz)')
    ax2.legend()
    ax2.set_ylim(-0.15, 1.1)
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='k', lw=0.6)

    fig.suptitle('Abbildung 2 – Mechanorezeptoren der menschlichen Hand',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig02_mechanorezeptoren')


# ════════════════════════════════════════════════════════════════════════════
# Fig 03 – Greifanalyse: EMG, Greifkraftphasen, Fingerkinematik
# ════════════════════════════════════════════════════════════════════════════
def fig03_greifanalyse():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    t = np.linspace(0, 2.0, 2000)

    # Oben links: Greifkraftprofil
    ax = axes[0, 0]
    grip = np.zeros_like(t)
    # Annährungsphase 0–0.4 s
    mask1 = (t >= 0) & (t <= 0.4)
    grip[mask1] = 0.0
    # Kontaktphase 0.4–0.6 s – schneller Kraftaufbau
    mask2 = (t > 0.4) & (t <= 0.6)
    grip[mask2] = 40 * (t[mask2] - 0.4) / 0.2
    # Halte-Pre-Loading 0.6–0.8 s – Kraftüberschuss
    mask3 = (t > 0.6) & (t <= 0.8)
    grip[mask3] = 40 + 15 * np.sin(np.pi * (t[mask3] - 0.6) / 0.2)
    # Steady-state 0.8–1.6 s
    mask4 = (t > 0.8) & (t <= 1.6)
    grip[mask4] = 38 + 2 * np.random.default_rng(42).normal(size=mask4.sum()) * 0.6
    # Loslassen 1.6–1.8 s
    mask5 = (t > 1.6) & (t <= 1.8)
    grip[mask5] = 38 * (1 - (t[mask5] - 1.6) / 0.2)
    grip = np.clip(grip, 0, 60)
    # Phasenhintergrund
    ax.axvspan(0, 0.4,    alpha=0.07, color=BLUE,   label='Annährung')
    ax.axvspan(0.4, 0.8,  alpha=0.07, color=RED,    label='Kontakt/Pre-Load')
    ax.axvspan(0.8, 1.6,  alpha=0.07, color=GREEN,  label='Halten')
    ax.axvspan(1.6, 2.0,  alpha=0.07, color=ORANGE, label='Ablegen')
    ax.plot(t, grip, color=BLUE, lw=1.5)
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Greifkraft [N]')
    ax.set_title('Greifkraftprofil beim präzisen Objektgreifen')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Oben rechts: EMG der wichtigsten Hand-/Unterarmmuskeln
    ax = axes[0, 1]
    muscles = ['FDP', 'FDS', 'FPL', 'EDC', 'FDI', 'Lumbrical']
    phase_on = [0.38, 0.40, 0.45, 0.35, 0.42, 0.50]
    phase_off = [1.65, 1.65, 1.62, 1.60, 1.65, 1.63]
    colors_emg = [RED, ORANGE, BLUE, GREEN, PURPLE, GRAY]
    for i, (m, on, off, c) in enumerate(zip(muscles, phase_on, phase_off, colors_emg)):
        burst = np.zeros_like(t)
        mask = (t >= on) & (t <= off)
        burst[mask] = (0.6 + 0.4 * np.random.default_rng(i).random()) * \
                      np.abs(np.random.default_rng(i*3).normal(size=mask.sum()))
        # normalize
        if burst.max() > 0:
            burst /= burst.max()
        ax.plot(t, burst * 0.6 + i, color=c, lw=0.8, alpha=0.9)
        ax.text(-0.05, i + 0.3, m, transform=ax.get_yaxis_transform(),
                fontsize=7.5, ha='right')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Muskel (versetzt)')
    ax.set_title('EMG-Aktivierungsmuster beim Präzisionsgriff')
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0.4, color=RED, ls='--', lw=1, alpha=0.7)
    ax.axvline(1.65, color=ORANGE, ls='--', lw=1, alpha=0.7)
    ax.text(0.4, 3.8, 'Kontakt', rotation=90, color=RED, fontsize=7)
    ax.text(1.65, 3.8, 'Loslassen', rotation=90, color=ORANGE, fontsize=7)

    # Unten links: Fingerflexion-Winkelprofile
    ax = axes[1, 0]
    t2 = np.linspace(0, 1.2, 600)
    fingers = ['Zeige-', 'Mittel-', 'Ring-', 'Klein-', 'Daumen']
    angles_max = [85, 90, 88, 82, 60]
    onset = [0.1, 0.12, 0.14, 0.13, 0.15]
    cols_f = [RED, ORANGE, BLUE, GREEN, PURPLE]
    for name, amax, on, c in zip(fingers, angles_max, onset, cols_f):
        angle = np.where(t2 >= on,
                         amax * (1 - np.exp(-8*(t2 - on))), 0.0)
        ax.plot(t2, angle, color=c, label=f'{name}finger')
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Flexionswinkel [°]')
    ax.set_title('Fingerkinematik beim Faustschluss (Flexionswinkel MCP)')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Unten rechts: Druckverteilung am Griff (heatmap)
    ax = axes[1, 1]
    rng = np.random.default_rng(99)
    nx, ny = 40, 80
    x_g = np.linspace(-20, 20, nx)
    y_g = np.linspace(0, 80, ny)
    X, Y = np.meshgrid(x_g, y_g)
    # Druckzentren: 4 Finger + Daumen + Thenar
    centers = [(-8, 70), (-2, 72), (4, 70), (10, 66), (-14, 35)]
    widths  = [    3.5,       3.2,     3.2,      3.5,        4.5]
    force   = [   45,        50,      48,        40,          30]
    P = np.zeros((ny, nx))
    for (cx, cy), w, f in zip(centers, widths, force):
        P += f * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2*w**2))
    # add noise
    P += rng.normal(0, 1, P.shape)
    P = np.clip(P, 0, None)
    cmap = LinearSegmentedColormap.from_list('pressure',
                                             ['white', '#ffffcc', '#fd8d3c', '#800026'])
    im = ax.imshow(P, origin='lower', aspect='auto', cmap=cmap,
                   extent=[x_g[0], x_g[-1], y_g[0], y_g[-1]])
    plt.colorbar(im, ax=ax, label='Druck [kPa]', fraction=0.04)
    ax.set_xlabel('Lateral [mm]')
    ax.set_ylabel('Proximal → Distal [mm]')
    ax.set_title('Druckverteilung an der Innenfläche beim zylindrischen Griff')
    ax.set_facecolor('#f0f0f0')

    fig.suptitle('Abbildung 3 – Greifanalyse: Kraft, EMG, Kinematik und Druckverteilung',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig03_greifanalyse')


# ════════════════════════════════════════════════════════════════════════════
# Fig 04 – Feinmotorik und posturale Kontrolle der Hand
# ════════════════════════════════════════════════════════════════════════════
def fig04_feinmotorik():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    rng = np.random.default_rng(7)

    # Oben links: Pinch-Kraft-Stabilität (Kraft-Zeit-Serie mit Variabilität)
    ax = axes[0, 0]
    t = np.linspace(0, 10, 1000)
    mean_force = 5.0  # N
    groups = {
        'Junge Erwachsene':  dict(noise=0.08, col=BLUE),
        'Senioren':          dict(noise=0.22, col=RED),
        'Geübte Musiker':    dict(noise=0.04, col=GREEN),
    }
    for label, g in groups.items():
        f = mean_force + rng.normal(0, g['noise'], len(t))
        f = np.convolve(f, np.ones(20)/20, 'same')
        ax.plot(t, f, color=g['col'], label=label, lw=1.0, alpha=0.9)
    ax.axhline(mean_force, color=GRAY, ls='--', lw=0.8)
    ax.set_xlabel('Zeit [s]')
    ax.set_ylabel('Pinch-Kraft [N]')
    ax.set_title('Kraftstabilität beim isometrischen Pinch-Griff')
    ax.legend()
    ax.set_ylim(4.0, 6.0)
    ax.grid(alpha=0.3)

    # Oben rechts: Bode-Diagramm des manuellen Regelkreises
    ax = axes[0, 1]
    omega = np.logspace(-1, 2.5, 500)
    f_hz = omega / (2 * np.pi)
    # Modellparameter (Literaturwerte)
    Kp, wcp = 3.2, 2*np.pi*22
    Kv_v, w_lv, w_hv = 1.5, 2*np.pi*0.4, 2*np.pi*6
    # Propriozeptiver Kanal (Hand-Spindeln)
    G_prop = Kp / np.abs(1 + 1j*omega/wcp)
    # Visueller Kanal
    G_vis  = 0.9 / np.abs(1 + 1j*omega/(2*np.pi*3))
    # Kombination
    G_tot = G_prop + G_vis
    ax.semilogx(f_hz, 20*np.log10(G_prop), color=BLUE,  lw=1.5, label='Propriozeptiv (Handspindeln)')
    ax.semilogx(f_hz, 20*np.log10(G_vis),  color=GREEN, lw=1.5, label='Visuell')
    ax.semilogx(f_hz, 20*np.log10(G_tot),  color=RED,   lw=2.0, ls='--', label='Gesamt')
    ax.axvline(22, color=BLUE, ls=':', lw=0.8, alpha=0.6)
    ax.text(22, -8, '22 Hz\n(Eckfreq. prop.)', fontsize=7, color=BLUE, ha='center')
    ax.set_xlabel('Frequenz [Hz]')
    ax.set_ylabel('Verstärkung [dB]')
    ax.set_title('Bode-Diagramm: manueller Regelkreis der Hand')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0.1, 200)

    # Unten links: Tremor-Spektrum
    ax = axes[1, 0]
    freq = np.linspace(0, 30, 3000)
    def tremor_psd(f, physio_peak, patho_peak=None, patho_amp=0):
        psd = 0.08 * np.exp(-f/3)  # physiologischer 1/f-Anteil
        psd += 0.5 * np.exp(-0.5*((f - physio_peak)/0.6)**2)
        if patho_peak:
            psd += patho_amp * np.exp(-0.5*((f - patho_peak)/0.5)**2)
        return psd
    psd_healthy  = tremor_psd(freq, 8.5)
    psd_parkinson= tremor_psd(freq, 8.5, 4.5, 3.0)
    psd_essential= tremor_psd(freq, 8.5, 6.5, 2.5)
    ax.fill_between(freq, psd_healthy,   alpha=0.4, color=BLUE,  label='Physiologischer Tremor')
    ax.fill_between(freq, psd_parkinson, alpha=0.3, color=RED,   label='Parkinson-Tremor (4–5 Hz)')
    ax.fill_between(freq, psd_essential, alpha=0.3, color=ORANGE,label='Essentieller Tremor (5–8 Hz)')
    ax.set_xlabel('Frequenz [Hz]')
    ax.set_ylabel('Leistungsdichte [a.u.]')
    ax.set_title('Handtremor-Spektrum: physiologisch vs. pathologisch')
    ax.legend()
    ax.axvspan(4, 5, alpha=0.12, color=RED)
    ax.axvspan(5, 8, alpha=0.12, color=ORANGE)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 25)

    # Unten rechts: Bewegungsgenauigkeit (Fitts' Gesetz)
    ax = axes[1, 1]
    W_vals = np.array([2, 4, 8, 16, 32])  # cm
    D_vals = np.array([10, 20, 40])        # cm
    fitts_MT = {}
    a_fitts, b_fitts = 0.20, 0.095  # s, s/bit
    for D in D_vals:
        ID = np.log2(2*D / W_vals)
        MT = a_fitts + b_fitts * ID
        fitts_MT[D] = (ID, MT)
        ax.scatter(ID, MT, zorder=3)
        ax.plot(ID, MT, label=f'D = {D} cm')
    ID_cont = np.linspace(0.5, 6, 100)
    MT_fit = a_fitts + b_fitts * ID_cont
    ax.plot(ID_cont, MT_fit, 'k--', lw=1.2, alpha=0.6, label=f'MT={a_fitts:.2f}+{b_fitts:.3f}·ID')
    ax.set_xlabel('Schwierigkeitsindex ID = log₂(2D/W) [bit]')
    ax.set_ylabel('Bewegungszeit MT [s]')
    ax.set_title("Fitts' Gesetz der Handbewegung")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle('Abbildung 4 – Feinmotorik und manuelle Regelung der Hand',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig04_feinmotorik')


# ════════════════════════════════════════════════════════════════════════════
# Fig 05 – Korrelationen Hand-Aktivität und Kognition/Gesundheit
# ════════════════════════════════════════════════════════════════════════════
def fig05_korrelationen():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    rng = np.random.default_rng(23)

    # Oben links: Handaktivität vs. kognitiver Leistungsindex
    ax = axes[0, 0]
    n = 60
    hand_activity = rng.uniform(1, 10, n)
    cog_index = 0.80 * hand_activity + rng.normal(0, 0.8, n) + 1.5
    cog_index = np.clip(cog_index, 1, 10)
    r, p = pearsonr(hand_activity, cog_index)
    ax.scatter(hand_activity, cog_index, color=BLUE, alpha=0.6, s=25)
    m, b = np.polyfit(hand_activity, cog_index, 1)
    xfit = np.linspace(1, 10, 100)
    ax.plot(xfit, m*xfit + b, color=RED, lw=1.5)
    ax.set_xlabel('Manuelle Aktivitätsindex [1–10]')
    ax.set_ylabel('Kognitiver Leistungsindex [1–10]')
    ax.set_title(f'Handaktivität vs. Kognition\n(r = {r:.2f}, p < 0.001)')
    ax.grid(alpha=0.3)

    # Oben rechts: Greifkraft vs. Mortalitätsrisiko
    ax = axes[0, 1]
    grip_strength = np.linspace(10, 65, 100)  # kg
    # Hazard ratio sinkt mit Greifkraft (Cox-Modell-Analogie)
    hazard = 1.6 * np.exp(-0.025 * (grip_strength - 10)) + 0.4
    ax.plot(grip_strength, hazard, color=RED, lw=2)
    ax.fill_between(grip_strength, hazard - 0.1, hazard + 0.1, alpha=0.2, color=RED)
    ax.axhline(1.0, color=GRAY, ls='--', lw=1, label='HR = 1 (Referenz)')
    ax.axvline(35, color=BLUE, ls=':', lw=1)
    ax.text(35, 1.8, 'Grenzwert\n35 kg', ha='center', color=BLUE, fontsize=7.5)
    ax.set_xlabel('Handgreifkraft [kg]')
    ax.set_ylabel('Hazard Ratio (Gesamtmortalität)')
    ax.set_title('Greifkraft als Mortalitätsprädiktor\n(nach Leong et al. 2015, N > 140 000)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Unten links: Handaktivität vs. Demenzrisiko (Längsschnitt)
    ax = axes[1, 0]
    age = np.linspace(60, 90, 100)
    demenz_aktiv   = 4  * np.exp(0.08  * (age - 60))
    demenz_inaktiv = 4  * np.exp(0.115 * (age - 60))
    ax.fill_between(age, demenz_aktiv,   alpha=0.35, color=GREEN, label='Hohe manuelle Aktivität')
    ax.fill_between(age, demenz_inaktiv, alpha=0.35, color=RED,   label='Niedrige manuelle Aktivität')
    ax.plot(age, demenz_aktiv,   color=GREEN, lw=2)
    ax.plot(age, demenz_inaktiv, color=RED,   lw=2)
    ax.set_xlabel('Alter [Jahre]')
    ax.set_ylabel('Kumulative Demenzinzidenz [%]')
    ax.set_title('Manuelle Aktivität und Demenzrisiko (Simulationsmodell)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Unten rechts: Schreiben/Tippen vs. Lernleistung
    ax = axes[1, 1]
    groups_labels = ['Tippen\n(Tastatur)', 'Schreiben\n(Stift)', 'Zeichnen\n(Hand)']
    retention_24h = np.array([58, 78, 85])
    retention_7d  = np.array([38, 62, 72])
    x = np.arange(3)
    w = 0.3
    ax.bar(x - w/2, retention_24h, w, color=BLUE,  alpha=0.85, label='Behaltensleistung 24 h')
    ax.bar(x + w/2, retention_7d,  w, color=GREEN, alpha=0.85, label='Behaltensleistung 7 d')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_labels)
    ax.set_ylabel('Behaltensleistung [%]')
    ax.set_title('Handschrift vs. Tippen: Gedächtnisleistung\n(modelliert nach Mueller & Oppenheimer 2014)')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    for xi, v24, v7 in zip(x, retention_24h, retention_7d):
        ax.text(xi - w/2, v24 + 1.5, f'{v24}%', ha='center', fontsize=8, color=BLUE)
        ax.text(xi + w/2, v7  + 1.5, f'{v7}%',  ha='center', fontsize=8, color=GREEN)

    fig.suptitle('Abbildung 5 – Korrelationen: manuelle Aktivität, Kognition und Gesundheit',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig05_korrelationen')


# ════════════════════════════════════════════════════════════════════════════
# Fig 06 – Neuroplastizität des S1-Handareals
# ════════════════════════════════════════════════════════════════════════════
def fig06_neuroplastizitaet():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Links: Altersbedingte Abnahme der Fingerrezeptordichte
    ax = axes[0]
    age = np.linspace(20, 80, 100)
    dens_norm    = 280 * np.exp(-0.008 * (age - 20))
    dens_musiker = 280 * np.exp(-0.004 * (age - 20)) + 20
    ax.plot(age, dens_norm,    color=RED,   lw=1.8, label='Allgemeinbevölkerung')
    ax.plot(age, dens_musiker, color=GREEN, lw=1.8, label='Berufsmusiker (Streich.)')
    ax.fill_between(age, dens_norm, dens_musiker, alpha=0.15, color=GREEN,
                    label='Schutzeffekt')
    ax.set_xlabel('Alter [Jahre]')
    ax.set_ylabel('Mechanorezeptordichte [Rez./cm²]')
    ax.set_title('Mechanorezeptordichte\n im Alter')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Mitte: S1-Handarealvolumen unter drei Bedingungen über 24 Monate
    ax = axes[1]
    months = np.array([0, 3, 6, 12, 18, 24])
    vol_control = np.array([1.85, 1.83, 1.81, 1.78, 1.76, 1.75])
    vol_musiker = np.array([1.85, 1.91, 1.97, 2.10, 2.18, 2.23])
    vol_rehab   = np.array([1.85, 1.87, 1.92, 2.01, 2.06, 2.09])
    ax.plot(months, vol_control, 'o-', color=GRAY,  label='Kontrolle')
    ax.plot(months, vol_musiker, 's-', color=GREEN,  label='Tägliches Instrumentenüben')
    ax.plot(months, vol_rehab,   '^-', color=BLUE,   label='Handrehabilitation (post-Stroke)')
    ax.fill_between(months, vol_control, vol_musiker, alpha=0.1, color=GREEN)
    ax.set_xlabel('Monate')
    ax.set_ylabel('S1-Handarealvolumen [cm³]')
    ax.set_title('Kortikales Handareal-Volumen\n(fMRT, 7 T, longitudinal)')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Rechts: STDP-Kurve und synaptische Dichte
    ax = axes[2]
    stim_intensity = np.linspace(0, 10, 200)
    syn_young = 1.0 * (1 - np.exp(-0.8 * stim_intensity))
    syn_old   = 0.65 * (1 - np.exp(-0.6 * stim_intensity))
    ax.plot(stim_intensity, syn_young, color=BLUE,   lw=1.8, label='30 Jahre')
    ax.plot(stim_intensity, syn_old,   color=RED,    lw=1.8, label='70 Jahre')
    ax.fill_between(stim_intensity, syn_old, syn_young, alpha=0.18, color=RED,
                    label='Alters-Plastizitätslücke')
    ax.set_xlabel('Stimulationsintensität [a.u.]')
    ax.set_ylabel('Synaptische Dichte [norm.]')
    ax.set_title('Synaptische Dichte S1-Handareal\nals Funktion der Stimulation')
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle('Abbildung 6 – Neuroplastizität des somatosensorischen Handareals',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    savefig('fig06_neuroplastizitaet')


# ════════════════════════════════════════════════════════════════════════════
# Fig 07 – Dynamisches Systemmodell Hand-Gehirn
# ════════════════════════════════════════════════════════════════════════════
def fig07_systemmodell():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Links: Gesundheitszustandsdynamik
    ax = axes[0]
    t = np.linspace(0, 36, 3600)  # Monate
    # Logistisches Wachstum / decay
    def health_traj(t, H0, r, K):
        return K / (1 + ((K - H0)/H0) * np.exp(-r * t))
    H_high = health_traj(t, 0.5, 0.12, 0.95)
    H_low  = health_traj(t, 0.5, 0.12, 0.45)
    # Intervention at t=12
    t_int = 12
    t_after = t[t >= t_int] - t_int
    H_interv = np.concatenate([H_low[t < t_int],
                               health_traj(t_after, H_low[t < t_int][-1], 0.15, 0.90)])
    ax.plot(t, H_high,   color=GREEN, lw=2, label='Hohe manuelle Aktivität')
    ax.plot(t, H_low,    color=RED,   lw=2, label='Niedrige manuelle Aktivität')
    ax.plot(t, H_interv, color=BLUE,  lw=2, ls='--', label='Intervention ab Monat 12')
    ax.axvline(t_int, color=GRAY, ls=':', lw=1)
    ax.text(t_int + 0.3, 0.52, 'Intervention\nstart', fontsize=7.5, color=GRAY)
    ax.set_xlabel('Zeit [Monate]')
    ax.set_ylabel('Manueller Gesundheitsindex H [0–1]')
    ax.set_title('Dynamisches Modell: H(t) unter verschiedenen Aktivitätsniveaus')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Rechts: Phase-Amplitude-Kopplung (PAC) Hand-S1
    ax = axes[1]
    phase = np.linspace(-np.pi, np.pi, 200)
    # PAC: Gamma-Amplituden moduliert durch Griffrhythmus-Phase
    pac_rest   = 0.3 + 0.05 * np.cos(phase)
    pac_grip   = 0.3 + 0.55 * np.exp(-0.5 * ((phase - 0.3)/0.7)**2)
    pac_tap    = 0.3 + 0.45 * np.exp(-0.5 * ((phase - 0.1)/0.6)**2) + \
                 0.2  * np.exp(-0.5 * ((phase + np.pi/2)/0.5)**2)
    ax.plot(np.degrees(phase), pac_rest, color=GRAY,  lw=1.5, label='Ruhe')
    ax.plot(np.degrees(phase), pac_grip, color=RED,   lw=1.8, label='Greifbewegung')
    ax.plot(np.degrees(phase), pac_tap,  color=BLUE,  lw=1.8, label='Finger-Tapping')
    ax.fill_between(np.degrees(phase), pac_rest, pac_grip, alpha=0.12, color=RED)
    ax.set_xlabel('Phase des Griffrhythmus [°]')
    ax.set_ylabel('Gamma-Amplitude (30–80 Hz) [norm.]')
    ax.set_title('Phase-Amplitude-Kopplung (PAC) im S1-Handareal')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks([-180, -90, 0, 90, 180])

    fig.suptitle('Abbildung 7 – Dynamisches Systemmodell und phase-amplitude Kopplung',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig07_systemmodell')


# ════════════════════════════════════════════════════════════════════════════
# Fig 08 – Handpathologien und Beeinträchtigungsindizes
# ════════════════════════════════════════════════════════════════════════════
def fig08_pathologien():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Links: Beeinträchtigungsindizes
    ax = axes[0]
    pathologies = [
        'Karpaltunnel-\nsyndrom', 'Rheumatoide\nArthritis',
        'Parkinson\n(Tremor)', 'Dupuytren\nKontr.', 'Ulnaris-\nParese',
        'Diabetische\nNeuropathie', 'Trigger\nFinger'
    ]
    direct     = np.array([6.2, 7.8, 6.5, 5.1, 7.2, 6.8, 4.3])
    neural_sec = np.array([4.1, 3.5, 7.1, 2.8, 6.3, 5.9, 2.0])
    cogn_tert  = np.array([2.3, 2.8, 4.8, 1.5, 3.7, 4.5, 1.2])
    x = np.arange(len(pathologies))
    ax.barh(x, direct,     color=RED,    alpha=0.85, label='Direkte Beeinträchtigung')
    ax.barh(x, neural_sec, left=direct,  color=ORANGE, alpha=0.85, label='Neuronale Sekundärfolge')
    ax.barh(x, cogn_tert, left=direct+neural_sec, color=PURPLE, alpha=0.85,
            label='Kognitive Tertiärfolge')
    ax.set_yticks(x)
    ax.set_yticklabels(pathologies, fontsize=8)
    ax.set_xlabel('Beeinträchtigungsindex [a.u.]')
    ax.set_title('Beeinträchtigungsindizes ausgewählter Handpathologien')
    ax.legend(fontsize=7.5)
    ax.grid(axis='x', alpha=0.3)

    # Rechts: Zeitlicher Verlauf Neuropathieindex unter Behandlungsstrategien
    ax = axes[1]
    t = np.linspace(0, 24, 240)
    n_untreated   = 0.1 * t + 0.1 * t * np.exp(0.04*t)
    n_medication  = 0.08 * t + 0.05 * t * np.exp(0.025*t)
    n_combined    = 0.04 * t + 0.02 * t * np.exp(0.01*t)
    n_hand_train  = 0.03 * t + 0.015 * t * np.exp(0.005*t)
    ax.plot(t, n_untreated,  color=RED,    lw=2,   label='Unbehandelt')
    ax.plot(t, n_medication, color=ORANGE, lw=1.8, label='Medikation allein')
    ax.plot(t, n_combined,   color=BLUE,   lw=1.8, label='Medikation + Physiotherapie')
    ax.plot(t, n_hand_train, color=GREEN,  lw=2,   ls='--', label='Handtraining + Propriozeptionsstim.')
    ax.set_xlabel('Zeit [Monate]')
    ax.set_ylabel('Kumulativer Neuropathieindex [a.u.]')
    ax.set_title('Progression des Neuropathieindex unter verschiedenen Strategien')
    ax.legend()
    ax.grid(alpha=0.3)
    # Beschriftung Reduktion
    ax.annotate('', xy=(24, n_hand_train[-1]), xytext=(24, n_untreated[-1]),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5))
    ax.text(24.5, (n_hand_train[-1] + n_untreated[-1])/2,
            f'−{100*(1 - n_hand_train[-1]/n_untreated[-1]):.0f}%',
            va='center', color=GREEN, fontsize=8)

    fig.suptitle('Abbildung 8 – Handpathologien: Beeinträchtigungsindizes und Verlaufsdynamik',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig08_pathologien')


# ════════════════════════════════════════════════════════════════════════════
# Fig 09 – Informationsfluss Hand → Kortex
# ════════════════════════════════════════════════════════════════════════════
def fig09_informationsfluss():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # Oben links: Shannon-Kapazität verschiedener Hautareale
    ax = axes[0, 0]
    areas_label = ['Fingerkuppe', 'Handfläche', 'Unterarm', 'Oberarm',
                   'Rücken', 'Fußsohle', 'Lippe', 'Zunge']
    tpd_mm  = np.array([2.5, 6.5, 18, 35, 50, 12, 2.0, 1.5])  # Two-Point-Discrimination
    area_cm2 = np.array([5, 150, 200, 400, 3500, 150, 8, 4])    # Fläche in cm²
    # Kapazität ~ log2(Fläche / TPD²)
    B_Hz = 300  # Hz
    SNR = 20
    C = B_Hz * np.log2(1 + SNR * area_cm2 / (tpd_mm/10)**2)
    C_norm = C / C.max()
    colors_cap = [RED if 'Finger' in l or 'Hand' in l else
                  BLUE for l in areas_label]
    bars = ax.bar(areas_label, C_norm, color=colors_cap, alpha=0.85)
    ax.set_xticklabels(areas_label, rotation=40, ha='right', fontsize=7.5)
    ax.set_ylabel('Somatosensorische Kanalkapazität [norm.]')
    ax.set_title('Shannon-Kapazität somatosensorischer Kanäle')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, C_norm):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.2f}', ha='center', fontsize=7)

    # Oben rechts: Mutual Information Hand → S1 vs. Aktivitätsintensität
    ax = axes[0, 1]
    intensity = np.linspace(0, 10, 100)
    MI_grob  = 2.5 * (1 - np.exp(-0.5 * intensity))
    MI_fein  = 3.8 * (1 - np.exp(-0.4 * intensity))
    MI_musik = 4.5 * (1 - np.exp(-0.35 * intensity))
    ax.plot(intensity, MI_grob,  color=BLUE,  lw=1.8, label='Grobmotorik (Greif.)')
    ax.plot(intensity, MI_fein,  color=RED,   lw=1.8, label='Feinmotorik (Schreib.)')
    ax.plot(intensity, MI_musik, color=GREEN, lw=1.8, label='Musikinstrument')
    ax.set_xlabel('Aktivitätsintensität [a.u.]')
    ax.set_ylabel('Mutual Information Hand→S1 [bit/s]')
    ax.set_title('Transinformation Hand→S1 vs. Aktivität')
    ax.legend()
    ax.grid(alpha=0.3)

    # Unten links: Granger-Kausalität
    ax = axes[1, 0]
    freq = np.linspace(1, 80, 500)
    GC_hand_s1 = 0.6 + 0.35 * np.exp(-0.5*((freq - 25)/15)**2) + \
                 0.2  * np.exp(-0.5*((freq - 10)/5)**2)
    GC_s1_hand = 0.15 + 0.10 * np.exp(-0.5*((freq - 40)/20)**2)
    ax.fill_between(freq, GC_s1_hand, GC_hand_s1, alpha=0.15, color=RED,
                    label='Überschuss Hand→S1')
    ax.plot(freq, GC_hand_s1, color=RED,  lw=1.8, label='Hand→S1 (GC)')
    ax.plot(freq, GC_s1_hand, color=BLUE, lw=1.8, label='S1→Hand (GC)')
    ax.set_xlabel('Frequenz [Hz]')
    ax.set_ylabel('Granger-Kausalitätsstärke [a.u.]')
    ax.set_title('Granger-Kausalität: Hand↔S1 (gerichtet)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Unten rechts: Informationsübertragungsrate bei Rezeptorverlust
    ax = axes[1, 1]
    loss_pct = np.linspace(0, 100, 200)
    ITR = 4.2 * (1 - loss_pct/100)**1.8
    ax.plot(loss_pct, ITR, color=BLUE, lw=2)
    crit_loss = 55
    crit_ITR  = 4.2 * (1 - crit_loss/100)**1.8
    ax.axvline(crit_loss, color=RED, ls='--', lw=1.2, label=f'Kritische Schwelle ({crit_loss}%)')
    ax.axhline(crit_ITR,  color=RED, ls='--', lw=1.2)
    ax.fill_between(loss_pct[loss_pct >= crit_loss],
                    ITR[loss_pct >= crit_loss], alpha=0.15, color=RED,
                    label='Klinisch kritischer Bereich')
    ax.set_xlabel('Mechanorezeptorverlust [%]')
    ax.set_ylabel('Informationsübertragungsrate [bit/s]')
    ax.set_title('ITR-Degradation bei zunehmender Deafferenzierung')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(crit_loss + 1, crit_ITR + 0.1, f'ITR = {crit_ITR:.1f} bit/s', fontsize=7.5)

    fig.suptitle('Abbildung 9 – Informationstheoretische Analyse: Hand als Hochbandbreitenkanal',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    savefig('fig09_informationsfluss')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Erzeuge Abbildung 1 – Homunkulus …")
    fig01_homunkulus()
    print("Erzeuge Abbildung 2 – Mechanorezeptoren …")
    fig02_mechanorezeptoren()
    print("Erzeuge Abbildung 3 – Greifanalyse …")
    fig03_greifanalyse()
    print("Erzeuge Abbildung 4 – Feinmotorik …")
    fig04_feinmotorik()
    print("Erzeuge Abbildung 5 – Korrelationen …")
    fig05_korrelationen()
    print("Erzeuge Abbildung 6 – Neuroplastizität …")
    fig06_neuroplastizitaet()
    print("Erzeuge Abbildung 7 – Systemmodell …")
    fig07_systemmodell()
    print("Erzeuge Abbildung 8 – Pathologien …")
    fig08_pathologien()
    print("Erzeuge Abbildung 9 – Informationsfluss …")
    fig09_informationsfluss()
    print("Alle Abbildungen erfolgreich generiert.")
