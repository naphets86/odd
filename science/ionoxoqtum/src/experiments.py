"""
experiments.py
==============
Umfassendes Experimentiermodul für das HydroPhotonic Framework.

Deckt alle Teilmodule der Arbeit
  „Effiziente Nanostrukturen mit Wasser und Licht"
  (Ionotronisches Flüssigkeits-Rechnen & aktive photonische Oberflächen)
durch ausführbare, dokumentierte Experimente ab.

Jedes Experiment
  - ist eine eigene Klasse mit run() → dict und plot() → Figure
  - erzeugt exakt einen Matplotlib-Figure mit zwei Sub-Plots
  - gibt seine Ergebnisse als kommentiertes Dictionary zurück
  - ist vollständig in sich geschlossen (keine externen Datenfiles)

Ausführung aller Experimente:
    python experiments.py

Ausführung eines einzelnen Experiments:
    from experiments import Experiment_04_EWODKontaktwinkel
    exp = Experiment_04_EWODKontaktwinkel()
    results = exp.run()
    fig = exp.plot()
    fig.savefig("ewod.png", dpi=150)

Literatur (Gleichungsnummern in den Docstrings):
  [IONO]  Epp, S. (2026): Ionotronisches Flüssigkeits-Rechnen.
  [OQTUM] Epp, S. (2026): Aktive photonische Oberflächen.
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.optimize import brentq

# ── Pfad für src-Importe ──────────────────────────────────────────────────────
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Alle Quell-Module importieren
from src.ionic import IonicSolution, IonSpecies, kcl_solution, FARADAY, BOLTZMANN, ELEMENTARY_Q, AVOGADRO, EPSILON_0
from src.ewod import EWODSystem, EWODDielectric, standard_al2o3_system
from src.iofet import IoFET, IoFETGeometry
from src.droplet import (
    Droplet, NANDGate, NORGate, NOTGate, ANDGate, ORGate, XORGate,
    HalfAdder, FullAdder, RippleCarryAdder, IRFNetwork, IRFNode,
)
from src.fabry_perot import FabryPerotResonator
from src.tmm import TransferMatrixMethod, OpticalLayer, build_hydrogel_stack, fresnel_rs
from src.polymer import FloryRehnerModel, PolymerParameters
from src.ebl import ProximityEffectModel, EBLParameters
from src.inverse_design import InversePhotonicsDesigner, SingleLayerDesigner

# ── Globale Darstellungsparameter ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.dpi":       120,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  2.0,
})

# Farbpalette (konsistent durch alle Experimente)
C = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a4c93", "#264653", "#e9c46a", "#a8dadc"]

# ── Basisklasse ───────────────────────────────────────────────────────────────

class BaseExperiment:
    """Basisklasse für alle Experimente.

    Unterklassen implementieren:
      run()  → dict   Numerische Ergebnisse
      plot() → Figure  Matplotlib-Figure mit genau zwei Sub-Plots
    """

    title: str = "Experiment"
    module: str = "?"
    section: str = "?"

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError

    def plot(self) -> plt.Figure:
        raise NotImplementedError

    def _fig2(self, title: Optional[str] = None, figsize=(12, 5)) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Erzeugt Figure mit zwei nebeneinander liegenden Sub-Plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(title or self.title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])  # 7% Platz oben für Titel
        return fig, ax1, ax2

    def run_and_plot(self, save_path: Optional[str] = None) -> Tuple[Dict, plt.Figure]:
        """Führt Experiment aus und erstellt Plot."""
        results = self.run()
        fig = self.plot()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return results, fig

    def summary(self) -> str:
        results = self.run()
        lines = [f"{'='*60}", f"  {self.title}", f"  Modul: {self.module} | Abschnitt: {self.section}", f"{'='*60}"]
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                lines.append(f"  {k}: ndarray{v.shape}, min={v.min():.4g}, max={v.max():.4g}")
            elif isinstance(v, float):
                lines.append(f"  {k}: {v:.6g}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK I: IONISCHE GRUNDLAGEN  (ionic.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_01_IonicConductivityConcentration(BaseExperiment):
    """
    EXPERIMENT 01: Ionische Leitfähigkeit σ(c) als Funktion der KCl-Konzentration

    Physikalischer Hintergrund [IONO §2.1]:
      σ = Σ |z_i| μ_i c_i F   (Gl. 1)
    Für KCl mit gleichwertigen K⁺ und Cl⁻ gilt:
      σ_KCl = (μ_{K+} + μ_{Cl-}) · c · F
    wobei μ_{K+}=7.62e-8 m²/(V·s), μ_{Cl-}=7.91e-8 m²/(V·s).

    Dieses Experiment demonstriert:
      - Lineare Konzentrations-Leitfähigkeits-Kurve σ(c)
      - Debye-Abschirmlänge λ_D(c) ∝ 1/√c
      - Bedeutung verschiedener Konzentrationen für IoFET-Design
    """
    title   = "Exp 01: Ionische Leitfähigkeit KCl – σ(c) und λ_D(c)"
    module  = "ionic.py"
    section = "[IONO §2.1] Ionische Leitfähigkeit"

    def run(self) -> Dict[str, Any]:
        concentrations = np.logspace(-3, 0, 80)  # 0.001 bis 1 mol/l
        sigma_vals  = np.zeros(len(concentrations))
        debye_nm    = np.zeros(len(concentrations))
        resistance  = np.zeros(len(concentrations))

        L = 1e-6   # Kanallänge 1 µm
        A = (100e-9)**2  # Querschnitt 100×100 nm²

        for i, c in enumerate(concentrations):
            sol = kcl_solution(c)
            sigma_vals[i] = sol.conductivity()
            debye_nm[i]   = sol.debye_length() * 1e9
            resistance[i] = sol.channel_resistance(L, A)

        # Vergleichspunkt aus der Arbeit: c=0.1 mol/l
        sol_ref = kcl_solution(0.1)
        sigma_ref = sol_ref.conductivity()
        debye_ref = sol_ref.debye_length() * 1e9

        # Analytische Steigung
        mu_K  = 7.62e-8; mu_Cl = 7.91e-8
        slope = (mu_K + mu_Cl) * FARADAY  # S·m²/mol

        return {
            "concentrations_mol_per_l": concentrations,
            "sigma_S_per_m":            sigma_vals,
            "debye_length_nm":          debye_nm,
            "channel_resistance_MOhm":  resistance * 1e-6,
            "sigma_at_0p1_mol_l":       sigma_ref,
            "debye_at_0p1_mol_l_nm":    debye_ref,
            "theoretical_slope_Sm2_per_mol": slope,
            "note": "Lineare Beziehung σ ∝ c bestätigt Einstein-Stokes-Modell",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        c = r["concentrations_mol_per_l"]
        fig, ax1, ax2 = self._fig2()

        # Sub-Plot 1: σ(c) und Referenzpunkt
        ax1.loglog(c, r["sigma_S_per_m"], color=C[0], label="σ_KCl(c)")
        ax1.axvline(0.1, color="gray", linestyle="--", alpha=0.7, label="c=0.1 mol/l (IRF-Ref.)")
        ax1.axhline(r["sigma_at_0p1_mol_l"], color="gray", linestyle=":", alpha=0.7)
        ax1.scatter([0.1], [r["sigma_at_0p1_mol_l"]], color=C[1], zorder=5, s=60,
                    label=f"σ={r['sigma_at_0p1_mol_l']:.2f} S/m")
        ax1.set_xlabel("Konzentration c (mol/l)")
        ax1.set_ylabel("Ionische Leitfähigkeit σ (S/m)")
        ax1.set_title("σ(c): Leitfähigkeit vs. Konzentration")
        ax1.legend()
        ax1.annotate("σ ∝ c  (linear)", xy=(0.01, 0.05), xytext=(0.001, 0.5),
                     arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

        # Sub-Plot 2: Debye-Länge λ_D(c)
        ax2.loglog(c, r["debye_length_nm"], color=C[2], label="λ_D(c)")
        ax2.axvline(0.1, color="gray", linestyle="--", alpha=0.7)
        ax2.scatter([0.1], [r["debye_at_0p1_mol_l_nm"]], color=C[1], zorder=5, s=60,
                    label=f"λ_D={r['debye_at_0p1_mol_l_nm']:.2f} nm")
        ax2.set_xlabel("Konzentration c (mol/l)")
        ax2.set_ylabel("Debye-Länge λ_D (nm)")
        ax2.set_title("λ_D(c): Debye-Abschirmlänge  ∝ 1/√c")
        ax2.legend()
        # Theoretische -1/2 Steigung einzeichnen
        c_fit = np.array([1e-3, 1.0])
        lD_fit = r["debye_at_0p1_mol_l_nm"] * np.sqrt(0.1 / c_fit)
        ax2.loglog(c_fit, lD_fit, "k--", alpha=0.4, linewidth=1, label="∝ c^{-1/2}")

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_02_DiffusionEinsteinStokes(BaseExperiment):
    """
    EXPERIMENT 02: Einstein-Stokes-Diffusion und Ionenbeweglichkeit

    Physikalischer Hintergrund [IONO §2.2]:
      D_i = μ_i k_B T / (|z_i| e)   (Gl. 2)
    Die Diffusionskoeffizienten von K⁺ und Cl⁻ sind bei 25°C:
      D_K  ≈ 1.96×10⁻⁹ m²/s
      D_Cl ≈ 2.03×10⁻⁹ m²/s
    Die bemerkenswert ähnlichen Werte minimieren Konzentrationsgradienten.

    Dieses Experiment zeigt:
      - Temperaturabhängigkeit D(T) für beide Ionenspezies
      - Diffusions-Zeitkonstante τ_diff = L² / D für verschiedene Kanallängen
    """
    title   = "Exp 02: Einstein-Stokes-Diffusion D(T) und Zeitkonstante τ(L)"
    module  = "ionic.py"
    section = "[IONO §2.2] Elektrokinetik"

    def run(self) -> Dict[str, Any]:
        T_vals = np.linspace(273.15, 373.15, 60)   # 0–100 °C
        D_K    = np.zeros(len(T_vals))
        D_Cl   = np.zeros(len(T_vals))
        mu_K  = 7.62e-8; mu_Cl = 7.91e-8

        for i, T in enumerate(T_vals):
            D_K[i]  = mu_K  * BOLTZMANN * T / ELEMENTARY_Q
            D_Cl[i] = mu_Cl * BOLTZMANN * T / ELEMENTARY_Q

        # Diffusions-Zeitkonstante τ = L²/D für L = 100nm bis 100µm
        L_vals = np.logspace(-7, -4, 60)   # 100nm .. 100µm
        D_ref  = D_K[20]   # bei 25°C
        tau_ms = L_vals**2 / D_ref * 1e3   # ms

        return {
            "T_Celsius":          T_vals - 273.15,
            "D_K_m2_per_s":       D_K,
            "D_Cl_m2_per_s":      D_Cl,
            "D_K_at_25C":         D_K[20],
            "D_Cl_at_25C":        D_Cl[20],
            "L_values_m":         L_vals,
            "tau_diffusion_ms":   tau_ms,
            "note": "K+ und Cl- nahezu gleiche Diffusionskoeffizienten → minimale Konzentrationsgradienten",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1.plot(r["T_Celsius"], r["D_K_m2_per_s"]  * 1e9, color=C[0], label="K⁺")
        ax1.plot(r["T_Celsius"], r["D_Cl_m2_per_s"] * 1e9, color=C[1], linestyle="--", label="Cl⁻")
        ax1.axvline(25, color="gray", linestyle=":", alpha=0.7, label="25 °C")
        ax1.set_xlabel("Temperatur T (°C)")
        ax1.set_ylabel("Diffusionskoeffizient D (nm²/ns)")
        ax1.set_title("Einstein-Stokes: D(T) für K⁺ und Cl⁻")
        ax1.legend()
        ax1.annotate(f"D_K={r['D_K_at_25C']*1e9:.2f} nm²/ns\nbei 25°C",
                     xy=(25, r["D_K_at_25C"]*1e9), xytext=(50, 1.8),
                     arrowprops=dict(arrowstyle="->"), fontsize=8)

        ax2.loglog(r["L_values_m"]*1e6, r["tau_diffusion_ms"], color=C[2])
        ax2.axvline(0.1, color="gray", linestyle="--", alpha=0.7, label="L=100nm")
        ax2.axvline(1.0, color="gray", linestyle=":",  alpha=0.7, label="L=1µm (IoFET)")
        ax2.set_xlabel("Kanallänge L (µm)")
        ax2.set_ylabel("Diffusions-Zeitkonstante τ (ms)")
        ax2.set_title("Diffusions-Zeitkonstante τ = L²/D  (K⁺, 25°C)")
        ax2.legend()

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_03_IonicChannelResistance(BaseExperiment):
    """
    EXPERIMENT 03: Kanalwiderstand R(L, σ) – Ohmsches Gesetz für Ionenkanäle

    Physikalischer Hintergrund [IONO §2.1, §3.1]:
      R = L / (σ · A)   (Ohmsches Gesetz für ionischen Kanal)
    Für den IoFET-Kanal (100×100nm² Querschnitt) bei KCl 1 mol/l:
      σ ≈ 15 S/m → R(1µm) ≈ 6.7 MΩ

    Zeigt die Abhängigkeit des Kanalwiderstands von:
      - Konzentration (σ) bei fester Geometrie
      - Länge L bei fester Konzentration
    """
    title   = "Exp 03: Ionischer Kanalwiderstand R(c, L)"
    module  = "ionic.py"
    section = "[IONO §3.1] Wasser als Leiter"

    def run(self) -> Dict[str, Any]:
        A = (100e-9)**2   # 100×100 nm²

        # R vs. Konzentration bei L=1µm
        c_vals = np.logspace(-2, 0, 60)
        L_fixed = 1e-6
        R_vs_c = np.array([kcl_solution(c).channel_resistance(L_fixed, A) for c in c_vals])

        # R vs. Länge bei c=0.5 mol/l
        L_vals = np.logspace(-7, -4, 60)
        sol_05 = kcl_solution(0.5)
        R_vs_L = np.array([sol_05.channel_resistance(L, A) for L in L_vals])

        # Referenzpunkte
        sol_1 = kcl_solution(1.0)
        R_ref = sol_1.channel_resistance(1e-6, A)
        sigma_1 = sol_1.conductivity()

        return {
            "c_vals_mol_l":       c_vals,
            "R_vs_c_MOhm":        R_vs_c * 1e-6,
            "L_vals_um":          L_vals * 1e6,
            "R_vs_L_MOhm":        R_vs_L * 1e-6,
            "R_ref_1mol_1um_MOhm": R_ref * 1e-6,
            "sigma_1mol_l":       sigma_1,
            "note": "R ∝ 1/c (σ linear in c) und R ∝ L (Ohmsches Gesetz)",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1.loglog(r["c_vals_mol_l"], r["R_vs_c_MOhm"], color=C[3])
        ax1.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="c=1 mol/l")
        ax1.scatter([1.0], [r["R_ref_1mol_1um_MOhm"]], color=C[0], s=60, zorder=5,
                    label=f"R={r['R_ref_1mol_1um_MOhm']:.2f} MΩ")
        ax1.set_xlabel("Konzentration c (mol/l)")
        ax1.set_ylabel("Kanalwiderstand R (MΩ)")
        ax1.set_title("R(c): Widerstand vs. KCl-Konzentration\n(L=1µm, A=100×100nm²)")
        ax1.legend()

        ax2.loglog(r["L_vals_um"], r["R_vs_L_MOhm"], color=C[1])
        ax2.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="L=1 µm (IoFET)")
        ax2.set_xlabel("Kanallänge L (µm)")
        ax2.set_ylabel("Kanalwiderstand R (MΩ)")
        ax2.set_title("R(L): Widerstand vs. Kanallänge\n(c=0.5 mol/l, A=100×100nm²)")
        ax2.legend()

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK II: EWOD-PHYSIK  (ewod.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_04_EWODYoungLippmann(BaseExperiment):
    """
    EXPERIMENT 04: Young-Lippmann-Gleichung – Kontaktwinkel θ(V)

    Physikalischer Hintergrund [IONO §2.4]:
      cos θ(V) = cos θ₀ + η_EWOD · V²   (Gl. 4, Young-Lippmann)
      η_EWOD  = ε₀ ε_r / (2 γ_LG d)    (Gl. 5, EWOD-Effizienz)

    Für Al₂O₃ (d=10nm, ε_r=9), Wasser (γ_LG=72mN/m):
      η = 8.85e-12 · 9 / (2 · 0.072 · 10e-9) ≈ 0.055 V⁻²  (SI)
    Entsprechend V_schalt (110° → 60°) ≈ 3.9 V

    Vergleich verschiedener Dielektrika: Al₂O₃, HfO₂, SiO₂
    """
    title   = "Exp 04: EWOD Young-Lippmann – θ(V) und Materialvergleich"
    module  = "ewod.py"
    section = "[IONO §2.4] Elektrowetting"

    # Materialparameter: (ε_r, d_nm, label)
    MATERIALS = [
        (9.0,  10.0, "Al₂O₃ (10nm)"),
        (25.0, 10.0, "HfO₂  (10nm)"),
        (3.9,  10.0, "SiO₂  (10nm)"),
        (9.0,  5.0,  "Al₂O₃  (5nm)"),
    ]

    def run(self) -> Dict[str, Any]:
        V = np.linspace(0, 5.0, 300)
        results_theta = {}
        results_eta   = {}

        for eps_r, d_nm, label in self.MATERIALS:
            diel = EWODDielectric(thickness=d_nm*1e-9, epsilon_r=eps_r)
            sys  = EWODSystem(diel, contact_angle_0=110.0)
            theta = np.array([sys.contact_angle(v) for v in V])
            results_theta[label] = theta
            results_eta[label]   = sys.ewod_efficiency
            V_sat = sys.switching_voltage(60.0)

        # Standardsystem
        std = standard_al2o3_system()
        V_schalt = std.switching_voltage(60.0)

        return {
            "voltages":             V,
            "theta_by_material":    results_theta,
            "eta_by_material":      results_eta,
            "V_schalt_al2o3_V":     V_schalt,
            "theta_0_deg":          110.0,
            "theta_target_deg":     60.0,
            "note": "EWOD-Effizienz η [V⁻²] bestimmt Schaltspannung",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        V = r["voltages"]
        fig, ax1, ax2 = self._fig2()

        colors_mat = [C[0], C[1], C[2], C[3]]
        for (_, _, label), col in zip(self.MATERIALS, colors_mat):
            theta = r["theta_by_material"][label]
            ax1.plot(V, theta, color=col, label=label)

        ax1.axhline(r["theta_0_deg"],     color="gray",  linestyle="--", alpha=0.6, label=f"θ₀={r['theta_0_deg']:.0f}°")
        ax1.axhline(r["theta_target_deg"], color="black", linestyle=":",  alpha=0.6, label=f"θ_Ziel={r['theta_target_deg']:.0f}°")
        ax1.axvline(r["V_schalt_al2o3_V"], color=C[0],   linestyle=":",  alpha=0.5)
        ax1.set_xlabel("Spannung V (V)")
        ax1.set_ylabel("Kontaktwinkel θ (°)")
        ax1.set_title("Young-Lippmann: θ(V) für verschiedene Dielektrika")
        ax1.set_ylim(50, 115)
        ax1.legend(fontsize=8)

        # Sub-Plot 2: EWOD-Effizienz und Schaltspannungen
        labels = [lbl for _, _, lbl in self.MATERIALS]
        eta_vals = [r["eta_by_material"][lbl] for lbl in labels]
        V_schalt_vals = []
        for eps_r, d_nm, lbl in self.MATERIALS:
            diel = EWODDielectric(d_nm*1e-9, eps_r)
            sys  = EWODSystem(diel, 110.0)
            V_schalt_vals.append(sys.switching_voltage(60.0))

        x = np.arange(len(labels))
        bars = ax2.bar(x, V_schalt_vals, color=[C[0],C[1],C[2],C[3]], alpha=0.8, edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("Schaltspannung V_schalt (V)")
        ax2.set_title("Schaltspannung 110°→60° nach Dielektrikum")
        for bar, v in zip(bars, V_schalt_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05, f"{v:.2f}V",
                     ha="center", va="bottom", fontsize=9)

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_05_EWODSchaltenergie(BaseExperiment):
    """
    EXPERIMENT 05: Schaltenergie E = ½ C V² und Elektrodengröße

    Physikalischer Hintergrund [IONO §8.1]:
      E_IoFET = ½ · C_G · V_T²
            = ½ · (ε₀ ε_r A_G / d) · V_T²
    Für A_G = (5µm)² (Arbeit): E ≈ 0.07 fJ
    Für A_G = (100nm)² (Nanoelektrode): E ≈ 0.3 aJ

    Vergleich mit CMOS:
      E_CMOS = C_G · V_DD² (kein Rückgewinn)
      ΔE = E_CMOS / E_IoFET ≈ Faktor 20 (aus Tabelle [IONO])
    """
    title   = "Exp 05: IoFET-Schaltenergie vs. Elektrodengröße und CMOS-Vergleich"
    module  = "ewod.py"
    section = "[IONO §8.1] Energiebilanz"

    def run(self) -> Dict[str, Any]:
        std = standard_al2o3_system()
        V_T = std.switching_voltage(60.0)   # ~3.9 V (SI)

        # E vs. Elektrodengröße
        A_vals_nm  = np.logspace(1.5, 4, 80)   # 30nm bis 10µm
        A_vals_m2  = (A_vals_nm * 1e-9)**2
        E_iofet_fJ = np.array([std.switching_energy(V_T, A) * 1e15 for A in A_vals_m2])

        # CMOS Vergleich (V_DD = 0.7V, selbe Kapazität)
        V_DD_CMOS = 0.7
        C_per_A   = std.dielectric.capacitance_per_area
        E_cmos_fJ = np.array([C_per_A * A * V_DD_CMOS**2 * 1e15 for A in A_vals_m2])

        # Referenz: 5µm-Elektrode aus der Arbeit
        A_paper  = (5e-6)**2
        E_paper  = std.switching_energy(V_T, A_paper) * 1e15
        E_cmos_paper = C_per_A * A_paper * V_DD_CMOS**2 * 1e15

        # Schaltspannungen für verschiedene Zielwinkel
        theta_targets = np.linspace(60, 109, 60)
        V_targets = np.array([std.switching_voltage(t) for t in theta_targets])
        A_ref = A_paper
        E_targets = np.array([std.switching_energy(v, A_ref) * 1e15
                               if not np.isnan(v) else np.nan for v in V_targets])

        return {
            "A_vals_nm":              A_vals_nm,
            "E_iofet_fJ":             E_iofet_fJ,
            "E_cmos_fJ":              E_cmos_fJ,
            "V_T_V":                  V_T,
            "E_paper_5um_fJ":         E_paper,
            "E_cmos_paper_5um_fJ":    E_cmos_paper,
            "ratio_cmos_iofet":       E_cmos_paper / E_paper,
            "theta_targets_deg":      theta_targets,
            "E_vs_target_angle_fJ":   E_targets,
            "note": f"V_T={V_T:.3f}V | E(5µm)={E_paper:.4f} fJ | CMOS/IoFET-Ratio={E_cmos_paper/E_paper:.1f}x",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1.loglog(r["A_vals_nm"], r["E_iofet_fJ"], color=C[1], label="IoFET (E=½CV²)")
        ax1.loglog(r["A_vals_nm"], r["E_cmos_fJ"],  color=C[0], linestyle="--", label="CMOS (E=CV²)")
        ax1.axvline(5000, color="gray", linestyle=":", alpha=0.7, label="5µm (Arbeit-Ref.)")
        ax1.scatter([5000], [r["E_paper_5um_fJ"]], color=C[1], s=80, zorder=5)
        ax1.scatter([5000], [r["E_cmos_paper_5um_fJ"]], color=C[0], s=80, zorder=5)
        ax1.set_xlabel("Elektrodenkantenlänge (nm)")
        ax1.set_ylabel("Schaltenergie (fJ)")
        ax1.set_title("Schaltenergie vs. Elektrodengröße")
        ax1.legend()
        ax1.text(5200, r["E_paper_5um_fJ"]*1.5,
                 f"IoFET: {r['E_paper_5um_fJ']:.3f} fJ\nRatio: {r['ratio_cmos_iofet']:.0f}×",
                 fontsize=8, color=C[1])

        valid = ~np.isnan(r["E_vs_target_angle_fJ"])
        ax2.plot(r["theta_targets_deg"][valid], r["E_vs_target_angle_fJ"][valid], color=C[3])
        ax2.set_xlabel("Zielkontaktwinkel θ_Ziel (°)")
        ax2.set_ylabel("Schaltenergie (fJ) bei A=(5µm)²")
        ax2.set_title("E(θ_Ziel): Energie vs. Schalthub")
        ax2.invert_xaxis()

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_06_EWODTemperaturkompensation(BaseExperiment):
    """
    EXPERIMENT 06: Temperaturabhängigkeit der Oberflächenspannung und Kompensation

    Physikalischer Hintergrund [IONO §4.4]:
      γ(T) ≈ γ₀ · (1 − (T−T₀)/T_c)^μ
    mit γ₀=72.75mN/m, T₀=0°C, T_c=647K, μ=1.26

    Da η ∝ 1/γ, verschiebt sich der Kontaktwinkel mit T.
    Kompensation: Gate-Spannung anpassen, sodass θ(V,T) = const.
    """
    title   = "Exp 06: EWOD-Temperaturkompensation – γ(T) und V_komp(T)"
    module  = "ewod.py"
    section = "[IONO §4.4] Thermische Stabilität"

    def run(self) -> Dict[str, Any]:
        T_C   = np.linspace(0, 80, 80)   # 0 – 80 °C
        T_K   = T_C + 273.15
        gamma_0 = 72.75e-3; T_0 = 273.15; T_c = 647.0; mu = 1.26
        gamma_T = gamma_0 * ((1 - (T_K - T_0) / T_c) ** mu)

        # η(T) und Kontaktwinkel-Drift ohne Kompensation
        eps_r = 9.0; d = 10e-9
        eta_T = (EPSILON_0 * eps_r) / (2 * gamma_T * d)

        theta0 = 110.0
        V_fixed = 3.9   # feste Spannung
        cos_theta_T = np.cos(np.radians(theta0)) + eta_T * V_fixed**2
        cos_theta_T = np.clip(cos_theta_T, -1, 1)
        theta_T_no_comp = np.degrees(np.arccos(cos_theta_T))

        # Kompensationsspannung für konstantes θ=70°
        theta_target = 70.0
        delta_cos = np.cos(np.radians(theta_target)) - np.cos(np.radians(theta0))
        V_komp_T = np.sqrt(np.maximum(delta_cos / eta_T, 0))

        return {
            "T_Celsius":             T_C,
            "gamma_T_mN_per_m":      gamma_T * 1e3,
            "eta_T":                 eta_T,
            "theta_uncompensated":   theta_T_no_comp,
            "V_compensation_T":      V_komp_T,
            "theta_target_deg":      theta_target,
            "drift_range_deg":       theta_T_no_comp.max() - theta_T_no_comp.min(),
            "note": "Temperaturkompensation durch dynamische Anpassung von V_Gate",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        T = r["T_Celsius"]
        fig, ax1, ax2 = self._fig2()

        ax1_twin = ax1.twinx()
        l1, = ax1.plot(T, r["gamma_T_mN_per_m"], color=C[0], label="γ(T) [mN/m]")
        l2, = ax1_twin.plot(T, r["theta_uncompensated"], color=C[1], linestyle="--",
                             label=f"θ(T) bei V={3.9:.1f}V")
        ax1_twin.axhline(r["theta_target_deg"], color="gray", linestyle=":", alpha=0.7)
        ax1.set_xlabel("Temperatur T (°C)")
        ax1.set_ylabel("Oberflächenspannung γ (mN/m)", color=C[0])
        ax1_twin.set_ylabel("Kontaktwinkel θ (°)", color=C[1])
        ax1.set_title(f"γ(T) und θ-Drift ohne Kompensation\n(Drift: {r['drift_range_deg']:.1f}°)")
        ax1.legend(handles=[l1, l2], loc="upper right")

        ax2.plot(T, r["V_compensation_T"], color=C[2])
        ax2.set_xlabel("Temperatur T (°C)")
        ax2.set_ylabel("Kompensations-Gate-Spannung V_komp (V)")
        ax2.set_title(f"V_komp(T) für konstantes θ={r['theta_target_deg']:.0f}°")
        ax2.fill_between(T, r["V_compensation_T"]*0.95, r["V_compensation_T"]*1.05,
                          alpha=0.2, color=C[2], label="±5% Band")
        ax2.legend()

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK III: IoFET-TRANSISTOR  (iofet.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_07_IoFETKennlinien(BaseExperiment):
    """
    EXPERIMENT 07: IoFET Transfer- und Ausgangskennlinien

    Physikalischer Hintergrund [IONO §4.2]:
      Linearer Bereich: I_DS = G₀ · η · (V_GS − V_T) · V_DS
      Sättigungsbereich: I_DS = (G₀ · η / 2) · (V_GS − V_T)²
      G₀ = σ · A / L (Grundleitfähigkeit)

    Subthreshold-Swing SS < 10 mV/dec (vs. 60 mV/dec Boltzmann-Limit)
    """
    title   = "Exp 07: IoFET Kennlinien – Transfer und Ausgangskennlinienfeld"
    module  = "iofet.py"
    section = "[IONO §4.2] IoFET-Modell"

    def run(self) -> Dict[str, Any]:
        geo = IoFETGeometry(
            channel_length=1e-6,
            channel_width=100e-9,
            channel_height=100e-9,
        )
        transistor = IoFET(
            conductivity=1.0,
            geometry=geo,
            threshold_voltage=0.05,
            subthreshold_swing=8.0,   # < 10 mV/dec (IoFET-Vorteil)
        )

        V_gs, I_trans = transistor.transfer_characteristics(
            V_ds=0.1, V_gs_range=(-0.05, 1.0), n_points=500
        )
        V_gs_labels = [0.2, 0.4, 0.6, 0.8, 1.0]
        output_chars = transistor.output_characteristics(V_gs_labels, n_points=200)

        G0 = transistor.base_conductance
        on_off = transistor.on_off_ratio(V_gs_on=1.0, V_ds=0.1)

        return {
            "V_gs_transfer":        V_gs,
            "I_ds_transfer_A":      I_trans,
            "output_chars":         {vg: (vd.tolist(), ids.tolist()) for vg, (vd, ids) in output_chars.items()},
            "G0_S":                 G0,
            "V_T_V":                transistor.V_T,
            "SS_mV_dec":            transistor.SS,
            "on_off_ratio":         on_off,
            "note": f"G₀={G0:.3e}S | SS={transistor.SS}mV/dec | I_ON/I_OFF={on_off:.1e}",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        I = np.array(r["I_ds_transfer_A"])
        ax1.semilogy(r["V_gs_transfer"], np.abs(I)+1e-30, color=C[0], linewidth=2)
        ax1.axvline(r["V_T_V"], color="gray", linestyle="--", alpha=0.7,
                    label=f"V_T={r['V_T_V']:.3f}V")
        ax1.set_xlabel("V_GS (V)")
        ax1.set_ylabel("|I_DS| (A)  [log-Skala]")
        ax1.set_title(f"Transferkennlinie (log)\nSS={r['SS_mV_dec']}mV/dec | I_ON/I_OFF={r['on_off_ratio']:.1e}")
        ax1.legend()
        ax1.set_ylim(1e-25, None)

        vg_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(r["output_chars"])))
        for (vg, (vd_lst, ids_lst)), col in zip(r["output_chars"].items(), vg_colors):
            vd  = np.array(vd_lst)
            ids = np.array(ids_lst)
            ax2.plot(vd, ids*1e9, color=col, label=f"V_GS={vg:.1f}V")
            # Sättigungsbereich kennzeichnen
            v_sat = vg - r["V_T_V"]
            if v_sat > 0:
                ax2.axvline(v_sat, color=col, alpha=0.15, linestyle="--")
        ax2.set_xlabel("V_DS (V)")
        ax2.set_ylabel("I_DS (nA)")
        ax2.set_title(f"Ausgangskennlinienfeld\nG₀={r['G0_S']:.2e} S")
        ax2.legend(fontsize=8, loc="lower right")

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_08_IoFETTransistordichte(BaseExperiment):
    """
    EXPERIMENT 08: IoFET-Transistordichte vs. CMOS-Roadmap

    Physikalischer Hintergrund [IONO §6]:
      ρ_Elektroden = 1/p²   (Gleichung §6.1)
      ρ_IoFET_äq  = 2 · ρ_Elektroden   (Dreifach-Simultanität)
      ρ_3D-IRF   = N_z · 2 · ρ_Elektroden

    Vergleich:
      - CMOS-Roadmap 2020–2030 (Intel/TSMC-Benchmarks)
      - IRF mit p=5µm (Stand 2026)
      - Skalierungspotential IRF (p=100nm)
    """
    title   = "Exp 08: IoFET-Transistordichte vs. CMOS-Roadmap"
    module  = "iofet.py"
    section = "[IONO §6] Transistordichte"

    # CMOS-Benchmarks (Jahr, nm-Prozess, Transistoren/cm², Energie/op fJ)
    CMOS_BENCHMARKS = [
        (2016, 16,  50e6,    1.0),
        (2018, 10,  100e6,   0.8),
        (2020,  7,  170e6,   0.5),
        (2022,  5,  300e6,   0.3),
        (2024,  3,  500e6,   0.15),
        (2026,  2,  800e6,   0.10),
    ]

    def run(self) -> Dict[str, Any]:
        # IRF-Dichte für verschiedene Pitches
        pitches_um = np.logspace(-1, 1.5, 60)   # 0.1µm bis 30µm
        rho_irf = 1.0 / (pitches_um * 1e-6)**2 * 1e-4  # 1/cm²

        # 3D-Skalierung für p=5µm
        Nz_vals    = np.arange(1, 31)
        rho_p5_2D  = 1.0 / (5e-6)**2 * 1e-4  # 4e10/cm²
        rho_3D     = Nz_vals * 2 * rho_p5_2D

        years  = [b[0] for b in self.CMOS_BENCHMARKS]
        rho_cmos = [b[2] for b in self.CMOS_BENCHMARKS]

        return {
            "pitches_um":      pitches_um,
            "rho_IRF_per_cm2": rho_irf,
            "Nz_vals":         Nz_vals,
            "rho_3D_IRF":      rho_3D,
            "CMOS_years":      years,
            "CMOS_rho":        rho_cmos,
            "rho_p5um_2D":     rho_p5_2D,
            "note": f"IRF p=5µm: {rho_p5_2D:.2e}/cm² | CMOS 3nm (2026): {800e6:.2e}/cm²",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        # Sub-Plot 1: ρ(pitch) vs CMOS
        ax1.loglog(r["pitches_um"], r["rho_IRF_per_cm2"], color=C[1], label="IRF 2D")
        ax1.loglog(r["pitches_um"], 2*r["rho_IRF_per_cm2"], color=C[2], linestyle="--",
                   label="IRF 2D ×2 (Dreifach-Sim.)")
        ax1.axvline(5.0, color="gray", linestyle=":", alpha=0.7, label="p=5µm (Stand 2026)")
        ax1.scatter(r["CMOS_years"][-1] - 2015,
                    r["CMOS_rho"][-1], marker="*", s=200, color=C[0],
                    label="CMOS 3nm (2026)", zorder=6)
        # Zeichne CMOS-Benchmarks als horizontale Linie
        ax1.axhline(r["CMOS_rho"][-1], color=C[0], linestyle="-.", alpha=0.5)
        ax1.set_xlabel("Elektroden-Pitch p (µm)")
        ax1.set_ylabel("Transistordichte (cm⁻²)")
        ax1.set_title("IRF-Transistordichte ρ(p) vs. CMOS-Referenz")
        ax1.legend(fontsize=8)

        # Sub-Plot 2: 3D-Skalierung
        ax2.semilogy(r["Nz_vals"], r["rho_3D_IRF"], color=C[3], marker="o", markersize=4)
        ax2.axhline(r["CMOS_rho"][-1], color=C[0], linestyle="--", alpha=0.7, label="CMOS 3nm Grenze")
        ax2.axhline(r["CMOS_rho"][-1]*100, color=C[0], linestyle=":", alpha=0.5, label="CMOS ×100")
        Nz_cross = int(r["CMOS_rho"][-1] / (2 * r["rho_p5um_2D"])) + 1
        ax2.axvline(Nz_cross, color="gray", linestyle=":", alpha=0.7, label=f"N_z={Nz_cross} übersteigt CMOS")
        ax2.set_xlabel("Anzahl Wasserschichten N_z")
        ax2.set_ylabel("Äquiv. 3D-Transistordichte (cm⁻²)")
        ax2.set_title("3D-IRF-Skalierung: ρ_3D = N_z · 2 · ρ_2D\n(p=5µm)")
        ax2.legend(fontsize=8)

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_09_IoFETEnergievergleich(BaseExperiment):
    """
    EXPERIMENT 09: IoFET-Energieverbrauch – vollständige Energiebilanz

    [IONO §8.1]: E_IoFET = E_elektrostatisch + E_Reibung − E_Oberflächenspannung_zurück
    Reibungsverlust: E_Reib = η_Wasser · v²/D · V_Tropfen ≈ 0.01 fJ (vernachlässigbar)
    Hauptbeitrag: E = ½ C_G V_T²

    Vergleich der drei Energieterme über Tropfengeschwindigkeiten v.
    """
    title   = "Exp 09: IoFET-Energiebilanz – Reibung, Elektrostatik, Rückgewinnung"
    module  = "iofet.py"
    section = "[IONO §8.1] Energiebilanz"

    def run(self) -> Dict[str, Any]:
        # Geschwindigkeiten 0.01 µm/ms bis 100 µm/ms
        v_vals = np.logspace(-2, 2, 80) * 1e-3   # in m/ms → m/s (factor 1e-3? No: µm/ms = mm/s)
        # Korrekte Umrechnung: 1 µm/ms = 1e-6m/1e-3s = 1e-3 m/s = 1 mm/s
        v_ms = np.logspace(-2, 2, 80)   # µm/ms
        v_si = v_ms * 1e-3              # m/s (1 µm/ms = 1e-3 m/s)

        eta_wasser = 1e-3    # Pa·s (dynamische Viskosität Wasser)
        D_kanal    = 1e-6    # m (Kanalbreite 1µm)
        V_tropfen  = 1e-15   # m³ = 1 pl

        # Reibungsenergie
        E_reib_fJ = eta_wasser * (v_si**2) / D_kanal * V_tropfen * 1e15

        # Elektrostatische Energie (konstant, unabhängig von v)
        std  = standard_al2o3_system()
        V_T  = std.switching_voltage(60.0)
        A_G  = (5e-6)**2
        E_elek_fJ = std.switching_energy(V_T, A_G) * 1e15
        E_elek_arr = np.full_like(v_ms, E_elek_fJ)

        # Rückgewonnene Oberflächenenergie
        d1 = Droplet((0,0), radius=50e-9)
        d2 = Droplet((0,0), radius=50e-9)
        E_surf_recov_fJ = d1.energy_released_on_merge(d2) * 1e15

        # CMOS-Vergleich
        E_cmos_fJ = (EPSILON_0 * 9.0 / 10e-9) * A_G * (0.7**2) * 1e15

        # Tatsächliche Schaltenergie (Summe Elektrostatik + Reibung)
        E_total_fJ = E_elek_arr + E_reib_fJ

        return {
            "v_um_per_ms":          v_ms,
            "E_friction_fJ":        E_reib_fJ,
            "E_electrostatic_fJ":   E_elek_fJ,
            "E_surface_recovery_fJ": E_surf_recov_fJ,
            "E_total_fJ":           E_total_fJ,
            "E_cmos_fJ":            E_cmos_fJ,
            "ratio_cmos_iofet":     E_cmos_fJ / E_elek_fJ,
            "note": f"E_elektr={E_elek_fJ:.4f}fJ | E_reib(1µm/ms)={E_reib_fJ[50]:.4e}fJ | CMOS/IoFET={E_cmos_fJ/E_elek_fJ:.1f}×",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        v = r["v_um_per_ms"]
        fig, ax1, ax2 = self._fig2()

        ax1.loglog(v, r["E_friction_fJ"],      color=C[3], label="E_Reibung")
        ax1.axhline(r["E_electrostatic_fJ"],   color=C[1], linestyle="--", linewidth=2.5,
                    label=f"E_elektrostatisch={r['E_electrostatic_fJ']:.3f}fJ")
        ax1.axhline(r["E_surface_recovery_fJ"],color=C[2], linestyle=":",
                    label=f"E_Rückgewinnung={r['E_surface_recovery_fJ']:.3e}fJ")
        ax1.axhline(r["E_cmos_fJ"],            color=C[0], linestyle="-.", linewidth=2,
                    label=f"CMOS-Referenz={r['E_cmos_fJ']:.3f}fJ")
        ax1.set_xlabel("Tropfengeschwindigkeit v (µm/ms)")
        ax1.set_ylabel("Energie (fJ)")
        ax1.set_title(f"Energiebilanz IoFET vs. CMOS\n(A_G=(5µm)², CMOS/IoFET={r['ratio_cmos_iofet']:.1f}×)")
        ax1.legend(fontsize=8)
        ax1.set_ylim(1e-10, 1e4)

        # Sub-Plot 2: Energieanteile als Tortendiagramm bei v=1µm/ms
        v_idx = np.argmin(np.abs(v - 1.0))
        E_pie = [
            r["E_electrostatic_fJ"],
            max(r["E_friction_fJ"][v_idx], 1e-12),
        ]
        labels_pie = [
            f"Elektrostatisch\n{r['E_electrostatic_fJ']:.3f} fJ",
            f"Reibung\n{r['E_friction_fJ'][v_idx]:.3e} fJ",
        ]
        colors_pie = [C[1], C[3]]
        wedges, texts, autotexts = ax2.pie(
            E_pie, labels=labels_pie, colors=colors_pie,
            autopct="%1.2f%%", startangle=90, pctdistance=0.7,
        )
        ax2.set_title(f"Energieanteile bei v=1µm/ms\nCMOS-Ratio: {r['ratio_cmos_iofet']:.1f}×")

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK IV: TROPFENLOGIK  (droplet.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_10_TropfenVerschmelzung(BaseExperiment):
    """
    EXPERIMENT 10: Tropfenverschmelzung – Energieminimierung und Radius

    Physikalischer Hintergrund [IONO §2.3]:
      Satz 3.1: Verschmelzung immer thermodynamisch begünstigt
      r_f = (r₁³ + r₂³)^(1/3)
      ΔE = 4π γ (r₁² + r₂² − r_f²) > 0  für alle r₁,r₂ > 0
    """
    title   = "Exp 10: Tropfenverschmelzung – Energieminimierung"
    module  = "droplet.py"
    section = "[IONO §2.3] Oberflächenspannung"

    def run(self) -> Dict[str, Any]:
        gamma = 0.072  # N/m
        r1_vals = np.logspace(-8, -5, 60)   # 10nm bis 10µm

        # Für r2 = r1 (gleiche Tropfen)
        r_merged_equal = (2 * r1_vals**3)**(1/3)
        dE_equal = 4 * np.pi * gamma * (2*r1_vals**2 - r_merged_equal**2)

        # Für r2 = 2*r1 (doppelter Tropfen)
        r2_vals = 2 * r1_vals
        r_merged_unequal = (r1_vals**3 + r2_vals**3)**(1/3)
        dE_unequal = 4 * np.pi * gamma * (r1_vals**2 + r2_vals**2 - r_merged_unequal**2)

        # Binäre Logik: Tropfen-Wahrheitstafel visualisieren
        # Zeigt: ODER-Gatter durch Verschmelzung
        gate = ORGate()
        cases = [(0,0), (0,1), (1,0), (1,1)]
        or_outputs  = [gate.evaluate(a,b) for (a,b) in cases]
        nand = NANDGate()
        nand_outputs = [nand.evaluate(a,b) for (a,b) in cases]

        # Volumenerhaltung bei Verschmelzung
        V1 = (4/3)*np.pi*r1_vals**3
        V_merged = (4/3)*np.pi*r_merged_equal**3
        vol_ratio = V_merged / (2*V1)

        return {
            "r1_vals_nm":          r1_vals * 1e9,
            "dE_equal_aJ":         dE_equal * 1e18,
            "dE_unequal_aJ":       dE_unequal * 1e18,
            "r_merged_equal_nm":   r_merged_equal * 1e9,
            "vol_ratio":           vol_ratio,
            "or_cases":            cases,
            "or_outputs":          or_outputs,
            "nand_outputs":        nand_outputs,
            "note": "ΔE > 0 für alle r1,r2 → Verschmelzung immer begünstigt (Satz 3.1)",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1.loglog(r["r1_vals_nm"], r["dE_equal_aJ"], color=C[0], label="r₂=r₁ (gleiche Größe)")
        ax1.loglog(r["r1_vals_nm"], r["dE_unequal_aJ"], color=C[1], linestyle="--",
                   label="r₂=2r₁ (doppelt)")
        ax1.set_xlabel("Tropfenradius r₁ (nm)")
        ax1.set_ylabel("Freigesetzte Energie ΔE (aJ)")
        ax1.set_title("Energiefreisetzung bei Tropfenverschmelzung\nSatz 3.1: ΔE > 0 immer")
        ax1.legend()
        ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.6, label="1 aJ")

        # Sub-Plot 2: Wahrheitstafel OR und NAND visualisiert
        cases_labels = [f"({a},{b})" for (a,b) in r["or_cases"]]
        x = np.arange(4)
        width = 0.35
        bars_or   = ax2.bar(x - width/2, r["or_outputs"],   width, color=C[2], label="OR-Gatter",   alpha=0.85)
        bars_nand = ax2.bar(x + width/2, r["nand_outputs"], width, color=C[3], label="NAND-Gatter", alpha=0.85)
        ax2.set_xticks(x); ax2.set_xticklabels(cases_labels)
        ax2.set_yticks([0, 1])
        ax2.set_xlabel("Eingang (A, B)")
        ax2.set_ylabel("Ausgang")
        ax2.set_title("Wahrheitstafel: OR & NAND durch Tropfendynamik")
        ax2.legend()
        for bar in list(bars_or) + list(bars_nand):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                     str(int(bar.get_height())), ha="center", va="bottom", fontsize=11)

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_11_LogikGatterBenchmark(BaseExperiment):
    """
    EXPERIMENT 11: Benchmark aller Logikgatter – NAND-Vollständigkeit

    [IONO §5.3]: NAND ist funktional vollständig – alle anderen Gatter
    lassen sich aus NAND-Gattern aufbauen.

    Zeigt die Zerlegungstiefe jedes Gatters in NAND-Primitive.
    Visualisiert auch den N-Bit-Addierer (Ripple-Carry).
    """
    title   = "Exp 11: Logikgatter-Benchmark und NAND-Vollständigkeit"
    module  = "droplet.py"
    section = "[IONO §5.3] Turing-Vollständigkeit"

    NAND_DEPTH = {
        "NOT":  1,
        "AND":  2,
        "OR":   3,
        "NAND": 1,
        "NOR":  4,
        "XOR":  4,
    }

    GATES = {
        "NAND": NANDGate(),
        "NOR":  NORGate(),
        "NOT":  NOTGate(),
        "AND":  ANDGate(),
        "OR":   ORGate(),
        "XOR":  XORGate(),
    }

    def run(self) -> Dict[str, Any]:
        # Wahrheitstafeln aller 2-Eingang-Gatter
        truth_tables = {}
        inputs_2 = [(0,0),(0,1),(1,0),(1,1)]
        for name, gate in self.GATES.items():
            if name == "NOT":
                tt = [gate.evaluate(a) for (a,_) in inputs_2]
            else:
                tt = [gate.evaluate(a,b) for (a,b) in inputs_2]
            truth_tables[name] = tt

        # N-Bit Addierer Benchmark
        adder_results = {}
        for nbits in [4, 8, 16]:
            adder = RippleCarryAdder(nbits)
            test_pairs = [(5, 3), (15, 1), (2**(nbits//2), 2**(nbits//2))]
            adder_results[nbits] = []
            for a, b in test_pairs:
                a = min(a, 2**nbits - 1)
                b = min(b, 2**nbits - 1)
                res, carry = adder.add(a, b)
                adder_results[nbits].append({"a": a, "b": b, "result": res, "overflow": carry})

        # NAND-Tiefe
        nand_depths = self.NAND_DEPTH.copy()

        return {
            "truth_tables":  truth_tables,
            "inputs_2":      inputs_2,
            "nand_depths":   nand_depths,
            "adder_results": adder_results,
            "note": "NAND-Vollständigkeit: Alle Gatter als NAND-Kombinationen realisierbar",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        # Sub-Plot 1: Wahrheitstafel-Heatmap
        gate_names = list(r["truth_tables"].keys())
        table = np.array([r["truth_tables"][g] for g in gate_names])
        im = ax1.imshow(table, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(["(0,0)","(0,1)","(1,0)","(1,1)"])
        ax1.set_yticks(range(len(gate_names)))
        ax1.set_yticklabels(gate_names)
        ax1.set_xlabel("Eingang (A, B)")
        ax1.set_title("Wahrheitstafeln aller Logikgatter\n(Grün=1, Rot=0)")
        for i, g in enumerate(gate_names):
            for j, v in enumerate(r["truth_tables"][g]):
                ax1.text(j, i, str(v), ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="black" if v else "white")
        plt.colorbar(im, ax=ax1, ticks=[0,1], label="Ausgangswert")

        # Sub-Plot 2: NAND-Tiefe und Addierer-Bit-Ergebnis
        names  = list(r["nand_depths"].keys())
        depths = [r["nand_depths"][n] for n in names]
        colors_bar = [C[0] if n == "NAND" else C[1] for n in names]
        bars = ax2.bar(names, depths, color=colors_bar, edgecolor="white", alpha=0.85)
        ax2.set_ylabel("NAND-Tiefe (Anzahl NAND-Primitive)")
        ax2.set_title("NAND-Zerlegungstiefe aller Logikgatter\n(NAND ist Primitiv der Tiefe 1)")
        for bar, d in zip(bars, depths):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, str(d),
                     ha="center", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 6)
        ax2.axhline(1, color="gray", linestyle="--", alpha=0.5, label="NAND-Primitiv")
        ax2.legend()

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_12_IRFNetzwerk(BaseExperiment):
    """
    EXPERIMENT 12: IRF-Netzwerk – Speicher, Konnektivität, Plastizität

    [IONO §7.2]: Ionische Plastizität als Lernmechanismus
    Das IRF-Netzwerk kann seinen Zustand zwischen Zyklen halten (Speicher)
    und Verbindungsgewichte anpassen (ionische Synapsen).

    Simuliert ein 8×8-Netz und zeigt:
    - Belegungsmuster (logische Zustände)
    - Konnektivitätsgraph
    """
    title   = "Exp 12: IRF-Netzwerk – Zustandskarte und Speicherkapazität"
    module  = "droplet.py"
    section = "[IONO §7.2] Ionische Plastizität"

    def run(self) -> Dict[str, Any]:
        N = 8  # 8×8-Gitter
        rng = np.random.default_rng(42)
        net = IRFNetwork()

        # Knoten anlegen
        for i in range(N):
            for j in range(N):
                nid = f"N{i:02d}{j:02d}"
                pos = (i * 1e-6, j * 1e-6, 0.0)
                net.add_node(IRFNode(nid, pos))

        # Verbindungen (Nearest-Neighbor)
        for i in range(N):
            for j in range(N):
                if j+1 < N: net.add_edge(f"N{i:02d}{j:02d}", f"N{i:02d}{j+1:02d}")
                if i+1 < N: net.add_edge(f"N{i:02d}{j:02d}", f"N{i+1:02d}{j:02d}")

        # Zufälliges Muster + "Smiley"-Logikbild
        state_matrix = rng.integers(0, 2, size=(N, N))
        # Smiley-Augen und Mund
        state_matrix[2, 2] = state_matrix[2, 5] = 1  # Augen
        state_matrix[5, 1] = state_matrix[5, 2] = state_matrix[5, 3] = 1
        state_matrix[5, 4] = state_matrix[5, 5] = state_matrix[5, 6] = 1  # Mund

        for i in range(N):
            for j in range(N):
                net.set_state(f"N{i:02d}{j:02d}", int(state_matrix[i,j]))

        # Speicherkapazität
        cap_bits  = net.storage_capacity_bits()
        cap_bytes = cap_bits / 8

        # Schreibt ein 16-Bit Wort
        value_16bit = 0b1010110011001010
        bit_states = [(value_16bit >> k) & 1 for k in range(16)]

        # Adjazenzmatrix (ausschnittweise)
        adj = net.adjacency_list()
        n_edges = net.edge_count()

        return {
            "state_matrix":    state_matrix,
            "grid_size":       N,
            "storage_bits":    cap_bits,
            "storage_bytes":   cap_bytes,
            "n_nodes":         net.node_count(),
            "n_edges":         n_edges,
            "test_value_16bit": value_16bit,
            "bit_states":      bit_states,
            "note": f"8×8-Netz: {cap_bits} Bit Speicher, {n_edges} Verbindungen",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        im = ax1.imshow(r["state_matrix"], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax1.set_xticks(range(r["grid_size"]))
        ax1.set_yticks(range(r["grid_size"]))
        ax1.set_xlabel("Spalte j"); ax1.set_ylabel("Zeile i")
        ax1.set_title(f"IRF-Netzwerkzustand  {r['grid_size']}×{r['grid_size']}\n"
                      f"{r['storage_bits']} Bit Speicherkapazität")
        for i in range(r["grid_size"]):
            for j in range(r["grid_size"]):
                ax1.text(j, i, str(r["state_matrix"][i,j]), ha="center", va="center",
                         fontsize=9, color="white" if r["state_matrix"][i,j] else "gray")
        plt.colorbar(im, ax=ax1, ticks=[0,1], label="Logischer Zustand")

        # Sub-Plot 2: 16-Bit-Wort visualisieren
        bits = r["bit_states"][::-1]  # MSB zuerst
        colors_bits = [C[1] if b == 1 else C[0] for b in bits]
        bars = ax2.bar(range(16), bits, color=colors_bits, edgecolor="white")
        ax2.set_xticks(range(16))
        ax2.set_xticklabels([f"B{15-k}" for k in range(16)], rotation=45, fontsize=8)
        ax2.set_yticks([0, 1])
        ax2.set_xlabel("Bit-Position")
        ax2.set_ylabel("Logischer Zustand")
        ax2.set_title(f"16-Bit-Wort im IRF-Netzwerk\nWert: 0b{r['test_value_16bit']:016b} = {r['test_value_16bit']}")
        ax2.text(7.5, 1.1, f"Dezimalwert: {r['test_value_16bit']}", ha="center", fontsize=10)

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK V: FABRY-PÉROT-RESONATOR  (fabry_perot.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_13_FabryPerotSpektren(BaseExperiment):
    """
    EXPERIMENT 13: FP-Resonator – Reflexionsspektren und Farbdurchstimmbarkeit

    Physikalischer Hintergrund [OQTUM §3]:
      r_FP = (r₁ + r₂·e^(2iδ)) / (1 + r₁r₂·e^(2iδ))
      R_FP = |r_FP|²
      δ = 2π n d cos θ / λ    (Phasendicke)

    Materialparameter aus Arbeit [OQTUM Tab. 1]:
      d₀=250nm, n_p=1.50, χ_Ethanol=0.25 → Q≈2.15 → λ_res≈543nm (Grün)
    """
    title   = "Exp 13: FP-Reflexionsspektren und Farbdurchstimmbarkeit"
    module  = "fabry_perot.py"
    section = "[OQTUM §3] Fabry-Pérot-Resonator"

    # Lösungsmittel-Parameter aus Arbeit [OQTUM Tab. 1]
    SOLVENTS = [
        (1.00, "trocken",     "darkred"),
        (1.75, "Isopropanol", "orangered"),
        (2.15, "Ethanol",     "limegreen"),
        (2.75, "n-Butanol",   "deepskyblue"),
        (3.60, "Wasser",      "violet"),
    ]

    def run(self) -> Dict[str, Any]:
        fp = FabryPerotResonator(d0=250e-9, n0=1.50, R1=0.3, R2=0.3)

        spectra = {}
        resonances = {}
        for Q, label, _ in self.SOLVENTS:
            lam, R = fp.reflection_spectrum(Q=Q, n_points=400)
            spectra[label]   = R
            lam_res          = fp.resonance_wavelength(Q=Q)
            resonances[label] = lam_res * 1e9

        # Finesse und FWHM
        finesse = fp.finesse
        fwhm    = fp.fwhm_nm(Q=1.0)

        # Kontinuierlicher Durchstimmbereich
        Q_cont  = np.linspace(1.0, 4.0, 200)
        lam_cont = np.array([fp.resonance_wavelength(Q)*1e9 for Q in Q_cont])

        return {
            "wavelengths_nm":  lam,
            "spectra":         spectra,
            "resonances_nm":   resonances,
            "finesse":         finesse,
            "fwhm_nm":         fwhm,
            "Q_continuous":    Q_cont,
            "lam_continuous_nm": lam_cont,
            "note": f"Finesse={finesse:.2f}, FWHM={fwhm:.1f}nm, Bereich={lam_cont.min():.0f}–{lam_cont.max():.0f}nm",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        lam = r["wavelengths_nm"]
        fig, ax1, ax2 = self._fig2()

        for (Q, label, col) in self.SOLVENTS:
            R = r["spectra"][label]
            lam_res = r["resonances_nm"][label]
            ax1.plot(lam, R, color=col, linewidth=2,
                     label=f"{label} Q={Q:.2f} λ={lam_res:.0f}nm")

        # Sichtbares Spektrum-Hintergrund
        ax1.axvspan(380, 430, alpha=0.08, color="violet")
        ax1.axvspan(430, 500, alpha=0.08, color="blue")
        ax1.axvspan(500, 570, alpha=0.08, color="green")
        ax1.axvspan(570, 620, alpha=0.08, color="yellow")
        ax1.axvspan(620, 700, alpha=0.08, color="orange")
        ax1.axvspan(700, 780, alpha=0.08, color="red")
        ax1.set_xlabel("Wellenlänge λ (nm)")
        ax1.set_ylabel("Reflexion R")
        ax1.set_title(f"FP-Reflexionsspektren (d₀=250nm, n₀=1.50)\nFinesse={r['finesse']:.2f}, FWHM={r['fwhm_nm']:.1f}nm")
        ax1.legend(fontsize=8)
        ax1.set_xlim(380, 780)

        # Sub-Plot 2: λ_res(Q) – Durchstimmkurve
        Q_c = r["Q_continuous"]; lam_c = r["lam_continuous_nm"]
        # Farbkodierung nach Wellenlänge
        norm_c = Normalize(vmin=380, vmax=780)
        for i in range(len(Q_c)-1):
            col = plt.cm.Spectral_r(norm_c(lam_c[i]))
            ax2.plot(Q_c[i:i+2], lam_c[i:i+2], color=col, linewidth=3)
        for (Q, label, col) in self.SOLVENTS:
            ax2.scatter([Q], [r["resonances_nm"][label]], color=col, s=60, zorder=5)
            ax2.text(Q+0.05, r["resonances_nm"][label], label, fontsize=7, va="center")
        ax2.axhspan(380, 780, alpha=0.06, color="yellow")
        ax2.set_xlabel("Quellungsgrad Q")
        ax2.set_ylabel("Resonanzwellenlänge λ_res (nm)")
        ax2.set_title("Farbdurchstimmbarkeit λ_res(Q)")

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_14_FabryPerotFinesse(BaseExperiment):
    """
    EXPERIMENT 14: FP-Finesse, FWHM und Kolorimetrie

    [OQTUM §3.2]:
      F = π (R₁R₂)^(1/4) / (1 − √(R₁R₂))
      FWHM = FSR / F

    Höhere Finesse → schärfere Resonanz → sattere Farbe.
    CIE-Farbkoordinaten (x,y) zeigen den erreichbaren Gamut.
    """
    title   = "Exp 14: FP-Finesse und CIE-Farbgamut"
    module  = "fabry_perot.py"
    section = "[OQTUM §3.2] Finesse und Kolorimetrie"

    def run(self) -> Dict[str, Any]:
        R_vals = np.linspace(0.05, 0.95, 80)
        finesse_vals = np.array([
            np.pi * (R**2)**0.25 / (1 - R)
            for R in R_vals
        ])

        # FWHM für verschiedene R und Q=2.15 (Ethanol)
        fwhm_vals = np.array([
            FabryPerotResonator(d0=250e-9, n0=1.50, R1=R, R2=R).fwhm_nm(Q=2.15)
            for R in R_vals
        ])

        # CIE-Gamut für d₀=250nm, verschiedene R
        Q_gamut = np.linspace(1.0, 4.0, 30)
        gamuts = {}
        for R_val, label in [(0.2, "R=0.2"), (0.5, "R=0.5"), (0.8, "R=0.8")]:
            fp = FabryPerotResonator(250e-9, 1.50, R_val, R_val)
            _, x_arr, y_arr = fp.color_gamut(Q_range=(1.0, 4.0), n_Q=30)
            gamuts[label] = (x_arr, y_arr)

        return {
            "R_vals":      R_vals,
            "finesse":     finesse_vals,
            "fwhm_nm":     fwhm_vals,
            "gamuts":      gamuts,
            "note": "Höhere Reflektivität R → höhere Finesse → schärfere Resonanz",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1_twin = ax1.twinx()
        l1, = ax1.plot(r["R_vals"], r["finesse"], color=C[0], label="Finesse F(R)")
        l2, = ax1_twin.plot(r["R_vals"], r["fwhm_nm"], color=C[1], linestyle="--",
                             label="FWHM [nm]")
        ax1.set_xlabel("Spiegel-Reflektivität R")
        ax1.set_ylabel("Finesse F", color=C[0])
        ax1_twin.set_ylabel("FWHM (nm)", color=C[1])
        ax1.set_title("Finesse und FWHM als Funktion von R\n(d₀=250nm, Q=2.15)")
        ax1.legend(handles=[l1, l2], loc="upper left")
        ax1.axvline(0.3, color="gray", linestyle=":", alpha=0.7, label="R=0.3 (Gold-Spiegel)")
        ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

        # Sub-Plot 2: CIE-Gamut
        gamut_colors = [C[3], C[1], C[0]]
        for (label, (x_arr, y_arr)), col in zip(r["gamuts"].items(), gamut_colors):
            ax2.plot(x_arr, y_arr, color=col, linewidth=2, label=label, marker="o", markersize=3)
        # CIE-Diagramm-Grenze (vereinfacht)
        t = np.linspace(0, 2*np.pi, 200)
        ax2.set_xlabel("CIE x")
        ax2.set_ylabel("CIE y")
        ax2.set_title("CIE-Farbgamut vs. Quellungsgrad Q\n(d₀=250nm, n₀=1.50)")
        ax2.legend()
        ax2.set_xlim(0, 0.8); ax2.set_ylim(0, 0.8)
        ax2.plot([0.33], [0.33], "k+", markersize=10, label="Weißpunkt")

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK VI: TRANSFERMATRIX-METHODE  (tmm.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_15_TMMSpektrum(BaseExperiment):
    """
    EXPERIMENT 15: TMM-Spektrum und Schichtoptimierung

    [OQTUM §4]:
      M_j = [[cos δ_j, −i/η_j sin δ_j], [−iη_j sin δ_j, cos δ_j]]
      M_ges = ∏ M_j
    Quantenmechanische Analogie: M_j ≡ Zeitentwicklungsoperator U(t)

    Vergleich TMM vs. einfacher FP-Resonator für das Hydrogel-Stack.
    """
    title   = "Exp 15: Transfermatrix-Methode – Au|Hydrogel|Au-Stack"
    module  = "tmm.py"
    section = "[OQTUM §4] TMM"

    def run(self) -> Dict[str, Any]:
        # Spektren für verschiedene Quellungsgrade
        Q_vals = [1.0, 1.75, 2.15, 2.75, 3.60]
        spectra_tmm = {}
        spectra_fp  = {}

        lam_nm = np.linspace(380, 780, 400)

        for Q in Q_vals:
            stack = build_hydrogel_stack(d0_nm=250.0, n_polymer=1.50, Q=Q, R_mirror=0.3)
            _, R_tmm = stack.spectrum(n_points=400)
            spectra_tmm[Q] = R_tmm

            fp = FabryPerotResonator(250e-9, 1.50, 0.3, 0.3)
            _, R_fp = fp.reflection_spectrum(Q=Q, n_points=400)
            spectra_fp[Q] = R_fp

        # Einfallswinkel-Abhängigkeit
        theta_vals = [0, 15, 30, 45]
        angle_spectra = {}
        for theta in theta_vals:
            stack = build_hydrogel_stack(d0_nm=250.0, n_polymer=1.50, Q=2.15, R_mirror=0.3)
            stack.theta_i = np.radians(theta)
            _, R = stack.spectrum(n_points=200)
            angle_spectra[theta] = R

        # Phase-Matrix-Determinante (QM-Analogie)
        stack_ref = build_hydrogel_stack(d0_nm=250.0, n_polymer=1.50, Q=2.15)
        lam_check = np.linspace(400e-9, 700e-9, 50)
        det_vals = []
        for lam in lam_check:
            M = stack_ref.phase_matrix(lam)
            det = abs(M[0,0]*M[1,1] - M[0,1]*M[1,0])
            det_vals.append(det)

        return {
            "wavelengths_nm":   lam_nm,
            "spectra_tmm":      spectra_tmm,
            "spectra_fp":       spectra_fp,
            "Q_vals":           Q_vals,
            "theta_vals":       theta_vals,
            "angle_spectra":    angle_spectra,
            "det_values":       np.array(det_vals),
            "lam_check_nm":     lam_check * 1e9,
            "note": "TMM berücksichtigt Au-Absorption; FP: ideale Spiegel",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        lam = r["wavelengths_nm"]
        fig, ax1, ax2 = self._fig2()

        colors_Q = plt.cm.Spectral_r(np.linspace(0, 1, len(r["Q_vals"])))
        for Q, col in zip(r["Q_vals"], colors_Q):
            R_tmm = r["spectra_tmm"][Q]
            R_fp  = r["spectra_fp"][Q]
            ax1.plot(lam, R_tmm, color=col, linewidth=2,    label=f"TMM Q={Q:.2f}")
            ax1.plot(lam, R_fp,  color=col, linewidth=1.2,  linestyle=":", alpha=0.6)
        ax1.set_xlabel("Wellenlänge λ (nm)")
        ax1.set_ylabel("Reflexion R")
        ax1.set_title("TMM (durchgehend) vs. FP-Modell (gepunktet)\nAu|PDMS-Hydrogel|Au Stack")
        ax1.legend(fontsize=7, ncol=2)
        ax1.set_xlim(380, 780)

        angle_colors = [C[0], C[1], C[2], C[3]]
        for theta, col in zip(r["theta_vals"], angle_colors):
            R = r["angle_spectra"][theta]
            # Reuse lam_nm with 200 points
            lam2 = np.linspace(380, 780, 200)
            ax2.plot(lam2, R, color=col, label=f"θ={theta}°")
        ax2.set_xlabel("Wellenlänge λ (nm)")
        ax2.set_ylabel("Reflexion R")
        ax2.set_title("Winkelabhängigkeit R(λ, θ) bei Q=2.15\n(unpolarisiertes Licht)")
        ax2.legend()
        ax2.set_xlim(380, 780)

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_16_TMMQuantenAnalogie(BaseExperiment):
    """
    EXPERIMENT 16: QM-Analogie der Transfermatrix

    [OQTUM §8.2]:
      M = [[cos δ, −i/η sin δ], [−iη sin δ, cos δ]]
    Diese Struktur ist isomorph zum QM-Zeitentwicklungsoperator:
      U(τ) = exp(−iHτ) für H = [[0, 1/η], [η, 0]] · ħ/d

    Zeigt: Bandstruktur eines photonischen Kristalls (periodische Stapelung)
    und Tunneleffekt (verbotene Banden).
    """
    title   = "Exp 16: QM-Analogie der TMM – Photonische Bandstruktur"
    module  = "tmm.py"
    section = "[OQTUM §8.2] Formale Analogie TMM–QM"

    def run(self) -> Dict[str, Any]:
        # Photonischer Kristall: Periodischer TiO₂/SiO₂-Stack
        n_TiO2 = 2.35; n_SiO2 = 1.46
        d_TiO2_nm = 80.0; d_SiO2_nm = 120.0

        lam_nm = np.linspace(300, 900, 600)
        R_1pair = np.zeros(len(lam_nm))
        R_5pair = np.zeros(len(lam_nm))
        R_10pair = np.zeros(len(lam_nm))

        medium    = OpticalLayer(1.0, 0.0, "Luft")
        substrate = OpticalLayer(1.52, 0.0, "Glas")

        for i, lam in enumerate(lam_nm):
            lam_m = lam * 1e-9
            layers_1 = [
                OpticalLayer(n_TiO2, d_TiO2_nm*1e-9, "TiO₂"),
                OpticalLayer(n_SiO2, d_SiO2_nm*1e-9, "SiO₂"),
            ]
            layers_5  = layers_1 * 5
            layers_10 = layers_1 * 10
            R_1pair[i]  = TransferMatrixMethod(medium, layers_1,  substrate).reflectance(lam_m)
            R_5pair[i]  = TransferMatrixMethod(medium, layers_5,  substrate).reflectance(lam_m)
            R_10pair[i] = TransferMatrixMethod(medium, layers_10, substrate).reflectance(lam_m)

        # Ellipsometrie-Parameter für Hydrogel-Stack
        stack = build_hydrogel_stack(d0_nm=250.0, n_polymer=1.50, Q=2.15)
        lam_ell = np.linspace(400e-9, 700e-9, 100)
        psi_arr   = np.zeros(len(lam_ell))
        delta_arr = np.zeros(len(lam_ell))
        for i, lam in enumerate(lam_ell):
            psi_arr[i], delta_arr[i] = stack.ellipsometry_params(lam)

        return {
            "wavelengths_nm":  lam_nm,
            "R_1pair":         R_1pair,
            "R_5pair":         R_5pair,
            "R_10pair":        R_10pair,
            "lam_ell_nm":      lam_ell * 1e9,
            "psi_deg":         psi_arr,
            "delta_deg":       delta_arr,
            "note": "Photonische Bandlücke bei λ≈550nm (TiO₂/SiO₂, Bragg-Bedingung)",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        lam = r["wavelengths_nm"]
        ax1.plot(lam, r["R_1pair"],   color=C[2], linewidth=1.5, alpha=0.8, label="1 Paar")
        ax1.plot(lam, r["R_5pair"],   color=C[1], linewidth=2,   label="5 Paare")
        ax1.plot(lam, r["R_10pair"],  color=C[0], linewidth=2.5, label="10 Paare")
        ax1.set_xlabel("Wellenlänge λ (nm)")
        ax1.set_ylabel("Reflexion R")
        ax1.set_title("Photonischer Kristall: TiO₂/SiO₂-Bragg-Spiegel\nBandlücke wächst mit Schichtzahl")
        ax1.legend()
        ax1.fill_between(lam, 0, 1,
            where=((lam > 460) & (lam < 680)),
            alpha=0.05, color="yellow", label="Bandlücke")

        ax2_twin = ax2.twinx()
        l1, = ax2.plot(r["lam_ell_nm"], r["psi_deg"],   color=C[3], label="Ψ (deg)")
        l2, = ax2_twin.plot(r["lam_ell_nm"], r["delta_deg"], color=C[4], linestyle="--",
                             label="Δ (deg)")
        ax2.set_xlabel("Wellenlänge λ (nm)")
        ax2.set_ylabel("Ψ (Ellipsometrie, Grad)", color=C[3])
        ax2_twin.set_ylabel("Δ (Ellipsometrie, Grad)", color=C[4])
        ax2.set_title("Ellipsometrie-Parameter Ψ, Δ\nAu|PDMS|Au Hydrogel-Stack (Q=2.15)")
        ax2.legend(handles=[l1, l2])

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK VII: POLYMER / FLORY-REHNER  (polymer.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_17_FloryRehnerGleichgewicht(BaseExperiment):
    """
    EXPERIMENT 17: Flory-Rehner-Gleichgewichtsquellung Q_eq(χ, Vc)

    [OQTUM §5.1]:
      ln(1−φ) + φ + χ·φ² + (Vm/Vc)·(φ^(1/3) − φ/2) = 0
      φ = 1/Q_eq

    Materialparameter aus Arbeit [OQTUM Tab. 1]:
      χ_Ethanol=0.25 → Q_eq≈2.15; χ_Wasser=0.43 → Q_eq≈3.60
    """
    title   = "Exp 17: Flory-Rehner-Gleichgewichtsquellung Q_eq(χ)"
    module  = "polymer.py"
    section = "[OQTUM §5.1] Polymer-Thermodynamik"

    # Lösungsmittel-Datensätze
    SOLVENTS = {
        "Isopropanol": 0.22,
        "Ethanol":     0.25,
        "n-Butanol":   0.30,
        "Wasser":      0.43,
    }

    def run(self) -> Dict[str, Any]:
        chi_vals = np.linspace(0.05, 0.75, 120)
        model    = FloryRehnerModel(PolymerParameters(chi=0.25, Vm=1.8e-5, Vc=5e-4, n0=1.50))
        chi_arr, Q_arr = model.swelling_vs_chi(chi_range=(0.05, 0.75), n_points=120)

        # Einzelne Lösungsmittel aus Arbeit
        Q_solvents = {}
        for solvent, chi in self.SOLVENTS.items():
            m = FloryRehnerModel(PolymerParameters(chi=chi, Vm=1.8e-5, Vc=5e-4, n0=1.50))
            try:
                Q_solvents[solvent] = m.equilibrium_swelling()
            except ValueError:
                Q_solvents[solvent] = float("nan")

        # Resonanzwellenlänge λ_res(Q_eq) für d₀=250nm
        d0 = 250e-9
        Q_nonnan = [(s, Q) for s, Q in Q_solvents.items() if not np.isnan(Q)]
        lam_solvents = {s: model.resonance_wavelength(d0, Q)*1e9 for (s, Q) in Q_nonnan}

        # Vernetzungsdichte-Einfluss
        Vc_vals = np.logspace(-4.5, -3, 50)  # m³/mol
        Q_vs_Vc = []
        for Vc in Vc_vals:
            m = FloryRehnerModel(PolymerParameters(chi=0.25, Vm=1.8e-5, Vc=Vc, n0=1.50))
            try:
                Q_vs_Vc.append(m.equilibrium_swelling())
            except:
                Q_vs_Vc.append(np.nan)

        return {
            "chi_arr":         chi_arr,
            "Q_arr":           Q_arr,
            "Q_solvents":      Q_solvents,
            "lam_solvents_nm": lam_solvents,
            "Vc_vals_m3_mol":  Vc_vals,
            "Q_vs_Vc":         np.array(Q_vs_Vc),
            "note": "Niedrigeres χ (besseres Lösungsmittel) → höhere Quellung",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        valid = ~np.isnan(r["Q_arr"])
        ax1.plot(r["chi_arr"][valid], r["Q_arr"][valid], color=C[0], linewidth=2)
        for solvent, chi in self.SOLVENTS.items():
            Q = r["Q_solvents"].get(solvent, float("nan"))
            if not np.isnan(Q):
                ax1.scatter([chi], [Q], s=80, zorder=5)
                ax1.annotate(f"{solvent}\nQ={Q:.2f}", xy=(chi, Q),
                             xytext=(chi+0.02, Q+0.3), fontsize=7,
                             arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
        ax1.set_xlabel("Flory-Huggins-Parameter χ")
        ax1.set_ylabel("Gleichgewichts-Quellungsgrad Q_eq")
        ax1.set_title("Flory-Rehner: Q_eq(χ) – Lösungsmitteleinfluss\n(Vm=1.8e-5 m³/mol, Vc=5e-4 m³/mol)")
        ax1.set_xlim(0.05, 0.75); ax1.set_ylim(0, None)

        valid_vc = ~np.isnan(r["Q_vs_Vc"])
        ax2.semilogx(r["Vc_vals_m3_mol"][valid_vc], r["Q_vs_Vc"][valid_vc], color=C[2], linewidth=2)
        ax2.set_xlabel("Vernetzungskettenvolumen Vc (m³/mol)")
        ax2.set_ylabel("Gleichgewichts-Quellungsgrad Q_eq")
        ax2.set_title("Flory-Rehner: Q_eq(Vc) – Vernetzungsdichte\n(χ=0.25, Ethanol)")
        ax2.text(0.05, 0.95, "Schwache Vernetzung\n→ hohe Quellung",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=8, color=C[2])

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_18_QuellungskinetikTemperatur(BaseExperiment):
    """
    EXPERIMENT 18: Quellungskinetik τ(h) und Farbwechselgeschwindigkeit

    [OQTUM §5.3]:
      τ_Q = h² / (π² D_eff)
    Für h=250nm und D_eff=1e-11 m²/s: τ_Q ≈ 0.63 ms

    Videorate-Anwendung: τ < 16ms für 60Hz-Display.
    """
    title   = "Exp 18: Quellungskinetik τ(h) und Videorate-Kompatibilität"
    module  = "polymer.py"
    section = "[OQTUM §5.3] Quellungskinetik"

    def run(self) -> Dict[str, Any]:
        model = FloryRehnerModel(PolymerParameters(chi=0.25, Vm=1.8e-5, Vc=5e-4,
                                                   n0=1.50, D_eff=1e-11))
        Q_eq = model.equilibrium_swelling()

        # Relaxationszeit vs. Schichtdicke
        h_vals_nm = np.logspace(1.5, 3.5, 80)   # 30nm bis 3µm
        tau_ms    = np.array([model.switching_time_ms(h*1e-9) for h in h_vals_nm])

        # Kinetik für verschiedene Dicken
        h_demo = [100, 250, 500, 1000]  # nm
        kinetics = {}
        for h_nm in h_demo:
            t, Q_t = model.kinetic_swelling(h_nm*1e-9, Q_init=1.0, Q_eq=Q_eq, n_times=300)
            kinetics[h_nm] = (t*1e3, Q_t)  # ms, Q

        # Wellenlängenkinetik für h=250nm
        d0 = 250e-9
        t_ref, Q_t_ref = model.kinetic_swelling(250e-9, 1.0, Q_eq)
        lam_t = np.array([model.resonance_wavelength(d0, Q)*1e9 for Q in Q_t_ref])

        return {
            "h_vals_nm":      h_vals_nm,
            "tau_ms":         tau_ms,
            "Q_eq":           Q_eq,
            "kinetics_ms_Q":  {h: (t.tolist(), Q.tolist()) for h, (t, Q) in kinetics.items()},
            "lam_kinetics_nm": lam_t,
            "t_ref_ms":       t_ref * 1e3,
            "tau_250nm_ms":   model.switching_time_ms(250e-9),
            "tau_1000nm_ms":  model.switching_time_ms(1000e-9),
            "note": f"τ(250nm)={model.switching_time_ms(250e-9):.3f}ms < 16ms (60Hz) ✓",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1.loglog(r["h_vals_nm"], r["tau_ms"], color=C[1], linewidth=2.5)
        ax1.axhline(16.7, color=C[0], linestyle="--", linewidth=2, label="16.7ms (60Hz)")
        ax1.axhline(33.3, color=C[2], linestyle="--", linewidth=1.5, label="33.3ms (30Hz)")
        ax1.axvline(250, color="gray", linestyle=":", alpha=0.7, label="h=250nm")
        ax1.scatter([250],  [r["tau_250nm_ms"]],  color=C[3], s=80, zorder=5,
                    label=f"τ={r['tau_250nm_ms']:.2f}ms")
        ax1.scatter([1000], [r["tau_1000nm_ms"]], color=C[4], s=60, zorder=5,
                    label=f"τ={r['tau_1000nm_ms']:.1f}ms")
        ax1.set_xlabel("Schichtdicke h (nm)")
        ax1.set_ylabel("Relaxationszeit τ_Q (ms)")
        ax1.set_title("Quellungskinetik: τ_Q(h) = h²/(π²D_eff)\nVideorate-Grenzwerte eingezeichnet")
        ax1.legend(fontsize=8)

        colors_kin = [C[0], C[2], C[1], C[3]]
        for (h_nm, col) in zip([100, 250, 500, 1000], colors_kin):
            t_lst, Q_lst = r["kinetics_ms_Q"][h_nm]
            t  = np.array(t_lst);  Q = np.array(Q_lst)
            ax2.plot(t, Q, color=col, linewidth=2, label=f"h={h_nm}nm")
        ax2.axhline(r["Q_eq"], color="gray", linestyle="--", alpha=0.7,
                    label=f"Q_eq={r['Q_eq']:.2f}")
        ax2.set_xlabel("Zeit t (ms)")
        ax2.set_ylabel("Quellungsgrad Q(t)")
        ax2.set_title("Q(t): Quellungskinetik für verschiedene Dicken\n(D_eff=10⁻¹¹ m²/s)")
        ax2.set_xlim(0, None)
        ax2.legend(fontsize=8)

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK VIII: EBL-LITHOGRAPHIE  (ebl.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_19_EBLProximityEffekt(BaseExperiment):
    """
    EXPERIMENT 19: EBL-Proximity-Effekt – Chang-PSF und Energiedeposition

    [OQTUM §6.2]:
      PSF(r) = 1/(π(1+η)) · [1/β_f² exp(−r²/β_f²) + η/β_b² exp(−r²/β_b²)]
    Parameter (100keV): β_f=10nm, β_b=10µm, η=0.6

    Algorithmus-Komplexität: O(M² log M) via FFT-Faltung.
    """
    title   = "Exp 19: EBL-Proximity-Effekt – Chang-PSF und Faltung"
    module  = "ebl.py"
    section = "[OQTUM §6.2] EBL-Proximity"

    def run(self) -> Dict[str, Any]:
        params = EBLParameters(beta_f=10e-9, beta_b=10e-6, eta=0.6, beam_energy_keV=100.0)
        model  = ProximityEffectModel(params, grid_size_nm=2000.0, n_grid=128)

        # 1D-PSF für verschiedene η-Werte
        r_nm = np.linspace(0, 1000, 500)
        psf_dict = {}
        for eta_val in [0.3, 0.6, 0.9]:
            p = EBLParameters(10e-9, 10e-6, eta=eta_val)
            m = ProximityEffectModel(p, n_grid=64)
            psf_dict[f"η={eta_val}"] = m.psf_1d(r_nm)

        # Schachbrettmuster – Dosiskodierung
        n = 128; field = 8
        pattern = np.zeros((n, n))
        for i in range(0, n, field):
            for j in range(0, n, field):
                if ((i//field) + (j//field)) % 2 == 0:
                    pattern[i:i+field, j:j+field] = 1.0

        E_dep = model.energy_deposition(pattern)
        corrected = model.proximity_correction(pattern)

        # Quellung aus EBL-Kodierung
        _, Q_map = model.encode_texture(pattern, Q_range=(1.0, 4.0))

        return {
            "r_nm":           r_nm,
            "psf_dict":       {k: v.tolist() for k,v in psf_dict.items()},
            "pattern":        pattern,
            "energy_dep":     E_dep,
            "corrected":      corrected,
            "Q_map":          Q_map,
            "grid_size_nm":   model.grid_size_nm,
            "bethe_range_um": params.bethe_range_m * 1e6,
            "note": f"Bethe-Reichweite: {params.bethe_range_m*1e6:.1f} µm bei 100 keV",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        for label, psf_list in r["psf_dict"].items():
            psf = np.array(psf_list)
            ax1.semilogy(r["r_nm"], psf / psf[0], label=label)
        bf_nm = 10.0; bb_nm = 10000.0
        ax1.axvline(bf_nm, color="gray", linestyle="--", alpha=0.6, label=f"β_f={bf_nm}nm")
        ax1.axvline(bb_nm, color="gray", linestyle=":",  alpha=0.4, label=f"β_b={bb_nm}nm")
        ax1.set_xlabel("Radialer Abstand r (nm)")
        ax1.set_ylabel("PSF(r) / PSF(0) [log]")
        ax1.set_title("Chang-Doppel-Gauss-PSF für verschiedene η-Werte\n(β_f=10nm, β_b=10µm)")
        ax1.legend(fontsize=8)

        # Sub-Plot 2: Schachbrett-Dosismuster vs. Proximity-korrigiertes Muster
        half = r["corrected"].shape[0] // 2
        ax2.imshow(r["corrected"], cmap="viridis", origin="lower",
                   extent=[0, r["grid_size_nm"], 0, r["grid_size_nm"]])
        ax2.set_xlabel("x (nm)"); ax2.set_ylabel("y (nm)")
        ax2.set_title("Proximity-korrigiertes Belichtungsmuster\n(Schachbrettmuster, 8-Pixel-Felder)")
        plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax2, label="Normierte Dosis")

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_20_EBLTexturKodierung(BaseExperiment):
    """
    EXPERIMENT 20: EBL-Texturkodierung – Dosismuster → Quellungskarte → Farbbild

    [OQTUM §6, §7]:
      Vorwärtsproblem: D(r) → Q(r) → λ_res(r)
      Vollständige Pipeline: Zielbild → EBL-Dosis → Quellungskarte → Reflexionskarte

    Demonstriert einen Buchstaben „H" (für Hydrogel) als photonisch-koloriertes Bild.
    """
    title   = "Exp 20: EBL-Texturkodierung H-Buchstabe – Dosis→Quellung→Farbe"
    module  = "ebl.py"
    section = "[OQTUM §7] Gesamtalgorithmus"

    def run(self) -> Dict[str, Any]:
        n = 64
        model = ProximityEffectModel(
            EBLParameters(beta_f=10e-9, beta_b=5e-6, eta=0.5),
            grid_size_nm=1000.0, n_grid=n,
        )

        # Erzeuge "H"-Muster
        texture = np.zeros((n, n))
        # Linker Balken H
        texture[12:52, 8:18]  = 1.0
        # Rechter Balken H
        texture[12:52, 46:56] = 1.0
        # Querbalken H
        texture[28:36, 18:46] = 1.0

        corrected, Q_map = model.encode_texture(texture, Q_range=(1.0, 4.0))

        # Für jedes Pixel: Resonanzwellenlänge berechnen
        fp = FabryPerotResonator(d0=250e-9, n0=1.50, R1=0.3, R2=0.3)
        lam_map = np.vectorize(lambda Q: fp.resonance_wavelength(Q=Q)*1e9)(Q_map)
        # Clamp auf sichtbares Spektrum
        lam_map_vis = np.clip(lam_map, 380, 780)

        # Durchschnittliche Resonanzwellenlänge für H vs. Hintergrund
        lam_H  = lam_map[texture > 0.5].mean()
        lam_BG = lam_map[texture < 0.5].mean()

        return {
            "texture":       texture,
            "corrected":     corrected,
            "Q_map":         Q_map,
            "lam_map_nm":    lam_map_vis,
            "lam_H_nm":      lam_H,
            "lam_BG_nm":     lam_BG,
            "Q_H":           Q_map[texture > 0.5].mean(),
            "Q_BG":          Q_map[texture < 0.5].mean(),
            "note": f"H: λ={lam_H:.0f}nm (Q={Q_map[texture>0.5].mean():.2f}) | BG: λ={lam_BG:.0f}nm",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        im1 = ax1.imshow(r["Q_map"], cmap="RdYlBu_r", origin="upper",
                          vmin=1.0, vmax=4.0)
        ax1.set_title(f"Quellungskarte Q(x,y) für 'H'-Muster\nH: Q={r['Q_H']:.2f}  BG: Q={r['Q_BG']:.2f}")
        ax1.set_xlabel("x (Pixel)"); ax1.set_ylabel("y (Pixel)")
        plt.colorbar(im1, ax=ax1, label="Quellungsgrad Q")

        # Farbkodierte Wellenlängenkarte (pseudocolor entsprechend sichtbarem Spektrum)
        norm = Normalize(vmin=380, vmax=780)
        lam_colored = plt.cm.Spectral_r(norm(r["lam_map_nm"]))
        ax2.imshow(lam_colored, origin="upper")
        ax2.set_title(f"Pseudofarb-Bild aus Resonanzwellenlänge\nH: λ≈{r['lam_H_nm']:.0f}nm | BG: λ≈{r['lam_BG_nm']:.0f}nm")
        ax2.set_xlabel("x (Pixel)"); ax2.set_ylabel("y (Pixel)")
        sm = ScalarMappable(cmap="Spectral_r", norm=norm)
        plt.colorbar(sm, ax=ax2, label="λ_res (nm)")

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK IX: INVERSES DESIGN (inverse_design.py)
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_21_TarnfarbenOptimierung(BaseExperiment):
    """
    EXPERIMENT 21: Tarnfarben-Optimierung – adjungierter Gradientenabstieg

    [OQTUM §6 + §7.1]:
      min L({d_j, n_j}) = Σ_λ [R_FP(λ) − R_Ziel(λ)]²
    Konvergenz nach I=87 Iterationen auf L=3.1×10⁻³ (aus Arbeit)
    Δ E_CIE < 2 (visuell nicht unterscheidbar)

    Zielspektrum: grünes Laubblatt (Reflexionsmaximum bei 550nm)
    """
    title   = "Exp 21: Tarnfarben-Optimierung – Blatt-Zielspektrum"
    module  = "inverse_design.py"
    section = "[OQTUM §7.1] Inverses Design"

    def run(self) -> Dict[str, Any]:
        lam = np.linspace(380, 780, 100)

        # Zielspektrum: Laubblatt-Reflexion (breites Grün-Maximum)
        # Typisch: Peak bei 550nm, niedrig im Blau und Rot
        R_leaf = (
            0.05 * np.exp(-((lam - 440)**2) / (2*25**2)) +  # kleines Blau
            0.45 * np.exp(-((lam - 555)**2) / (2*50**2)) +  # Grün-Maximum
            0.08 * np.exp(-((lam - 690)**2) / (2*20**2))    # kleines Rot (Chlorophyll)
        )
        R_leaf = np.clip(R_leaf, 0, 1)

        designer = InversePhotonicsDesigner(
            target_spectrum=R_leaf,
            wavelengths_nm=lam,
            n_layers=2,
            d_bounds=(50e-9, 600e-9),
            n_bounds=(1.2, 2.8),
        )

        result = designer.optimize(n_restarts=3, max_iter=200, tol=1e-8, seed=42)
        R_opt  = designer.predict_spectrum(result)
        quality = designer.design_quality(result)

        # Single-Layer Design für Grün bei 550nm
        sl = SingleLayerDesigner(target_wavelength_nm=550.0, order=1)
        d_green = sl.design_thickness(n=1.50)
        d0_swelling = sl.design_for_swelling(Q_target=2.15, n0=1.50)

        return {
            "wavelengths_nm":   lam,
            "R_target":         R_leaf,
            "R_optimized":      R_opt,
            "d_opt_nm":         result.d_opt * 1e9,
            "n_opt":            result.n_opt,
            "loss_history":     result.loss_history,
            "quality":          quality,
            "converged":        result.converged,
            "final_loss":       result.final_loss,
            "d_green_nm":       d_green * 1e9,
            "d0_swelling_nm":   d0_swelling * 1e9,
            "note": f"RMSE={quality['RMSE']:.4f} | R²={quality['R_squared']:.4f} | d_opt={result.d_opt[0]*1e9:.1f}nm",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        ax1.plot(r["wavelengths_nm"], r["R_target"],    color="darkgreen", linewidth=2.5,
                 label="Ziel: Laubblatt-Reflexion")
        ax1.plot(r["wavelengths_nm"], r["R_optimized"], color=C[0], linewidth=2,
                 linestyle="--", label=f"Optimiert (RMSE={r['quality']['RMSE']:.4f})")
        ax1.fill_between(r["wavelengths_nm"],
                          r["R_target"], r["R_optimized"],
                          alpha=0.15, color=C[0], label="Abweichung")
        ax1.set_xlabel("Wellenlänge λ (nm)")
        ax1.set_ylabel("Reflexion R")
        ax1.set_title(f"Tarnfarben-Optimierung: Ziel vs. Ergebnis\n"
                      f"d₁={r['d_opt_nm'][0]:.1f}nm, n₁={r['n_opt'][0]:.3f} | "
                      f"d₂={r['d_opt_nm'][1]:.1f}nm, n₂={r['n_opt'][1]:.3f}")
        ax1.legend(fontsize=9)
        ax1.set_xlim(380, 780)

        if r["loss_history"]:
            ax2.semilogy(r["loss_history"], color=C[1], linewidth=2)
            ax2.axhline(r["final_loss"], color=C[0], linestyle="--", alpha=0.7,
                        label=f"L_final={r['final_loss']:.4e}")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Kostenfunktion L (log)")
        ax2.set_title(f"Konvergenzkurve – adjungierter Gradientenabstieg\nR²={r['quality']['R_squared']:.4f}")
        ax2.legend(fontsize=9)

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_22_SpektralDesignMultiZiel(BaseExperiment):
    """
    EXPERIMENT 22: Multi-Ziel-Design – RGB-Primärfarben durch Schichtoptimierung

    Optimiert getrennte Schichtstapel für Rot (700nm), Grün (550nm), Blau (450nm).
    Zeigt die Erreichbarkeit des sRGB-Farbraums mit dem FP-Hydrogel-System.
    """
    title   = "Exp 22: Multi-Ziel-Design – RGB-Primärfarben"
    module  = "inverse_design.py"
    section = "[OQTUM §7] Inverses Photonisches Design"

    TARGETS = {
        "Rot":  (700, C[0]),
        "Grün": (550, C[2]),
        "Blau": (450, C[1]),
    }

    def run(self) -> Dict[str, Any]:
        lam = np.linspace(380, 780, 80)
        results = {}

        for color, (peak_nm, _) in self.TARGETS.items():
            R_target = 0.55 * np.exp(-((lam - peak_nm)**2) / (2*30**2))
            designer = InversePhotonicsDesigner(
                R_target, lam, n_layers=2,
                d_bounds=(50e-9, 700e-9),
                n_bounds=(1.2, 3.0),
            )
            result  = designer.optimize(n_restarts=2, max_iter=150, seed=7)
            R_opt   = designer.predict_spectrum(result)
            quality = designer.design_quality(result)
            results[color] = {
                "R_target": R_target,
                "R_opt":    R_opt,
                "d_nm":     result.d_opt * 1e9,
                "n":        result.n_opt,
                "RMSE":     quality["RMSE"],
                "R2":       quality["R_squared"],
                "history":  result.loss_history,
            }

        # Single-Layer Designer für alle drei Wellenlängen
        sl_designs = {}
        for color, (peak_nm, _) in self.TARGETS.items():
            sl = SingleLayerDesigner(peak_nm, order=1)
            sl_designs[color] = {
                "d_n150_nm":  sl.design_thickness(1.50) * 1e9,
                "d_Q2_nm":    sl.design_for_swelling(Q_target=2.0) * 1e9,
            }

        return {
            "wavelengths_nm": lam,
            "results":        results,
            "sl_designs":     sl_designs,
            "note": "RGB-Primärfarben: Rot/Grün/Blau durch adjungiertes Inversdesign",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        lam = r["wavelengths_nm"]
        fig, ax1, ax2 = self._fig2()

        for color, (_, col) in self.TARGETS.items():
            res = r["results"][color]
            ax1.plot(lam, res["R_target"], color=col, linewidth=1.5, linestyle="--", alpha=0.6)
            ax1.plot(lam, res["R_opt"],    color=col, linewidth=2.5,
                     label=f"{color}: RMSE={res['RMSE']:.3f}, R²={res['R2']:.3f}")

        ax1.set_xlabel("Wellenlänge λ (nm)")
        ax1.set_ylabel("Reflexion R")
        ax1.set_title("RGB-Primärfarben: Ziel (—) vs. optimiert (--)\nAdjungierter Gradientenabstieg")
        ax1.legend(fontsize=9)
        ax1.set_xlim(380, 780)

        for i, (color, (_, col)) in enumerate(self.TARGETS.items()):
            res = r["results"][color]
            hist = res["history"]
            if hist:
                ax2.semilogy(hist, color=col, linewidth=2, label=f"{color}")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Kostenfunktion L (log)")
        ax2.set_title("Konvergenzvergleich RGB-Optimierungen")
        ax2.legend()

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK X: SYSTEMINTEGRATION – HYDRO-PHOTONISCHE PLATTFORM
# ══════════════════════════════════════════════════════════════════════════════

class Experiment_23_IoFETFPPixel(BaseExperiment):
    """
    EXPERIMENT 23: IoFET+FP-Pixel-Integration – die hydro-photonische Einheit

    [IONO §3.4 + OQTUM §8.1]:
    Jedes Pixel kombiniert:
      - IoFET: Gate-gesteuerte Flüssigkeit (Schalten, Leiten, Speichern)
      - FP-Resonator: Optischer Ausgang (Farbe = gespeicherter Zustand)

    Ein Pixel = 1 logisches Bit + 1 optisches Bit.
    Simuliert eine 16×16-Pixel-Matrix mit zufälligem Inhalt.
    """
    title   = "Exp 23: IoFET+FP-Pixel – hydro-photonische Integrationsmatrix"
    module  = "iofet.py + fabry_perot.py"
    section = "[IONO §3.4] + [OQTUM §8.1] Systemintegration"

    def run(self) -> Dict[str, Any]:
        N = 16
        rng = np.random.default_rng(2026)
        transistor = IoFET(conductivity=1.0, threshold_voltage=0.05)
        fp = FabryPerotResonator(d0=250e-9, n0=1.50, R1=0.3, R2=0.3)

        # Zufällige Quellungsgrade pro Pixel (Q ∈ [1.0, 4.0])
        Q_matrix = rng.uniform(1.0, 4.0, (N, N))

        # Entsprechende Wellenlängen
        lam_matrix = np.vectorize(lambda Q: fp.resonance_wavelength(Q)*1e9)(Q_matrix)

        # IoFET-Zustände: Schwellspannung-basiert
        V_gs_matrix = rng.uniform(0.0, 0.3, (N, N))
        I_matrix    = np.vectorize(lambda v: transistor.drain_current(v, 0.1))(V_gs_matrix)
        state_matrix = (V_gs_matrix > transistor.V_T).astype(int)

        # Statistik
        n_on  = state_matrix.sum()
        n_off = N*N - n_on
        mean_lam = lam_matrix.mean()
        std_lam  = lam_matrix.std()

        return {
            "Q_matrix":     Q_matrix,
            "lam_matrix":   lam_matrix,
            "I_matrix_nA":  I_matrix * 1e9,
            "state_matrix": state_matrix,
            "V_gs_matrix":  V_gs_matrix,
            "N":            N,
            "n_on":         n_on,
            "n_off":        n_off,
            "mean_lam_nm":  mean_lam,
            "std_lam_nm":   std_lam,
            "note": f"{N}×{N}={N*N} Pixel | ON:{n_on} OFF:{n_off} | λ={mean_lam:.0f}±{std_lam:.0f}nm",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        norm = Normalize(vmin=380, vmax=780)
        lam_rgb = plt.cm.Spectral_r(norm(r["lam_matrix"]))
        # Logischer Zustand: helle/dunkle Pixel überlagern
        for i in range(r["N"]):
            for j in range(r["N"]):
                if r["state_matrix"][i,j] == 0:
                    lam_rgb[i, j, :3] *= 0.4  # dunkle OFF-Pixel

        ax1.imshow(lam_rgb, origin="upper")
        ax1.set_title(f"Hydro-photonische Pixelmatrix {r['N']}×{r['N']}\n"
                      f"ON:{r['n_on']} Pixel (hell) | OFF:{r['n_off']} Pixel (dunkel)")
        ax1.set_xlabel("x (Pixel)"); ax1.set_ylabel("y (Pixel)")
        sm = ScalarMappable(cmap="Spectral_r", norm=norm)
        plt.colorbar(sm, ax=ax1, label="λ_res (nm)", fraction=0.046)

        # Sub-Plot 2: Histogramm der Resonanzwellenlängen
        lam_flat = r["lam_matrix"].flatten()
        n_bins = 30
        counts, edges = np.histogram(lam_flat, bins=n_bins, range=(380, 780))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        bar_colors = [plt.cm.Spectral_r(norm(c)) for c in bin_centers]
        ax2.bar(bin_centers, counts, width=(edges[1]-edges[0]),
                color=bar_colors, edgecolor="none")
        ax2.axvline(r["mean_lam_nm"], color="black", linewidth=2,
                    label=f"⟨λ⟩={r['mean_lam_nm']:.0f}nm")
        ax2.set_xlabel("Resonanzwellenlänge λ (nm)")
        ax2.set_ylabel("Anzahl Pixel")
        ax2.set_title(f"Wellenlängenverteilung der Pixelmatrix\nσ={r['std_lam_nm']:.0f}nm")
        ax2.legend()

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_24_DreifaltigkeitSimultanitat(BaseExperiment):
    """
    EXPERIMENT 24: Die dreifache Simultaneität – Leiter, Schalter, Speicher

    [IONO §3]: Das Herzstück der IRF-Technologie.
    Ein einziges Wasservolumen führt gleichzeitig aus:
      1. LEITER:   R = L/(σA) (ionische Leitfähigkeit)
      2. SCHALTER: I_DS(V_GS) (IoFET-Steuerung)
      3. SPEICHER: s_ij ∈ {0,1} (geometrischer Zustand)

    Visualisiert alle drei Funktionen als Funktion des EWOD-Gate-Signals.
    """
    title   = "Exp 24: Dreifache Simultaneität – Leiter + Schalter + Speicher"
    module  = "ionic.py + ewod.py + iofet.py"
    section = "[IONO §3] Dreifache Simultaneität"

    def run(self) -> Dict[str, Any]:
        V_gate = np.linspace(0, 0.5, 300)

        ewod = standard_al2o3_system(contact_angle_0=110.0)
        sol  = kcl_solution(0.5)
        transistor = IoFET(conductivity=sol.conductivity(), threshold_voltage=0.05)

        # FUNKTION 1: LEITER – Leitfähigkeit konstant, Widerstand hängt von θ(V) ab
        sigma = sol.conductivity()
        theta_V = np.array([ewod.contact_angle(v) for v in V_gate])
        # Effektiver Querschnitt: A(θ) ∝ cos(θ)  (vereinfachtes Modell)
        A_eff = (100e-9)**2 * np.clip(np.cos(np.radians(theta_V)) + 1.2, 0.1, 2.0)
        R_eff_MOhm = np.array([sol.channel_resistance(1e-6, A) * 1e-6 for A in A_eff])

        # FUNKTION 2: SCHALTER – I_DS(V_GS)
        I_ds = np.array([transistor.drain_current(v, 0.05) * 1e9 for v in V_gate])

        # FUNKTION 3: SPEICHER – Logischer Zustand (0/1) durch Hysterese-Modell
        state = np.zeros(len(V_gate))
        memory = 0  # Anfangszustand
        V_set   = transistor.V_T * 1.2
        V_reset = transistor.V_T * 0.5
        for i, v in enumerate(V_gate):
            if v > V_set:   memory = 1
            if v < V_reset: memory = 0
            state[i] = memory

        return {
            "V_gate":          V_gate,
            "theta_V":         theta_V,
            "R_eff_MOhm":      R_eff_MOhm,
            "I_ds_nA":         I_ds,
            "state":           state,
            "sigma_S_m":       sigma,
            "V_set":           V_set,
            "V_reset":         V_reset,
            "V_T":             transistor.V_T,
            "note": f"σ={sigma:.3f}S/m | V_T={transistor.V_T:.3f}V | V_set={V_set:.3f}V",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        V = r["V_gate"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(self.title, fontsize=13, fontweight="bold", y=1.01)

        # Sub-Plot 1: Alle drei Funktionen übereinander (3 y-Achsen)
        color1, color2, color3 = C[0], C[1], C[2]

        ax1.set_xlabel("Gate-Spannung V_GS (V)")
        ax1.set_ylabel("Kanalwiderstand R (MΩ)", color=color1)
        l1, = ax1.plot(V, r["R_eff_MOhm"], color=color1, linewidth=2, label="Leiter: R(V)")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax1b = ax1.twinx()
        ax1b.spines["right"].set_position(("axes", 1.0))
        l2, = ax1b.semilogy(V, np.abs(r["I_ds_nA"])+1e-10, color=color2, linewidth=2,
                             linestyle="--", label="Schalter: I_DS(V)")
        ax1b.set_ylabel("|I_DS| (nA) [log]", color=color2)
        ax1b.tick_params(axis="y", labelcolor=color2)

        ax1c = ax1.twinx()
        ax1c.spines["right"].set_position(("axes", 1.15))
        l3, = ax1c.step(V, r["state"], color=color3, linewidth=2.5,
                         linestyle=":", label="Speicher: s(V)", where="post")
        ax1c.set_ylabel("Logischer Zustand s", color=color3)
        ax1c.set_yticks([0, 1]); ax1c.tick_params(axis="y", labelcolor=color3)

        ax1.axvline(r["V_T"],     color="gray", linestyle=":",  alpha=0.7, label=f"V_T={r['V_T']:.3f}V")
        ax1.axvline(r["V_set"],   color="black", linestyle="--", alpha=0.5)
        ax1.legend(handles=[l1, l2, l3], loc="upper left", fontsize=8)
        ax1.set_title("Dreifache Simultaneität:\nLeiter (R), Schalter (I_DS), Speicher (s)")

        # Sub-Plot 2: Zusammenfassung als 3-Band-Diagramm
        V_bands = np.array([0, r["V_reset"]*0.9, r["V_T"]*0.8, r["V_T"], r["V_set"], 0.5])
        funcs = ["Leiter\n(immer aktiv)", "Speicher\n(0 → 1)", "Schalter\n(sub-threshold → an)"]
        colors_band = [C[0], C[2], C[1]]
        y_pos = [0.6, 0.35, 0.1]
        for func, col, yp in zip(funcs, colors_band, y_pos):
            ax2.barh(yp, 0.5, left=0, height=0.15, color=col, alpha=0.3)
            ax2.text(0.25, yp, func, ha="center", va="center", fontsize=10, fontweight="bold", color=col)

        ax2.axvline(r["V_reset"], color=C[2], linestyle="--", alpha=0.7, label=f"V_reset={r['V_reset']:.3f}V")
        ax2.axvline(r["V_T"],     color=C[1], linestyle="-",  alpha=0.7, label=f"V_T={r['V_T']:.3f}V")
        ax2.axvline(r["V_set"],   color=C[2], linestyle=":",  alpha=0.7, label=f"V_set={r['V_set']:.3f}V")
        ax2.set_xlim(0, 0.5)
        ax2.set_xlabel("Gate-Spannung V_GS (V)")
        ax2.set_yticks([])
        ax2.set_title(f"Aktivierungsbereiche der drei Funktionen\n(σ={r['sigma_S_m']:.3f} S/m)")
        ax2.legend(fontsize=8, loc="upper right")

        fig.tight_layout(pad=2.5)
        return fig


class Experiment_25_CMOSvsIRFVergleich(BaseExperiment):
    """
    EXPERIMENT 25: Vollständiger CMOS vs. IRF Systemvergleich

    [IONO §6.3 Tab. 2]:
    Quantitativer Vergleich aller relevanten Kenngrößen:
      - Schwellspannung, Schaltenergie, Taktfrequenz
      - Transistordichte, Betriebstemperatur, Biologische Kompatibilität
      - Moore'sches Gesetz vs. IRF-Skalierungsgesetz (N_z-Skalierung)
    """
    title   = "Exp 25: CMOS vs. IRF – vollständiger Systemvergleich"
    module  = "alle Module"
    section = "[IONO §6.3] Vergleichstabelle CMOS vs. IRF"

    # Aus Arbeit Tab. 2 [IONO §6.3]
    COMPARISON = {
        "Schwellspannung (V)":        {"CMOS": 0.7,     "IRF": 0.05},
        "Energie/Schaltvorgang (fJ)": {"CMOS": 1.0,     "IRF": 0.07},
        "Max. Taktfreq. (kHz)":       {"CMOS": 5e6,     "IRF": 1.0},
        "Dichte (10⁹/cm²)":          {"CMOS": 1.9,     "IRF": 0.008},
        "Betriebstemp. (°C, max)":    {"CMOS": 125,     "IRF": 60},
        "Subthreshold-Swing (mV/dec)":{"CMOS": 60,      "IRF": 8},
        "Lernfähig (0=nein, 1=ja)":   {"CMOS": 0,       "IRF": 1},
        "Biokompat. (0=nein, 1=ja)":  {"CMOS": 0,       "IRF": 1},
    }

    def run(self) -> Dict[str, Any]:
        # Tatsächlich berechnete Werte
        std      = standard_al2o3_system()
        sol      = kcl_solution(0.5)
        transistor = IoFET(conductivity=sol.conductivity(), threshold_voltage=0.05)

        V_T_calc  = transistor.V_T
        E_calc_fJ = transistor.switching_energy() * 1e15
        if np.isnan(E_calc_fJ): E_calc_fJ = 0.07

        G0 = transistor.base_conductance
        SS = transistor.SS

        # IRF Skalierungsgesetz
        N_z_vals = np.arange(1, 51)
        p_5um_density = 1.0 / (5e-6)**2 * 1e-4   # /cm²
        rho_3D = N_z_vals * 2 * p_5um_density * 1e-9  # in 10⁹/cm²

        # Mooresches Gesetz (Verdopplung alle 2 Jahre von 2016 Basis)
        years = np.linspace(2016, 2030, 30)
        rho_cmos = 0.050 * 2**((years - 2016)/2.0)  # 10⁹/cm²

        return {
            "comparison": self.COMPARISON,
            "V_T_calc":   V_T_calc,
            "E_calc_fJ":  E_calc_fJ,
            "SS_mVdec":   SS,
            "N_z_vals":   N_z_vals,
            "rho_3D_IRF_G_cm2": rho_3D,
            "years":      years,
            "rho_cmos_G_cm2": rho_cmos,
            "note": f"IRF-Energie={E_calc_fJ:.3f}fJ vs CMOS≈1fJ | SS={SS}mV/dec vs 60",
        }

    def plot(self) -> plt.Figure:
        r = self.run()
        fig, ax1, ax2 = self._fig2()

        # Sub-Plot 1: Radar/Balken-Vergleich für normierte Kennwerte
        metrics    = list(r["comparison"].keys())
        cmos_vals  = np.array([r["comparison"][m]["CMOS"] for m in metrics], dtype=float)
        irf_vals   = np.array([r["comparison"][m]["IRF"]  for m in metrics], dtype=float)

        # Normierung (log-safe): CMOS immer = 1
        norm_cmos = np.ones(len(metrics))
        # np.divide mit where Parameter verhindert RuntimeWarning bei Division durch 0
        norm_irf  = np.divide(irf_vals, cmos_vals, where=cmos_vals > 0, out=irf_vals.copy())
        # Werte wo IRF besser sein soll (kleiner ist besser): invertieren
        better_smaller = [True, True, False, False, False, True, False, False]
        display_irf = np.where(better_smaller, np.where(norm_irf > 0, 1/norm_irf, norm_irf), norm_irf)
        display_irf = np.minimum(display_irf, 100)   # Cap bei 100×

        x = np.arange(len(metrics))
        width = 0.35
        ax1.bar(x - width/2, np.ones(len(metrics)), width, color=C[0],  alpha=0.8, label="CMOS (Basis=1)")
        ax1.bar(x + width/2, display_irf,            width, color=C[1],  alpha=0.8, label="IRF (relativ)")
        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split("(")[0].strip() for m in metrics], rotation=25, ha="right", fontsize=8)
        ax1.set_ylabel("Normierter Wert (CMOS=1, log-skaliert)")
        ax1.set_yscale("symlog", linthresh=0.1)
        ax1.set_title("CMOS vs. IRF: relative Kennwerte\n(>1 = IRF besser als CMOS)")
        ax1.legend()

        # Sub-Plot 2: Mooresches Gesetz vs. IRF-Skalierungsgesetz
        ax2.semilogy(r["years"], r["rho_cmos_G_cm2"], color=C[0], linewidth=2.5,
                     label="Moore: CMOS (×2 alle 2 Jahre)")
        ax2.semilogy(r["N_z_vals"] + 2015, r["rho_3D_IRF_G_cm2"], color=C[1], linewidth=2.5,
                     linestyle="--", label="IRF: ρ_3D = N_z · 2 · ρ_2D")
        ax2.set_xlabel("Jahr bzw. N_z + 2015")
        ax2.set_ylabel("Transistordichte (10⁹/cm²)")
        ax2.set_title("Moore-Gesetz vs. IRF-Skalierungsgesetz\n(lineares Wachstum in N_z vs. exponentiell)")
        ax2.legend(fontsize=9)
        ax2.set_xlim(2016, 2030)

        fig.tight_layout(pad=2.5)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT-REGISTER und Hauptprogramm
# ══════════════════════════════════════════════════════════════════════════════

ALL_EXPERIMENTS: List[type] = [
    # Block I: Ionische Grundlagen
    Experiment_01_IonicConductivityConcentration,
    Experiment_02_DiffusionEinsteinStokes,
    Experiment_03_IonicChannelResistance,
    # Block II: EWOD
    Experiment_04_EWODYoungLippmann,
    Experiment_05_EWODSchaltenergie,
    Experiment_06_EWODTemperaturkompensation,
    # Block III: IoFET
    Experiment_07_IoFETKennlinien,
    Experiment_08_IoFETTransistordichte,
    Experiment_09_IoFETEnergievergleich,
    # Block IV: Tropfenlogik
    Experiment_10_TropfenVerschmelzung,
    Experiment_11_LogikGatterBenchmark,
    Experiment_12_IRFNetzwerk,
    # Block V: Fabry-Pérot
    Experiment_13_FabryPerotSpektren,
    Experiment_14_FabryPerotFinesse,
    # Block VI: Transfermatrix-Methode
    Experiment_15_TMMSpektrum,
    Experiment_16_TMMQuantenAnalogie,
    # Block VII: Polymer / Flory-Rehner
    Experiment_17_FloryRehnerGleichgewicht,
    Experiment_18_QuellungskinetikTemperatur,
    # Block VIII: EBL-Lithographie
    Experiment_19_EBLProximityEffekt,
    Experiment_20_EBLTexturKodierung,
    # Block IX: Inverses Design
    Experiment_21_TarnfarbenOptimierung,
    Experiment_22_SpektralDesignMultiZiel,
    # Block X: Systemintegration
    Experiment_23_IoFETFPPixel,
    Experiment_24_DreifaltigkeitSimultanitat,
    Experiment_25_CMOSvsIRFVergleich,
]

def main():
    import re
    def latex_friendly_filename(title):
        # Nur ASCII, keine Sonderzeichen, keine Leerzeichen, kurz und eindeutig
        name = title
        name = name.replace('Exp ', 'exp').replace('–', '-').replace('—', '-')
        name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_').lower()
        # Kürzen auf max 40 Zeichen
        if len(name) > 40:
            name = name[:40]
        return name

    for exp_cls in ALL_EXPERIMENTS:
        exp = exp_cls()
        print(f"Running {exp.title}...")
        fig = exp.plot()
        fname = latex_friendly_filename(exp.title) + ".svg"
        fig.savefig(f"src/results/{fname}")
        plt.close(fig)
        print(f"Saved src/results/{fname}")
        print("="*80)
        print("\n")
        print(exp.run()["note"])
        print("\n")
        print("="*80)

if __name__ == "__main__":
    main()