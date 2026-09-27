"""
demo.py – Ausführliche Demonstration der IRF-Bibliothek
=======================================================

Ionotronic Reconfigurable Fabric (IRF)
Demonstriert alle Module der Bibliothek mit Berechnungen und Plots.

Module:
  - utils.py         Konstanten, Einheitenkonvertierung, Logging
  - physics.py       EWOD, Ionenleitfähigkeit, Oberflächenspannung, Elektrokinetik
  - transistor.py    IoFET-Transistormodell, Kennlinien
  - droplet.py       Tropfendynamik, Verschmelzung, Selbstorganisation
  - logic.py         Tropfenlogik, Gatter, Netzwerke, Wahrheitstabellen
  - memory.py        Pinning-Wells, Ionische Speicherzellen, MemoryArray
  - matrix.py        Elektrodenmatrix, Konfiguration, Topologie
  - fabrication.py   Fertigungsprozess, Technologiegenerationen, Validierung
  - architecture.py  Von-Neumann-Flasche, HybridController, Durchsatzanalyse
  - simulation.py    Simulator, Zeitverlauf, SimulationResult
"""

import sys
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import Normalize
import matplotlib.cm as cm

matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#F9F9F9",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Pfad-Setup: src-Ordner simulieren (falls nicht als Paket installiert)
# ---------------------------------------------------------------------------
# Alle Module liegen im gleichen Verzeichnis; wir erzeugen ein synthetisches
# 'src'-Package durch Hinzufügen des aktuellen Verzeichnisses zum Pfad.
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEMO_DIR)

# Da die Module untereinander 'from src.xyz import ...' nutzen, erzeugen wir
# ein temporäres src-Package, das auf das Demo-Verzeichnis zeigt.
import types

src_pkg = types.ModuleType("src")
src_pkg.__path__ = [DEMO_DIR]
src_pkg.__package__ = "src"
sys.modules["src"] = src_pkg

# Jetzt können wir die echten Module als src.xyz importieren:
import importlib

for mod_name in [
    "utils", "physics", "transistor", "droplet",
    "logic", "memory", "matrix", "fabrication",
    "architecture", "simulation",
]:
    full = f"src.{mod_name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(DEMO_DIR, f"{mod_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        setattr(src_pkg, mod_name, module)

# ---------------------------------------------------------------------------
# Importe der IRF-Bibliothek
# ---------------------------------------------------------------------------
from src.utils import CONST, UnitConverter, Logger, ValidationError
from src.physics import (
    EWODModel, IonicConductivity, IonSpecies, SurfaceTension,
    ElectrokineticTransport,
)
from src.transistor import IoFET, IoFETState, TransferCharacteristic
from src.droplet import Droplet, DropletMerger, SelfOrganizer
from src.logic import (
    GateType, DropletGate, LogicNetwork, TruthTable,
)
from src.memory import PinningWell, IonicMemoryCell, MemoryArray
from src.matrix import MatrixConfig, ElectrodeMatrix, ElectrodeState
from src.fabrication import (
    TechnologyNode, FabricationProcess, ProcessValidator,
    TECHNOLOGY_PARAMETERS,
)
from src.architecture import (
    VonNeumannBottleneck, HybridController, ThroughputAnalyzer, IRFSystem,
)
from src.simulation import SimulationConfig, Simulator

# ---------------------------------------------------------------------------
# Hilfsfunktion: Kapitel-Header
# ---------------------------------------------------------------------------

def header(title: str, level: int = 1) -> None:
    width = 72
    if level == 1:
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)
    else:
        print(f"\n  --- {title} ---")


# ===========================================================================
# 1) utils.py – Konstanten, Einheitenkonvertierung, Logging
# ===========================================================================

def demo_utils() -> None:
    header("MODUL: utils.py – Physikalische Konstanten & Einheiten")

    # Physikalische Konstanten
    print(f"  Vakuumpermittivität ε₀  = {CONST.epsilon_0:.4e} F/m")
    print(f"  Boltzmann-Konstante k_B = {CONST.k_B:.4e} J/K")
    print(f"  Elementarladung e       = {CONST.e:.4e} C")
    print(f"  Faraday-Konstante F     = {CONST.F:.2f} C/mol")
    print(f"  Avogadro-Konstante N_A  = {CONST.N_A:.4e} 1/mol")
    print(f"  Oberflächenspannung γ   = {CONST.gamma_water*1000:.2f} mN/m")
    print(f"  Viskosität η_Wasser     = {CONST.eta_water:.2e} Pa·s")
    print(f"  ε_r Wasser              = {CONST.epsilon_r_water}")

    # Einheitenkonvertierung
    header("Einheitenkonvertierung", 2)
    for val, unit in [(5.0, "um"), (200.0, "nm"), (1.0, "mm")]:
        m = UnitConverter.to_meters(val, unit)
        print(f"    {val} {unit} = {m:.2e} m")

    for val, unit in [(120.0, "mV"), (0.5, "V"), (1.0, "kV")]:
        v = UnitConverter.to_volts(val, unit)
        print(f"    {val} {unit} = {v:.4f} V")

    for val, unit in [(10.0, "pl"), (1.0, "nl"), (50.0, "fl")]:
        m3 = UnitConverter.to_cubic_meters(val, unit)
        print(f"    {val} {unit} = {m3:.2e} m³")

    print(f"    25 °C = {UnitConverter.celsius_to_kelvin(25):.2f} K")
    print(f"    π rad = {UnitConverter.radians_to_degrees(math.pi):.1f}°")

    # Logger
    logger = Logger.get("irf.demo")
    logger.info("IRF-Demo gestartet – Logger funktioniert korrekt.")

    # ValidationError
    try:
        raise ValidationError("voltage_V", -0.5, "> 0")
    except ValidationError as e:
        print(f"\n  ValidationError korrekt ausgelöst: {e}")


# ===========================================================================
# 2) physics.py – EWOD, Ionenleitfähigkeit, Oberflächenspannung
# ===========================================================================

def demo_physics() -> None:
    header("MODUL: physics.py – Physikalische Grundmodelle")

    # -----------------------------------------------------------------------
    # 2a) EWOD-Modell
    # -----------------------------------------------------------------------
    header("EWOD-Modell (Young-Lippmann-Gleichung)", 2)

    ewod = EWODModel(
        dielectric_thickness_m=10e-9,  # 10 nm Al₂O₃
        dielectric_epsilon_r=9.0,
        contact_angle_0_deg=110.0,
        surface_tension_N_m=CONST.gamma_water,
    )
    print(f"  EWOD-Effizienz η = {ewod.ewod_efficiency:.4e} V⁻²")
    print(f"  Schwellspannung (θ=90°): {ewod.threshold_voltage(90.0)*1000:.1f} mV")

    vgs_range = np.linspace(0, 0.5, 200)
    angles = [ewod.contact_angle_deg(v) for v in vgs_range]
    sw_energies = [ewod.switching_energy_J(v, (5e-6)**2) * 1e18 for v in vgs_range]  # in aJ

    # -----------------------------------------------------------------------
    # 2b) Ionische Leitfähigkeit
    # -----------------------------------------------------------------------
    header("Ionische Leitfähigkeit (KCl)", 2)

    concs = [0.001, 0.01, 0.1, 0.5, 1.0]  # mol/L
    for c in concs:
        ic = IonicConductivity.from_kcl(c)
        sigma = ic.conductivity_S_m()
        r = ic.channel_resistance_Ohm(10e-6, 5e-12)
        print(f"    c = {c:.3f} mol/L → σ = {sigma:.4f} S/m, R_Kanal = {r:.2f} Ω")

    sigma_vals = [IonicConductivity.from_kcl(c).conductivity_S_m() for c in np.linspace(0.001, 1, 200)]

    # -----------------------------------------------------------------------
    # 2c) Oberflächenspannung
    # -----------------------------------------------------------------------
    header("Oberflächenspannung (temperaturabhängig)", 2)

    temps_C = np.linspace(0, 90, 200)
    gammas = [SurfaceTension.at_celsius(t) * 1000 for t in temps_C]  # mN/m

    print(f"    γ(0 °C)  = {SurfaceTension.at_celsius(0)*1000:.2f} mN/m")
    print(f"    γ(25 °C) = {SurfaceTension.at_celsius(25)*1000:.2f} mN/m")
    print(f"    γ(80 °C) = {SurfaceTension.at_celsius(80)*1000:.2f} mN/m")

    # Verschmelzungsenergie zweier Tropfen
    r1, r2 = 50e-6, 30e-6  # 50 µm, 30 µm
    e_merge = SurfaceTension.droplet_merge_energy_J(r1, r2, 298.15)
    print(f"\n  Verschmelzungsenergie (r1=50µm, r2=30µm): ΔE = {e_merge*1e15:.4f} fJ")

    # -----------------------------------------------------------------------
    # 2d) Elektrokinetischer Transport
    # -----------------------------------------------------------------------
    header("Elektrokinetischer Transport", 2)

    k_ion = IonSpecies("K+", +1, 7.62e-8, 100.0)
    ekt = ElectrokineticTransport(k_ion, temperature_K=298.15)
    D = ekt.diffusion_coefficient_m2_s
    E_field = 1e4  # V/m
    v_drift = ekt.drift_velocity_m_s(E_field)
    v_eo = ekt.electroosmotic_velocity_m_s(E_field, zeta_potential_V=-0.05)

    print(f"    Diffusionskoeffizient D(K+) = {D:.4e} m²/s")
    print(f"    Driftgeschwindigkeit (E=10⁴ V/m): v = {v_drift*1000:.4f} mm/s")
    print(f"    Elektroosmose (ζ=-50 mV):         v_eo = {v_eo*1000:.4f} mm/s")
    print(f"    Diffusionsfluss (dc/dx=10⁶ mol/m⁴): J = {ekt.diffusion_flux_mol_m2s(1e6):.4e} mol/(m²·s)")

    # -----------------------------------------------------------------------
    # Plot: Physikalische Grundmodelle
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Physikalische Grundmodelle des IRF", fontsize=13, fontweight="bold")

    # Sub-Plot 1: EWOD – Kontaktwinkel + Schaltenergie
    ax1 = axes[0]
    color1, color2 = "#1f77b4", "#d62728"
    line1, = ax1.plot(vgs_range * 1000, angles, color=color1, lw=2, label="Kontaktwinkel θ(V)")
    ax1.axhline(90, color="gray", ls="--", lw=1, label="θ = 90° (Schwelle)")
    ax1.set_xlabel("Gate-Spannung V_GS [mV]")
    ax1.set_ylabel("Kontaktwinkel [°]", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 120)
    ax1.set_title("EWOD: Kontaktwinkel & Schaltenergie\n(Young-Lippmann-Gleichung)")

    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    line2, = ax1b.plot(vgs_range * 1000, sw_energies, color=color2, lw=2, ls="--", label="Schaltenergie E [aJ]")
    ax1b.set_ylabel("Schaltenergie [aJ]", color=color2)
    ax1b.tick_params(axis="y", labelcolor=color2)
    ax1b.set_ylim(0, max(sw_energies) * 1.2)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", fontsize=8)

    # Sub-Plot 2: Ionenleitfähigkeit & Oberflächenspannung
    ax2 = axes[1]
    conc_range = np.linspace(0.001, 1.0, 200)
    sigmas = [IonicConductivity.from_kcl(c).conductivity_S_m() for c in conc_range]

    ax2.plot(conc_range, sigmas, color=color1, lw=2, label="σ (KCl) [S/m]")
    ax2.set_xlabel("KCl-Konzentration [mol/L]")
    ax2.set_ylabel("Ionenleitfähigkeit σ [S/m]", color=color1)
    ax2.tick_params(axis="y", labelcolor=color1)
    ax2.set_title("Ionenleitfähigkeit & Oberflächenspannung\nvs. Konzentration / Temperatur")

    ax2b = ax2.twinx()
    ax2b.spines["right"].set_visible(True)
    ax2b.plot(temps_C / 90, gammas, color=color2, lw=2, ls="--", label="γ(T) [mN/m]")
    ax2b.set_ylabel("Oberflächenspannung γ [mN/m]", color=color2)
    ax2b.tick_params(axis="y", labelcolor=color2)
    ax2b.set_ylim(55, 80)

    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color=color1, lw=2),
        Line2D([0], [0], color=color2, lw=2, ls="--"),
    ]
    ax2.legend(legend_lines, ["σ(c) [S/m]", "γ(T) [mN/m] (x-Achse: T/90°C)"],
               loc="center right", fontsize=8)

    plt.tight_layout()
    plt.savefig("src/results/plot_01_physics.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_01_physics.svg")


# ===========================================================================
# 3) transistor.py – IoFET-Transistormodell
# ===========================================================================

def demo_transistor() -> None:
    header("MODUL: transistor.py – Ionotronic Field-Effect Transistor (IoFET)")

    ewod = EWODModel(
        dielectric_thickness_m=10e-9,
        dielectric_epsilon_r=9.0,
        contact_angle_0_deg=110.0,
    )
    ic = IonicConductivity.from_kcl(0.1)  # 0,1 mol/L KCl

    iofet = IoFET(
        ewod=ewod,
        conductivity_model=ic,
        channel_length_m=5e-6,
        channel_width_m=5e-6,
        channel_height_m=50e-6,
        threshold_angle_deg=90.0,
    )

    vt = iofet.threshold_voltage_V
    g0 = iofet.g0
    eta = iofet.eta_ewod

    print(f"  Schwellspannung V_T     = {vt*1000:.2f} mV")
    print(f"  Grundleitfähigkeit G₀   = {g0:.4e} S")
    print(f"  EWOD-Effizienz η        = {eta:.4e} V⁻²")
    print(f"  Querschnitt A           = {iofet.channel_cross_section_m2*1e12:.1f} µm²")

    # Betriebszustände
    header("Betriebszustände", 2)
    test_points = [
        (0.0, 0.0), (0.05, 0.05), (0.13, 0.05), (0.5, 0.05), (0.5, 0.3),
    ]
    for vgs, vds in test_points:
        s = iofet.state(vgs, vds)
        ids = iofet.drain_current_A(vgs, vds)
        gm = iofet.transconductance_S(vgs, vds)
        print(f"    V_GS={vgs:.2f}V  V_DS={vds:.2f}V → {s.name:<15} "
              f"I_DS={ids*1e9:.4f} nA  g_m={gm*1e6:.4f} µS")

    # Schaltenergie
    e_sw = iofet.switching_energy_J(vt)
    print(f"\n  Schaltenergie bei V_T   = {e_sw*1e18:.4f} aJ")

    # Übertragungskennlinie
    tc = iofet.transfer_characteristic(
        vds=0.05, vgs_start=0.0, vgs_stop=0.8, n_points=300
    )
    print(f"  Ein/Aus-Verhältnis      = {tc.on_off_ratio:.2e}")

    # Ausgangskennlinien für verschiedene V_GS
    vgs_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    output_curves = {
        vgs: iofet.output_characteristic(vgs, vds_start=0.0, vds_stop=0.8, n_points=200)
        for vgs in vgs_list
    }

    # -----------------------------------------------------------------------
    # Plot: IoFET Kennlinien
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IoFET – Ionotronic Field-Effect Transistor Kennlinien", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Übertragungskennlinie (log + lin)
    ax1 = axes[0]
    vgs_arr = tc.vgs_values
    ids_arr_nA = [i * 1e9 for i in tc.ids_values]
    ids_arr_log = [max(i, 1e-6) for i in ids_arr_nA]

    ax1.semilogy(vgs_arr, ids_arr_log, color="#1f77b4", lw=2, label="I_DS (log)")
    ax1.axvline(vt, color="#d62728", ls="--", lw=1.5, label=f"V_T = {vt*1000:.0f} mV")
    ax1.set_xlabel("V_GS [V]")
    ax1.set_ylabel("I_DS [nA] (log)")
    ax1.set_title("Übertragungskennlinie I_DS(V_GS)\nV_DS = 50 mV")
    ax1.legend()

    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    ax1b.plot(vgs_arr, ids_arr_nA, color="#2ca02c", lw=1.5, ls="--", alpha=0.7, label="I_DS (linear)")
    ax1b.set_ylabel("I_DS [nA] (linear)", color="#2ca02c")
    ax1b.tick_params(axis="y", labelcolor="#2ca02c")

    # Sub-Plot 2: Ausgangskennlinien
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(vgs_list)))
    for vgs, color in zip(vgs_list, colors):
        curve = output_curves[vgs]
        vds_arr = [pt[0] for pt in curve]
        ids_arr = [pt[1] * 1e9 for pt in curve]
        ax2.plot(vds_arr, ids_arr, color=color, lw=2, label=f"V_GS = {vgs:.1f} V")

    # Sättigungsgrenze einzeichnen
    vds_sat = np.linspace(0.0, 0.5, 100)
    ax2.plot(vds_sat, vds_sat * 0 + 0, "k--", lw=0.5)
    ax2.set_xlabel("V_DS [V]")
    ax2.set_ylabel("I_DS [nA]")
    ax2.set_title("Ausgangskennlinien I_DS(V_DS)\nfür verschiedene V_GS")
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("src/results/plot_02_transistor.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_02_transistor.svg")


# ===========================================================================
# 4) droplet.py – Tropfendynamik und Selbstorganisation
# ===========================================================================

def demo_droplet() -> None:
    header("MODUL: droplet.py – Tropfendynamik und Selbstorganisation")

    # Einzelner Tropfen
    header("Einzelner Tropfen (Pikoliter-Maßstab)", 2)
    d = Droplet(
        volume_m3=1e-15,          # 1 pL
        position=(5, 5),
        ion_concentration_mol_m3=100.0,
        temperature_K=298.15,
    )
    print(f"  Volumen               = {d.volume_m3*1e15:.1f} pL")
    print(f"  Äquiv. Radius         = {d.radius_m*1e6:.2f} µm")
    print(f"  Oberfläche            = {d.surface_area_m2*1e12:.4f} µm²")
    print(f"  Oberflächenenergie    = {d.surface_energy_J*1e15:.4f} fJ")
    print(f"  Ionengehalt           = {d.total_ion_content_mol:.4e} mol")
    print(f"  Laplace-Druck ΔP      = {d.laplace_pressure_Pa():.2f} Pa")
    print(f"  Leitfähigkeit (Kanal) = {d.conductance_S(5e-6, 25e-12):.4e} S")

    # Verschmelzung
    header("Tropfen-Verschmelzung (OR-Gatter)", 2)
    d1 = Droplet(volume_m3=1e-15, position=(3, 3), label="A")
    d2 = Droplet(volume_m3=0.5e-15, position=(3, 4), label="B")
    print(f"  Tropfen A: r = {d1.radius_m*1e6:.2f} µm  V = {d1.volume_m3*1e15:.1f} pL")
    print(f"  Tropfen B: r = {d2.radius_m*1e6:.2f} µm  V = {d2.volume_m3*1e15:.1f} pL")
    print(f"  Kann verschmelzen?    = {DropletMerger.can_merge(d1, d2)}")

    merged = DropletMerger.merge(d1, d2)
    e_release = DropletMerger.merge_energy_released_J(d1, d2)
    print(f"  Fusionierter Tropfen: r = {merged.radius_m*1e6:.2f} µm  "
          f"c = {merged.ion_concentration_mol_m3:.1f} mol/m³  Label = '{merged.label}'")
    print(f"  Freigesetzte Oberflächenenergie: ΔE = {e_release*1e18:.4f} aJ")

    # Selbstorganisation
    header("Selbstorganisation (5 × 5 Matrix, 4 Schritte)", 2)
    organizer = SelfOrganizer(threshold_voltage_V=0.12)

    # Ausgangstropfen auf 5×5-Matrix verteilen
    initial_positions = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 2)]
    vols = [2e-15, 1.5e-15, 1e-15, 2.5e-15, 1e-15]
    for pos, vol in zip(initial_positions, vols):
        organizer.add_droplet(Droplet(volume_m3=vol, position=pos, label=str(pos)))

    print(f"  Startzustand: {organizer.num_droplets} Tropfen")

    # Spannungsschema: mittlere Elektroden aktivieren
    voltage_pattern_step = {
        (2, 1): 0.15, (2, 2): 0.15, (2, 3): 0.15,
        (1, 2): 0.15, (3, 2): 0.15,
    }

    sim_data = []
    for step in range(4):
        droplets_after = organizer.step(voltage_pattern_step)
        n = len(droplets_after)
        sim_data.append(n)
        print(f"  Schritt {step+1}: {n} Tropfen  "
              f"(Positionen: {[d.position for d in droplets_after]})")

    print(f"  Relaxationszeit (L=1mm): τ = {organizer.relaxation_time_s(1e-3):.2e} s")

    # -----------------------------------------------------------------------
    # Plot: Tropfendynamik
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tropfendynamik – Selbstorganisation & Verschmelzung", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Laplace-Druck vs. Radius
    ax1 = axes[0]
    radii_um = np.linspace(1, 500, 300)
    radii_m = radii_um * 1e-6

    gamma_25 = SurfaceTension.at_celsius(25)
    lp_25 = [2 * gamma_25 / r for r in radii_m]  # Pa
    gamma_60 = SurfaceTension.at_celsius(60)
    lp_60 = [2 * gamma_60 / r for r in radii_m]
    gamma_80 = SurfaceTension.at_celsius(80)
    lp_80 = [2 * gamma_80 / r for r in radii_m]

    ax1.loglog(radii_um, lp_25, lw=2, label="25 °C", color="#1f77b4")
    ax1.loglog(radii_um, lp_60, lw=2, label="60 °C", color="#ff7f0e", ls="--")
    ax1.loglog(radii_um, lp_80, lw=2, label="80 °C", color="#d62728", ls=":")
    ax1.set_xlabel("Tropfenradius [µm]")
    ax1.set_ylabel("Laplace-Druck ΔP [Pa]")
    ax1.set_title("Laplace-Druck vs. Tropfenradius\nbei verschiedenen Temperaturen")
    ax1.legend()

    # Sub-Plot 2: Verschmelzungsenergie vs. Radienverhältnis
    ax2 = axes[1]
    r_base = 50e-6  # µm
    r2_factors = np.linspace(0.1, 2.0, 200)
    e_merges_aJ = [
        SurfaceTension.droplet_merge_energy_J(r_base, r_base * f, 298.15) * 1e18
        for f in r2_factors
    ]
    ax2.plot(r2_factors, e_merges_aJ, lw=2, color="#2ca02c")
    ax2.axvline(1.0, color="gray", ls="--", lw=1, label="r₂ = r₁ (gleich groß)")
    ax2.fill_between(r2_factors, 0, e_merges_aJ, alpha=0.15, color="#2ca02c")
    ax2.set_xlabel("Radienverhältnis r₂/r₁  (r₁ = 50 µm)")
    ax2.set_ylabel("Freigesetzte Oberflächenenergie ΔE [aJ]")
    ax2.set_title("Verschmelzungsenergie zweier Tropfen\nvs. Radienverhältnis")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("src/results/plot_03_droplet.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_03_droplet.svg")


# ===========================================================================
# 5) logic.py – Tropfenlogik, Gatter, Netzwerke
# ===========================================================================

def demo_logic() -> None:
    header("MODUL: logic.py – Tropfenlogik und Gatter-Netzwerke")

    # Alle Gatter-Typen demonstrieren
    header("Alle Gatter-Typen im Überblick", 2)

    gate_configs = [
        (GateType.AND,  "AND",  ["A", "B"], "Y_AND"),
        (GateType.OR,   "OR",   ["A", "B"], "Y_OR"),
        (GateType.NOT,  "NOT",  ["A"],       "Y_NOT"),
        (GateType.NAND, "NAND", ["A", "B"], "Y_NAND"),
        (GateType.NOR,  "NOR",  ["A", "B"], "Y_NOR"),
        (GateType.XOR,  "XOR",  ["A", "B"], "Y_XOR"),
        (GateType.XNOR, "XNOR", ["A", "B"], "Y_XNOR"),
        (GateType.BUFFER, "BUF", ["A"],     "Y_BUF"),
    ]

    gates = []
    for gtype, name, inputs, output in gate_configs:
        g = DropletGate(gate_type=gtype, name=name, inputs=inputs, output=output)
        gates.append(g)
        print(f"  {name:<6}: universal={g.is_universal}  Energie: {g.energy_model}")

    # Wahrheitstabelle für AND, OR, XOR
    header("Wahrheitstabellen ausgewählter Gatter", 2)

    for gtype, name in [(GateType.AND, "AND"), (GateType.OR, "OR"), (GateType.XOR, "XOR")]:
        g = DropletGate(gtype, name, ["X", "Y"], f"Z_{name}")
        print(f"\n  {name}-Gatter:")
        print(f"  X Y | Z")
        print(f"  -----")
        for x in [False, True]:
            for y in [False, True]:
                z = g.evaluate({"X": x, "Y": y})
                print(f"  {int(x)} {int(y)} | {int(z)}")

    # Logisches Netzwerk: Halbaddierer
    header("Logisches Netzwerk: Halbaddierer (AND + XOR)", 2)

    net = LogicNetwork(name="Halbaddierer")
    net.declare_primary_input("A")
    net.declare_primary_input("B")

    net.add_gate(DropletGate(GateType.XOR,  "XOR1", ["A", "B"], "Sum"))
    net.add_gate(DropletGate(GateType.AND,  "AND1", ["A", "B"], "Carry"))

    print(f"  Netzwerk: '{net.name}'  Gatter: {net.num_gates}  Turing-vollständig: {net.is_turing_complete()}")
    print(f"  Maximale Signallaufzeit: {net.total_propagation_delay_s*1000:.1f} ms")
    print(f"  Passive OR-Gatter (Null-Energie): {net.or_gate_count}")

    tt = TruthTable.generate(net, ["A", "B"], ["Sum", "Carry"])
    print(f"\n  Wahrheitstabelle ({tt.num_rows} Zeilen):")
    print("  " + "\n  ".join(tt.format().split("\n")))

    # Vollständigeres Netzwerk: NAND-basierter Inverter
    header("NAND-basiertes Netzwerk (Turing-vollständig)", 2)

    net2 = LogicNetwork(name="NAND-Test")
    net2.declare_primary_input("P")
    net2.declare_primary_input("Q")
    net2.add_gate(DropletGate(GateType.NAND, "N1", ["P", "Q"],   "NAND_PQ"))
    net2.add_gate(DropletGate(GateType.NAND, "N2", ["NAND_PQ", "NAND_PQ"], "NOT_NAND"))
    print(f"  Turing-vollständig: {net2.is_turing_complete()}")

    result = net2.evaluate({"P": True, "Q": False})
    print(f"  P=1, Q=0 → NAND(P,Q)={int(result['NAND_PQ'])}, NOT_NAND={int(result['NOT_NAND'])}")

    # -----------------------------------------------------------------------
    # Plot: Gatter-Wahrheitstabellen und Energievergleich
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IRF Tropfenlogik – Gatter-Übersicht", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Heatmap der Wahrheitstabellen (AND, OR, XOR, NAND)
    ax1 = axes[0]
    gate_types_vis = [GateType.AND, GateType.OR, GateType.XOR, GateType.NAND, GateType.NOR, GateType.XNOR]
    gate_names_vis = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
    truth_matrix = np.zeros((6, 4))

    combos = [(False, False), (False, True), (True, False), (True, True)]
    for i, gtype in enumerate(gate_types_vis):
        g = DropletGate(gtype, gtype.name, ["X", "Y"], "Z")
        for j, (x, y) in enumerate(combos):
            truth_matrix[i, j] = int(g.evaluate({"X": x, "Y": y}))

    im = ax1.imshow(truth_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(["00", "01", "10", "11"])
    ax1.set_yticks(range(6))
    ax1.set_yticklabels(gate_names_vis)
    ax1.set_xlabel("Eingangskombination (A, B)")
    ax1.set_title("Wahrheitstabellen aller 2-Eingang-Gatter\n(grün=1, rot=0)")

    for i in range(6):
        for j in range(4):
            ax1.text(j, i, str(int(truth_matrix[i, j])),
                     ha="center", va="center", fontsize=13, fontweight="bold",
                     color="white" if truth_matrix[i, j] == 0 else "black")

    # Sub-Plot 2: Propagationszeit & Gatter-Energie-Modell (qualitativ)
    ax2 = axes[1]
    gate_labels = [g.name for g in gates]
    # Propagationszeiten (alle gleich 10ms außer bei NOT und BUFFER mit 5ms)
    delays_ms = [g.propagation_delay_s * 1000 for g in gates]
    is_passive = [g.gate_type == GateType.OR or g.gate_type == GateType.BUFFER for g in gates]
    bar_colors = ["#2ca02c" if p else "#1f77b4" for p in is_passive]

    bars = ax2.bar(gate_labels, delays_ms, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Gatter-Typ")
    ax2.set_ylabel("Propagationszeit [ms]")
    ax2.set_title("Signallaufzeiten pro Gatter-Typ\n(grün = passiv / Null-Energie)")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="#2ca02c", label="Passiv (Null-Energie)"),
        Patch(color="#1f77b4", label="Aktiv"),
    ]
    ax2.legend(handles=legend_patches)
    ax2.set_ylim(0, max(delays_ms) * 1.3)

    plt.tight_layout()
    plt.savefig("src/results/plot_04_logic.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_04_logic.svg")


# ===========================================================================
# 6) memory.py – Pinning-Wells, Ionische Speicherzellen, MemoryArray
# ===========================================================================

def demo_memory() -> None:
    header("MODUL: memory.py – Speichertechnologien des IRF")

    # -----------------------------------------------------------------------
    # 6a) Pinning-Well (geometrischer Speicher)
    # -----------------------------------------------------------------------
    header("Pinning-Well – Geometrischer Speicher", 2)

    well = PinningWell(
        position=(2, 3),
        well_area_m2=(5e-6)**2,       # 5 µm × 5 µm
        pinning_energy_J=1e-15,
        label="W[2,3]",
    )

    print(f"  Position            : {well.position}")
    print(f"  Well-Fläche         : {well.well_area_m2*1e12:.1f} µm²")
    print(f"  Pinning-Energie     : {well.pinning_energy_J*1e15:.1f} fJ")
    print(f"  Anfangs belegt      : {well.is_occupied} (logisch: {well.logical_value})")

    d = Droplet(volume_m3=1e-15, position=(2, 3))
    well.write(d)
    print(f"  Nach Schreiben      : {well.is_occupied} (logisch: {well.logical_value})")
    print(f"  Kann gehalten werden (E=0.5fJ < 1fJ)? {well.can_hold(d, 0.5e-15)}")
    print(f"  Kann gehalten werden (E=2fJ > 1fJ)?   {well.can_hold(d, 2e-15)}")

    read_d = well.read()
    print(f"  Gelesener Tropfen   : V = {read_d.volume_m3*1e15:.1f} pL")

    erased = well.erase()
    print(f"  Nach Löschen        : {well.is_occupied}")

    # -----------------------------------------------------------------------
    # 6b) Ionische Speicherzelle (Analogspeicher / Synaptische Plastizität)
    # -----------------------------------------------------------------------
    header("Ionische Speicherzelle – Synaptische Plastizität", 2)

    cell = IonicMemoryCell(
        position=(1, 1),
        volume_m3=1e-15,
        base_concentration_mol_m3=100.0,
        plasticity_rate=0.1,
    )

    activation_history = []
    decay_history = []
    steps_act = list(range(10))
    steps_dec = list(range(10, 30))

    # Aktivierungsphase
    for _ in steps_act:
        c = cell.activate()
        activation_history.append((cell.activation_count, c, cell.normalized_strength))

    print(f"  Nach 10 Aktivierungen: c = {cell.concentration_mol_m3:.2f} mol/m³  "
          f"Stärke = {cell.normalized_strength:.3f}")

    # Zerfallsphase
    for _ in steps_dec:
        c = cell.decay(decay_factor=0.05)
        decay_history.append(c)

    print(f"  Nach 20 Zerfallsschritten: c = {cell.concentration_mol_m3:.2f} mol/m³  "
          f"Stärke = {cell.normalized_strength:.3f}")
    cell.reset()
    print(f"  Nach Reset: c = {cell.concentration_mol_m3:.2f} mol/m³")

    # -----------------------------------------------------------------------
    # 6c) MemoryArray
    # -----------------------------------------------------------------------
    header("MemoryArray – Digitaler Bit-Speicher (4×8)", 2)

    mem = MemoryArray(rows=4, cols=8)
    print(f"  Kapazität           : {mem.capacity_bits} Bits")

    # Wörter schreiben
    words = [42, 127, 0, 255]
    for r, w in enumerate(words):
        mem.write_word(r, w)
        readback = mem.read_word(r)
        print(f"  Zeile {r}: schreibe {w:3d} (0b{w:08b}) → lese {readback:3d} (0b{readback:08b})  "
              f"{'✓' if readback == w else '✗'}")

    print(f"  Gesetzte Bits gesamt: {mem.count_ones()} / {mem.capacity_bits}")

    # Einzelbit-Operationen
    mem.write_bit(1, 3, False)
    print(f"  Bit [1,3] nach Löschen: {mem.read_bit(1, 3)}")
    mem.clear()
    print(f"  Nach Clear: {mem.count_ones()} gesetzte Bits")

    # -----------------------------------------------------------------------
    # Plot: Speichertechnologien
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IRF Speichertechnologien", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Ionische Plastizität – Aktivierung & Zerfall
    ax1 = axes[0]
    # Neu simulieren für Plot
    cell2 = IonicMemoryCell(position=(0,0), volume_m3=1e-15,
                            base_concentration_mol_m3=100.0, plasticity_rate=0.1)
    conc_plot = []
    strength_plot = []
    steps_plot = []

    for i in range(15):  # Aktivierung
        cell2.activate()
        conc_plot.append(cell2.concentration_mol_m3)
        strength_plot.append(cell2.normalized_strength)
        steps_plot.append(i)

    for i in range(15, 50):  # Zerfall
        cell2.decay(decay_factor=0.08)
        conc_plot.append(cell2.concentration_mol_m3)
        strength_plot.append(cell2.normalized_strength)
        steps_plot.append(i)

    ax1.plot(steps_plot, conc_plot, lw=2, color="#1f77b4", label="Konzentration [mol/m³]")
    ax1.axvline(14.5, color="gray", ls="--", lw=1.5, label="Aktivierungs-Ende")
    ax1.axhline(100, color="#d62728", ls=":", lw=1, label="Ruhekonzentration")
    ax1.fill_between(steps_plot[:15], 100, conc_plot[:15], alpha=0.2, color="#2ca02c", label="Lernphase")
    ax1.fill_between(steps_plot[15:], 100, conc_plot[15:], alpha=0.2, color="#ff7f0e", label="Zerfallsphase")
    ax1.set_xlabel("Zeitschritt")
    ax1.set_ylabel("Ionenkonzentration [mol/m³]")
    ax1.set_title("Ionische Speicherzelle: Hebbsches Lernen\n& exponentieller Zerfall")
    ax1.legend(fontsize=8)

    # Sub-Plot 2: MemoryArray – Bit-Muster als Heatmap
    ax2 = axes[1]
    mem2 = MemoryArray(rows=8, cols=8)
    # Muster: IRF-Logo (simples Bitmuster)
    pattern_words = [0b01100110, 0b10011001, 0b10000001, 0b11000011,
                     0b10000001, 0b10011001, 0b01100110, 0b00000000]
    for r, w in enumerate(pattern_words):
        mem2.write_word(r, w)

    bit_matrix = np.array([[int(mem2.read_bit(r, c)) for c in range(8)] for r in range(8)])
    ax2.imshow(bit_matrix, cmap="RdYlGn", aspect="equal", vmin=0, vmax=1,
               interpolation="nearest")
    ax2.set_title("MemoryArray 8×8 – Bit-Muster\n(Pinning-Wells: grün=1, rot=0)")
    ax2.set_xlabel("Spalte (Bit)")
    ax2.set_ylabel("Zeile (Wort)")
    ax2.set_xticks(range(8))
    ax2.set_yticks(range(8))
    ax2.grid(True, color="white", linewidth=1.5)

    for r in range(8):
        for c in range(8):
            ax2.text(c, r, str(int(mem2.read_bit(r, c))),
                     ha="center", va="center", fontsize=10, fontweight="bold",
                     color="white" if bit_matrix[r, c] == 0 else "black")

    plt.tight_layout()
    plt.savefig("src/results/plot_05_memory.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_05_memory.svg")


# ===========================================================================
# 7) matrix.py – Elektrodenmatrix
# ===========================================================================

def demo_matrix() -> None:
    header("MODUL: matrix.py – Elektrodenmatrix und Topologie")

    # MatrixConfig
    cfg = MatrixConfig(
        rows=10,
        cols=10,
        electrode_pitch_m=5e-6,
        electrode_size_m=3.5e-6,
        channel_height_m=50e-6,
        num_layers=3,
    )

    print(f"  Dimensionen         : {cfg.rows} × {cfg.cols} × {cfg.num_layers} Schichten")
    print(f"  Elektrodenanzahl    : {cfg.num_electrodes}")
    print(f"  Chipfläche          : {cfg.chip_area_m2*1e6:.2f} mm²")
    print(f"  Elektrodendichte    : {cfg.electrode_density_per_m2:.4e} /m²")
    print(f"  Äquiv. Transistordichte: {cfg.equivalent_transistor_density_per_m2:.4e} /m²")
    print(f"                        = {cfg.equivalent_transistor_density_per_m2*1e-4:.4e} /cm²")

    # ElectrodeMatrix
    matrix = ElectrodeMatrix(cfg)
    print(f"\n  Matrix erzeugt – Layer 0 initialer Zustand:")

    # Spannungsmuster anlegen
    pattern = [
        [0.15 if (r + c) % 2 == 0 else 0.0 for c in range(10)]
        for r in range(10)
    ]
    matrix.apply_voltage_pattern(pattern, layer=0)

    # Zustände setzen
    for r in range(10):
        for c in range(10):
            if pattern[r][c] > 0:
                matrix.set_state(0, r, c, ElectrodeState.ON)

    active = matrix.active_electrodes(layer=0)
    util = matrix.utilization(layer=0)
    print(f"  Aktive Elektroden   : {len(active)}")
    print(f"  Auslastung (Layer 0): {util:.2f}")

    # Nachbarn abfragen
    neighbors = matrix.neighbors(0, 5, 5)
    print(f"  Nachbarn von (0, 5, 5): {neighbors}")

    # Spannungen auslesen
    v_sample = matrix.get_voltage(0, 3, 3)
    print(f"  Spannung an (0,3,3) : {v_sample:.3f} V")

    # Layer-Reset
    matrix.reset_layer(0)
    print(f"  Nach Layer-Reset: Auslastung = {matrix.utilization(0):.2f}")

    # -----------------------------------------------------------------------
    # Plot: Elektrodenmatrix-Visualisierung
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Elektrodenmatrix – Topologie & Spannungsverteilung", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Schachbrettmuster auf 10×10-Matrix
    ax1 = axes[0]
    v_matrix = np.array([[0.15 if (r + c) % 2 == 0 else 0.0 for c in range(10)] for r in range(10)])
    im1 = ax1.imshow(v_matrix, cmap="Blues", aspect="equal", vmin=0, vmax=0.2)
    plt.colorbar(im1, ax=ax1, label="Spannung [V]")
    ax1.set_title("Spannungsmuster Layer 0\n(Schachbrettmuster – alternierend aktiv)")
    ax1.set_xlabel("Spalte")
    ax1.set_ylabel("Zeile")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.grid(True, color="white", lw=0.5)

    # Sub-Plot 2: Äquivalente Transistordichte vs. Pitch (für alle Generationen)
    ax2 = axes[1]
    pitches_nm = []
    densities_cm2 = []
    gen_labels = []

    for node, params in TECHNOLOGY_PARAMETERS.items():
        p = params["electrode_pitch_m"]
        cfg_temp = MatrixConfig(
            rows=100, cols=100,
            electrode_pitch_m=p,
            electrode_size_m=p * 0.7,
            channel_height_m=params["channel_height_m"],
            num_layers=3,
        )
        pitches_nm.append(p * 1e9)
        densities_cm2.append(cfg_temp.equivalent_transistor_density_per_m2 * 1e-4)
        gen_labels.append(node.name)

    colors_gen = plt.cm.viridis(np.linspace(0.1, 0.9, len(pitches_nm)))
    ax2.scatter(pitches_nm, densities_cm2, s=80, c=colors_gen, zorder=5)
    ax2.plot(pitches_nm, densities_cm2, "k--", lw=1, alpha=0.5)

    for i, label in enumerate(gen_labels):
        ax2.annotate(label, (pitches_nm[i], densities_cm2[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    # CMOS-Referenz (3nm-Prozess: ~1.7×10⁸ Transistoren/cm²)
    ax2.axhline(1.7e8, color="#d62728", ls="--", lw=1.5, label="CMOS 3nm (~1.7×10⁸/cm²)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Elektrodenpitch [nm]")
    ax2.set_ylabel("Äquiv. Transistordichte [cm⁻²]")
    ax2.set_title("IRF Transistordichte vs. Pitch\n(alle Technologiegenerationen, 3 Schichten)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("src/results/plot_06_matrix.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_06_matrix.svg")


# ===========================================================================
# 8) fabrication.py – Fertigungsprozess
# ===========================================================================

def demo_fabrication() -> None:
    header("MODUL: fabrication.py – Fertigungsprozess & Technologiegenerationen")

    # Technologiegenerationen
    header("Technologieparameter aller Generationen", 2)
    for node, params in TECHNOLOGY_PARAMETERS.items():
        pitch_nm = params["electrode_pitch_m"] * 1e9
        v_sw = params["switching_voltage_V"] * 1000
        f_sw = params["switching_frequency_Hz"]
        e_sw = params["energy_per_switch_J"] * 1e18
        print(f"  {node.value}: pitch={pitch_nm:.0f}nm  V_T={v_sw:.0f}mV  "
              f"f={f_sw:.0e}Hz  E={e_sw:.1f}aJ")

    # Vollständiger Fertigungsprozess für IRF-1
    header("7-Schritt-Fertigungsprozess (IRF-1)", 2)

    fab = FabricationProcess(node=TechnologyNode.IRF_1)
    print(f"  Schrittanzahl       : {fab.num_steps}")
    print(f"  Gesamtdauer         : {fab.total_duration_s/3600:.1f} h")
    print(f"  Gesamtausbeute      : {fab.total_yield*100:.3f} %")

    warnings = fab.validate()
    print(f"  Validierungswarnungen: {len(warnings)} (Warnungen: {warnings if warnings else 'keine'})")

    results = fab.execute_all()
    for r in results:
        print(f"  [{r['step_number']}] {r['name']:<40} → "
              f"Ausbeute={r['yield_factor']*100:.1f}%  "
              f"Dauer={r['duration_s']/3600:.1f}h")

    # Validierung aller Generationen
    header("Parametervalidierung alle Generationen", 2)
    for node in TechnologyNode:
        warnings = ProcessValidator.validate_technology_node(node)
        status = "✓ OK" if not warnings else f"⚠ {len(warnings)} Warnungen"
        print(f"  {node.value}: {status}")

    # -----------------------------------------------------------------------
    # Plot: Fertigungsprozess
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IRF-Fertigungsprozess & Technologie-Roadmap", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Ausbeute pro Schritt und kumulative Ausbeute
    ax1 = axes[0]
    step_nums = [s.step_number for s in fab.steps]
    step_names_short = ["Sub.Präp.", "Gate-Dep.", "Dielektrikum\nALD",
                        "Oberflächen-\nfunkt.", "Ion.Flüss.\nBef.", "Herm.\nVersi.", "CMOS\nIntegr."]
    yield_per_step = [s.yield_factor * 100 for s in fab.steps]

    # Kumulative Ausbeute
    cum_yield = [100.0]
    for y in yield_per_step:
        cum_yield.append(cum_yield[-1] * y / 100)
    cum_yield = cum_yield[1:]

    x_pos = np.arange(len(step_nums))
    bar_colors = ["#2ca02c" if y > 99.5 else "#ff7f0e" if y > 99.0 else "#d62728"
                  for y in yield_per_step]

    bars = ax1.bar(x_pos, yield_per_step, color=bar_colors, alpha=0.8, edgecolor="white")
    ax1.plot(x_pos, cum_yield, "k--o", lw=2, ms=5, label="Kumul. Ausbeute [%]")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(step_names_short, fontsize=7)
    ax1.set_ylim(98.5, 100.05)
    ax1.set_ylabel("Ausbeute [%]")
    ax1.set_title("Prozessausbeute pro Schritt\n(IRF-1 Laborgeneration)")
    ax1.legend(fontsize=8)

    for bar, y in zip(bars, yield_per_step):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{y:.2f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Sub-Plot 2: Technologie-Roadmap (Schaltfrequenz vs. Schaltenergie)
    ax2 = axes[1]
    nodes = list(TechnologyNode)
    freqs = [TECHNOLOGY_PARAMETERS[n]["switching_frequency_Hz"] for n in nodes]
    energies_aJ = [TECHNOLOGY_PARAMETERS[n]["energy_per_switch_J"] * 1e18 for n in nodes]
    pitches_nm2 = [TECHNOLOGY_PARAMETERS[n]["electrode_pitch_m"] * 1e9 for n in nodes]
    years = [2026, 2027, 2029, 2031, 2033, 2035]

    scatter_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(nodes)))
    sc = ax2.scatter(freqs, energies_aJ, s=[200 * p / 50 for p in pitches_nm2],
                     c=scatter_colors, zorder=5, edgecolors="black", lw=0.5)

    for i, (node, year) in enumerate(zip(nodes, years)):
        ax2.annotate(f"{year}\n({pitches_nm2[i]:.0f}nm)",
                     (freqs[i], energies_aJ[i]),
                     textcoords="offset points", xytext=(8, 4), fontsize=7)

    ax2.plot(freqs, energies_aJ, "k--", lw=1, alpha=0.4)

    # CMOS-Referenz
    ax2.scatter([1e9], [1000], s=200, marker="*", color="#d62728", zorder=6, label="CMOS-Referenz (~1fJ, 1GHz)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Schaltfrequenz [Hz]")
    ax2.set_ylabel("Schaltenergie pro Vorgang [aJ]")
    ax2.set_title("IRF Technologie-Roadmap 2026–2035\n(Schaltfrequenz vs. Schaltenergie)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("src/results/plot_07_fabrication.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_07_fabrication.svg")


# ===========================================================================
# 9) architecture.py – Von-Neumann-Flasche, Durchsatzanalyse, IRFSystem
# ===========================================================================

def demo_architecture() -> None:
    header("MODUL: architecture.py – Systemarchitektur und Durchsatz")

    # Von-Neumann-Bottleneck
    header("Von-Neumann-Flasche (CMOS vs. IRF)", 2)

    # Typisches Hochleistungs-CMOS-System
    bottleneck = VonNeumannBottleneck(
        cpu_frequency_Hz=3e9,             # 3 GHz CPU
        memory_bandwidth_bytes_s=64e9,    # DDR5 64 GB/s
        word_size_bytes=8,
    )
    print(f"  CPU-Frequenz         : {bottleneck.cpu_frequency_Hz/1e9:.1f} GHz")
    print(f"  Speicherbandbreite   : {bottleneck.memory_bandwidth_bytes_s/1e9:.0f} GB/s")
    print(f"  Max. Speicherzugriffe/Takt: {bottleneck.max_memory_accesses_per_cycle:.3f}")
    print(f"  Flaschenhals-Ratio   : {bottleneck.bottleneck_ratio:.2f}x  (>1 = Speicher limitiert)")
    print(f"  Stall-Zyklen (8 Zugriffe): {bottleneck.cycles_stalled_per_operation(8):.1f}")

    # IRF-Systemkonfiguration
    header("IRF-Systemkonfiguration (10×10×3 Matrix)", 2)

    cfg = MatrixConfig(
        rows=10, cols=10,
        electrode_pitch_m=5e-6,
        electrode_size_m=3.5e-6,
        channel_height_m=50e-6,
        num_layers=3,
    )
    system = IRFSystem(
        matrix_config=cfg,
        cmos_frequency_Hz=1e9,
        configuration_time_s=0.01,
    )

    status = system.status_report()
    print(f"  Matrix               : {status['matrix_rows']}×{status['matrix_cols']}×{status['num_layers']}")
    print(f"  Gesamtelektroden     : {status['total_electrodes']}")
    print(f"  Chipfläche           : {status['chip_area_mm2']:.4f} mm²")
    print(f"  Äquiv. Trans.-Dichte : {status['equivalent_transistor_density_cm2']:.4e} /cm²")
    print(f"  Speicherkapazität    : {status['memory_capacity_bits']} Bits")
    print(f"  Konfigurationsrate   : {status['config_rate_Hz']:.1f} Hz")

    # HybridController
    header("HybridController – Durchsatz", 2)
    ctrl = system.controller
    tp = ctrl.throughput_ops_per_second(utilization=0.8)
    e_cfg = ctrl.energy_per_config_J(0.12, switching_fraction=0.5)
    print(f"  Elektroden           : {ctrl.num_electrodes}")
    print(f"  Parallele Ops/Konfig : {ctrl.parallel_operations_per_config}")
    print(f"  Durchsatz (80%)      : {tp:.2e} ops/s")
    print(f"  Energie/Konfig.      : {e_cfg*1e18:.4f} aJ")

    # ThroughputAnalyzer
    header("ThroughputAnalyzer – Skalierungsanalyse", 2)
    analyzer = ThroughputAnalyzer(cfg)
    cmos_ref = 1.7e8  # Transistoren/cm² bei 3nm CMOS
    ratio = analyzer.compare_to_cmos_density(cmos_ref)
    total_tp = analyzer.total_throughput_ops_s(100.0, configs_per_computation=1)
    energy_eff = analyzer.energy_efficiency_ops_J(100.0, 0.12, 0.5)
    scaling = analyzer.irf_scaling_exponent()

    print(f"  IRF/CMOS-Dichteverhältnis: {ratio:.2f}x")
    print(f"  Gesamtdurchsatz      : {total_tp:.2e} ops/s")
    print(f"  Energieeffizienz     : {energy_eff:.2e} ops/J")
    print(f"  Skalierungsexponent  : {scaling:.1f} (kubisch)")

    # Konfigurationszyklus ausführen
    header("Konfigurationszyklus", 2)
    vpattern = [[0.12 if (r + c) % 2 == 0 else 0.0 for c in range(10)] for r in range(10)]
    metrics = system.execute_configuration(vpattern, layer=0)
    print(f"  Zyklus {metrics['cycle']}: util={metrics['utilization']:.2f}  "
          f"E={metrics['energy_J']*1e18:.4f}aJ  ops={metrics['throughput_ops']:.1f}")

    # -----------------------------------------------------------------------
    # Plot: Architektur & Skalierung
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IRF-Systemarchitektur – Von-Neumann-Analyse & Skalierung", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Von-Neumann-Bottleneck bei verschiedenen CPU/RAM-Konstellationen
    ax1 = axes[0]
    cpu_freqs = [1e9, 2e9, 3e9, 4e9, 5e9]
    bandwidths_GB = [16, 32, 64, 128, 256]  # GB/s

    # Heatmap: Bottleneck-Ratio
    ratio_matrix = np.zeros((len(cpu_freqs), len(bandwidths_GB)))
    for i, cpu in enumerate(cpu_freqs):
        for j, bw in enumerate(bandwidths_GB):
            bn = VonNeumannBottleneck(cpu, bw * 1e9, 8)
            ratio_matrix[i, j] = bn.bottleneck_ratio

    im = ax1.imshow(ratio_matrix, cmap="RdYlGn_r", aspect="auto",
                    norm=Normalize(vmin=0, vmax=8))
    plt.colorbar(im, ax=ax1, label="Bottleneck-Ratio (>1 = Speicher limitiert)")
    ax1.set_xticks(range(len(bandwidths_GB)))
    ax1.set_xticklabels([f"{b} GB/s" for b in bandwidths_GB], rotation=30, fontsize=8)
    ax1.set_yticks(range(len(cpu_freqs)))
    ax1.set_yticklabels([f"{f/1e9:.0f} GHz" for f in cpu_freqs])
    ax1.set_xlabel("Speicherbandbreite")
    ax1.set_ylabel("CPU-Frequenz")
    ax1.set_title("Von-Neumann-Bottleneck-Ratio\n(rot = stark limitiert, grün = OK)")

    for i in range(len(cpu_freqs)):
        for j in range(len(bandwidths_GB)):
            ax1.text(j, i, f"{ratio_matrix[i,j]:.1f}",
                     ha="center", va="center", fontsize=8, fontweight="bold",
                     color="white" if ratio_matrix[i,j] > 4 else "black")

    # Sub-Plot 2: Skalierung IRF vs. CMOS
    ax2 = axes[1]
    pitches = np.logspace(-8, -5, 200)  # 10nm bis 10µm

    # IRF: ρ ∝ 1/p³ (kubisch)
    p_ref = 5e-6
    rho_ref = cfg.equivalent_transistor_density_per_m2 * 1e-4  # /cm²
    irf_densities = rho_ref * (p_ref / pitches) ** 3

    # CMOS: ρ ∝ 1/p² (quadratisch)
    cmos_ref_density = 1.7e8  # /cm² bei 3nm
    p_cmos_ref = 3e-9
    cmos_densities = cmos_ref_density * (p_cmos_ref / pitches) ** 2

    ax2.loglog(pitches * 1e9, irf_densities, lw=2.5, color="#1f77b4", label="IRF (kubisch, p⁻³)")
    ax2.loglog(pitches * 1e9, cmos_densities, lw=2.5, color="#d62728", ls="--", label="CMOS (quadratisch, p⁻²)")

    ax2.axvline(5000, color="#1f77b4", ls=":", lw=1, alpha=0.5)
    ax2.axvline(3, color="#d62728", ls=":", lw=1, alpha=0.5)

    ax2.set_xlabel("Strukturgröße / Pitch [nm]")
    ax2.set_ylabel("Transistordichte [cm⁻²]")
    ax2.set_title("Skalierungsgesetze: IRF vs. CMOS\n(kubisch vs. quadratisch)")
    ax2.legend()
    ax2.set_xlim(1, 1e4)

    plt.tight_layout()
    plt.savefig("src/results/plot_08_architecture.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_08_architecture.svg")


# ===========================================================================
# 10) simulation.py – Simulator und Zeitverlauf
# ===========================================================================

def demo_simulation() -> None:
    header("MODUL: simulation.py – IRF-Simulationsengine")

    # System aufsetzen
    cfg = MatrixConfig(
        rows=6, cols=6,
        electrode_pitch_m=5e-6,
        electrode_size_m=3.5e-6,
        channel_height_m=50e-6,
        num_layers=1,
    )
    system = IRFSystem(
        matrix_config=cfg,
        cmos_frequency_Hz=1e9,
        configuration_time_s=0.02,  # 50 Hz Konfigurationsrate
    )

    sim_cfg = SimulationConfig(
        total_time_s=1.0,
        time_step_s=0.02,
        threshold_voltage_V=0.12,
        record_all_steps=True,
    )

    simulator = Simulator(system, sim_cfg)

    # Anfangstropfen hinzufügen
    initial_droplets = [
        Droplet(volume_m3=2e-15, position=(0, 0), label="D1"),
        Droplet(volume_m3=1.5e-15, position=(0, 5), label="D2"),
        Droplet(volume_m3=1e-15, position=(5, 0), label="D3"),
        Droplet(volume_m3=2e-15, position=(5, 5), label="D4"),
        Droplet(volume_m3=0.5e-15, position=(2, 2), label="D5"),
        Droplet(volume_m3=0.5e-15, position=(3, 3), label="D6"),
    ]
    for d in initial_droplets:
        simulator.add_droplet(d)

    print(f"  Simulationsdauer    : {sim_cfg.total_time_s:.1f} s")
    print(f"  Zeitschritt         : {sim_cfg.time_step_s*1000:.0f} ms")
    print(f"  Schrittanzahl       : {sim_cfg.num_steps}")
    print(f"  Anfangstropfen      : {len(initial_droplets)}")
    print(f"  Schaltspannung      : {sim_cfg.threshold_voltage_V*1000:.0f} mV")

    # Spannungsschema: Pulsierende Zentralaktivierung
    def voltage_schedule(t: float) -> dict[tuple[int, int], float]:
        """Rotierendes Aktivierungsmuster für 6×6-Matrix."""
        phase = int(t / 0.1) % 4
        patterns = [
            {(2, 2): 0.15, (2, 3): 0.15, (3, 2): 0.15, (3, 3): 0.15},
            {(1, 1): 0.15, (1, 4): 0.15, (4, 1): 0.15, (4, 4): 0.15},
            {(2, 2): 0.15, (3, 3): 0.15, (1, 1): 0.15, (4, 4): 0.15},
            {(0, 3): 0.15, (5, 2): 0.15, (2, 0): 0.15, (3, 5): 0.15},
        ]
        return patterns[phase]

    print("\n  Starte Simulation ...")
    result = simulator.run(voltage_schedule)

    # Ergebnis auswerten
    print(f"\n  Simulation abgeschlossen:")
    print(f"  Aufgezeichnete Schritte : {result.num_steps_recorded}")
    print(f"  Gesamtenergie           : {result.total_energy_J*1e18:.4f} aJ")
    print(f"  Gesamtverschmelzungen   : {result.total_merges}")
    print(f"  Mittlere Auslastung     : {result.mean_utilization:.3f}")
    print(f"  Maximale Tropfenanzahl  : {result.peak_droplet_count()}")
    print(f"  Tropfen am Ende         : {len(result.final_droplets)}")

    if result.final_droplets:
        for d in result.final_droplets:
            print(f"    - {d.label}: V={d.volume_m3*1e15:.2f}pL  r={d.radius_m*1e6:.2f}µm  "
                  f"pos={d.position}")

    # Zeitreihen
    ts_droplets = result.time_series("num_droplets")
    ts_energy = result.time_series("energy_J")
    ts_util = result.time_series("utilization")
    ts_merges = result.time_series("merges")

    # -----------------------------------------------------------------------
    # Plot: Simulationszeitverlauf
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IRF-Simulation – Zeitverlauf (6×6-Matrix, 1 Sekunde)", fontsize=13, fontweight="bold")

    times_d = [t for t, _ in ts_droplets]
    vals_d = [v for _, v in ts_droplets]
    times_e = [t for t, _ in ts_energy]
    vals_e = [v * 1e18 for _, v in ts_energy]  # aJ
    times_u = [t for t, _ in ts_util]
    vals_u = [v for _, v in ts_util]
    times_m = [t for t, _ in ts_merges]
    vals_m = [v for _, v in ts_merges]

    # Sub-Plot 1: Tropfenanzahl + Verschmelzungen
    ax1 = axes[0]
    color_d, color_m = "#1f77b4", "#d62728"
    ax1.step(times_d, vals_d, where="post", lw=2, color=color_d, label="Tropfenanzahl")
    ax1.set_xlabel("Zeit [s]")
    ax1.set_ylabel("Tropfenanzahl", color=color_d)
    ax1.tick_params(axis="y", labelcolor=color_d)
    ax1.set_title("Tropfendynamik & Verschmelzungen\nim Simulationsverlauf")
    ax1.set_ylim(bottom=0)

    ax1b = ax1.twinx()
    ax1b.spines["right"].set_visible(True)
    ax1b.bar(times_m, vals_m, width=sim_cfg.time_step_s * 0.8,
             color=color_m, alpha=0.6, label="Verschmelzungen/Schritt")
    ax1b.set_ylabel("Verschmelzungen / Schritt", color=color_m)
    ax1b.tick_params(axis="y", labelcolor=color_m)
    ax1b.set_ylim(bottom=0)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color=color_d, lw=2),
        Patch(color=color_m, alpha=0.6),
    ]
    ax1.legend(legend_handles, ["Tropfenanzahl", "Verschmelzungen/Schritt"],
               loc="upper right", fontsize=8)

    # Sub-Plot 2: Energie + Auslastung
    ax2 = axes[1]
    color_e, color_u = "#2ca02c", "#ff7f0e"
    ax2.plot(times_e, vals_e, lw=2, color=color_e, label="Energie/Schritt [aJ]")
    ax2.set_xlabel("Zeit [s]")
    ax2.set_ylabel("Energie pro Schritt [aJ]", color=color_e)
    ax2.tick_params(axis="y", labelcolor=color_e)
    ax2.set_title("Energieverbrauch & Matrixauslastung\nim Simulationsverlauf")

    ax2b = ax2.twinx()
    ax2b.spines["right"].set_visible(True)
    ax2b.plot(times_u, [v * 100 for v in vals_u], lw=2, color=color_u,
              ls="--", label="Auslastung [%]")
    ax2b.set_ylabel("Matrixauslastung [%]", color=color_u)
    ax2b.tick_params(axis="y", labelcolor=color_u)
    ax2b.set_ylim(0, 100)

    legend_handles2 = [
        Line2D([0], [0], color=color_e, lw=2),
        Line2D([0], [0], color=color_u, lw=2, ls="--"),
    ]
    ax2.legend(legend_handles2, ["Energie/Schritt [aJ]", "Auslastung [%]"],
               loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("src/results/plot_09_simulation.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_09_simulation.svg")


# ===========================================================================
# 11) Integrierter Systemvergleich – IRF vs. CMOS
# ===========================================================================

def demo_system_comparison() -> None:
    header("SYSTEMVERGLEICH: IRF-Roadmap vs. CMOS-Benchmarks")

    print("\n  Vergleich IRF-Generationen vs. CMOS:")
    print(f"  {'Generation':<30} {'Pitch':>8} {'Dichte':>14} {'Energie':>12} {'Freq.':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*14} {'-'*12} {'-'*10}")

    for node, params in TECHNOLOGY_PARAMETERS.items():
        pitch = params["electrode_pitch_m"]
        cfg_t = MatrixConfig(
            rows=100, cols=100,
            electrode_pitch_m=pitch,
            electrode_size_m=pitch * 0.7,
            channel_height_m=params["channel_height_m"],
            num_layers=3,
        )
        density = cfg_t.equivalent_transistor_density_per_m2 * 1e-4
        energy_aJ = params["energy_per_switch_J"] * 1e18
        freq = params["switching_frequency_Hz"]
        print(f"  {node.value:<30} {pitch*1e9:>6.0f}nm {density:>12.2e}/cm² "
              f"{energy_aJ:>10.3f}aJ {freq:>8.0e}Hz")

    # CMOS Referenz
    print(f"\n  {'CMOS 3nm (TSMC 2024)':<30} {'3':>6}nm  "
          f"{'1.70e+08':>12}/cm²     {'1.000':>6}aJ  {'1.00e+09':>8}Hz")

    # -----------------------------------------------------------------------
    # Plot: Zusammenfassender Systemvergleich
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IRF vs. CMOS – Umfassender Systemvergleich", fontsize=13, fontweight="bold")

    # Sub-Plot 1: Radardiagramm (auf Rechteck-Achsen normiert)
    ax1 = axes[0]
    categories = ["Trans.-Dichte", "Schaltenergie\n(niedrig=gut)", "Schaltfrequenz",
                  "Betriebsspannung\n(niedrig=gut)", "Fertigungs-\nkomplexität\n(niedrig=gut)"]

    # IRF-4 (200nm) vs CMOS 3nm – normierte Werte [0,1]
    irf4_params = TECHNOLOGY_PARAMETERS[TechnologyNode.IRF_4]
    irf4_cfg = MatrixConfig(
        rows=100, cols=100,
        electrode_pitch_m=irf4_params["electrode_pitch_m"],
        electrode_size_m=irf4_params["electrode_pitch_m"] * 0.7,
        channel_height_m=irf4_params["channel_height_m"],
        num_layers=3,
    )
    irf4_density = irf4_cfg.equivalent_transistor_density_per_m2 * 1e-4

    # Normierung: CMOS als Referenz (0.5 = gleichauf, >0.5 = IRF besser)
    irf4_scores = [
        min(1.0, irf4_density / (1.7e10)),        # Dichte (IRF 200nm)
        min(1.0, 1.0 / (irf4_params["energy_per_switch_J"] / 1e-18)),  # Energie-Effizienz
        min(1.0, irf4_params["switching_frequency_Hz"] / 1e9),          # Frequenz
        min(1.0, 0.7 / irf4_params["switching_voltage_V"]),             # Niedriger = besser
        0.85,                                                             # Fertigungskomplexität
    ]
    cmos_scores = [0.5, 0.5, 1.0, 0.4, 0.3]

    N = len(categories)
    x_cat = np.arange(N)
    width = 0.35

    bars1 = ax1.bar(x_cat - width/2, irf4_scores, width, color="#1f77b4",
                    alpha=0.8, label="IRF-4 (200nm, 3 Layer)")
    bars2 = ax1.bar(x_cat + width/2, cmos_scores, width, color="#d62728",
                    alpha=0.8, label="CMOS 3nm")

    ax1.set_xticks(x_cat)
    ax1.set_xticklabels(categories, fontsize=7.5)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel("Normierter Wert (höher = besser)")
    ax1.set_title("Normierter Technologievergleich\nIRF-4 vs. CMOS 3nm")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="CMOS-Referenz")
    ax1.legend(fontsize=8)

    # Sub-Plot 2: Energieeffizienz-Karte (Ops/J vs. Ops/s)
    ax2 = axes[1]

    # Alle IRF-Generationen
    irf_ops_s = []
    irf_ops_J = []
    irf_labels_plot = []

    for node, params in TECHNOLOGY_PARAMETERS.items():
        freq = params["switching_frequency_Hz"]
        energy = params["energy_per_switch_J"]
        cfg_t = MatrixConfig(
            rows=100, cols=100,
            electrode_pitch_m=params["electrode_pitch_m"],
            electrode_size_m=params["electrode_pitch_m"] * 0.7,
            channel_height_m=params["channel_height_m"],
            num_layers=3,
        )
        n_el = cfg_t.num_electrodes
        ops_s = freq * n_el
        ops_J = 1.0 / energy if energy > 0 else float("inf")
        irf_ops_s.append(ops_s)
        irf_ops_J.append(ops_J)
        irf_labels_plot.append(node.name)

    colors_gen2 = plt.cm.viridis(np.linspace(0.1, 0.9, len(irf_ops_s)))
    sc = ax2.scatter(irf_ops_s, irf_ops_J, s=120, c=colors_gen2,
                     zorder=5, edgecolors="black", lw=0.5)
    ax2.plot(irf_ops_s, irf_ops_J, "k--", lw=1, alpha=0.4)

    for i, label in enumerate(irf_labels_plot):
        ax2.annotate(label, (irf_ops_s[i], irf_ops_J[i]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)

    # CMOS-Referenzpunkt (1GHz, 1fJ → 10¹⁸ ops/J theoretisch, ~10²⁷ ops/s für ganzen Chip?)
    ax2.scatter([1e9 * 1e10], [1e18], s=200, marker="*", color="#d62728",
                zorder=6, label="CMOS-Ref. (1GHz, 1fJ, 10¹⁰ Transistoren)")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Chip-Durchsatz [Ops/s]")
    ax2.set_ylabel("Energieeffizienz [Ops/J]")
    ax2.set_title("Energieeffizienz-Karte\nIRF-Generationen (100×100×3 Chip)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("src/results/plot_10_system_comparison.svg", bbox_inches="tight")
    plt.close()
    print("\n  → Plot gespeichert: plot_10_system_comparison.svg")


# ===========================================================================
# Hauptprogramm
# ===========================================================================

def main() -> None:
    print("""
══════════════════════════════════════════════════════════════════════════
    IRF – Ionotronic Reconfigurable Fabric                                 
    Ausführliche Bibliotheks-Demo                                          
                                                                           
    Demonstriert alle 10 Module:                                           
    utils, physics, transistor, droplet, logic, memory, matrix,            
    fabrication, architecture, simulation                                  
══════════════════════════════════════════════════════════════════════════
    """)

    os.makedirs("src/results/", exist_ok=True)

    demo_utils()
    demo_physics()
    demo_transistor()
    demo_droplet()
    demo_logic()
    demo_memory()
    demo_matrix()
    demo_fabrication()
    demo_architecture()
    demo_simulation()
    demo_system_comparison()

    print("\n")
    print("=" * 72)
    print("  Demo abgeschlossen. Generierte Plots:")
    plots = [
        "plot_01_physics.svg         – Physikalische Grundmodelle",
        "plot_02_transistor.svg      – IoFET Kennlinien",
        "plot_03_droplet.svg         – Tropfendynamik",
        "plot_04_logic.svg           – Gatter-Logik",
        "plot_05_memory.svg          – Speichertechnologien",
        "plot_06_matrix.svg          – Elektrodenmatrix",
        "plot_07_fabrication.svg     – Fertigungsprozess",
        "plot_08_architecture.svg    – Systemarchitektur",
        "plot_09_simulation.svg      – Simulationszeitverlauf",
        "plot_10_system_comparison.svg – IRF vs. CMOS",
    ]
    for p in plots:
        print(f"    ✓ {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()
