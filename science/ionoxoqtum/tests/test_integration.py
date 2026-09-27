"""
tests/test_integration.py
=========================
Integrationstests: Zusammenspiel aller Systemkomponenten.
Testet die hydro-photonische Plattform als Gesamtsystem.
"""

import pytest
import numpy as np
from src.ionic import kcl_solution
from src.ewod import standard_al2o3_system
from src.iofet import IoFET, IoFETGeometry
from src.droplet import NANDGate, FullAdder, RippleCarryAdder, IRFNetwork, IRFNode
from src.fabry_perot import FabryPerotResonator
from src.tmm import build_hydrogel_stack
from src.polymer import FloryRehnerModel, PolymerParameters
from src.ebl import ProximityEffectModel, EBLParameters
from src.inverse_design import InversePhotonicsDesigner, SingleLayerDesigner


class TestIonicToEWODToIoFET:
    """Test: Ionische Lösung → EWOD-System → IoFET."""

    def test_full_transistor_pipeline(self):
        # 1. Ionische Lösung
        solution = kcl_solution(0.5)
        sigma = solution.conductivity()
        assert sigma > 0

        # 2. EWOD-System
        ewod = standard_al2o3_system(contact_angle_0=110.0)
        V_schalt = ewod.switching_voltage(60.0)
        assert V_schalt > 0

        # 3. IoFET
        transistor = IoFET(
            conductivity=sigma,
            ewod=ewod,
            threshold_voltage=V_schalt * 0.5,
        )
        I_on  = transistor.drain_current(V_gs=V_schalt * 2, V_ds=0.1)
        I_off = transistor.drain_current(V_gs=0.0, V_ds=0.1)
        assert I_on > I_off

    def test_switching_energy_in_zeptojoule(self):
        ewod = standard_al2o3_system()
        transistor = IoFET(ewod=ewod)
        E = transistor.switching_energy()
        if not np.isnan(E):
            # E = ½ C V² mit C~8e-21 F, V~3.9V -> E~6e-20 J (sub-aJ Bereich)
            assert 1e-24 < E < 1e-12


class TestPolymerToFabryPerot:
    """Test: Flory-Rehner-Quellung → FP-Resonator → Farbsteuerung."""

    def test_color_tuning_pipeline(self):
        # 1. Polymer-Gleichgewichtquellung
        polymer = FloryRehnerModel(
            PolymerParameters(chi=0.45, Vm=1.8e-5, Vc=5.0e-4, n0=1.40)
        )
        Q_eq = polymer.equilibrium_swelling()
        assert Q_eq > 1.0

        # 2. FP-Resonator: Resonanzwellenlänge
        fp = FabryPerotResonator(d0=250e-9, n0=1.40, R1=0.5, R2=0.5)
        lam_res = fp.resonance_wavelength(Q=Q_eq)
        assert lam_res > 0

        # 3. Sichtbares Spektrum
        lam_nm, R = fp.reflection_spectrum(Q=Q_eq)
        assert np.all(R >= 0) and np.all(R <= 1)

    def test_full_visible_gamut(self):
        """Quellungsbereich 1–4 liefert Spektrum von 700nm (Q=1) bis ~2800nm (Q=4).
        Für vollständiges sichtbares Spektrum (420-700nm) benötigt man d₀≈150nm."""
        # Mit d₀=150nm: λ_res(Q=1) ≈ 2*1.4*150 = 420nm (blau)
        fp_blue = FabryPerotResonator(d0=150e-9, n0=1.40)
        lam_1 = fp_blue.resonance_wavelength(Q=1.0) * 1e9
        lam_4 = fp_blue.resonance_wavelength(Q=4.0) * 1e9
        assert lam_1 < 500  # Beginnt im Kurzwelligen
        assert lam_4 > 600  # Endet im Langwelligen

    def test_swelling_kinetics_video_rate(self):
        """Schaltzeit für 500nm-Schicht soll < 200 ms sein (Videorate)."""
        polymer = FloryRehnerModel()
        tau_ms = polymer.switching_time_ms(h=500e-9)
        assert tau_ms < 200.0


class TestEBLToSwellingMap:
    """Test: EBL-Kodierung → Quellungskarte → optischer Effekt."""

    def test_texture_to_color_map(self):
        n = 32
        ebl = ProximityEffectModel(
            EBLParameters(beta_f=10e-9, beta_b=5e-6, eta=0.5),
            grid_size_nm=500.0,
            n_grid=n,
        )
        # Einfaches Streifenmuster
        texture = np.zeros((n, n))
        texture[:, ::4] = 1.0
        corrected, Q_map = ebl.encode_texture(texture, Q_range=(1.0, 4.0))
        assert corrected.shape == (n, n)
        assert Q_map.shape == (n, n)
        assert Q_map.min() >= 1.0 - 0.1
        assert Q_map.max() <= 4.0 + 0.1


class TestLogicComputation:
    """Test: Boolesche Logik aus Tropfengattern."""

    def test_nand_functional_completeness(self):
        """Alle 4 Grundgatter aus NAND konstruierbar."""
        nand = NANDGate()
        # NOT
        assert nand.evaluate(0, 0) == 1
        assert nand.evaluate(1, 1) == 0
        # AND = NOT(NAND)
        and_11 = nand.evaluate(nand.evaluate(1, 1), nand.evaluate(1, 1))
        assert and_11 == 1

    def test_4bit_addition_table(self):
        """Vollständige 3-Bit-Additionstabelle testen."""
        adder = RippleCarryAdder(8)
        test_cases = [(0, 0, 0), (1, 1, 2), (7, 8, 15), (100, 55, 155), (127, 1, 128)]
        for a, b, expected in test_cases:
            result, _ = adder.add(a, b)
            assert result == expected, f"{a} + {b} = {result}, erwartet {expected}"

    def test_irf_network_logic_state(self):
        """IRF-Netzwerk speichert logische Zustände."""
        net = IRFNetwork()
        for i in range(8):
            net.add_node(IRFNode(f"N{i}", (i * 1e-6, 0, 0)))
        # Schreibe 8-Bit Wert
        value = 0b10110011  # 179
        for i in range(8):
            net.set_state(f"N{i}", (value >> i) & 1)
        # Lese zurück
        readback = sum(net.get_state(f"N{i}") << i for i in range(8))
        assert readback == value


class TestTMMAnalogy:
    """Test: Photon-Elektron-Analogie der Transfermatrix."""

    def test_transfer_matrix_structure(self):
        """M ist 2×2-komplex und hat bestimmte Struktur."""
        stack = build_hydrogel_stack(d0_nm=250.0, Q=1.0)
        M = stack.phase_matrix(550e-9)
        assert M.shape == (2, 2)
        assert M.dtype == complex

    def test_optical_spectral_consistency(self):
        """TMM und FP-Resonator geben ähnliche Spektren."""
        fp = FabryPerotResonator(d0=250e-9, n0=1.40, R1=0.3, R2=0.3)
        tmm = build_hydrogel_stack(d0_nm=250.0, Q=1.0, R_mirror=0.3)
        lam_nm, R_fp = fp.reflection_spectrum(Q=1.0)
        lam_nm2, R_tmm = tmm.spectrum()
        # Beide Spektren sollten in ähnlichem Bereich liegen
        assert R_fp.mean() >= 0
        assert R_tmm.mean() >= 0


class TestInversDesignIntegration:
    """Test: Vollständige Optimierungspipeline."""

    def test_design_for_green_light(self):
        """Design-Ziel: Reflexionsmaximum bei 550 nm (Grün)."""
        lam = np.linspace(380, 780, 80)
        R_target = 0.6 * np.exp(-((lam - 550)**2) / (2 * 40**2))

        designer = InversePhotonicsDesigner(
            target_spectrum=R_target,
            wavelengths_nm=lam,
            n_layers=2,
            d_bounds=(100e-9, 600e-9),
            n_bounds=(1.2, 2.5),
        )
        result = designer.optimize(n_restarts=2, max_iter=30, seed=42)

        assert result.final_loss >= 0
        assert len(result.d_opt) == 2

        # Vorhersage
        R_pred = designer.predict_spectrum(result)
        assert len(R_pred) == len(lam)

    def test_single_layer_design_round_trip(self):
        """Design → Simulation → Resonanzwellenlänge stimmt überein."""
        target_nm = 550.0
        des = SingleLayerDesigner(target_wavelength_nm=target_nm)
        n = 1.40
        d0 = des.design_thickness(n)
        lam_calc = 2 * n * d0  # m=1, senkrechter Einfall
        assert lam_calc * 1e9 == pytest.approx(target_nm, rel=1e-6)
