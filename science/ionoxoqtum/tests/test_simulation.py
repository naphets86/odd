"""
tests/test_simulation.py
=========================
Tests für simulation/fabry_perot.py, tmm.py, polymer.py, ebl.py
"""

import pytest
import numpy as np
from src.fabry_perot import FabryPerotResonator
from src.tmm import (
    TransferMatrixMethod,
    OpticalLayer,
    fresnel_rs,
    fresnel_rp,
    build_hydrogel_stack,
)
from src.polymer import FloryRehnerModel, PolymerParameters
from src.ebl import ProximityEffectModel, EBLParameters


# ══════════════════════════════════════════════════════════════════════════════
# FABRY-PÉROT
# ══════════════════════════════════════════════════════════════════════════════

class TestFabryPerotResonator:

    def setup_method(self):
        self.fp = FabryPerotResonator(d0=250e-9, n0=1.40, R1=0.5, R2=0.5)

    def test_cavity_thickness_dry(self):
        assert self.fp.cavity_thickness(Q=1.0) == pytest.approx(250e-9)

    def test_cavity_thickness_swollen(self):
        assert self.fp.cavity_thickness(Q=3.0) == pytest.approx(750e-9)

    def test_effective_index_dry(self):
        assert self.fp.effective_index(Q=1.0) == pytest.approx(1.40, rel=1e-6)

    def test_effective_index_swollen_approaches_1(self):
        n = self.fp.effective_index(Q=100.0)
        assert abs(n - 1.0) < 0.01

    def test_effective_index_formula(self):
        """n(Q) = 1 + (n₀ − 1) / Q."""
        Q = 2.5
        expected = 1.0 + (1.40 - 1.0) / Q
        assert self.fp.effective_index(Q) == pytest.approx(expected, rel=1e-10)

    def test_resonance_wavelength_dry(self):
        """λ = 2 n d für m=1."""
        lam = self.fp.resonance_wavelength(Q=1.0, order=1)
        expected = 2 * 1.40 * 250e-9
        assert lam == pytest.approx(expected, rel=1e-6)

    def test_resonance_wavelength_scales_with_Q(self):
        lam1 = self.fp.resonance_wavelength(Q=1.0)
        lam2 = self.fp.resonance_wavelength(Q=2.0)
        # Näherung: λ_res ≈ 2 d₀ Q → ratio ≈ 2
        assert lam2 > lam1

    def test_reflection_spectrum_length(self):
        lam, R = self.fp.reflection_spectrum(Q=1.0, n_points=200)
        assert len(lam) == 200
        assert len(R) == 200

    def test_reflection_in_range(self):
        lam, R = self.fp.reflection_spectrum(Q=1.0)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0)

    def test_reflection_has_minimum(self):
        """Resonanz = Reflexionsminimum."""
        lam, R = self.fp.reflection_spectrum(Q=1.0, n_points=1000)
        assert R.min() < R.max()

    def test_resonance_shifts_with_Q(self):
        """Höheres Q → Resonanz bei längerer Wellenlänge."""
        lam1, R1 = self.fp.reflection_spectrum(Q=1.0, n_points=500)
        lam2, R2 = self.fp.reflection_spectrum(Q=3.0, n_points=500)
        pos1 = lam1[np.argmin(R1)]
        pos2 = lam2[np.argmin(R2)]
        assert pos2 > pos1

    def test_finesse_formula(self):
        """F = π (R1 R2)^(1/4) / (1 − √(R1 R2))."""
        R1 = R2 = 0.5
        expected = np.pi * (R1 * R2)**0.25 / (1.0 - np.sqrt(R1 * R2))
        assert self.fp.finesse == pytest.approx(expected, rel=1e-10)

    def test_finesse_positive(self):
        assert self.fp.finesse > 0

    def test_fwhm_positive(self):
        fwhm = self.fp.fwhm_nm(Q=1.0)
        assert fwhm > 0

    def test_cie_xy_in_range(self):
        x, y = self.fp.cie_xy(Q=1.0)
        assert 0 <= x <= 1
        assert 0 <= y <= 1

    def test_tunable_spectra_returns_dict(self):
        d = self.fp.tunable_spectra([1.0, 2.0, 3.0])
        assert len(d) == 3

    def test_color_gamut_shape(self):
        Q_arr, x_arr, y_arr = self.fp.color_gamut(Q_range=(1.0, 4.0), n_Q=20)
        assert len(Q_arr) == 20
        assert len(x_arr) == 20
        assert len(y_arr) == 20

    def test_repr(self):
        assert "FabryPerotResonator" in repr(self.fp)


# ══════════════════════════════════════════════════════════════════════════════
# TMM
# ══════════════════════════════════════════════════════════════════════════════

class TestTMM:

    def test_single_air_glass_reflectance(self):
        """Luft/Glas Grenzfläche: R ≈ 4% (Fresnel)."""
        medium    = OpticalLayer(1.0,  0.0)
        substrate = OpticalLayer(1.52, 0.0)
        tmm = TransferMatrixMethod(medium, [], substrate)
        R = tmm.reflectance(550e-9)
        assert 0.03 < R < 0.06

    def test_reflectance_between_0_and_1(self):
        stack = build_hydrogel_stack(d0_nm=250.0, Q=1.0)
        lam_nm, R = stack.spectrum()
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0)

    def test_transmittance_plus_reflectance_le_1(self):
        """R + T ≤ 1 (Absorptionsverluste möglich)."""
        stack = build_hydrogel_stack()
        lam = 550e-9
        R = stack.reflectance(lam)
        T = stack.transmittance(lam)
        assert R + T <= 1.0 + 1e-10

    def test_spectrum_length(self):
        stack = build_hydrogel_stack()
        lam, R = stack.spectrum(n_points=200)
        assert len(lam) == 200
        assert len(R) == 200

    def test_swollen_vs_dry_spectrum_different(self):
        dry   = build_hydrogel_stack(Q=1.0)
        swollen = build_hydrogel_stack(Q=3.0)
        _, R_dry = dry.spectrum()
        _, R_swollen = swollen.spectrum()
        assert not np.allclose(R_dry, R_swollen, atol=0.01)

    def test_phase_matrix_shape(self):
        stack = build_hydrogel_stack()
        M = stack.phase_matrix(550e-9)
        assert M.shape == (2, 2)

    def test_phase_matrix_unimodular(self):
        """Det(M) ≈ 1 für verlustfreie Schichten (exakt für ideale Medien)."""
        medium    = OpticalLayer(1.0,  0.0)
        lossless  = OpticalLayer(1.5,  100e-9)
        substrate = OpticalLayer(1.0,  0.0)
        tmm = TransferMatrixMethod(medium, [lossless], substrate)
        M = tmm.phase_matrix(550e-9)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert abs(abs(det) - 1.0) < 0.01

    def test_ellipsometry_params_types(self):
        stack = build_hydrogel_stack()
        psi, delta = stack.ellipsometry_params(550e-9)
        assert isinstance(psi, float)
        assert isinstance(delta, float)

    def test_repr(self):
        stack = build_hydrogel_stack()
        assert "TMM" in repr(stack)


class TestFresnelCoefficients:

    def test_normal_incidence_rs(self):
        """r_s bei senkrechtem Einfall = (n1−n2)/(n1+n2)."""
        r = fresnel_rs(1.0, 1.5, theta_i_deg=0.0)
        expected = (1.0 - 1.5) / (1.0 + 1.5)
        assert abs(r - expected) < 1e-10

    def test_normal_incidence_rp(self):
        """r_p bei senkrechtem Einfall = (n2−n1)/(n2+n1)."""
        r = fresnel_rp(1.0, 1.5, theta_i_deg=0.0)
        expected = (1.5 - 1.0) / (1.5 + 1.0)
        assert abs(r - expected) < 1e-10

    def test_reflectance_glass_air(self):
        R = abs(fresnel_rs(1.0, 1.52))**2
        assert 0.035 < R < 0.05

    def test_symmetric_reflection(self):
        """R(n1,n2) = R(n2,n1) bei senkrechtem Einfall."""
        R1 = abs(fresnel_rs(1.0, 1.5))**2
        R2 = abs(fresnel_rs(1.5, 1.0))**2
        assert R1 == pytest.approx(R2, rel=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# POLYMER
# ══════════════════════════════════════════════════════════════════════════════

class TestFloryRehner:

    def setup_method(self):
        self.model = FloryRehnerModel(
            PolymerParameters(chi=0.45, Vm=1.8e-5, Vc=5.0e-4, n0=1.40)
        )

    def test_equilibrium_swelling_positive(self):
        Q = self.model.equilibrium_swelling()
        assert Q > 1.0

    def test_equilibrium_swelling_chi_0_45(self):
        """Typisches Q für PDMS/Wasser: 2–6."""
        Q = self.model.equilibrium_swelling()
        assert 1.5 < Q < 20.0

    def test_higher_chi_lower_swelling(self):
        """Höheres χ → geringere Quellung (schlechteres Lösungsmittel)."""
        m_good = FloryRehnerModel(PolymerParameters(chi=0.3, Vm=1.8e-5, Vc=5.0e-4))
        m_bad  = FloryRehnerModel(PolymerParameters(chi=0.6, Vm=1.8e-5, Vc=5.0e-4))
        Q_good = m_good.equilibrium_swelling()
        Q_bad  = m_bad.equilibrium_swelling()
        assert Q_good > Q_bad

    def test_refractive_index_dry(self):
        """n(Q=1) = n₀."""
        n = self.model.refractive_index(Q=1.0)
        assert n == pytest.approx(1.40, rel=1e-10)

    def test_refractive_index_approaches_1_for_large_Q(self):
        n = self.model.refractive_index(Q=1000.0)
        assert abs(n - 1.0) < 0.001

    def test_resonance_wavelength_scales_with_Q(self):
        d0 = 250e-9
        lam1 = self.model.resonance_wavelength(d0, Q=1.0)
        lam2 = self.model.resonance_wavelength(d0, Q=2.0)
        assert lam2 > lam1

    def test_tunable_range_positive_width(self):
        d0 = 250e-9
        lam_min, lam_max = self.model.tunable_range_nm(d0, Q_min=1.0, Q_max=4.0)
        assert lam_max > lam_min

    def test_relaxation_time_formula(self):
        """τ = h² / (π² D)."""
        h = 500e-9
        tau = self.model.relaxation_time(h)
        expected = h**2 / (np.pi**2 * self.model.p.D_eff)
        assert tau == pytest.approx(expected, rel=1e-10)

    def test_relaxation_time_quadratic_in_h(self):
        """τ ∝ h²."""
        tau1 = self.model.relaxation_time(500e-9)
        tau2 = self.model.relaxation_time(1000e-9)
        assert tau2 == pytest.approx(4 * tau1, rel=1e-8)

    def test_kinetic_swelling_shape(self):
        t, Q_t = self.model.kinetic_swelling(
            h=500e-9, Q_init=1.0, Q_eq=3.0, n_times=100
        )
        assert len(t) == 100
        assert len(Q_t) == 100

    def test_kinetic_swelling_starts_at_Q_init(self):
        t, Q_t = self.model.kinetic_swelling(
            h=500e-9, Q_init=1.0, Q_eq=3.0, n_times=100
        )
        assert Q_t[0] == pytest.approx(1.0, abs=0.001)

    def test_kinetic_swelling_approaches_Q_eq(self):
        t, Q_t = self.model.kinetic_swelling(
            h=500e-9, Q_init=1.0, Q_eq=3.0, n_times=500
        )
        # Nach 5τ: Q(5τ) = Q_eq + (Q_init-Q_eq)*e^(-5) ≈ Q_eq ± 0.014
        assert abs(Q_t[-1] - 3.0) < 0.02

    def test_kinetic_swelling_monotone(self):
        t, Q_t = self.model.kinetic_swelling(
            h=500e-9, Q_init=1.0, Q_eq=3.0
        )
        diffs = np.diff(Q_t)
        assert np.all(diffs >= -1e-12)

    def test_switching_time_ms_positive(self):
        tau_ms = self.model.switching_time_ms(500e-9)
        assert tau_ms > 0

    def test_lorentz_lorenz_n_approaches_n0(self):
        """Lorentz-Lorenz bei Q=1 sollte n₀ ergeben."""
        n_ll = self.model.lorentz_lorenz_n(Q=1.0)
        assert abs(n_ll - self.model.p.n0) < 0.05

    def test_swelling_vs_chi_shape(self):
        chi_arr, Q_arr = self.model.swelling_vs_chi(chi_range=(0.2, 0.6), n_points=20)
        assert len(chi_arr) == 20
        assert len(Q_arr) == 20

    def test_repr(self):
        assert "FloryRehner" in repr(self.model)


# ══════════════════════════════════════════════════════════════════════════════
# EBL
# ══════════════════════════════════════════════════════════════════════════════

class TestEBLParameters:

    def test_bethe_range_positive(self):
        params = EBLParameters(beam_energy_keV=100.0)
        assert params.bethe_range_m > 0

    def test_bethe_range_increases_with_energy(self):
        p1 = EBLParameters(beam_energy_keV=50.0)
        p2 = EBLParameters(beam_energy_keV=100.0)
        assert p2.bethe_range_m > p1.bethe_range_m


class TestProximityEffect:

    def setup_method(self):
        self.model = ProximityEffectModel(
            EBLParameters(beta_f=10e-9, beta_b=10e-6, eta=0.6),
            grid_size_nm=1000.0,
            n_grid=64,
        )

    def test_psf_1d_positive(self):
        r = np.linspace(0, 500, 100)
        psf = self.model.psf_1d(r)
        assert np.all(psf >= 0)

    def test_psf_1d_maximum_at_zero(self):
        r = np.array([0.0, 10.0, 100.0])
        psf = self.model.psf_1d(r)
        assert psf[0] >= psf[1] >= psf[2]

    def test_psf_2d_shape(self):
        psf = self.model.psf_2d()
        assert psf.shape == (64, 64)

    def test_psf_2d_normalized(self):
        psf = self.model.psf_2d()
        total = psf.sum() * self.model.dx_nm**2
        assert abs(total - 1.0) < 0.1  # Näherungsweise normiert

    def test_psf_2d_symmetric(self):
        psf = self.model.psf_2d()
        n = self.model.n_grid
        # Symmetrie: PSF[i,j] ≈ PSF[j,i]
        assert np.allclose(psf, psf.T, atol=1e-10)

    def test_energy_deposition_shape(self):
        n = self.model.n_grid
        pattern = np.zeros((n, n))
        pattern[n//2, n//2] = 1.0
        E = self.model.energy_deposition(pattern)
        assert E.shape == (n, n)

    def test_energy_deposition_spreads(self):
        """Energiedeposition breiter als Eingangsmuster."""
        n = self.model.n_grid
        pattern = np.zeros((n, n))
        pattern[n//2, n//2] = 1.0
        E = self.model.energy_deposition(pattern)
        # Mehr als 3 Pixel haben signifikante Energie (PSF breitet aus)
        significant = (E > E.max() * 0.01).sum()
        assert significant >= 4

    def test_proximity_correction_shape(self):
        n = self.model.n_grid
        target = np.ones((n, n)) * 0.5
        corrected = self.model.proximity_correction(target)
        assert corrected.shape == (n, n)

    def test_proximity_correction_nonnegative(self):
        n = self.model.n_grid
        target = np.random.default_rng(0).uniform(0, 1, (n, n))
        corrected = self.model.proximity_correction(target)
        assert np.all(corrected >= 0)

    def test_dose_to_swelling_range(self):
        n = self.model.n_grid
        dose = np.random.default_rng(0).uniform(0, 1, (n, n))
        Q_map = self.model.dose_to_swelling(dose, Q_min=1.0, Q_max=4.0)
        assert np.all(Q_map >= 1.0 - 1e-10)
        assert np.all(Q_map <= 4.0 + 1e-10)

    def test_encode_texture_output_shapes(self):
        n = self.model.n_grid
        texture = np.random.default_rng(0).uniform(0, 1, (n, n))
        corrected, Q_map = self.model.encode_texture(texture)
        assert corrected.shape == (n, n)
        assert Q_map.shape == (n, n)

    def test_repr(self):
        assert "ProximityEffectModel" in repr(self.model)
