"""
Tests für Cochlear Model und FilterBank Module.

Deckt ab:
- BasilarMembrane Mechanik
- Greenwood-Mapping
- Gammatone-Filter
- CochlearFilterBank
"""

import pytest
import numpy as np
from acoustcs.cochlear_model import BasilarMembrane, CochlearModel
from acoustcs.filterbank import GammaToneFilter, CochlearFilterBank


class TestBasilarMembraneInitialization:
    """Tests für BasilarMembrane-Initialisierung."""
    
    def test_default_initialization(self):
        """Test Standard-Initialisierung."""
        bm = BasilarMembrane()
        
        assert bm.L == 35.0
        assert bm.f_min == 20.0
        assert bm.f_max == 20000.0
        assert bm.n_positions == 100
    
    def test_custom_parameters(self):
        """Test mit benutzerdefinierten Parametern."""
        bm = BasilarMembrane(
            cochlea_length_mm=32.0,
            f_min_hz=55.0,
            f_max_hz=77000.0,
            n_positions=50
        )
        
        assert bm.L == 32.0
        assert bm.f_min == 55.0
        assert bm.f_max == 77000.0
        assert bm.n_positions == 50


class TestGreenwood:
    """Tests für Greenwood-Frequenzkarte (Eq. 10.4-10.5)."""
    
    def test_greenwood_mapping_boundaries(self):
        """Test Grenzen der Greenwood-Abbildung."""
        bm = BasilarMembrane(f_min_hz=20.0, f_max_hz=20000.0)
        
        # Bei x=0 (basal): sollte f_max sein
        freqs = bm._greenwood_map(np.array([0.0]))
        assert freqs[0] > 15000.0  # Nah bei f_max
        
        # Bei x=L (apikal): sollte f_min sein
        freqs = bm._greenwood_map(np.array([bm.L]))
        assert freqs[0] < 100.0  # Nah bei f_min
    
    def test_greenwood_monotonicity(self):
        """Test dass Frequenzen monoton abnehmen mit Position."""
        bm = BasilarMembrane()
        
        x = np.linspace(0, bm.L, 50)
        freqs = bm._greenwood_map(x)
        
        # Frequenz sollte monoton abnehmen
        for i in range(len(freqs) - 1):
            assert freqs[i] > freqs[i + 1]
    
    def test_inverse_greenwood(self):
        """Test dass inverse Greenwood-Abbildung konsistent ist."""
        bm = BasilarMembrane()
        
        # Teste für verschiedene Frequenzen
        test_freqs = [100.0, 1000.0, 5000.0, 10000.0]
        
        for f_test in test_freqs:
            # Berechne Position
            x = bm.L * np.log(bm.f_max / f_test) / np.log(
                bm.f_max / bm.f_min
            )
            
            # Rekonstruiere Frequenz
            f_reconstructed = bm._greenwood_map(np.array([x]))[0]
            
            assert f_reconstructed == pytest.approx(f_test, rel=0.01)


class TestBasilarMembraneMechanics:
    """Tests für mechanische Eigenschaften (Eq. 10.1-10.3)."""
    
    def test_mass_per_length_positive(self):
        """Test dass Massenbelegung positiv ist."""
        bm = BasilarMembrane()
        
        for x in np.linspace(0, bm.L, 10):
            m = bm.mass_per_length(x)
            assert m > 0
    
    def test_mass_increases_apical(self):
        """Test dass Masse von basal zu apikal zunimmt."""
        bm = BasilarMembrane()
        
        m_basal = bm.mass_per_length(0.0)
        m_apical = bm.mass_per_length(bm.L)
        
        assert m_apical > m_basal
    
    def test_stiffness_decreases_apical(self):
        """Test dass Steifigkeit von basal zu apikal abnimmt."""
        bm = BasilarMembrane()
        
        k_basal = bm.stiffness_per_length(0.0)
        k_apical = bm.stiffness_per_length(bm.L)
        
        assert k_basal > k_apical
    
    def test_quality_factor_increases_apical(self):
        """Test dass Q-Faktor von basal zu apikal zunimmt."""
        bm = BasilarMembrane()
        
        q_basal = bm.quality_factor(0.0)
        q_apical = bm.quality_factor(bm.L)
        
        assert q_apical > q_basal
    
    def test_mechanical_impedance_complex(self):
        """Test dass mechanische Impedanz komplex ist."""
        bm = BasilarMembrane()
        
        z = bm.mechanical_impedance(bm.L / 2, 1000.0)
        
        # Sollte komplex sein
        assert isinstance(z, complex)
        assert np.abs(z) > 0


class TestGammaToneFilter:
    """Tests für Gammatone-Filter (Eq. 10.7-10.8)."""
    
    def test_gammatone_initialization(self):
        """Test Gammatone-Filter-Initialisierung."""
        filt = GammaToneFilter(
            center_freq_hz=1000.0,
            bandwidth_hz=100.0,
            sample_rate_hz=16000.0
        )
        
        assert filt.f_c == 1000.0
        assert filt.bandwidth == 100.0
        assert filt.fs == 16000.0
    
    def test_filter_signal_shape(self):
        """Test dass Filter-Output die richtige Form hat."""
        filt = GammaToneFilter(1000.0, 100.0)
        
        # Erstelle Testsignal
        x = np.random.randn(1000)
        
        # Filtere
        y = filt.filter_signal(x)
        
        assert y.shape == x.shape
        assert not np.all(np.isnan(y))
        assert not np.all(np.isinf(y))
    
    def test_impulse_response(self):
        """Test Impulsantwort."""
        filt = GammaToneFilter(1000.0, 100.0, sample_rate_hz=16000.0)
        
        t, h = filt.impulse_response(duration_ms=50.0)
        
        assert len(t) == len(h)
        assert t[0] >= 0
        assert t[-1] <= 0.05  # 50 ms
    
    def test_frequency_response(self):
        """Test Frequenzantwort."""
        filt = GammaToneFilter(1000.0, 100.0)
        
        freqs = np.array([100.0, 500.0, 1000.0, 2000.0])
        f_out, mag, phase = filt.frequency_response(freqs)
        
        assert len(f_out) == len(freqs)
        assert len(mag) == len(freqs)
        assert len(phase) == len(freqs)
        
        # Maximum sollte bei oder nahe der Centerfrequenz sein
        max_idx = np.argmax(mag)
        assert freqs[max_idx] >= 500.0


class TestCochlearFilterBank:
    """Tests für Cochlear FilterBank (Eq. 10.10)."""
    
    def test_filterbank_initialization(self):
        """Test FilterBank-Initialisierung."""
        fb = CochlearFilterBank(
            f_min_hz=20.0,
            f_max_hz=20000.0,
            n_channels=128,
            sample_rate_hz=96000.0
        )
        
        assert fb.f_min == 20.0
        assert fb.f_max == 20000.0
        assert fb.n_channels == 128
        assert len(fb.center_freqs) == 128
        assert len(fb.filters) == 128
    
    def test_logarithmic_frequency_spacing(self):
        """Test dass Frequenzen logarithmisch verteilt sind."""
        fb = CochlearFilterBank(n_channels=50)
        
        # Verhältnisse zwischen aufeinanderfolgenden Kanälen sollten gleich sein
        ratios = fb.center_freqs[1:] / fb.center_freqs[:-1]
        
        # Alle Verhältnisse sollten ungefähr gleich sein
        mean_ratio = np.mean(ratios)
        for ratio in ratios:
            assert ratio == pytest.approx(mean_ratio, rel=0.01)
    
    def test_apply_filterbank_shape(self):
        """Test Shape der FilterBank-Ausgabe."""
        fb = CochlearFilterBank(n_channels=64)
        
        # Erstelle Testsignal
        x = np.random.randn(1000)
        
        # Wende FilterBank an
        y = fb.apply_filterbank(x)
        
        assert y.shape == (64, 1000)
    
    def test_tonotopic_position_mapping(self):
        """Test tonotopische Position-Abbildung."""
        fb = CochlearFilterBank()
        
        freq_test = 1000.0
        x_pos = fb.tonotopic_position(freq_test)
        
        # Position sollte im Bereich der Cochlea-Länge sein
        assert 0 <= x_pos <= fb.L
    
    def test_frequency_for_position(self):
        """Test Frequenz-Berechnung aus Position."""
        fb = CochlearFilterBank()
        
        x_pos = fb.L / 2
        freq = fb.frequency_for_position(x_pos)
        
        # Frequenz sollte im erwarteten Bereich sein
        assert fb.f_min <= freq <= fb.f_max
    
    def test_get_channel_for_frequency(self):
        """Test Channel-Index für Frequenz."""
        fb = CochlearFilterBank()
        
        # Teste mehrere Frequenzen
        for freq in [100.0, 1000.0, 10000.0]:
            idx = fb.get_channel_for_frequency(freq)
            
            assert 0 <= idx < fb.n_channels
            # Center-Frequenz des Kanals sollte nah bei test-freq sein
            assert np.abs(fb.center_freqs[idx] - freq) < 1000.0


class TestCochlearModel:
    """Tests für integriertes CochlearModel."""
    
    def test_ohc_amplification_range(self):
        """Test OHC-Verstärkungsbereich (Eq. 10.13)."""
        model = CochlearModel()
        
        # Teste verschiedene Auslenkungen
        displacements = np.linspace(-0.2, 0.3, 10)
        gains = [model.ohc_amplification(d) for d in displacements]
        
        # Gains sollten im erwarteten Bereich liegen
        assert all(0.1 <= g <= 3.0 for g in gains)
    
    def test_ihc_response_sigmoidal(self):
        """Test sigmoidale IHC-Antwort (Eq. 10.14)."""
        model = CochlearModel()
        
        # Bei sehr negativen Auslenkungen: nahe 0
        response_neg = model.ihc_response(-0.2)
        assert 0 <= response_neg < 0.1
        
        # Bei sehr positiven Auslenkungen: nahe 1
        response_pos = model.ihc_response(0.2)
        assert 0.9 < response_pos <= 1.0
    
    def test_afferent_firing_rate(self):
        """Test afferente Feuerrate."""
        model = CochlearModel()
        
        # IHC-Antwort 0 -> Feuerrate 0
        rate_0 = model.afferent_firing_rate(0.0)
        assert rate_0 == 0.0
        
        # IHC-Antwort 1 -> Feuerrate max
        rate_1 = model.afferent_firing_rate(1.0)
        assert rate_1 > 0.0


# Integrationstests
class TestCochlearIntegration:
    """Integrationstests für Cochlear-Module."""
    
    def test_basilar_membrane_to_filterbank(self):
        """Test dass BasilarMembrane und FilterBank konsistent sind."""
        bm = BasilarMembrane()
        fb = CochlearFilterBank(
            n_channels=len(bm.positions),
            cochlea_length_mm=bm.L
        )
        
        # Frequenzen sollten in ähnlicher Reihenfolge sein
        assert bm.resonance_frequencies[0] > bm.resonance_frequencies[-1]
        assert fb.center_freqs[0] > fb.center_freqs[-1]
    
    def test_gammatone_critical_bandwidth_consistency(self):
        """Test Konsistenz zwischen Gammatone und kritischen Bandbreiten."""
        model = CochlearModel()
        filt = GammaToneFilter(1000.0, 100.0)
        
        erb = model.critical_band_erb(1000.0)
        
        # Filter-Bandbreite sollte in ähnlicher Größenordnung sein
        assert erb > 0
        assert filt.bandwidth > 0
