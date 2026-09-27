"""
Tests für Audio Converter und Hearing Comparison.

Deckt ab:
- A/D-Wandlung mit Quantisierung
- Sigma-Delta-Modulation
- Hearing-Vergleiche zwischen Spezies
"""

import pytest
import numpy as np
from acoustcs.audio_converter import AudioConverter
from acoustcs.hearing_comparison import HearingComparison
from acoustcs.feline_auditory import Species


class TestAudioConverterInitialization:
    """Tests für AudioConverter-Initialisierung."""
    
    def test_default_initialization(self):
        """Test Standard-Initialisierung."""
        converter = AudioConverter()
        
        assert converter.fs == 96000.0
        assert converter.bit_depth == 24
        assert converter.v_max == 10.0
    
    def test_bit_depth_validation(self):
        """Test Validierung der Bit-Tiefe."""
        # Gültige Werte
        AudioConverter(bit_depth=16)
        AudioConverter(bit_depth=24)
        AudioConverter(bit_depth=32)
        
        # Ungültige Werte
        with pytest.raises(ValueError):
            AudioConverter(bit_depth=12)
        
        with pytest.raises(ValueError):
            AudioConverter(bit_depth=20)


class TestQuantization:
    """Tests für Quantisierung (Eq. 10.17-10.19)."""
    
    def test_quantization_delta(self):
        """Test Quantisierungsstufe."""
        converter = AudioConverter(bit_depth=16, v_max_v=1.0)
        
        # Δ = 2 * V_max / 2^N_b
        expected_delta = 2.0 * 1.0 / (2**16)
        assert converter.delta == pytest.approx(expected_delta)
    
    def test_quantize_operation(self):
        """Test Quantisierungs-Operation."""
        converter = AudioConverter(bit_depth=16, v_max_v=10.0)
        
        # Erstelle Test-Signal
        x = np.array([-5.0, 0.0, 5.0, 9.9])
        
        quantized = converter.quantize(x)
        
        # Quantisierte Werte sollten Vielfache von Δ sein
        assert np.all(np.abs(quantized) <= converter.v_max)
    
    def test_quantization_clipping(self):
        """Test dass Übersteuerung geclippt wird."""
        converter = AudioConverter(v_max_v=1.0)
        
        x = np.array([-5.0, 0.0, 5.0])  # Außerhalb [-1, 1]
        
        quantized = converter.quantize(x)
        
        # Alle geclippt zu [-1, 1]
        assert np.all(quantized >= -1.0)
        assert np.all(quantized <= 1.0)
    
    def test_quantization_error(self):
        """Test Quantisierungsfehler."""
        converter = AudioConverter(bit_depth=16)
        
        x = np.array([0.0, 1.0, -1.0, 0.5])
        quantized = converter.quantize(x)
        
        error = converter.quantization_error(x, quantized)
        
        # Fehler sollte begrenzt sein
        assert np.all(np.abs(error) <= converter.delta)
    
    def test_snr_quantization(self):
        """Test Signal-zu-Quantisierungsrausch-Verhältnis."""
        converter = AudioConverter(bit_depth=24)
        
        snr = converter.quantization_snr()
        
        # Theoretisch: SNR ≈ 6.02 * N_b dB
        expected_snr = 6.02 * 24 + 1.76
        assert snr == pytest.approx(expected_snr, rel=0.01)


class TestSampleAndHold:
    """Tests für Sample-and-Hold."""
    
    def test_sample_and_hold_output_length(self):
        """Test dass S&H die richtige Länge hat."""
        converter = AudioConverter(sample_rate_hz=1000.0)
        
        # Analoges Signal
        t_analog = np.linspace(0, 1.0, 10000)
        x_analog = np.sin(2 * np.pi * 10 * t_analog)
        
        samples, t_digital = converter.sample_and_hold(x_analog, t_analog)
        
        assert len(samples) > 0
        assert len(samples) == len(t_digital)
        assert samples.min() >= x_analog.min()
        assert samples.max() <= x_analog.max()


class TestADConversion:
    """Tests für komplette A/D-Wandlung."""
    
    def test_ad_conversion_output_structure(self):
        """Test Struktur der A/D-Wandlungs-Ausgabe."""
        converter = AudioConverter()
        
        # Test-Signal
        t_analog = np.linspace(0, 0.1, 10000)
        x_analog = np.sin(2 * np.pi * 100 * t_analog)
        
        digital, t_digital, metadata = converter.convert_analog_to_digital(
            x_analog, t_analog
        )
        
        assert len(digital) > 0
        assert len(t_digital) == len(digital)
        assert isinstance(metadata, dict)
        assert 'final' in metadata
    
    def test_ad_conversion_with_sdm(self):
        """Test A/D-Wandlung mit Sigma-Delta-Modulation."""
        converter = AudioConverter(use_sigma_delta=True, oversampling_factor=64)
        
        # Test-Signal
        t_analog = np.linspace(0, 0.01, 1000)
        x_analog = np.sin(2 * np.pi * 1000 * t_analog)
        
        digital, t_digital, metadata = converter.convert_analog_to_digital(
            x_analog, t_analog
        )
        
        assert 'sdm' in metadata
        assert 'oversampling_factor' in metadata['sdm']


class TestMemoryUsage:
    """Tests für Speicherverbrauch-Berechnung."""
    
    def test_memory_usage_calculation(self):
        """Test Speicherverbrauch-Berechnung."""
        converter_16 = AudioConverter(bit_depth=16)
        converter_24 = AudioConverter(bit_depth=24)
        
        mb_16, str_16 = converter_16.memory_usage(100000)
        mb_24, str_24 = converter_24.memory_usage(100000)
        
        # 24-Bit sollte 50% mehr als 16-Bit sein
        assert mb_24 / mb_16 == pytest.approx(1.5, rel=0.01)
    
    def test_data_rate(self):
        """Test Datenrate-Berechnung."""
        converter = AudioConverter(sample_rate_hz=96000.0, bit_depth=24)
        
        data_rate = converter.data_rate_mbps()
        
        # 96000 * 24 / 1e6 = 2.304 Mbps
        assert data_rate == pytest.approx(2.304, rel=0.01)


class TestHearingComparison:
    """Tests für HearingComparison."""
    
    def test_hearing_comparison_initialization(self):
        """Test HearingComparison-Initialisierung."""
        comparison = HearingComparison()
        
        assert len(comparison.systems) == 3
        assert Species.CAT in comparison.systems
        assert Species.DOG in comparison.systems
        assert Species.HUMAN in comparison.systems
    
    def test_frequency_bandwidth_comparison(self):
        """Test Frequenzbandbreiten-Vergleich."""
        comparison = HearingComparison()
        
        results = comparison.frequency_bandwidth_comparison()
        
        # Katze sollte höhere f_max haben
        cat_f_max = results[Species.CAT.value][1]
        dog_f_max = results[Species.DOG.value][1]
        
        assert cat_f_max > dog_f_max
    
    def test_upper_frequency_ratios(self):
        """Test Verhältnisse oberer Frequenzen (Eq. 9.8)."""
        comparison = HearingComparison()
        
        ratios = comparison.upper_frequency_ratios()
        
        cat_ratio = ratios[Species.CAT.value]
        dog_ratio = ratios[Species.DOG.value]
        
        # Katze sollte ~1.71x höher sein als Hund
        assert cat_ratio / dog_ratio == pytest.approx(1.71, rel=0.05)
    
    def test_spectral_selectivity_ratio(self):
        """Test Q-Faktor-Verhältnis (Eq. 9.12)."""
        comparison = HearingComparison()
        
        ratio = comparison.spectral_selectivity_ratio()
        
        # Katze sollte ~1.5x höher sein
        assert ratio == pytest.approx(1.5, rel=0.1)
    
    def test_cochlear_amplification_ratio(self):
        """Test cochleäre Verstärkungsverhältnis (Eq. 9.13)."""
        comparison = HearingComparison()
        
        ratio = comparison.cochlear_amplification_ratio()
        
        # Katze sollte ~3.16x höher sein (10 dB Unterschied)
        assert ratio == pytest.approx(3.16, rel=0.1)
    
    def test_localization_accuracy_comparison(self):
        """Test Lokalisierungsgenauigkeits-Vergleich."""
        comparison = HearingComparison()
        
        results = comparison.localization_accuracy_comparison()
        
        cat_fixed, cat_mobile = results[Species.CAT.value]
        dog_fixed, dog_mobile = results[Species.DOG.value]
        
        # Katze sollte bessere Genauigkeit haben
        assert cat_fixed < dog_fixed
        assert cat_mobile < dog_mobile
    
    def test_superiority_table_structure(self):
        """Test Struktur der Überlegenheits-Tabelle."""
        comparison = HearingComparison()
        
        table = comparison.superiority_table()
        
        # Sollte für jede Spezies einen Eintrag haben
        assert Species.CAT.value in table
        assert Species.DOG.value in table
        assert Species.HUMAN.value in table
        
        # Katze vs. Hund sollte > 1.0 sein
        cat_factors = table[Species.CAT.value]
        assert cat_factors['geometric_mean_mobile'] > 1.0
    
    def test_geometric_mean_superiority(self):
        """Test geometrisches Mittel der Überlegenheit."""
        comparison = HearingComparison()
        
        gm = comparison.geometric_mean_superiority(include_mobile=True)
        
        # Hund = 1.0 (Referenz)
        assert gm[Species.DOG.value] == 1.0
        
        # Katze > 1.0
        assert gm[Species.CAT.value] > 1.0
    
    def test_summary_output(self):
        """Test Summary-Ausgabe."""
        comparison = HearingComparison()
        
        summary = comparison.summary_table()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "AUDITORY SYSTEM COMPARISON" in summary
    
    def test_dataframe_format_output(self):
        """Test DataFrame-Format-Ausgabe."""
        comparison = HearingComparison()
        
        output = comparison.comparison_dataframe_format()
        
        assert isinstance(output, str)
        assert len(output) > 0
        # Sollte Tabellen-ähnlich sein
        assert "|" in output


# Integrationstests
class TestADConversionIntegration:
    """Integrationstests für A/D-Wandlung."""
    
    def test_full_conversion_pipeline(self):
        """Test komplette A/D-Wandlungs-Pipeline."""
        # Erstelle Testsignal
        fs = 48000.0
        t = np.arange(0, 1.0, 1/fs)
        
        # Sinuswelle mit 1 kHz
        x = 5.0 * np.sin(2 * np.pi * 1000 * t)
        
        # Wandle
        converter = AudioConverter(sample_rate_hz=fs, bit_depth=24)
        digital, t_digital, metadata = converter.convert_analog_to_digital(x, t)
        
        # Überprüfungen
        assert len(digital) > 0
        assert np.min(digital) >= -converter.v_max
        assert np.max(digital) <= converter.v_max
        assert 'final' in metadata
    
    def test_snr_with_noise(self):
        """Test SNR mit Rauschen."""
        # Sauberes Signal
        x_clean = np.sin(np.linspace(0, 4*np.pi, 1000))
        
        # Mit Rauschen
        noise = 0.1 * np.random.randn(1000)
        x_noisy = x_clean + noise
        
        converter = AudioConverter()
        
        # Quantisiere beide
        q_clean = converter.quantize(x_clean)
        q_noisy = converter.quantize(x_noisy)
        
        # SNR sollte für sauberes Signal höher sein
        snr_clean = converter.estimate_snr(x_clean, q_clean)
        snr_noisy = converter.estimate_snr(x_noisy, q_noisy)
        
        # Der Vergleich macht Sinn
        assert isinstance(snr_clean, (float, np.floating))
        assert isinstance(snr_noisy, (float, np.floating))


class TestComparisonIntegration:
    """Integrationstests für Hearing Comparison."""
    
    def test_all_comparison_methods(self):
        """Test dass alle Vergleichsmethoden konsistent sind."""
        comparison = HearingComparison()
        
        # Teste mehrere Vergleichsmethoden
        freq_comp = comparison.frequency_bandwidth_comparison()
        q_comp = comparison.spectral_selectivity_comparison()
        loc_comp = comparison.localization_accuracy_comparison()
        
        # Alle sollten Einträge für alle Spezies haben
        for species in [Species.CAT, Species.DOG, Species.HUMAN]:
            species_name = species.value
            assert species_name in freq_comp
            assert species_name in q_comp
            assert species_name in loc_comp
    
    def test_superiority_consistency(self):
        """Test dass Überlegenheits-Faktoren konsistent sind."""
        comparison = HearingComparison()
        
        # Katze superiority_index relativ zu Hund
        cat = comparison.systems[Species.CAT]
        dog = comparison.systems[Species.DOG]
        
        factors = cat.superiority_index(dog)
        
        # Geometrisches Mittel sollte Produkt aller Faktoren widerspiegeln
        gm_check = np.exp(np.mean(np.log([
            factors['upper_frequency'],
            factors['cochlear_scaling'],
            factors['spectral_selectivity'],
            factors['cochlear_amplification'],
            factors['temporal_resolution'],
            factors['localization_mobile']
        ])))
        
        assert gm_check == pytest.approx(
            factors['geometric_mean_mobile'], rel=0.05
        )
