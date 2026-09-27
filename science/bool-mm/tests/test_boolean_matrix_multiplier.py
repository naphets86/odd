"""
Tests für Boolean Matrix Multiplikation.

Umfasst Unit-Tests, Integration-Tests und Benchmarks für maximale Code Coverage.
"""

import pytest
import numpy as np
from src.boolean_matrix_multiplier import BooleanMatrixMultiplier


class TestSignatureComputation:
    """Tests für Signatur-Berechnung."""
    
    def test_compute_row_signature_simple(self):
        """Test: Einfache Zeilen-Signatur."""
        multiplier = BooleanMatrixMultiplier()
        row = np.array([1, 0, 1, 1])
        signature = multiplier.compute_row_signature(row)
        # 1*2^0 + 0*2^1 + 1*2^2 + 1*2^3 = 1 + 4 + 8 = 13
        assert signature == 13
    
    def test_compute_row_signature_all_zeros(self):
        """Test: Zeile mit nur Nullen."""
        multiplier = BooleanMatrixMultiplier()
        row = np.array([0, 0, 0, 0])
        signature = multiplier.compute_row_signature(row)
        assert signature == 0
    
    def test_compute_row_signature_all_ones(self):
        """Test: Zeile mit nur Einsen."""
        multiplier = BooleanMatrixMultiplier()
        row = np.array([1, 1, 1])
        signature = multiplier.compute_row_signature(row)
        # 1*2^0 + 1*2^1 + 1*2^2 = 1 + 2 + 4 = 7
        assert signature == 7
    
    def test_compute_column_signature_simple(self):
        """Test: Einfache Spalten-Signatur."""
        multiplier = BooleanMatrixMultiplier()
        col = np.array([0, 1, 1])
        signature = multiplier.compute_column_signature(col)
        # 0*2^0 + 1*2^1 + 1*2^2 = 2 + 4 = 6
        assert signature == 6
    
    def test_signature_uniqueness(self):
        """Test: Verschiedene Vektoren haben verschiedene Signaturen."""
        multiplier = BooleanMatrixMultiplier()
        vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]),
            np.array([1, 0, 1]),
        ]
        signatures = [multiplier.compute_row_signature(v) for v in vectors]
        assert len(set(signatures)) == len(signatures)  # Alle unterschiedlich


class TestMatrixValidation:
    """Tests für Matrix-Validierung."""
    
    def test_validate_matrix_invalid_type(self):
        """Test: Fehler bei falschem Typ."""
        multiplier = BooleanMatrixMultiplier()
        with pytest.raises(ValueError, match="muss ein numpy.ndarray sein"):
            multiplier._validate_matrix([[1, 0], [0, 1]], "Test")
    
    def test_validate_matrix_wrong_dimensions(self):
        """Test: Fehler bei falschen Dimensionen."""
        multiplier = BooleanMatrixMultiplier()
        with pytest.raises(ValueError, match="muss 2-dimensional sein"):
            multiplier._validate_matrix(np.array([1, 0, 1]), "Test")
    
    def test_validate_matrix_non_boolean_values(self):
        """Test: Fehler bei Nicht-Boolean-Werten."""
        multiplier = BooleanMatrixMultiplier()
        matrix = np.array([[1, 2], [0, 1]])
        with pytest.raises(ValueError, match="muss nur Werte 0 und 1 enthalten"):
            multiplier._validate_matrix(matrix, "Test")
    
    def test_validate_dimensions_incompatible(self):
        """Test: Fehler bei inkompatiblen Dimensionen."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="Inkompatible Dimensionen"):
            multiplier._validate_dimensions(A, B)
    
    def test_validate_dimensions_compatible(self):
        """Test: Kompatible Dimensionen werden akzeptiert."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1], [0]])
        # Sollte keine Exception werfen
        multiplier._validate_dimensions(A, B)


class TestBooleanOperations:
    """Tests für Boolean-Operationen."""
    
    def test_boolean_and_via_signature(self):
        """Test: Bitweise AND-Operation."""
        multiplier = BooleanMatrixMultiplier()
        # 5 = 0101, 6 = 0110, 5 & 6 = 0100 = 4
        result = multiplier.boolean_and_via_signature(5, 6)
        assert result == 4
    
    def test_boolean_and_zero_result(self):
        """Test: AND ergibt Null."""
        multiplier = BooleanMatrixMultiplier()
        # 5 = 0101, 2 = 0010, 5 & 2 = 0000 = 0
        result = multiplier.boolean_and_via_signature(5, 2)
        assert result == 0
    
    def test_boolean_or_check_true(self):
        """Test: OR-Check gibt True zurück."""
        multiplier = BooleanMatrixMultiplier()
        assert multiplier.boolean_or_check(4) is True
        assert multiplier.boolean_or_check(1) is True
    
    def test_boolean_or_check_false(self):
        """Test: OR-Check gibt False zurück."""
        multiplier = BooleanMatrixMultiplier()
        assert multiplier.boolean_or_check(0) is False


class TestPrecomputeSignatures:
    """Tests für Signatur-Vorberechnung."""
    
    def test_precompute_signatures_simple(self):
        """Test: Einfache Signatur-Vorberechnung."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        
        row_sigs, col_sigs = multiplier.precompute_signatures(A, B)
        
        assert len(row_sigs) == 2
        assert len(col_sigs) == 2
        assert row_sigs[0] == 1  # [1, 0] -> 1*2^0 = 1
        assert row_sigs[1] == 2  # [0, 1] -> 1*2^1 = 2
    
    def test_precompute_signatures_with_cache(self):
        """Test: Cache-Nutzung."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        
        # Erste Berechnung
        row_sigs1, col_sigs1 = multiplier.precompute_signatures(A, B, use_cache=True)
        
        # Zweite Berechnung sollte Cache nutzen
        row_sigs2, col_sigs2 = multiplier.precompute_signatures(A, B, use_cache=True)
        
        assert row_sigs1 == row_sigs2
        assert col_sigs1 == col_sigs2
    
    def test_clear_cache(self):
        """Test: Cache wird geleert."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        
        multiplier.precompute_signatures(A, B, use_cache=True)
        assert len(multiplier._row_signatures_cache) > 0
        
        multiplier.clear_cache()
        assert len(multiplier._row_signatures_cache) == 0
        assert len(multiplier._col_signatures_cache) == 0


class TestMatrixMultiplication:
    """Tests für Matrixmultiplikation."""
    
    def test_multiply_identity_matrices(self):
        """Test: Multiplikation mit Einheitsmatrizen."""
        multiplier = BooleanMatrixMultiplier()
        I = np.array([[1, 0], [0, 1]])
        
        result_optimized = multiplier.multiply_optimized(I, I)
        result_naive = multiplier.multiply_naive(I, I)
        
        np.testing.assert_array_equal(result_optimized, I)
        np.testing.assert_array_equal(result_naive, I)
    
    def test_multiply_zero_matrices(self):
        """Test: Multiplikation mit Nullmatrizen."""
        multiplier = BooleanMatrixMultiplier()
        Z = np.array([[0, 0], [0, 0]])
        
        result = multiplier.multiply_optimized(Z, Z)
        
        np.testing.assert_array_equal(result, Z)
    
    def test_multiply_example_from_paper(self):
        """Test: Beispiel aus der wissenschaftlichen Arbeit."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ])
        B = np.array([
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        
        expected = np.array([
            [1, 1],
            [1, 0],
            [1, 1]
        ])
        
        result_optimized = multiplier.multiply_optimized(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_optimized, expected)
        np.testing.assert_array_equal(result_naive, expected)
    
    def test_multiply_rectangular_matrices(self):
        """Test: Rechteckige Matrizen."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0, 1]])  # 1x3
        B = np.array([[1], [0], [1]])  # 3x1
        
        result = multiplier.multiply_optimized(A, B)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == 1  # 1*1 OR 0*0 OR 1*1 = 1
    
    def test_multiply_larger_matrix(self):
        """Test: Größere Matrix."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        B = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        
        result_optimized = multiplier.multiply_optimized(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_optimized, result_naive)
    
    def test_multiply_with_cache(self):
        """Test: Multiplikation mit Cache."""
        multiplier = BooleanMatrixMultiplier()
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        
        result1 = multiplier.multiply_optimized(A, B, use_cache=True)
        result2 = multiplier.multiply_optimized(A, B, use_cache=True)
        
        np.testing.assert_array_equal(result1, result2)


class TestRandomMatrices:
    """Tests mit zufälligen Matrizen."""
    
    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_random_square_matrices(self, n):
        """Test: Zufällige quadratische Matrizen verschiedener Größen."""
        multiplier = BooleanMatrixMultiplier()
        np.random.seed(42)
        
        A = np.random.randint(0, 2, (n, n))
        B = np.random.randint(0, 2, (n, n))
        
        result_optimized = multiplier.multiply_optimized(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_optimized, result_naive)
    
    @pytest.mark.parametrize("shape", [(3, 5, 4), (5, 3, 6), (10, 8, 7)])
    def test_random_rectangular_matrices(self, shape):
        """Test: Zufällige rechteckige Matrizen."""
        multiplier = BooleanMatrixMultiplier()
        n, k, m = shape
        np.random.seed(42)
        
        A = np.random.randint(0, 2, (n, k))
        B = np.random.randint(0, 2, (k, m))
        
        result_optimized = multiplier.multiply_optimized(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_optimized, result_naive)


class TestEdgeCases:
    """Tests für Grenzfälle."""
    
    def test_single_element_matrices(self):
        """Test: 1x1 Matrizen."""
        multiplier = BooleanMatrixMultiplier()
        
        cases = [
            (np.array([[0]]), np.array([[0]]), np.array([[0]])),
            (np.array([[1]]), np.array([[0]]), np.array([[0]])),
            (np.array([[0]]), np.array([[1]]), np.array([[0]])),
            (np.array([[1]]), np.array([[1]]), np.array([[1]])),
        ]
        
        for A, B, expected in cases:
            result = multiplier.multiply_optimized(A, B)
            np.testing.assert_array_equal(result, expected)
    
    def test_all_zeros(self):
        """Test: Alle Einträge Null."""
        multiplier = BooleanMatrixMultiplier()
        A = np.zeros((5, 5), dtype=int)
        B = np.zeros((5, 5), dtype=int)
        
        result = multiplier.multiply_optimized(A, B)
        
        np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=int))
    
    def test_all_ones(self):
        """Test: Alle Einträge Eins."""
        multiplier = BooleanMatrixMultiplier()
        A = np.ones((3, 3), dtype=int)
        B = np.ones((3, 3), dtype=int)
        
        result = multiplier.multiply_optimized(A, B)
        
        # Bei Boolean: Alle Einträge sollten 1 sein
        np.testing.assert_array_equal(result, np.ones((3, 3), dtype=int))


class TestPerformanceComparison:
    """Performance-Tests (werden nicht für Coverage gezählt, aber für Benchmarking)."""
    
    def test_performance_comparison_small(self):
        """Test: Performance-Vergleich für kleine Matrizen."""
        multiplier = BooleanMatrixMultiplier()
        n = 10
        np.random.seed(42)
        A = np.random.randint(0, 2, (n, n))
        B = np.random.randint(0, 2, (n, n))
        
        # Beide Methoden sollten gleiches Ergebnis liefern
        result_optimized = multiplier.multiply_optimized(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_optimized, result_naive)
