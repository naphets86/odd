"""
Tests für Strassen Matrixmultiplikation.
"""

import pytest
import numpy as np
from src.strassen_multiplier import StrassenMultiplier, KBoundedStrassenMultiplier


class TestStrassenMultiplier:
    """Tests für Strassen-Algorithmus."""
    
    def test_next_power_of_2(self):
        """Test: Nächste Zweierpotenz."""
        multiplier = StrassenMultiplier()
        assert multiplier._next_power_of_2(1) == 1
        assert multiplier._next_power_of_2(2) == 2
        assert multiplier._next_power_of_2(3) == 4
        assert multiplier._next_power_of_2(5) == 8
        assert multiplier._next_power_of_2(64) == 64
        assert multiplier._next_power_of_2(65) == 128
    
    def test_pad_matrix(self):
        """Test: Matrix-Padding."""
        multiplier = StrassenMultiplier()
        M = np.array([[1, 2], [3, 4]])
        padded = multiplier._pad_matrix(M, 4)
        
        assert padded.shape == (4, 4)
        np.testing.assert_array_equal(padded[:2, :2], M)
        np.testing.assert_array_equal(padded[2:, :], np.zeros((2, 4)))
    
    def test_unpad_matrix(self):
        """Test: Matrix-Unpadding."""
        multiplier = StrassenMultiplier()
        M = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        unpadded = multiplier._unpad_matrix(M, (2, 2))
        
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(unpadded, expected)
    
    def test_strassen_identity_matrices(self):
        """Test: Strassen mit Einheitsmatrizen."""
        multiplier = StrassenMultiplier(threshold=2)
        I = np.eye(4, dtype=int)
        
        result = multiplier.multiply_strassen(I, I, boolean=False)
        
        np.testing.assert_array_equal(result, I)
    
    def test_strassen_vs_naive_small(self):
        """Test: Strassen vs. naive für kleine Matrix."""
        multiplier = StrassenMultiplier(threshold=2)
        
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        result_strassen = multiplier.multiply_strassen(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_strassen, result_naive)
    
    def test_strassen_boolean(self):
        """Test: Strassen für Boolean-Matrizen."""
        multiplier = StrassenMultiplier(threshold=2)
        
        A = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        B = np.array([[0, 1], [1, 0], [1, 1]])
        
        # Erwarte auf größe 4 gepaddet
        result = multiplier.multiply_strassen(A, B, boolean=True)
        
        # Korrekt für Boolean (nur 0 oder 1)
        assert np.all(np.isin(result, [0, 1]))
    
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_strassen_random_matrices(self, n):
        """Test: Strassen mit zufälligen Matrizen."""
        multiplier = StrassenMultiplier(threshold=4)
        np.random.seed(42)
        
        A = np.random.randint(0, 10, (n, n))
        B = np.random.randint(0, 10, (n, n))
        
        result_strassen = multiplier.multiply_strassen(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_almost_equal(result_strassen, result_naive)
    
    def test_strassen_non_power_of_2(self):
        """Test: Strassen mit Nicht-Zweierpotenz-Größe."""
        multiplier = StrassenMultiplier(threshold=2)
        
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        
        result_strassen = multiplier.multiply_strassen(A, B)
        result_naive = multiplier.multiply_naive(A, B)
        
        np.testing.assert_array_equal(result_strassen, result_naive)


class TestKBoundedStrassenMultiplier:
    """Tests für k-beschränkten Strassen-Algorithmus."""
    
    def test_k_bounded_strassen_simple(self):
        """Test: Einfache k-beschränkte Multiplikation."""
        multiplier = KBoundedStrassenMultiplier(threshold=2)
        
        A = np.array([[1, 2], [0, 1]])
        B = np.array([[1, 0], [1, 2]])
        
        result = multiplier.multiply(A, B)
        expected = np.array([[3, 4], [1, 2]])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_k_bounded_strassen_larger(self):
        """Test: Größere k-beschränkte Matrix."""
        multiplier = KBoundedStrassenMultiplier(threshold=2)
        
        A = np.array([[1, 0, 2], [0, 1, 0], [1, 1, 0]])
        B = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 2]])
        
        result = multiplier.multiply(A, B)
        
        # Vergleich mit numpy
        expected = np.matmul(A, B)
        np.testing.assert_array_equal(result, expected)