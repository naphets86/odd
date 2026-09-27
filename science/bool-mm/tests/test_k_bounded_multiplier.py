"""
Tests für k-beschränkte Matrixmultiplikation.
"""

import pytest
import numpy as np
from src.k_bounded_multiplier import KBoundedMatrixMultiplier


class TestKBoundedMultiplier:
    """Tests für k-beschränkte Multiplikation mit Signaturen."""
    
    def test_popcount(self):
        """Test: Population Count."""
        multiplier = KBoundedMatrixMultiplier()
        
        assert multiplier._popcount(0) == 0
        assert multiplier._popcount(1) == 1
        assert multiplier._popcount(7) == 3  # 0b111
        assert multiplier._popcount(15) == 4  # 0b1111
    
    def test_compute_layer_signature(self):
        """Test: Schicht-Signatur."""
        multiplier = KBoundedMatrixMultiplier()
        
        vector = np.array([2, 0, 3, 1])
        # Threshold 1: [1, 0, 1, 1] -> 1 + 4 + 8 = 13
        sig1 = multiplier._compute_layer_signature(vector, 1)
        assert sig1 == 13
        
        # Threshold 2: [1, 0, 1, 0] -> 1 + 4 = 5
        sig2 = multiplier._compute_layer_signature(vector, 2)
        assert sig2 == 5
    
    def test_k_bounded_simple(self):
        """Test: Einfache k-beschränkte Multiplikation."""
        multiplier = KBoundedMatrixMultiplier()
        
        A = np.array([[1, 2], [0, 1]])
        B = np.array([[1, 0], [1, 2]])
        
        result = multiplier.multiply_k_bounded(A, B, k=2)
        expected = np.matmul(A, B)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_k_bounded_vs_numpy(self):
        """Test: Vergleich mit NumPy."""
        multiplier = KBoundedMatrixMultiplier()
        
        A = np.array([
            [1, 0, 2],
            [0, 1, 0],
            [1, 1, 0]
        ])
        B = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 2]
        ])
        
        result = multiplier.multiply_k_bounded(A, B, k=2)
        expected = np.matmul(A, B)
        
        np.testing.assert_array_equal(result, expected)
    
    @pytest.mark.parametrize("k", [1, 2, 3, 5])
    def test_k_bounded_random(self, k):
        """Test: Zufällige k-beschränkte Matrizen."""
        multiplier = KBoundedMatrixMultiplier()
        np.random.seed(42)
        
        n = 5
        A = np.random.randint(0, k + 1, (n, n))
        B = np.random.randint(0, k + 1, (n, n))
        
        result = multiplier.multiply_k_bounded(A, B, k)
        expected = np.matmul(A, B)
        
        np.testing.assert_array_equal(result, expected)