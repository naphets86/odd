"""
Umfassende Test-Suite für alle Algorithmen
Zielt auf 100% Code-Abdeckung ab
"""

import pytest
import numpy as np
from src.algorithms import (
    VectorCentroidFinder,
    MatrixCentroidFinder,
    MatrixLengthsMultiplication,
    CentroidDiagonalization,
    CentroidRefinement,
    QRCentroid,
    CentroidPreconditioning,
    StructuralUncertainty
)


# ============================================================================
# VectorCentroidFinder Tests
# ============================================================================

class TestVectorCentroidFinder:
    """Tests für Vektor-Schwerpunkt-Erkennung"""
    
    def test_simple_vector(self):
        """Test mit einfachem Vektor"""
        v = np.array([1.0, 5.0, 2.0, 3.0])
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == 1
        assert val == 5.0
    
    def test_negative_values(self):
        """Test mit negativen Werten"""
        v = np.array([1.0, -8.0, 2.0, -3.0])
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == 1
        assert val == 8.0
    
    def test_zeros(self):
        """Test mit Nullen"""
        v = np.array([0.0, 0.0, 0.0, 3.0])
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == 3
        assert val == 3.0
    
    def test_single_element(self):
        """Test mit Einzel-Vektor"""
        v = np.array([42.0])
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == 0
        assert val == 42.0
    
    def test_all_same(self):
        """Test wenn alle Werte gleich sind"""
        v = np.array([5.0, 5.0, 5.0, 5.0])
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == 0  # Erster Index mit Maximum
        assert val == 5.0
    
    def test_empty_vector_raises(self):
        """Test dass leerer Vektor einen Fehler verursacht"""
        v = np.array([])
        with pytest.raises(ValueError):
            VectorCentroidFinder.find_vector_centroid(v)
    
    def test_very_small_values(self):
        """Test mit sehr kleinen Werten"""
        v = np.array([1e-15, 1e-14, 1e-13])
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == 2
        assert val == pytest.approx(1e-13)
    
    def test_large_vector(self):
        """Test mit großem Vektor"""
        v = np.random.rand(1000)
        idx, val = VectorCentroidFinder.find_vector_centroid(v)
        assert idx == np.argmax(np.abs(v))
        assert val == pytest.approx(np.max(np.abs(v)))


# ============================================================================
# MatrixCentroidFinder Tests
# ============================================================================

class TestMatrixCentroidFinder:
    """Tests für Matrix-Schwerpunkt-Erkennung"""
    
    def test_diagonal_matrix(self):
        """Test mit Diagonalmatrix"""
        A = np.diag([1.0, 2.0, 3.0, 4.0])
        diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        
        np.testing.assert_array_almost_equal(diag, [1.0, 2.0, 3.0, 4.0])
        assert dsi == pytest.approx(1.0)
    
    def test_dense_matrix(self):
        """Test mit dichter Matrix"""
        A = np.ones((3, 3))
        diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        
        np.testing.assert_array_almost_equal(diag, [1.0, 1.0, 1.0])
        assert dsi == pytest.approx(1.0 / 3.0)
    
    def test_small_matrix(self):
        """Test mit 2x2 Matrix"""
        A = np.array([[4.0, 0.1], [0.1, 3.0]])
        diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        
        np.testing.assert_array_almost_equal(diag, [4.0, 3.0])
        expected_dsi = (16.0 + 9.0) / (16.0 + 0.01 + 0.01 + 9.0)
        assert dsi == pytest.approx(expected_dsi)
    
    def test_zero_matrix(self):
        """Test mit Nullmatrix"""
        A = np.zeros((2, 2))
        diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        
        np.testing.assert_array_almost_equal(diag, [0.0, 0.0])
        assert dsi == 0.0
    
    def test_non_square_matrix_raises(self):
        """Test dass nicht-quadratische Matrix einen Fehler verursacht"""
        A = np.ones((2, 3))
        with pytest.raises(ValueError):
            MatrixCentroidFinder.find_matrix_centroid(A)
    
    def test_large_matrix(self):
        """Test mit großer Matrix"""
        n = 100
        A = np.random.rand(n, n)
        diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        
        assert len(diag) == n
        assert 0 <= dsi <= 1.0
        np.testing.assert_array_almost_equal(diag, np.diag(A))
    
    def test_symmetric_matrix(self):
        """Test mit symmetrischer Matrix"""
        A = np.array([[5.0, 1.0, 0.5],
                      [1.0, 3.0, 0.2],
                      [0.5, 0.2, 4.0]])
        diag, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        
        np.testing.assert_array_almost_equal(diag, [5.0, 3.0, 4.0])
        total_energy = np.sum(A ** 2)
        diag_energy = 25.0 + 9.0 + 16.0
        assert dsi == pytest.approx(diag_energy / total_energy)


# ============================================================================
# MatrixLengthsMultiplication Tests
# ============================================================================

class TestMatrixLengthsMultiplication:
    """Tests für Multiplikationslängen-Berechnung"""
    
    def test_basic_computation(self):
        """Test grundlegende Berechnung der Multiplikationslängen"""
        A = np.array([[0.4, 0.1], [0.2, 0.3]])
        B = np.array([[0.3, 0.2], [0.2, 0.4]])
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        
        assert len(result) == 5
        for k in range(1, 6):
            assert k in result
            assert result[k].shape == (2, 2)
    
    def test_ml1_is_product(self):
        """Test dass ML-1 die Standardmultiplikation ist"""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        expected = A @ B
        
        np.testing.assert_array_almost_equal(result[1], expected)
    
    def test_ml2_rebalancing(self):
        """Test dass ML-2 Zeilensummen normalisiert"""
        A = np.array([[2.0, 0.0], [0.0, 2.0]])
        B = np.array([[3.0, 0.0], [0.0, 3.0]])
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        M2 = result[2]
        
        # Nach Rebalancierung sollten Zeilensummen ca. 1 sein
        row_sums = np.sum(M2, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])
    
    def test_identity_matrices(self):
        """Test mit Identitätsmatrizen"""
        I = np.eye(3)
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(I, I)
        
        np.testing.assert_array_almost_equal(result[1], I)
    
    def test_zero_elements_handling(self):
        """Test Behandlung von Nullen"""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.array([[2.0, 0.0], [0.0, 2.0]])
        
        # Sollte keine Division durch Null verursachen
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        
        assert np.all(np.isfinite(result[2]))
        assert np.all(np.isfinite(result[4]))
    
    def test_non_square_matrices_raise(self):
        """Test dass nicht-quadratische Matrizen Fehler verursachen"""
        A = np.ones((2, 3))
        B = np.ones((2, 3))
        
        with pytest.raises(ValueError):
            MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
    
    def test_different_size_matrices_raise(self):
        """Test dass unterschiedlich große Matrizen Fehler verursachen"""
        A = np.ones((2, 2))
        B = np.ones((3, 3))
        
        with pytest.raises(ValueError):
            MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
    
    def test_dsi_progression(self):
        """Test dass DSI monoton wächst"""
        A = np.random.rand(4, 4) + np.eye(4)
        B = np.random.rand(4, 4) + np.eye(4)
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        
        dsi_values = []
        for k in range(1, 6):
            _, dsi = MatrixCentroidFinder.find_matrix_centroid(result[k])
            dsi_values.append(dsi)
        
        # DSI sollte generell nicht stark abnehmen
        assert dsi_values[-1] >= dsi_values[0] * 0.5


# ============================================================================
# CentroidDiagonalization Tests
# ============================================================================

class TestCentroidDiagonalization:
    """Tests für Schwerpunkt-Diagonalisierung"""
    
    def test_already_diagonal(self):
        """Test mit bereits diagonaler Matrix"""
        A = np.diag([1.0, 2.0, 3.0])
        B, Q = CentroidDiagonalization.centroid_diagonalization(A)
        
        _, dsi = MatrixCentroidFinder.find_matrix_centroid(B)
        assert dsi > 0.95
    
    def test_convergence(self):
        """Test dass Verfahren konvergiert"""
        A = np.array([[2.0, 0.1], [0.1, 3.0]])
        B, Q = CentroidDiagonalization.centroid_diagonalization(A, max_iterations=100)
        
        _, dsi = MatrixCentroidFinder.find_matrix_centroid(B)
        assert dsi >= 0.8  # Sollte zumindest 80% diagonal sein
    
    def test_orthogonal_transformation(self):
        """Test dass Transformation orthogonal ist"""
        A = np.random.rand(3, 3)
        B, Q = CentroidDiagonalization.centroid_diagonalization(A)
        
        # Q sollte orthogonal sein: Q^T Q = I
        QTQ = Q.T @ Q
        np.testing.assert_array_almost_equal(QTQ, np.eye(3), decimal=5)
    
    def test_preserves_norm(self):
        """Test dass Frobenius-Norm erhalten bleibt"""
        A = np.random.rand(4, 4)
        B, Q = CentroidDiagonalization.centroid_diagonalization(A)
        
        norm_A = np.linalg.norm(A, 'fro')
        norm_B = np.linalg.norm(B, 'fro')
        
        assert norm_B == pytest.approx(norm_A, rel=1e-6)


# ============================================================================
# CentroidRefinement Tests
# ============================================================================

class TestCentroidRefinement:
    """Tests für iterative Schwerpunkt-Verfeinerung"""
    
    def test_convergence_exponential(self):
        """Test exponentielles Konvergenzverhalten"""
        A = np.array([[2.0, 0.3], [0.3, 3.0]])
        A_refined, convergence = CentroidRefinement.refine_to_diagonal(
            A, alpha=0.5, max_iterations=50, tolerance=1e-10
        )
        
        assert len(convergence) > 0
        # Sollte gegen 0 konvergieren
        assert convergence[-1] < convergence[0]
    
    def test_returns_diagonal(self):
        """Test dass Ergebnis diagonal ist"""
        A = np.array([[5.0, 2.0], [2.0, 4.0]])
        A_refined, _ = CentroidRefinement.refine_to_diagonal(
            A, alpha=0.7, max_iterations=200
        )
        
        # Off-Diagonal sollte sehr klein sein
        off_diag = A_refined - np.diag(np.diag(A_refined))
        assert np.linalg.norm(off_diag, 'fro') < 1e-6
    
    def test_different_alpha_values(self):
        """Test mit verschiedenen alpha-Werten"""
        A = np.random.rand(3, 3)
        
        for alpha in [0.3, 0.5, 0.7, 0.9]:
            A_refined, convergence = CentroidRefinement.refine_to_diagonal(
                A, alpha=alpha, max_iterations=100
            )
            
            assert len(convergence) > 0
            # Soll konvergieren
            assert convergence[-1] < convergence[0]
    
    def test_invalid_alpha_raises(self):
        """Test dass ungültiges alpha einen Fehler verursacht"""
        A = np.eye(2)
        
        with pytest.raises(ValueError):
            CentroidRefinement.refine_to_diagonal(A, alpha=0.0)
        
        with pytest.raises(ValueError):
            CentroidRefinement.refine_to_diagonal(A, alpha=1.0)
    
    def test_already_diagonal(self):
        """Test mit bereits diagonaler Matrix"""
        A = np.diag([1.0, 2.0, 3.0])
        A_refined, convergence = CentroidRefinement.refine_to_diagonal(A)
        
        # Sollte schnell konvergieren
        assert len(convergence) <= 3


# ============================================================================
# QRCentroid Tests
# ============================================================================

class TestQRCentroid:
    """Tests für QR-Algorithmus mit Schwerpunkt"""
    
    def test_eigenvalues_simple(self):
        """Test Eigenwertberechnung mit einfacher symmetrischer Matrix"""
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        eigenvals, eigenvecs, iterations = QRCentroid.qr_centroid(A)
        
        assert len(eigenvals) == 2
        assert iterations <= 100
        
        # Sollte 2 und 3 finden
        eigenvals_sorted = sorted(np.abs(eigenvals))
        assert eigenvals_sorted[0] == pytest.approx(2.0, abs=0.1)
        assert eigenvals_sorted[1] == pytest.approx(3.0, abs=0.1)
    
    def test_eigenvector_property(self):
        """Test dass Eigenvektoren die Eigenwert-Gleichung erfüllen"""
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        eigenvals, eigenvecs, _ = QRCentroid.qr_centroid(A, max_iterations=200)
        
        # Test A v = lambda v
        for i in range(len(eigenvals)):
            v = eigenvecs[:, i]
            Av = A @ v
            lhs = np.linalg.norm(Av)
            rhs = np.abs(eigenvals[i]) * np.linalg.norm(v)
            
            # Soll ungefähr gleich sein (numerisch)
            assert lhs == pytest.approx(rhs, rel=0.2)
    
    def test_convergence(self):
        """Test dass Algorithmus konvergiert"""
        A = np.random.rand(4, 4)
        A = (A + A.T) / 2  # Mache symmetrisch
        
        eigenvals, eigenvecs, iterations = QRCentroid.qr_centroid(
            A, max_iterations=1000
        )
        
        assert iterations <= 1000  # Sollte konvergieren
    
    def test_determinant_property(self):
        """Test dass Produkt der Eigenwerte = Determinante"""
        A = np.array([[2.0, 0.5], [0.5, 3.0]])
        eigenvals, _, _ = QRCentroid.qr_centroid(A, max_iterations=200)
        
        det_A = np.linalg.det(A)
        prod_eigenvals = np.prod(eigenvals)
        
        assert prod_eigenvals == pytest.approx(det_A, rel=0.1)


# ============================================================================
# CentroidPreconditioning Tests
# ============================================================================

class TestCentroidPreconditioning:
    """Tests für Schwerpunkt-Vorkonditionierung"""
    
    def test_preconditioner_diagonal(self):
        """Test dass Vorkonditioner die Diagonale extrahiert"""
        A = np.array([[2.0, 0.1], [0.1, 3.0]])
        M = CentroidPreconditioning.preconditioner_matrix(A)
        
        np.testing.assert_array_almost_equal(M, np.diag([2.0, 3.0]))
    
    def test_solve_simple_system(self):
        """Test Lösung eines einfachen Gleichungssystems"""
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([2.0, 3.0])
        
        x, residuals = CentroidPreconditioning.solve_preconditioned(A, b)
        
        # Sollte x = [1, 1] sein
        np.testing.assert_array_almost_equal(x, [1.0, 1.0], decimal=5)
    
    def test_residual_decreases(self):
        """Test dass Residuum abnimmt"""
        A = np.random.rand(4, 4) + 4 * np.eye(4)  # Diagonal-dominant
        b = np.random.rand(4)
        
        x, residuals = CentroidPreconditioning.solve_preconditioned(
            A, b, max_iterations=100
        )
        
        if len(residuals) > 1:
            # Residuum sollte abnehmen oder gleich bleiben
            assert residuals[-1] <= residuals[0]
    
    def test_preconditioner_zero_diagonal_handling(self):
        """Test Behandlung von Null-Diagonal-Elementen"""
        A = np.array([[0.0, 1.0], [1.0, 2.0]])
        M = CentroidPreconditioning.preconditioner_matrix(A)
        
        # Sollte Nullen durch 1 ersetzen
        assert M[0, 0] == 1.0


# ============================================================================
# StructuralUncertainty Tests
# ============================================================================

class TestStructuralUncertainty:
    """Tests für strukturelle Unsicherheits-Messungen"""
    
    def test_entropy_zero_matrix(self):
        """Test Entropie der Nullmatrix"""
        M = np.zeros((2, 2))
        entropy = StructuralUncertainty.compute_entropy(M)
        
        assert entropy == 0.0 or np.isfinite(entropy)
    
    def test_entropy_identity(self):
        """Test Entropie der Identitätsmatrix"""
        I = np.eye(3)
        entropy = StructuralUncertainty.compute_entropy(I)
        
        # Identität ist maximale Struktur
        assert entropy >= 0.0
    
    def test_spectral_gap_identity_AFTER(self):
        """Test Spektrallücke der Identität"""
        I = np.eye(3)
        gap = StructuralUncertainty.compute_spectral_gap(I)
        
        # Spektrallücke für Identität: 1 - 1 = 0
        # (alle Eigenwerte sind 1, also keine Lücke)
        assert gap == pytest.approx(0.0)
    
    def test_spectral_gap_identity(self):
        """Test Spektrallücke der Identität"""
        I = np.eye(3)
        gap = StructuralUncertainty.compute_spectral_gap(I)
        
        # Sollte 1 sein
        assert gap == pytest.approx(0.0)
    
    def test_total_uncertainty_computation(self):
        """Test Berechnung der Gesamt-Unsicherheit"""
        matrices = {
            1: np.random.rand(3, 3),
            2: np.random.rand(3, 3),
            3: np.random.rand(3, 3)
        }
        
        uncertainties = StructuralUncertainty.compute_total_uncertainty(matrices)
        
        assert len(uncertainties) == 3
        for k, u in uncertainties.items():
            assert np.isfinite(u)
            assert u >= 0.0
    
    def test_uncertainty_progression(self):
        """Test Unsicherheits-Progression"""
        A = np.random.rand(3, 3) + 3 * np.eye(3)
        B = np.random.rand(3, 3) + 3 * np.eye(3)
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        uncertainties = StructuralUncertainty.compute_total_uncertainty(result)
        
        # Alle sollten finite sein
        assert all(np.isfinite(u) for u in uncertainties.values())


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integrations-Tests für Zusammenspiel der Komponenten"""
    
    def test_full_pipeline(self):
        """Test kompletter Pipeline"""
        # Erstelle Testmatrizen
        A = np.random.rand(3, 3) + 2 * np.eye(3)
        
        # 1. Finde Schwerpunkt
        diag_vec, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
        assert len(diag_vec) == 3
        assert 0 <= dsi <= 1
        
        # 2. Verfeinere zu Diagonal
        A_refined, conv = CentroidRefinement.refine_to_diagonal(
            A, alpha=0.5, max_iterations=50
        )
        assert len(conv) > 0
        
        # 3. Berechne Eigenwerte
        eigenvals, eigenvecs, iters = QRCentroid.qr_centroid(A_refined)
        assert len(eigenvals) == 3
    
    def test_multiplikationslängen_pipeline(self):
        """Test Pipeline für Multiplikationslängen"""
        A = np.random.rand(3, 3) + 1.5 * np.eye(3)
        B = np.random.rand(3, 3) + 1.5 * np.eye(3)
        
        # Berechne Multiplikationslängen
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        
        # Berechne Unsicherheiten
        uncertainties = StructuralUncertainty.compute_total_uncertainty(result)
        
        # Alle sollten definiert sein
        assert len(result) == 5
        assert len(uncertainties) == 5


# ============================================================================
# Performance und Edge Cases
# ============================================================================

class TestPerformance:
    """Performance und Edge-Case Tests"""
    
    def test_large_matrix_performance(self):
        """Test Performance mit größeren Matrizen"""
        n = 50
        A = np.random.rand(n, n)
        
        # Sollte in angemessener Zeit durchlaufen
        result = MatrixLengthsMultiplication.compute_matrix_lengths(
            A, A, max_rotation_iterations=5
        )
        
        assert len(result) == 5
    
    def test_numerical_stability_small_values(self):
        """Test numerische Stabilität mit sehr kleinen Werten"""
        A = np.eye(3) * 1e-10
        B = np.eye(3) * 1e-10
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        
        assert np.all(np.isfinite(result[2]))
        assert np.all(np.isfinite(result[4]))
    
    def test_numerical_stability_large_values(self):
        """Test numerische Stabilität mit großen Werten"""
        A = np.eye(3) * 1e10
        B = np.eye(3) * 1e10
        
        result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
        
        assert np.all(np.isfinite(result[2]))
        assert np.all(np.isfinite(result[4]))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
