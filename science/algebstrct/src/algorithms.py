"""
Algorithmen aus "Strukturen in der Algebra"
Schwerpunkterkennung und effiziente Lösungsmethoden
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import warnings


class VectorCentroidFinder:
    """Findet den Schwerpunkt eines Vektors (Magnituden-Schwerpunkt)"""
    
    @staticmethod
    def find_vector_centroid(v: np.ndarray) -> Tuple[int, float]:
        """
        Findet den Vektor-Schwerpunkt durch maximale Magnitude.
        
        Args:
            v: Vektor in R^n
            
        Returns:
            Tuple: (max_index, max_value) - Index und Wert des Elements mit maximaler Magnitude
            
        Komplexität: O(n) Zeit, O(1) Speicher
        """
        if len(v) == 0:
            raise ValueError("Vektor darf nicht leer sein")
        
        max_value = 0.0
        max_index = 0
        
        for i in range(len(v)):
            if abs(v[i]) > max_value:
                max_value = abs(v[i])
                max_index = i
        
        return max_index, max_value


class MatrixCentroidFinder:
    """Findet den Schwerpunkt einer Matrix (Diagonale und DSI)"""
    
    @staticmethod
    def find_matrix_centroid(A: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Findet den Matrix-Schwerpunkt und berechnet den Diagonalisierungs-Schwerpunkt-Index.
        
        Args:
            A: Quadratische Matrix in R^(n x n)
            
        Returns:
            Tuple: (diag_vector, dsi) 
                - diag_vector: Diagonalelemente
                - dsi: Diagonalisierungs-Schwerpunkt-Index in [0, 1]
                
        Komplexität: O(n^2) Zeit, O(n) Speicher
        """
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError("Matrix muss quadratisch sein")
        
        diag_energy = 0.0
        total_energy = 0.0
        diag_vector = np.zeros(n)
        
        # Diagonale auslesen
        for i in range(n):
            diag_vector[i] = A[i, i]
            diag_energy += A[i, i] ** 2
        
        # Gesamtenergie berechnen (Frobenius-Norm^2)
        for i in range(n):
            for j in range(n):
                total_energy += A[i, j] ** 2
        
        # DSI berechnen
        if total_energy == 0:
            dsi = 0.0
        else:
            dsi = diag_energy / total_energy
        
        return diag_vector, dsi


class MatrixLengthsMultiplication:
    """Berechnet die fünf Multiplikationslängen nach der MLM-Methode"""
    
    @staticmethod
    def compute_matrix_lengths(A: np.ndarray, B: np.ndarray, 
                              max_rotation_iterations: int = 10) -> Dict[int, np.ndarray]:
        """
        Berechnet alle fünf Multiplikationslängen M_1 bis M_5.
        
        Args:
            A, B: Grundmatrizen in R^(n x n), typisch nicht-negativ
            max_rotation_iterations: Max. Iterationen für Optimal-Rotation
            
        Returns:
            Dict mit Keys 1-5, Werte sind die Multiplikationslängen M_k
            
        Komplexität: O(n^3) für Matrix-Multiplikation, O(n^4) für Rotation
        """
        if A.shape != B.shape or A.shape[0] != A.shape[1]:
            raise ValueError("A und B müssen quadratisch und gleich groß sein")
        
        n = A.shape[0]
        
        # ML-1: Direkte Multiplikation
        M1 = A @ B
        
        # ML-2: Schwerpunkt-Rebalancierung (Zeilen-Normalisierung)
        # FIXED: Normalize rows only to sum to 1, then columns to sum to 1
        row_sums = np.sum(M1, axis=1)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        M2_temp = M1 / row_sums[:, np.newaxis]
        
        col_sums = np.sum(M2_temp, axis=0)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        M2 = M2_temp / col_sums[np.newaxis, :]
        
        # ML-3: Strukturelle Rotation
        R = MatrixLengthsMultiplication._optimal_rotation(M2, max_rotation_iterations)
        M3 = R.T @ M2 @ R
        
        # ML-4: Zweite Rebalancierung
        row_sums_3 = np.sum(M3, axis=1)
        col_sums_3 = np.sum(M3, axis=0)
        
        row_sums_3 = np.where(row_sums_3 == 0, 1.0, row_sums_3)
        col_sums_3 = np.where(col_sums_3 == 0, 1.0, col_sums_3)
        
        M4_temp = M3 / row_sums_3[:, np.newaxis]
        col_sums_4 = np.sum(M4_temp, axis=0)
        col_sums_4 = np.where(col_sums_4 == 0, 1.0, col_sums_4)
        M4 = M4_temp / col_sums_4[np.newaxis, :]
        
        # ML-5: Indirekte Effekte (Selbstmultiplikation)
        M5 = M4 @ M4
        
        return {1: M1, 2: M2, 3: M3, 4: M4, 5: M5}
    
    @staticmethod
    def _optimal_rotation(M: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Findet optimale Rotationsmatrix zur Blockstruktur-Minimierung.
        Nutzt Givens-Rotationen.
        
        Args:
            M: Matrix zur Rotation
            max_iterations: Maximale Iterationen
            
        Returns:
            Orthogonale Rotationsmatrix R
        """
        n = M.shape[0]
        R = np.eye(n)
        
        for iteration in range(max_iterations):
            cost_old = MatrixLengthsMultiplication._compute_off_diagonal_norm(R.T @ M @ R)
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Finde optimalen Winkel für Givens-Rotation
                    theta_opt = MatrixLengthsMultiplication._optimal_givens_angle(M, R, i, j)
                    G = MatrixLengthsMultiplication._givens_rotation(n, i, j, theta_opt)
                    R = R @ G
            
            cost_new = MatrixLengthsMultiplication._compute_off_diagonal_norm(R.T @ M @ R)
            
            if abs(cost_old - cost_new) < 1e-8:
                break
        
        return R
    
    @staticmethod
    def _compute_off_diagonal_norm(M: np.ndarray) -> float:
        """Berechnet Frobenius-Norm der Off-Diagonalelemente."""
        M_copy = M.copy()
        np.fill_diagonal(M_copy, 0)
        return np.linalg.norm(M_copy, 'fro')
    
    @staticmethod
    def _optimal_givens_angle(M: np.ndarray, R: np.ndarray, i: int, j: int) -> float:
        """Findet optimalen Winkel für Givens-Rotation zwischen i und j."""
        # Vereinfachte Version: teste mehrere Winkel
        M_rot = R.T @ M @ R
        best_angle = 0.0
        best_cost = MatrixLengthsMultiplication._compute_off_diagonal_norm(M_rot)
        
        for angle in np.linspace(0, 2*np.pi, 16):
            G = MatrixLengthsMultiplication._givens_rotation(M.shape[0], i, j, angle)
            M_test = (R @ G).T @ M @ (R @ G)
            cost = MatrixLengthsMultiplication._compute_off_diagonal_norm(M_test)
            
            if cost < best_cost:
                best_cost = cost
                best_angle = angle
        
        return best_angle
    
    @staticmethod
    def _givens_rotation(n: int, i: int, j: int, theta: float) -> np.ndarray:
        """Erstellt Givens-Rotationsmatrix für Rotation in (i,j)-Ebene."""
        G = np.eye(n)
        c = np.cos(theta)
        s = np.sin(theta)
        
        G[i, i] = c
        G[i, j] = -s
        G[j, i] = s
        G[j, j] = c
        
        return G


class CentroidDiagonalization:
    """Schwerpunkt-offensive Diagonalisierung mit Power-Iteration"""
    
    @staticmethod
    def centroid_diagonalization(A: np.ndarray, max_iterations: int = 50,
                                 dsi_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Versucht, Matrix durch Schwerpunkt-fokussierte Transformationen zu diagonalisieren.
        
        Args:
            A: Eingabematrix
            max_iterations: Maximale Iterationen
            dsi_threshold: DSI-Schwelle für Konvergenz
            
        Returns:
            Tuple: (B_diag, Q) - Diagonalisierte Matrix und Transformationsmatrix
        """
        Q = np.eye(A.shape[0])
        B = A.copy()
        
        for k in range(max_iterations):
            # Finde Schwerpunkt
            diag_vec, dsi = MatrixCentroidFinder.find_matrix_centroid(B)
            
            if dsi > dsi_threshold:
                return B, Q
            
            # SVD durchführen
            try:
                U, S, VT = np.linalg.svd(B)
                
                # Transformieren
                B = U.T @ B @ VT.T
                Q = Q @ VT.T
            except np.linalg.LinAlgError:
                # Fallback wenn SVD scheitert
                return B, Q
        
        return B, Q


class CentroidRefinement:
    """Iterative Schwerpunkt-Verfeinerung gegen Diagonalmatrix"""
    
    @staticmethod
    def refine_to_diagonal(A: np.ndarray, alpha: float = 0.5, 
                          max_iterations: int = 100,
                          tolerance: float = 1e-8) -> Tuple[np.ndarray, List[float]]:
        """
        Verfeinert Matrix iterativ zu ihrer Diagonalen.
        
        Args:
            A: Eingabematrix
            alpha: Mischungsparameter in (0, 1)
            max_iterations: Maximale Iterationen
            tolerance: Konvergenz-Toleranz
            
        Returns:
            Tuple: (A_diag, convergence_history) - Finale Matrix und Fehlerhistorie
            
        Konvergenzgeschwindigkeit: exponentiell mit Rate alpha
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha muss in (0, 1) sein")
        
        A_k = A.copy()
        convergence = []
        
        for k in range(max_iterations):
            D_k = np.diag(np.diag(A_k))  # Extrahiere Diagonale
            A_k_new = alpha * A_k + (1 - alpha) * D_k
            
            # Fehler messen
            error = np.linalg.norm(A_k_new - D_k, 'fro') / np.linalg.norm(D_k, 'fro')
            convergence.append(error)
            
            if error < tolerance:
                return A_k_new, convergence
            
            A_k = A_k_new
        
        return A_k, convergence


class QRCentroid:
    """QR-Algorithmus mit Schwerpunkt-Vorkonditionierung"""
    
    @staticmethod
    def qr_centroid(A: np.ndarray, max_iterations: int = 100,
                   tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        QR-Algorithmus mit Schwerpunkt-Vorkonditionierung für Eigenwertberechnung.
        
        Args:
            A: Eingabematrix
            max_iterations: Maximale Iterationen
            tolerance: Konvergenz-Toleranz (Off-Diagonal-Norm)
            
        Returns:
            Tuple: (eigenvalues, eigenvectors, iterations_used)
            
        FIXED: Changed tolerance from 1e-10 to 1e-8 for better convergence behavior
        """
        n = A.shape[0]
        
        # Schwerpunkt-Matrix (Diagonale)
        diag = np.diag(A)
        M = np.diag(np.where(diag == 0, 1.0, diag))
        
        # Vorkonditionierung
        M_inv = np.diag(1.0 / np.diag(M))
        A_prime = M_inv @ A @ M
        
        Q_total = np.eye(n)
        B = A_prime.copy()
        
        for iteration in range(max_iterations):
            Q, R = np.linalg.qr(B)
            B = R @ Q
            Q_total = Q_total @ Q
            
            # Konvergenz-Check
            B_diag = np.diag(np.diag(B))
            off_diag_norm = np.linalg.norm(B - B_diag, 'fro')
            
            if off_diag_norm < tolerance:
                eigenvalues = np.diag(B)
                eigenvectors = M @ Q_total
                return eigenvalues, eigenvectors, iteration + 1
        
        eigenvalues = np.diag(B)
        eigenvectors = M @ Q_total
        return eigenvalues, eigenvectors, max_iterations


class CentroidPreconditioning:
    """Schwerpunkt-Vorkonditionierung für lineare Gleichungssysteme"""
    
    @staticmethod
    def preconditioner_matrix(A: np.ndarray) -> np.ndarray:
        """
        Erstellt Vorkonditionierungsmatrix aus Diagonale.
        
        Args:
            A: Koeffizientenmatrix
            
        Returns:
            Diagonale Vorkonditionierungsmatrix M
        """
        diag = np.diag(A)
        diag = np.where(np.abs(diag) < 1e-14, 1.0, diag)
        return np.diag(diag)
    
    @staticmethod
    def solve_preconditioned(A: np.ndarray, b: np.ndarray,
                           max_iterations: int = 1000,
                           tolerance: float = 1e-10) -> Tuple[np.ndarray, List[float]]:
        """
        Löst Ax = b mit GMRES-ähnlicher Vorkonditionierung.
        
        Args:
            A: Koeffizientenmatrix
            b: Rechte Seite
            max_iterations: Maximale Iterationen
            tolerance: Konvergenz-Toleranz
            
        Returns:
            Tuple: (solution, residual_history)
        """
        M = CentroidPreconditioning.preconditioner_matrix(A)
        M_inv = np.linalg.inv(M)
        
        # Transformiertes System: M^{-1} A x = M^{-1} b
        A_precond = M_inv @ A
        b_precond = M_inv @ b
        
        # Konjugierte Gradienten (wenn A positiv definit, sonst GMRES)
        try:
            x = np.linalg.solve(A_precond, b_precond)
            residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
            return x, [residual]
        except np.linalg.LinAlgError:
            # Fallback: einfache Iteration
            x = np.zeros_like(b)
            residuals = []
            
            for k in range(max_iterations):
                r = b_precond - A_precond @ x
                x = x + r
                
                residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
                residuals.append(residual)
                
                if residual < tolerance:
                    break
            
            return x, residuals


class StructuralUncertainty:
    """Berechnet strukturelle Unsicherheit in Multiplikationslängen"""
    
    @staticmethod
    def compute_entropy(M: np.ndarray) -> float:
        """
        Berechnet Shannon-Entropie der zeilennormalisierten Matrix.
        
        Args:
            M: Matrix
            
        Returns:
            Entropie-Wert
        """
        M_norm = M.copy()
        
        # Zeilennormalisierung
        row_sums = np.sum(M_norm, axis=1)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        M_norm = M_norm / row_sums[:, np.newaxis]
        
        # Shannon-Entropie
        entropy = 0.0
        epsilon = 1e-15
        
        for i in range(M_norm.shape[0]):
            for j in range(M_norm.shape[1]):
                p = M_norm[i, j]
                if p > epsilon:
                    entropy -= p * np.log(p)
        
        return entropy
    
    @staticmethod
    def compute_spectral_gap(M: np.ndarray) -> float:
        """
        Berechnet Spektrallücke (1 - lambda_2) für Konvergenzanalyse.
        
        Args:
            M: Matrix
            
        Returns:
            Spektrallücke
            
        FIXED: Correctly handle case where all eigenvalues are identical
        """
        if M.shape[0] < 2:
            return 1.0
        
        try:
            eigenvalues = np.linalg.eigvalsh(M)
            eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
            
            if len(eigenvalues) >= 2:
                gap = 1.0 - eigenvalues[1]
                return max(0.0, gap)
            return 1.0
        except:
            return 1.0
    
    @staticmethod
    def compute_total_uncertainty(matrices: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Berechnet Gesamt-Unsicherheitsmaß U_k für alle Multiplikationslängen.
        
        Args:
            matrices: Dict mit Multiplikationslängen
            
        Returns:
            Dict mit Unsicherheitsmaßen für jedes k
        """
        uncertainties = {}
        
        for k, M in matrices.items():
            entropy = StructuralUncertainty.compute_entropy(M)
            gap = StructuralUncertainty.compute_spectral_gap(M)
            
            # Vereinfachtes Unsicherheitsmaß
            uncertainty = entropy + gap
            uncertainties[k] = uncertainty
        
        return uncertainties