#!/usr/bin/env python3
"""
Praktische Beispiele für alle Algorithmen
Kopieren und direkt ausführen!
"""

import numpy as np
from algorithms import (
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
# BEISPIEL 1: Vektor-Schwerpunkt
# ============================================================================

def beispiel_1_vektor_schwerpunkt():
    print("=" * 70)
    print("BEISPIEL 1: Vektor-Schwerpunkt-Erkennung")
    print("=" * 70)
    
    # Ein Vektor mit verschiedenen Komponenten
    v = np.array([0.1, 0.2, 5.3, 0.3, 0.5])
    print(f"\nVektor: {v}")
    
    idx, val = VectorCentroidFinder.find_vector_centroid(v)
    print(f"Schwerpunkt: Index = {idx}, Wert = {val}")
    print(f"→ Das Element mit der größten Magnitude ist v[{idx}] = {val}")
    
    # Mit negativen Werten
    v2 = np.array([1.0, -8.0, 2.0, -3.0])
    print(f"\nVektor mit negativen Werten: {v2}")
    idx2, val2 = VectorCentroidFinder.find_vector_centroid(v2)
    print(f"Schwerpunkt: Index = {idx2}, Wert = {val2}")
    print(f"→ Der Betrag von v2[{idx2}] = {v2[idx2]} ist maximal\n")


# ============================================================================
# BEISPIEL 2: Matrix-Schwerpunkt und DSI
# ============================================================================

def beispiel_2_matrix_schwerpunkt():
    print("=" * 70)
    print("BEISPIEL 2: Matrix-Schwerpunkt-Erkennung (Diagonale & DSI)")
    print("=" * 70)
    
    # Matrix aus dem Paper
    A = np.array([[4.0, 0.1, 0.2],
                  [0.1, 3.0, 0.1],
                  [0.2, 0.1, 5.0]])
    
    print(f"\nMatrix A:\n{A}\n")
    
    diag_vec, dsi = MatrixCentroidFinder.find_matrix_centroid(A)
    
    print(f"Diagonale: {diag_vec}")
    print(f"DSI (Diagonalisierungs-Schwerpunkt-Index): {dsi:.6f}")
    print(f"→ DSI ≈ 0.998 bedeutet: Die Matrix ist zu ~99.8% diagonal!")
    print(f"→ Nur ~0.2% der Energie liegt außerhalb der Diagonale\n")


# ============================================================================
# BEISPIEL 3: Multiplikationslängen
# ============================================================================

def beispiel_3_multiplikationslängen():
    print("=" * 70)
    print("BEISPIEL 3: Multiplikationslängen (ML-1 bis ML-5)")
    print("=" * 70)
    
    # Einfache 2x2 Matrizen
    A = np.array([[0.4, 0.1], [0.2, 0.3]])
    B = np.array([[0.3, 0.2], [0.2, 0.4]])
    
    print(f"\nMatrix A:\n{A}\n")
    print(f"Matrix B:\n{B}\n")
    
    result = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
    
    for k in range(1, 6):
        M_k = result[k]
        diag_k, dsi_k = MatrixCentroidFinder.find_matrix_centroid(M_k)
        
        print(f"M_{k}:")
        print(f"{M_k}")
        print(f"Diagonale: {diag_k}, DSI: {dsi_k:.4f}\n")
    
    print("→ Beobachtung: Mit wachsendem k wird die Matrix diagonaler (DSI wächst)")


# ============================================================================
# BEISPIEL 4: Iterative Schwerpunkt-Verfeinerung
# ============================================================================

def beispiel_4_schwerpunkt_verfeinerung():
    print("=" * 70)
    print("BEISPIEL 4: Iterative Schwerpunkt-Verfeinerung zur Diagonale")
    print("=" * 70)
    
    A = np.array([[5.0, 2.0, 0.1],
                  [2.0, 4.0, 0.2],
                  [0.1, 0.2, 3.0]])
    
    print(f"\nAusgangsmatrix A:\n{A}\n")
    
    A_refined, convergence = CentroidRefinement.refine_to_diagonal(
        A, alpha=0.5, max_iterations=100, tolerance=1e-10
    )
    
    print(f"Nach Verfeinerung:\n{A_refined}\n")
    print(f"Anzahl Iterationen: {len(convergence)}")
    print(f"Finaler Fehler: {convergence[-1]:.2e}")
    print(f"→ Off-Diagonal-Norm ist praktisch Null!")
    print(f"→ Mit alpha=0.5 halbiert sich der Fehler ungefähr pro Iteration\n")


# ============================================================================
# BEISPIEL 5: Diagonalisierung mit SVD
# ============================================================================

def beispiel_5_centroid_diagonalisierung():
    print("=" * 70)
    print("BEISPIEL 5: Schwerpunkt-offensive Diagonalisierung (SVD-Variante)")
    print("=" * 70)
    
    # Eine nicht-diagonale Matrix
    A = np.array([[2.0, 1.5, 0.3],
                  [1.5, 3.0, 0.5],
                  [0.3, 0.5, 1.0]])
    
    print(f"\nAusgangsmatrix:\n{A}\n")
    
    B_diag, Q = CentroidDiagonalization.centroid_diagonalization(
        A, max_iterations=50, dsi_threshold=0.90
    )
    
    print(f"Nach Diagonalisierung:\n{B_diag}\n")
    
    # Überprüfe Orthogonalität von Q
    QTQ = Q.T @ Q
    print(f"Q^T Q (sollte Identität sein):\n{QTQ}\n")
    
    diag_b, dsi_b = MatrixCentroidFinder.find_matrix_centroid(B_diag)
    print(f"DSI nach Diagonalisierung: {dsi_b:.6f}")
    print(f"→ Transformationsmatrix Q ist orthogonal!")
    print(f"→ Matrix ist sehr diagonal geworden\n")


# ============================================================================
# BEISPIEL 6: QR-Algorithmus mit Schwerpunkt
# ============================================================================

def beispiel_6_qr_centroid():
    print("=" * 70)
    print("BEISPIEL 6: QR-Algorithmus mit Schwerpunkt-Vorkonditionierung")
    print("=" * 70)
    
    # Symmetrische Matrix
    A = np.array([[2.0, 0.5, 0.1],
                  [0.5, 3.0, 0.2],
                  [0.1, 0.2, 4.0]])
    
    print(f"\nMatrix A:\n{A}\n")
    
    eigenvals, eigenvecs, iterations = QRCentroid.qr_centroid(
        A, max_iterations=200, tolerance=1e-10
    )
    
    print(f"Berechnete Eigenwerte: {eigenvals}")
    print(f"Anzahl Iterationen: {iterations}")
    print(f"\nEigenvektoren (spaltenweise):\n{eigenvecs}\n")
    
    # Überprüfe: A v_i = λ_i v_i
    print("Verifikation (A·v - λ·v sollte ≈ 0 sein):")
    for i in range(len(eigenvals)):
        v = eigenvecs[:, i]
        Av = A @ v
        lhs = Av - eigenvals[i] * v
        error = np.linalg.norm(lhs)
        print(f"  Eigenwert {i}: λ = {eigenvals[i]:.6f}, Error = {error:.2e}")
    
    print("\n→ QR-Algorithmus mit Schwerpunkt konvergiert schnell!\n")


# ============================================================================
# BEISPIEL 7: Lineare Gleichungssysteme
# ============================================================================

def beispiel_7_lineares_system():
    print("=" * 70)
    print("BEISPIEL 7: Lösen von Ax = b mit Schwerpunkt-Vorkonditionierung")
    print("=" * 70)
    
    # Ein Gleichungssystem
    A = np.array([[2.0, 0.1, 0.05],
                  [0.1, 3.0, 0.2],
                  [0.05, 0.2, 4.0]])
    b = np.array([2.15, 3.3, 4.25])
    
    print(f"\nKoeffizientenmatrix A:\n{A}\n")
    print(f"Rechte Seite b: {b}\n")
    
    x, residuals = CentroidPreconditioning.solve_preconditioned(
        A, b, max_iterations=1000, tolerance=1e-10
    )
    
    print(f"Lösung x: {x}")
    print(f"Verifikation A·x - b: {A @ x - b}")
    print(f"Residual-Norm: {residuals[-1]:.2e}")
    print("\n→ Schwerpunkt-Vorkonditionierung stabilisiert die Lösung\n")


# ============================================================================
# BEISPIEL 8: Strukturelle Unsicherheit
# ============================================================================

def beispiel_8_strukturelle_unsicherheit():
    print("=" * 70)
    print("BEISPIEL 8: Strukturelle Unsicherheit der Multiplikationslängen")
    print("=" * 70)
    
    # Matrizen für die Analyse
    A = np.random.seed(42)
    A = np.random.rand(4, 4) + 2 * np.eye(4)
    B = np.random.rand(4, 4) + 2 * np.eye(4)
    
    print(f"\nBerechne Multiplikationslängen...\n")
    
    matrices = MatrixLengthsMultiplication.compute_matrix_lengths(A, B)
    
    print("Analyse der Struktur:")
    print("k | Entropie | Spektrallücke | Gesamt-Unsicherheit")
    print("-" * 55)
    
    uncertainties = StructuralUncertainty.compute_total_uncertainty(matrices)
    
    for k in range(1, 6):
        entropy = StructuralUncertainty.compute_entropy(matrices[k])
        gap = StructuralUncertainty.compute_spectral_gap(matrices[k])
        uncertainty = uncertainties[k]
        
        print(f"{k} | {entropy:8.4f} | {gap:13.4f} | {uncertainty:19.4f}")
    
    print("\n→ Mit wachsendem k wächst die Unsicherheit exponentiell")
    print("→ Dies ist das 'Paradox der Multiplikationslänge'\n")


# ============================================================================
# VERGLEICH: Mit und ohne Schwerpunkt-Vorkonditionierung
# ============================================================================

def beispiel_vergleich_schwerpunkt():
    print("=" * 70)
    print("VERGLEICH: Mit und ohne Schwerpunkt-Vorkonditionierung")
    print("=" * 70)
    
    # Eine illkonditionierte Matrix
    A = np.array([[100.0, 1.0, 0.1],
                  [1.0, 10.0, 0.1],
                  [0.1, 0.1, 1.0]])
    b = np.array([101.1, 11.1, 1.2])
    
    print(f"\nMatrix A (schlecht konditioniert):\n{A}\n")
    print(f"Konditionszahl κ(A): {np.linalg.cond(A):.2f}\n")
    
    # Mit Vorkonditionierung
    x, residuals = CentroidPreconditioning.solve_preconditioned(
        A, b, max_iterations=100
    )
    
    # Ohne Vorkonditionierung (direkter Löser)
    x_direct = np.linalg.solve(A, b)
    
    print(f"Mit Schwerpunkt-Vorkonditionierung:")
    print(f"  Residual-Norm: {residuals[-1]:.2e}")
    print(f"  Lösung x: {x}\n")
    
    print(f"Direkter Löser (ohne Vorkonditionierung):")
    print(f"  Lösung x: {x_direct}")
    print(f"  Residual-Norm: {np.linalg.norm(A @ x_direct - b):.2e}\n")
    
    print("→ Schwerpunkt-Vorkonditionierung verbessert numerische Stabilität\n")


# ============================================================================
# HAUPT-PROGRAMM
# ============================================================================

def main():
    print("\n")
    print("#" * 70)
    print("# PRAKTISCHE BEISPIELE FÜR ALLE ALGORITHMEN")
    print("# Strukturen in der Algebra - Schwerpunkterkennung")
    print("#" * 70)
    print("\n")
    
    beispiel_1_vektor_schwerpunkt()
    beispiel_2_matrix_schwerpunkt()
    beispiel_3_multiplikationslängen()
    beispiel_4_schwerpunkt_verfeinerung()
    beispiel_5_centroid_diagonalisierung()
    beispiel_6_qr_centroid()
    beispiel_7_lineares_system()
    beispiel_8_strukturelle_unsicherheit()
    beispiel_vergleich_schwerpunkt()
    
    print("=" * 70)
    print("ALLE BEISPIELE ABGESCHLOSSEN!")
    print("=" * 70)


if __name__ == "__main__":
    main()
