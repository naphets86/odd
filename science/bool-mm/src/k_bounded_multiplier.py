"""
k-beschränkte Matrixmultiplikation mit Schichten-Signaturen.

Implementiert Algorithmus 4.1 aus der wissenschaftlichen Arbeit.
"""

import numpy as np
from typing import List


class KBoundedMatrixMultiplier:
    """
    Matrixmultiplikation für k-beschränkte Werte mit Signaturen.
    
    Für Matrizen mit Einträgen aus {0, 1, ..., k} wird die Signatur-Technik
    auf Schichten angewendet, um O(k·n²) Laufzeit zu erreichen.
    """
    
    def __init__(self):
        """Initialisiert den k-beschränkten Matrix Multiplier."""
        pass
    
    def _popcount(self, x: int) -> int:
        """
        Zählt die Anzahl gesetzter Bits (Population Count).
        
        Args:
            x: Integer-Wert
            
        Returns:
            Anzahl der 1-Bits
        """
        return bin(x).count('1')
    
    def _compute_layer_signature(
        self, 
        vector: np.ndarray, 
        threshold: int
    ) -> int:
        """
        Berechnet Signatur für eine Schicht.
        
        Args:
            vector: Vektor mit Werten aus {0, ..., k}
            threshold: Schwellwert für binäre Schicht
            
        Returns:
            Signatur der Schicht
        """
        n = len(vector)
        signature = sum(
            (1 if vector[i] >= threshold else 0) * (2 ** i)
            for i in range(n)
        )
        return signature
    
    def multiply_k_bounded(
        self, 
        A: np.ndarray, 
        B: np.ndarray, 
        k: int
    ) -> np.ndarray:
        """
        Matrixmultiplikation für k-beschränkte Matrizen.
        
        Implementiert Algorithmus 4.1 mit O(k·n²) Komplexität.
        
        Args:
            A: Matrix mit Werten aus {0, ..., k} (n x n)
            B: Matrix mit Werten aus {0, ..., k} (n x n)
            k: Maximaler Wert in den Matrizen
            
        Returns:
            Produktmatrix C (n x n)
        """
        n = A.shape[0]
        
        # Schritt 1: Berechne Schichten-Signaturen
        row_sigs = [[0] * n for _ in range(k + 1)]
        col_sigs = [[0] * n for _ in range(k + 1)]
        
        # Für jede Schicht x von 1 bis k
        for x in range(1, k + 1):
            # Zeilen-Signaturen
            for i in range(n):
                row_sigs[x][i] = self._compute_layer_signature(A[i, :], x)
            
            # Spalten-Signaturen
            for j in range(n):
                col_sigs[x][j] = self._compute_layer_signature(B[:, j], x)
        
        # Schritt 2: Berechne Produktmatrix
        C = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(n):
                sum_val = 0
                
                # Für alle Schichtkombinationen
                for x in range(1, k + 1):
                    for y in range(1, k + 1):
                        # Bitweise AND
                        bit_match = row_sigs[x][i] & col_sigs[y][j]
                        
                        # Zähle übereinstimmende Bits
                        sum_val += self._popcount(bit_match)
                
                C[i, j] = sum_val
        
        return C