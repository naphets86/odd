"""
Strassen Matrixmultiplikation für Boolean und k-beschränkte Matrizen.

Implementiert den Strassen-Algorithmus (1969) mit O(n^2.807) Komplexität
für Vergleichszwecke mit der Signatur-Methode.
"""

import numpy as np
from typing import Tuple


class StrassenMultiplier:
    """
    Implementierung des Strassen-Algorithmus für Matrixmultiplikation.
    
    Der Algorithmus teilt Matrizen rekursiv in Quadranten auf und reduziert
    die Anzahl der benötigten Multiplikationen von 8 auf 7.
    
    Komplexität: O(n^2.807)
    """
    
    def __init__(self, threshold: int = 64):
        """
        Initialisiert den Strassen Multiplier.
        
        Args:
            threshold: Minimale Größe für rekursive Teilung.
                      Kleinere Matrizen werden naiv multipliziert.
        """
        self.threshold = threshold
        self.multiplication_count = 0
    
    def _next_power_of_2(self, n: int) -> int:
        """Berechnet die nächste Zweierpotenz >= n."""
        power = 1
        while power < n:
            power *= 2
        return power
    
    def _pad_matrix(self, M: np.ndarray, size: int) -> np.ndarray:
        """
        Erweitert Matrix auf Größe size x size mit Nullen.
        
        Args:
            M: Eingabematrix
            size: Zielgröße
            
        Returns:
            Gepaddete Matrix
        """
        n, m = M.shape
        padded = np.zeros((size, size), dtype=M.dtype)
        padded[:n, :m] = M
        return padded
    
    def _unpad_matrix(self, M: np.ndarray, orig_shape: Tuple[int, int]) -> np.ndarray:
        """
        Entfernt Padding von Matrix.
        
        Args:
            M: Gepaddete Matrix
            orig_shape: Ursprüngliche Form
            
        Returns:
            Matrix in ursprünglicher Größe
        """
        n, m = orig_shape
        return M[:n, :m]
    
    def multiply_naive(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive Matrixmultiplikation O(n^3).
        
        Wird für kleine Matrizen (< threshold) verwendet.
        
        Args:
            A: Matrix (n x k)
            B: Matrix (k x m)
            
        Returns:
            Produktmatrix C (n x m)
        """
        return np.matmul(A, B)
    
    def multiply_strassen(
        self, 
        A: np.ndarray, 
        B: np.ndarray,
        boolean: bool = False
    ) -> np.ndarray:
        """
        Strassen Matrixmultiplikation.
        
        Args:
            A: Erste Matrix (n x n)
            B: Zweite Matrix (n x n)
            boolean: Ob Boolean-Operationen verwendet werden sollen
            
        Returns:
            Produktmatrix C (n x n)
        """
        self.multiplication_count = 0
        
        # Speichere ursprüngliche Form
        orig_shape = A.shape
        n = A.shape[0]
        
        # Erweitere auf nächste Zweierpotenz
        size = self._next_power_of_2(n)
        
        if size != n:
            A = self._pad_matrix(A, size)
            B = self._pad_matrix(B, size)
        
        # Rekursive Multiplikation
        C = self._strassen_recursive(A, B, boolean)
        
        # Entferne Padding
        if size != orig_shape[0]:
            C = self._unpad_matrix(C, orig_shape)
        
        return C
    
    def _strassen_recursive(
        self, 
        A: np.ndarray, 
        B: np.ndarray,
        boolean: bool
    ) -> np.ndarray:
        """
        Rekursive Strassen-Multiplikation.
        
        Args:
            A: Matrix (n x n), n ist Zweierpotenz
            B: Matrix (n x n), n ist Zweierpotenz
            boolean: Ob Boolean-Operationen verwendet werden
            
        Returns:
            Produktmatrix C (n x n)
        """
        n = A.shape[0]
        
        # Basisfall: Nutze naive Multiplikation für kleine Matrizen
        if n <= self.threshold:
            self.multiplication_count += 1
            if boolean:
                return self._multiply_boolean_naive(A, B)
            else:
                return self.multiply_naive(A, B)
        
        # Teile Matrizen in Quadranten
        mid = n // 2
        
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
        
        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]
        
        # Berechne die 7 Strassen-Produkte
        if boolean:
            # Für Boolean: OR statt +, AND bleibt AND
            M1 = self._strassen_recursive(A11 + A22, B11 + B22, boolean)
            M2 = self._strassen_recursive(A21 + A22, B11, boolean)
            M3 = self._strassen_recursive(A11, B12 - B22, boolean)
            M4 = self._strassen_recursive(A22, B21 - B11, boolean)
            M5 = self._strassen_recursive(A11 + A12, B22, boolean)
            M6 = self._strassen_recursive(A21 - A11, B11 + B12, boolean)
            M7 = self._strassen_recursive(A12 - A22, B21 + B22, boolean)
            
            # Kombiniere zu Quadranten (mit modulo für Boolean)
            C11 = (M1 + M4 - M5 + M7) % 2
            C12 = (M3 + M5) % 2
            C21 = (M2 + M4) % 2
            C22 = (M1 - M2 + M3 + M6) % 2
        else:
            # Standard Strassen für reelle Zahlen
            M1 = self._strassen_recursive(A11 + A22, B11 + B22, boolean)
            M2 = self._strassen_recursive(A21 + A22, B11, boolean)
            M3 = self._strassen_recursive(A11, B12 - B22, boolean)
            M4 = self._strassen_recursive(A22, B21 - B11, boolean)
            M5 = self._strassen_recursive(A11 + A12, B22, boolean)
            M6 = self._strassen_recursive(A21 - A11, B11 + B12, boolean)
            M7 = self._strassen_recursive(A12 - A22, B21 + B22, boolean)
            
            # Kombiniere zu Quadranten
            C11 = M1 + M4 - M5 + M7
            C12 = M3 + M5
            C21 = M2 + M4
            C22 = M1 - M2 + M3 + M6
        
        # Setze Ergebnis zusammen
        C = np.zeros((n, n), dtype=A.dtype)
        C[:mid, :mid] = C11
        C[:mid, mid:] = C12
        C[mid:, :mid] = C21
        C[mid:, mid:] = C22
        
        return C
    
    def _multiply_boolean_naive(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Naive Boolean Matrixmultiplikation.
        
        Args:
            A: Boolean Matrix (n x k)
            B: Boolean Matrix (k x m)
            
        Returns:
            Boolean Produktmatrix C (n x m)
        """
        n, k = A.shape
        _, m = B.shape
        C = np.zeros((n, m), dtype=int)
        
        for i in range(n):
            for j in range(m):
                for k_idx in range(k):
                    if A[i, k_idx] and B[k_idx, j]:
                        C[i, j] = 1
                        break
        
        return C


class KBoundedStrassenMultiplier:
    """
    Strassen-Algorithmus für k-beschränkte Matrizen.
    
    Für Matrizen mit Werten aus {0, ..., k}.
    """
    
    def __init__(self, threshold: int = 64):
        """
        Initialisiert den k-beschränkten Strassen Multiplier.
        
        Args:
            threshold: Minimale Größe für rekursive Teilung
        """
        self.strassen = StrassenMultiplier(threshold)
    
    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matrixmultiplikation für k-beschränkte Matrizen mit Strassen.
        
        Args:
            A: Matrix mit Werten aus {0, ..., k} (n x n)
            B: Matrix mit Werten aus {0, ..., k} (n x n)
            
        Returns:
            Produktmatrix C (n x n)
        """
        # Nutze Standard-Strassen für ganzzahlige Arithmetik
        return self.strassen.multiply_strassen(A, B, boolean=False)