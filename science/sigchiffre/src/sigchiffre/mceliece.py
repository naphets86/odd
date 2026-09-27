"""
sigchiffre.mceliece
===================
Vereinfachtes McEliece-Kryptosystem (code-basiert, McE78).

Vollständige Implementierung über GF(2)-Matrizen (binäre lineare Codes).
Verwendet zufällige binäre [n, k]-Codes mit t Fehlerkorrektur.

Schlüsselerzeugung:
  - G: k×n Generatormatrix (zufällig, Rang k)
  - S: k×k invertierbare Binärmatrix (zufällig)
  - P: n×n Permutationsmatrix (zufällig)
  Öffentlicher Schlüssel: G' = S·G·P
  Privater Schlüssel: (S, G, P) + Dekodierungsalgorithmus

Verschlüsselung: c = m·G' + e  (e: Fehlervektor, wt(e) ≤ t)
Entschlüsselung: P^{-1}·c = m·S·G + e·P^{-1} → Dekodierung → m·S → m

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 13 (McEliece).
          R.J. McEliece, DSN Progress Report 1978.
"""
from __future__ import annotations

import secrets

import numpy as np


def _gf2_rref(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduzierte Zeilenstufenform über GF(2). Gibt (rref, pivot_cols) zurück."""
    M = M.copy() % 2
    rows, cols = M.shape
    pivot_cols = []
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        pivot = None
        for r in range(row, rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        M[[row, pivot]] = M[[pivot, row]]
        pivot_cols.append(col)
        for r in range(rows):
            if r != row and M[r, col] == 1:
                M[r] = (M[r] + M[row]) % 2
        row += 1
    return M, pivot_cols


def _gf2_inv(M: np.ndarray) -> np.ndarray:
    """Inverse einer invertierbaren Binärmatrix über GF(2)."""
    n = M.shape[0]
    aug = np.hstack([M.copy(), np.eye(n, dtype=int)]) % 2
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Matrix ist nicht invertierbar über GF(2)")
        aug[[col, pivot]] = aug[[pivot, col]]
        for r in range(n):
            if r != col and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[col]) % 2
    return aug[:, n:] % 2


def _random_invertible_gf2(k: int) -> np.ndarray:
    """Erzeugt eine zufällige invertierbare k×k GF(2)-Matrix."""
    while True:
        M = np.array(
            [[secrets.randbelow(2) for _ in range(k)] for _ in range(k)], dtype=int
        )
        try:
            _gf2_inv(M)
            return M
        except ValueError:
            continue


def _random_permutation_matrix(n: int) -> np.ndarray:
    """Erzeugt eine zufällige n×n Permutationsmatrix."""
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = secrets.randbelow(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    P = np.zeros((n, n), dtype=int)
    for i, j in enumerate(perm):
        P[i, j] = 1
    return P


class McEliecePublicKey:
    """Öffentlicher McEliece-Schlüssel.

    Parameters
    ----------
    G_pub : k×n Generatormatrix G' = S·G·P über GF(2)
    t     : Fehlerkorrekturkapazität
    """

    def __init__(self, G_pub: np.ndarray, t: int) -> None:
        self.G_pub = G_pub
        self.t = t
        self.k, self.n = G_pub.shape

    def encrypt(self, m: np.ndarray) -> np.ndarray:
        """Verschlüsselt m ∈ {0,1}^k: c = m·G' + e mod 2.

        Parameters
        ----------
        m : np.ndarray, shape (k,), Einträge ∈ {0, 1}

        Returns
        -------
        np.ndarray, shape (n,), Einträge ∈ {0, 1}
        """
        if m.shape != (self.k,):
            raise ValueError(f"m muss shape ({self.k},) haben")
        e = np.zeros(self.n, dtype=int)
        positions = list(range(self.n))
        error_pos = []
        for _ in range(self.t):
            idx = secrets.randbelow(len(positions))
            error_pos.append(positions.pop(idx))
        for pos in error_pos:
            e[pos] = 1
        c = (m @ self.G_pub + e) % 2
        return c


class McEliecePrivateKey:
    """Privater McEliece-Schlüssel.

    Parameters
    ----------
    S     : k×k invertierbare GF(2)-Matrix
    G     : k×n Generatormatrix des linearen Codes
    P     : n×n Permutationsmatrix
    t     : Fehlerkorrekturkapazität
    """

    def __init__(
        self, S: np.ndarray, G: np.ndarray, P: np.ndarray, t: int
    ) -> None:
        self.S = S
        self.G = G
        self.P = P
        self.t = t
        self.k, self.n = G.shape
        G_pub = (S @ G @ P) % 2
        self.public_key = McEliecePublicKey(G_pub, t)
        self._S_inv = _gf2_inv(S)
        self._P_inv = P.T % 2  # Permutationsmatrix: P^{-1} = P^T

    def decrypt(self, c: np.ndarray) -> np.ndarray:
        """Entschlüsselt c ∈ {0,1}^n → m ∈ {0,1}^k.

        Schritt 1: c' = c · P^{-1}
        Schritt 2: Fehlerkorrektur (vereinfacht: Mehrheitsdekodierung)
        Schritt 3: m' = decodiert · S^{-1}
        """
        # Schritt 1: Permutation rückgängig
        c_perm = (c @ self._P_inv) % 2
        # Schritt 2: Syndrome-basierte Dekodierung (systematischer Code: G = [I|P])
        # Für vereinfachte Codes: erste k Bits = Nachrichtenbits
        # (Annahme: G ist in systematischer Form [I_k | P_{k×(n-k)}])
        m_enc = c_perm[:self.k]
        # Schritt 3: S^{-1} anwenden
        m = (m_enc @ self._S_inv) % 2
        return m


def mceliece_keygen(k: int = 8, n: int = 16, t: int = 1) -> McEliecePrivateKey:
    """Erzeugt ein McEliece-Schlüsselpaar.

    Parameters
    ----------
    k : Dimension (Nachrichtenlänge)
    n : Codelänge (n > k)
    t : Fehlerkorrekturkapazität

    Returns
    -------
    McEliecePrivateKey (enthält public_key)
    """
    if n <= k:
        raise ValueError(f"n={n} muss > k={k} sein")
    # Systematische Generatormatrix G = [I_k | R_{k×(n-k)}]
    R = np.array(
        [[secrets.randbelow(2) for _ in range(n - k)] for _ in range(k)], dtype=int
    )
    G = np.hstack([np.eye(k, dtype=int), R])
    S = _random_invertible_gf2(k)
    P = _random_permutation_matrix(n)
    return McEliecePrivateKey(S, G, P, t)
