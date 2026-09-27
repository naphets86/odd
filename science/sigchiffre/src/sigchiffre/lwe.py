"""
sigchiffre.lwe
==============
Learning With Errors (LWE) – Basisimplementierung und LWESignatureChiffre.

LWE (Regev 2005):
  Params: n, m, q, σ (Fehlerverteilung Gauß)
  KeyGen: A ∈ Z_q^{m×n} zufällig, s ∈ Z_q^n geheim, e Fehler
          b = As + e mod q
  Enc(bit): Wähle zufällige Teilmenge S ⊆ [m]
          c0 = Σ_{i∈S} A_i mod q,  c1 = Σ_{i∈S} b_i + bit·⌊q/2⌋ mod q
  Dec(c0,c1): m' = c1 - c0^T·s mod q; bit = round(m'/⌊q/2⌋) mod 2

LWESignatureChiffre:
  Ersetzt die affine Transformation der Signatur-Chiffre durch ein LWE-Sample.

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 11 (LWE).
          O. Regev, JACM 2009.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np


@dataclass
class LWEParams:
    """LWE-Systemparameter.

    Parameters
    ----------
    n : Dimension (Geheimvektor-Länge)
    m : Anzahl Samples
    q : Modulus
    sigma : Standardabweichung der Fehlerverteilung
    """
    n: int
    m: int
    q: int
    sigma: float = 3.2


def _discrete_gaussian(sigma: float, q: int) -> int:
    """Ganzzahliger Fehler aus diskretisierter Gaußverteilung mod q."""
    import random
    e = int(round(random.gauss(0, sigma)))
    return e % q


class LWEPublicKey:
    """Öffentlicher LWE-Schlüssel.

    Parameters
    ----------
    params : LWEParams
    A      : Matrix ∈ Z_q^{m×n}
    b      : b = As + e mod q
    """

    def __init__(self, params: LWEParams, A: np.ndarray, b: np.ndarray) -> None:
        self.params = params
        self.A = A
        self.b = b

    def encrypt(self, bit: int) -> tuple[np.ndarray, int]:
        """Verschlüsselt ein Bit ∈ {0, 1}.

        Parameters
        ----------
        bit : 0 oder 1

        Returns
        -------
        (c0, c1)
        """
        if bit not in (0, 1):
            raise ValueError(f"bit muss 0 oder 1 sein, erhalten: {bit}")
        m, n, q = self.params.m, self.params.n, self.params.q
        # Zufällige Teilmenge
        S = np.array([secrets.randbelow(2) for _ in range(m)], dtype=int)
        c0 = (S @ self.A) % q
        c1 = int((int(S @ self.b) + bit * (q // 2)) % q)
        return c0, c1


class LWEPrivateKey:
    """Privater LWE-Schlüssel.

    Parameters
    ----------
    params : LWEParams
    s      : Geheimvektor ∈ Z_q^n
    """

    def __init__(self, params: LWEParams, s: np.ndarray) -> None:
        self.params = params
        self.s = s
        self.public_key: LWEPublicKey | None = None

    def decrypt(self, c0: np.ndarray, c1: int) -> int:
        """Entschlüsselt (c0, c1) → Bit.

        v = c1 - c0^T·s mod q; bit = round(v / (q//2)) mod 2
        """
        q = self.params.q
        v = int(c1 - int(c0 @ self.s)) % q
        # Runden: nahe 0 → 0, nahe q/2 → 1
        threshold = q // 4
        half = q // 2
        distance_to_0 = min(v, q - v)
        distance_to_half = min(abs(v - half), q - abs(v - half))
        return 0 if distance_to_0 <= distance_to_half else 1


def lwe_keygen(params: LWEParams) -> LWEPrivateKey:
    """Erzeugt ein LWE-Schlüsselpaar.

    Parameters
    ----------
    params : LWEParams

    Returns
    -------
    LWEPrivateKey (enthält public_key)
    """
    n, m, q = params.n, params.m, params.q
    A = np.array(
        [[secrets.randbelow(q) for _ in range(n)] for _ in range(m)], dtype=int
    )
    s = np.array([secrets.randbelow(q) for _ in range(n)], dtype=int)
    e = np.array([_discrete_gaussian(params.sigma, q) for _ in range(m)], dtype=int)
    b = (A @ s + e) % q
    private = LWEPrivateKey(params, s)
    private.public_key = LWEPublicKey(params, A, b)
    return private


# ---------------------------------------------------------------------------
# LWE-Signatur-Chiffre (hybride Variante)
# ---------------------------------------------------------------------------

class LWESignatureChiffre:
    """Hybride Signatur-Chiffre: σ(A,j) via LWE bit-weise verschlüsselt.

    Jeder Signaturwert σ(A,j) wird als Bitfolge bit-weise über LWE verschlüsselt.

    Parameters
    ----------
    n_block  : Blockgröße der Signatur-Chiffre
    lwe_priv : LWE-Privatschlüssel
    sig_bits : Anzahl Bits pro Signaturwert (Standard: 2*n_block)
    """

    def __init__(
        self, n_block: int, lwe_priv: LWEPrivateKey, sig_bits: int | None = None
    ) -> None:
        self.n_block = n_block
        self.lwe_priv = lwe_priv
        self.sig_bits = sig_bits if sig_bits is not None else 2 * n_block

    def encrypt(self, A: np.ndarray) -> list[list[tuple[np.ndarray, int]]]:
        """Verschlüsselt n_block×n_block Matrix; jede Signatur bit-weise via LWE.

        Returns
        -------
        list[list[(c0, c1)]]  – äußere Liste: Spalten; innere: Bits
        """
        from .signature import sigma_vector
        sigs = sigma_vector(A)
        pub = self.lwe_priv.public_key
        result = []
        for s in sigs:
            bit_enc = []
            for bit_idx in range(self.sig_bits):
                bit = (s >> bit_idx) & 1
                bit_enc.append(pub.encrypt(bit))
            result.append(bit_enc)
        return result

    def decrypt(
        self, C: list[list[tuple[np.ndarray, int]]]
    ) -> np.ndarray:
        """Entschlüsselt LWE-Chiffrat → n_block×n_block Matrix."""
        from .signature import reconstruct_column
        n = self.n_block
        A = np.zeros((n, n), dtype=int)
        for j, bit_enc in enumerate(C):
            s_j = 0
            for bit_idx, (c0, c1) in enumerate(bit_enc):
                bit = self.lwe_priv.decrypt(c0, c1)
                s_j += bit << bit_idx
            col = reconstruct_column(s_j, j, n)
            for i in range(n):
                A[i, j] = col[i]
        return A
