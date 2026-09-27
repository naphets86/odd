"""
sigchiffre.elgamal
==================
ElGamal-Public-Key-Kryptosystem als eigenständiges Modul sowie
PublicKeySignatureChiffre als ElGamal-Hybrid mit Signatur-Chiffre-Kern.

ElGamal (ElG85):
  KeyGen: p prim, g Erzeuger, x ∈ {2,…,p-2} privat, h = g^x mod p öffentlich.
  Enc(m): k zufällig, C = (g^k mod p, m·h^k mod p)
  Dec(c1,c2): m = c2 · (c1^x)^{-1} mod p

Homomorphie: Enc(m1)·Enc(m2) = Enc(m1·m2) (multiplikativ).

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 10 (ElGamal).
"""
from __future__ import annotations

import secrets
from math import gcd

from .keygen import generate_prime, is_prime


class ElGamalPublicKey:
    """Öffentlicher ElGamal-Schlüssel.

    Parameters
    ----------
    p : Primzahl
    g : Erzeuger von Z_p*
    h : h = g^x mod p (öffentlicher Schlüssel)
    """

    def __init__(self, p: int, g: int, h: int) -> None:
        self.p = p
        self.g = g
        self.h = h

    def encrypt(self, m: int) -> tuple[int, int]:
        """Verschlüsselt m ∈ Z_p*: C = (g^k, m·h^k) mod p."""
        if not (1 <= m < self.p):
            raise ValueError(f"m muss in {{1,…,{self.p-1}}} liegen")
        k = secrets.randbelow(self.p - 3) + 2  # k ∈ {2, …, p-2}
        c1 = pow(self.g, k, self.p)
        c2 = (m * pow(self.h, k, self.p)) % self.p
        return c1, c2

    def homomorphic_multiply(
        self, c1: tuple[int, int], c2: tuple[int, int]
    ) -> tuple[int, int]:
        """Homomorphe Multiplikation: Enc(m1·m2) = Enc(m1)·Enc(m2) mod p."""
        return (
            (c1[0] * c2[0]) % self.p,
            (c1[1] * c2[1]) % self.p,
        )


class ElGamalPrivateKey:
    """Privater ElGamal-Schlüssel (enthält öffentlichen Schlüssel).

    Parameters
    ----------
    p : Primzahl
    g : Erzeuger
    x : privater Exponent ∈ {2, …, p-2}
    """

    def __init__(self, p: int, g: int, x: int) -> None:
        self.p = p
        self.g = g
        self.x = x
        h = pow(g, x, p)
        self.public_key = ElGamalPublicKey(p, g, h)

    def decrypt(self, c1: int, c2: int) -> int:
        """Entschlüsselt: m = c2 · (c1^x)^{-1} mod p."""
        s = pow(c1, self.x, self.p)
        s_inv = pow(s, -1, self.p)
        return (c2 * s_inv) % self.p


def elgamal_keygen(bits: int = 256) -> ElGamalPrivateKey:
    """Erzeugt ein ElGamal-Schlüsselpaar.

    Parameters
    ----------
    bits : Bit-Länge der Primzahl p

    Returns
    -------
    ElGamalPrivateKey (enthält public_key)
    """
    p = generate_prime(bits)
    # g = 2 ist für viele Primzahlen ein guter Generator (ausreichend für Beispiele)
    g = 2
    x = secrets.randbelow(p - 3) + 2  # ∈ {2, …, p-2}
    return ElGamalPrivateKey(p, g, x)


class PublicKeySignatureChiffre:
    """Asymmetrische Erweiterung der Signatur-Chiffre via ElGamal-Hybrid.

    Der Klartext-Signaturwert σ(A,j) wird als ElGamal-Nachricht verschlüsselt.
    Dies macht die Signatur-Chiffre asymmetrisch: Verschlüsseln benötigt nur
    den öffentlichen ElGamal-Schlüssel; Entschlüsseln den privaten.

    Achtung: Für σ(A,j) ∈ Z_p* muss σ(A,j) ≠ 0; Null-Spalten werden als 1 kodiert.
    """

    def __init__(
        self, n: int, elgamal_private: ElGamalPrivateKey
    ) -> None:
        self.n = n
        self.private_key = elgamal_private
        self.public_key = elgamal_private.public_key

    def encrypt(self, A) -> list[tuple[int, int]]:
        """Verschlüsselt A (n×n) mit öffentlichem ElGamal-Schlüssel."""
        import numpy as np
        from .signature import sigma_vector
        sigs = sigma_vector(A)
        result = []
        for s in sigs:
            m = max(s, 1)  # Null-Werte auf 1 mappen
            result.append(self.public_key.encrypt(m))
        return result

    def decrypt(self, C: list[tuple[int, int]]) -> "np.ndarray":
        """Entschlüsselt C mit privatem ElGamal-Schlüssel → n×n Matrix."""
        import numpy as np
        from .signature import reconstruct_column
        n = self.n
        A = np.zeros((n, n), dtype=int)
        for j, (c1, c2) in enumerate(C):
            s_j = self.private_key.decrypt(c1, c2)
            col = reconstruct_column(s_j, j, n)
            for i in range(n):
                A[i, j] = col[i]
        return A
