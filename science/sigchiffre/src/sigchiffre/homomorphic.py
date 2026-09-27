"""
sigchiffre.homomorphic
======================
Homomorphe Erweiterungen:
  - HomomorphicSignatureChiffre: additive Homomorphie der Signatur-Chiffre
  - PaillierPublicKey / PaillierPrivateKey: Paillier-Kryptosystem (Gen09)

Homomorphie der Signatur-Chiffre:
  Enc(s1) + Enc(s2) = (a·s1+b + a·s2+b) mod p ≠ Enc(s1+s2) im Allgemeinen.
  Aber: Enc(s1) + Enc(s2) - b ≡ a·(s1+s2) + b (mod p)  →  begrenzte Homomorphie.

Paillier (1999):
  KeyGen: n=p·q, g=n+1, λ=lcm(p-1,q-1), μ=(λ mod n²)^{-1} mod n
  Enc(m): C = g^m · r^n mod n²
  Dec(C): m = L(C^λ mod n²) · μ mod n, L(x) = (x-1)/n
  Homomorphie: Enc(m1)·Enc(m2) = Enc(m1+m2) mod n²

Referenz: S. Epp, "Die Signatur-Chiffre", Kapitel 12 (Homomorphe Versch.).
"""
from __future__ import annotations

import secrets
from math import gcd, lcm

import numpy as np

from .chiffre import SignatureChiffre
from .keygen import generate_prime


# ---------------------------------------------------------------------------
# Homomorphe Signatur-Chiffre
# ---------------------------------------------------------------------------

class HomomorphicSignatureChiffre:
    """Additive Homomorphie über Z_q auf Basis der Signatur-Chiffre.

    Für Chiffratwerte c1, c2 gilt:
      (c1 + c2 - b) mod p = a·(s1 + s2) + b  mod p
    wenn s1 + s2 < p (kein Überlauf).

    Parameters
    ----------
    chiffre : SignatureChiffre
    q       : Modulus für homomorphe Operationen (typisch = p)
    """

    def __init__(self, chiffre: SignatureChiffre, q: int | None = None) -> None:
        self.chiffre = chiffre
        self.q = q if q is not None else chiffre.p

    def encrypt(self, A: np.ndarray) -> list[int]:
        return self.chiffre.encrypt(A)

    def decrypt(self, C: list[int]) -> np.ndarray:
        return self.chiffre.decrypt(C)

    def add_ciphertexts(self, C1: list[int], C2: list[int]) -> list[int]:
        """Addiert zwei Chiffrate komponentenweise mod q.

        Ergebnis entspricht Enc(s1+s2) wenn s1+s2 < p.
        """
        return [(c1 + c2) % self.q for c1, c2 in zip(C1, C2)]

    def scalar_multiply(self, C: list[int], k: int) -> list[int]:
        """Skalare Multiplikation: k·C mod q.

        Entspricht Enc(k·s) wenn k·s < p.
        """
        return [(k * c) % self.q for c in C]


# ---------------------------------------------------------------------------
# Paillier-Kryptosystem
# ---------------------------------------------------------------------------

def _l_function(x: int, n: int) -> int:
    """L(x) = (x - 1) / n (ganzzahlig)."""
    return (x - 1) // n


class PaillierPublicKey:
    """Öffentlicher Paillier-Schlüssel.

    Parameters
    ----------
    n : n = p·q
    g : g = n + 1 (Standard-Wahl)
    """

    def __init__(self, n: int, g: int) -> None:
        self.n = n
        self.g = g
        self.n_sq = n * n

    def encrypt(self, m: int) -> int:
        """Verschlüsselt m ∈ {0, …, n-1}: C = g^m · r^n mod n²."""
        if not (0 <= m < self.n):
            raise ValueError(f"m={m} muss in {{0,…,{self.n-1}}} liegen")
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if gcd(r, self.n) == 1:
                break
        return (pow(self.g, m, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq

    def add(self, c1: int, c2: int) -> int:
        """Homomorphe Addition: Enc(m1+m2) = c1·c2 mod n²."""
        return (c1 * c2) % self.n_sq

    def scalar_multiply(self, c: int, k: int) -> int:
        """Homomorphe skalare Multiplikation: Enc(k·m) = c^k mod n²."""
        return pow(c, k, self.n_sq)


class PaillierPrivateKey:
    """Privater Paillier-Schlüssel.

    Parameters
    ----------
    p, q : Primfaktoren von n
    """

    def __init__(self, p: int, q: int) -> None:
        self.p = p
        self.q = q
        n = p * q
        lam = lcm(p - 1, q - 1)
        g = n + 1
        n_sq = n * n
        mu = pow(_l_function(pow(g, lam, n_sq), n), -1, n)
        self.public_key = PaillierPublicKey(n, g)
        self._lam = lam
        self._mu = mu

    def decrypt(self, c: int) -> int:
        """Entschlüsselt c: m = L(c^λ mod n²) · μ mod n."""
        n = self.public_key.n
        n_sq = self.public_key.n_sq
        x = pow(c, self._lam, n_sq)
        return (_l_function(x, n) * self._mu) % n


def paillier_keygen(bits: int = 512) -> PaillierPrivateKey:
    """Erzeugt ein Paillier-Schlüsselpaar.

    Parameters
    ----------
    bits : Bit-Länge je Primfaktor

    Returns
    -------
    PaillierPrivateKey (enthält public_key)
    """
    p = generate_prime(bits)
    q = generate_prime(bits)
    while q == p:
        q = generate_prime(bits)
    return PaillierPrivateKey(p, q)
