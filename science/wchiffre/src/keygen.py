"""
Schluesselgenerierung fuer die Fenster-Chiffre (FC).

Gen(1^lambda) -> (pk, sk)

  pk = (n, e, k)   -- oeffentlicher Schluessel
  sk = (d, s)      -- privater Schluessel (Exponent + Offset)

Lehrbeispiel aus der Arbeit:  n=55, e=23, d=7, p=5, q=11
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from src.math_utils import mod_inverse, euler_phi, generate_prime, is_prime_miller_rabin


@dataclass(frozen=True)
class PublicKey:
    """Oeffentlicher Schluessel der Fenster-Chiffre.

    Attributes
    ----------
    n : int  RSA-Modulus (n = p * q).
    e : int  Oeffentlicher Exponent (gcd(e, phi(n)) = 1).
    k : int  Fenster-Abstand (Dummies jedes k-te Paket, k >= 2).
    """
    n: int
    e: int
    k: int

    def __post_init__(self) -> None:
        if self.n < 4:
            raise ValueError(f"RSA-Modulus n muss >= 4 sein, erhalten: {self.n}")
        if self.e < 2:
            raise ValueError(f"Exponent e muss >= 2 sein, erhalten: {self.e}")
        if self.k < 2:
            raise ValueError(f"Fenster-Abstand k muss >= 2 sein, erhalten: {self.k}")
        if math.gcd(self.e, self.n) != 1:
            raise ValueError(f"gcd(e={self.e}, n={self.n}) != 1")


@dataclass(frozen=True)
class PrivateKey:
    """Privater Schluessel der Fenster-Chiffre.

    Attributes
    ----------
    d     : int  Privater RSA-Exponent (e*d = 1 mod phi(n)).
    s     : int  Geheimer Offset (0 <= s < k).
    n     : int  RSA-Modulus (benoetigt fuer Entschluesselung).
    k     : int  Fenster-Abstand (konsistent mit PublicKey).
    phi_n : int  Eulersche Phi-Funktion phi(n) = (p-1)*(q-1).
                 Wird intern gespeichert, um teure Nachfaktorisierung
                 in Tests und Demo zu vermeiden.
    """
    d: int
    s: int
    n: int
    k: int
    phi_n: int = 0  # (p-1)*(q-1); optional fuer manuell erstellte Testkeys

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError(f"Privater Exponent d muss >= 1 sein, erhalten: {self.d}")
        if not (0 <= self.s < self.k):
            raise ValueError(
                f"Offset s muss in [0, k-1] liegen, erhalten: s={self.s}, k={self.k}"
            )


class KeygenFC:
    """Schluesselgenerierung fuer die Fenster-Chiffre.

    Unterstuetzt:
    - Lehrbeispiel (n=55, e=23, d=7) aus der Arbeit
    - Beliebige RSA-Parameter mit vorgegebenen Primzahlen p, q
    - Zufaellige RSA-Primzahlen fuer praktische Anwendungen

    Examples
    --------
    Lehrbeispiel (Grundversion der Arbeit):
    >>> pk, sk = KeygenFC.toy_example()
    >>> pk
    PublicKey(n=55, e=23, k=3)
    >>> sk
    PrivateKey(d=7, s=..., n=55, k=3, phi_n=40)

    Eigene Primzahlen:
    >>> pk, sk = KeygenFC.from_primes(p=5, q=11, e=23, k=3, s=0)
    >>> pk.n
    55
    """

    @staticmethod
    def from_primes(
        p: int,
        q: int,
        e: int,
        k: int = 3,
        s: Optional[int] = None,
    ) -> tuple[PublicKey, PrivateKey]:
        """Erstellt ein Schluesselpaar aus gegebenen Primzahlen p, q.

        Parameters
        ----------
        p, q : int  Primzahlen (p != q).
        e    : int  Oeffentlicher Exponent; muss gcd(e, phi(n)) = 1 erfuellen.
        k    : int  Fenster-Abstand (Standard: 3).
        s    : int  Geheimer Offset; wird zufaellig aus {0,...,k-1} gewaehlt
                    wenn None.

        Returns
        -------
        (PublicKey, PrivateKey)
        """
        if p == q:
            raise ValueError("p und q muessen verschieden sein.")
        if not is_prime_miller_rabin(p):
            raise ValueError(f"p={p} ist keine Primzahl.")
        if not is_prime_miller_rabin(q):
            raise ValueError(f"q={q} ist keine Primzahl.")

        n = p * q
        phi_n = (p - 1) * (q - 1)

        if math.gcd(e, phi_n) != 1:
            raise ValueError(
                f"gcd(e={e}, phi(n)={phi_n}) != 1 – e ist kein gueltiger Exponent."
            )

        d = mod_inverse(e, phi_n)

        if s is None:
            s = random.randrange(0, k)

        pk = PublicKey(n=n, e=e, k=k)
        sk = PrivateKey(d=d, s=s, n=n, k=k, phi_n=phi_n)
        return pk, sk

    @staticmethod
    def toy_example(k: int = 3, s: Optional[int] = None) -> tuple[PublicKey, PrivateKey]:
        """Lehrbeispiel aus der Arbeit: (e=23, d=7, n=55).

        p=5, q=11, phi(55)=40, 23*7=161=4*40+1 => 23*7 = 1 mod 40.

        Parameters
        ----------
        k : int  Fenster-Abstand (Standard: 3).
        s : int  Offset (zufaellig wenn None).

        Returns
        -------
        (PublicKey, PrivateKey)
        """
        return KeygenFC.from_primes(p=5, q=11, e=23, k=k, s=s)

    @staticmethod
    def generate(
        bits: int = 512,
        e: int = 65537,
        k: int = 3,
        s: Optional[int] = None,
    ) -> tuple[PublicKey, PrivateKey]:
        """Generiert ein Schluesselpaar mit zufaelligen Primzahlen.

        Parameters
        ----------
        bits : int  Bitlaenge jeder Primzahl (n hat ca. 2*bits Bits).
        e    : int  Oeffentlicher Exponent (Standard: 65537).
        k    : int  Fenster-Abstand (Standard: 3).
        s    : int  Offset (zufaellig wenn None).

        Returns
        -------
        (PublicKey, PrivateKey)
        """
        while True:
            p = generate_prime(bits)
            q = generate_prime(bits)
            if p == q:
                continue
            phi_n = (p - 1) * (q - 1)
            if math.gcd(e, phi_n) == 1:
                break

        return KeygenFC.from_primes(p=p, q=q, e=e, k=k, s=s)
