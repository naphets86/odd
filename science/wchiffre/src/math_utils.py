"""
Zahlentheoretische Hilfsfunktionen fuer die Fenster-Chiffre.

Enthaelt:
- Erweiterter Euklidischer Algorithmus (Lemma 2.3 / Bezout)
- Modulares Inverses (Satz 2.2)
- Eulersche Phi-Funktion
- Primzahlpruefung (Miller-Rabin)
"""

from __future__ import annotations
import math
import random
from typing import Tuple, Optional


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Erweiterter Euklidischer Algorithmus.

    Berechnet (g, x, y) mit g = gcd(a, b) und a*x + b*y = g.

    Parameters
    ----------
    a, b : int
        Eingabezahlen (a, b >= 0).

    Returns
    -------
    (g, x, y) : Tuple[int, int, int]
        g = gcd(a, b), Bezout-Koeffizienten x und y.

    Examples
    --------
    >>> extended_gcd(32, 23)
    (1, -5, 7)          # -5*32 + 7*23 = 1
    """
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    # Ruecksubstitution
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def mod_inverse(a: int, n: int) -> int:
    """Berechnet das multiplikative Inverse von a modulo n.

    Wirft ValueError wenn gcd(a, n) != 1.

    Parameters
    ----------
    a : int  Zahl, deren Inverses gesucht wird.
    n : int  Modulus.

    Returns
    -------
    int  a^{-1} mod n in {0, ..., n-1}.

    Examples
    --------
    >>> mod_inverse(23, 32)
    7
    >>> mod_inverse(7, 32)
    23
    """
    g, x, _ = extended_gcd(a % n, n)
    if g != 1:
        raise ValueError(f"Kein Inverses: gcd({a}, {n}) = {g} != 1")
    return x % n


def _pollard_rho(n: int) -> int:
    """Findet einen nicht-trivialen Teiler von n via Pollard-Rho (Floyd-Zyklus).

    Verwendet den Batch-GCD-Trick (alle 128 Schritte) fuer bessere Performance.

    Parameters
    ----------
    n : int  Zusammengesetzte Zahl (n > 1, n nicht prim).

    Returns
    -------
    int  Ein echter Teiler von n (1 < d < n).
    """
    if n % 2 == 0:
        return 2

    while True:
        x = random.randrange(2, n - 1)
        y = x
        c = random.randrange(1, n - 1)
        product = 1
        step = 0

        while True:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            diff = abs(x - y)
            if diff == 0:
                break  # Zyklus ohne Ergebnis -> neue Runde
            product = product * diff % n
            step += 1

            # Batch-GCD alle 128 Schritte
            if step % 128 == 0:
                d = math.gcd(product, n)
                if 1 < d < n:
                    return d
                product = 1

            d = math.gcd(product, n)
            if d == n:
                break  # Rueckfall -> neue Runde mit anderem c
            if 1 < d < n:
                return d


def _distinct_prime_factors(n: int) -> list[int]:
    """Gibt alle verschiedenen Primfaktoren von n zurueck.

    Strategie:
    1. Trial Division bis 10^6 (fuer kleine Faktoren)
    2. Miller-Rabin: falls Rest prim -> fertig
    3. Sonst: Pollard-Rho rekursiv

    Parameters
    ----------
    n : int  n >= 2.

    Returns
    -------
    list[int]  Sortierte Liste verschiedener Primfaktoren.
    """
    if n <= 1:
        return []
    if is_prime_miller_rabin(n):
        return [n]

    factors: set[int] = set()
    stack = [n]

    while stack:
        m = stack.pop()
        if m <= 1:
            continue
        if is_prime_miller_rabin(m):
            factors.add(m)
            continue

        # Trial Division bis 10^6 fuer kleine Faktoren
        divided = False
        for p in range(2, min(1_000_001, m)):
            if p * p > m:
                break
            if m % p == 0:
                factors.add(p)
                while m % p == 0:
                    m //= p
                if m > 1:
                    stack.append(m)
                divided = True
                break

        if not divided:
            # Pollard-Rho fuer grosse zusammengesetzte Reste
            d = _pollard_rho(m)
            stack.append(d)
            stack.append(m // d)

    return sorted(factors)


def euler_phi(n: int, factors: Optional[tuple] = None) -> int:
    """Eulersche Phi-Funktion phi(n).

    Fuer n = p*q (RSA-Modul): phi(n) = (p-1)*(q-1).
    Fuer n = 2^k:              phi(n) = 2^(k-1).
    Allgemein:                 phi(n) via Primfaktorzerlegung.

    Parameters
    ----------
    n       : int            n >= 1.
    factors : tuple, optional
        Bekannte Primfaktoren von n (z.B. (p, q) bei RSA-Moduln).
        Wenn angegeben, wird die teure Faktorisierung uebersprungen.
        Dopplungen sind erlaubt; nur eindeutige Primfaktoren werden genutzt.

    Returns
    -------
    int  phi(n).

    Examples
    --------
    >>> euler_phi(32)
    16
    >>> euler_phi(55)
    40
    >>> euler_phi(55, factors=(5, 11))
    40
    """
    if n < 1:
        raise ValueError(f"phi ist nur fuer n >= 1 definiert, erhalten: {n}")

    # Schnellpfad: Primfaktoren bereits bekannt (z.B. RSA: factors=(p, q))
    if factors is not None:
        result = n
        for p in set(factors):
            result -= result // p
        return result

    # Allgemeiner Pfad: Trial Division bis 10^6, dann Pollard-Rho
    result = n
    temp = n
    p = 2
    while p <= 1_000_000 and p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1

    if temp > 1:
        # temp ist groesser als 10^6 und moeglicherweise zusammengesetzt
        for pf in _distinct_prime_factors(temp):
            result -= result // pf

    return result


def is_prime_miller_rabin(n: int, rounds: int = 20) -> bool:
    """Miller-Rabin-Primzahltest.

    Parameters
    ----------
    n      : int  Zu pruefende Zahl (n >= 2).
    rounds : int  Anzahl zufaelliger Basen (default 20).

    Returns
    -------
    bool  True wenn n wahrscheinlich prim.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # n-1 = 2^r * d schreiben
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(rounds):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int) -> int:
    """Erzeugt eine zufaellige Primzahl der angegebenen Bitlaenge.

    Parameters
    ----------
    bits : int  Mindestbitlaenge der Primzahl (bits >= 4).

    Returns
    -------
    int  Eine wahrscheinliche Primzahl.
    """
    if bits < 4:
        raise ValueError(f"bits muss >= 4 sein, erhalten: {bits}")
    while True:
        candidate = random.getrandbits(bits)
        candidate |= (1 << (bits - 1)) | 1  # MSB setzen, ungerade machen
        if is_prime_miller_rabin(candidate):
            return candidate
