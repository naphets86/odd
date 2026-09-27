"""
Tests fuer src.math_utils

Abgedeckt:
- extended_gcd: Basisfaelle, Bezout-Koeffizienten, Symmetrie
- mod_inverse: Lehrbeispiel 23^-1=7 mod 32, 7^-1=23 mod 32, Fehlerfall
- euler_phi: phi(32)=16, phi(55)=40, phi(1)=1, phi(prime)
- is_prime_miller_rabin: bekannte Primzahlen und Nichtprimzahlen
- generate_prime: Bitlaenge und Primalitaet
"""

import math
import pytest

from src.math_utils import (
    extended_gcd,
    mod_inverse,
    euler_phi,
    is_prime_miller_rabin,
    generate_prime,
)


# ---------------------------------------------------------------------------
# extended_gcd
# ---------------------------------------------------------------------------

class TestExtendedGcd:

    def test_gcd_32_23_gives_bezout(self):
        """Lemma 2.3 der Arbeit: 7*23 - 5*32 = 1."""
        g, x, y = extended_gcd(32, 23)
        assert g == 1
        assert 32 * x + 23 * y == 1

    def test_gcd_value_matches_math_gcd(self):
        for a, b in [(48, 18), (100, 75), (17, 13), (1024, 768)]:
            g, x, y = extended_gcd(a, b)
            assert g == math.gcd(a, b)
            assert a * x + b * y == g

    def test_gcd_with_zero_b(self):
        g, x, y = extended_gcd(7, 0)
        assert g == 7
        assert x == 1
        assert y == 0

    def test_gcd_with_zero_a(self):
        g, x, y = extended_gcd(0, 5)
        assert g == 5

    def test_gcd_coprime_pair(self):
        g, x, y = extended_gcd(23, 32)
        assert g == 1
        assert 23 * x + 32 * y == 1

    def test_bezout_identity_holds_for_large(self):
        a, b = 1234567, 7654321
        g, x, y = extended_gcd(a, b)
        assert g == math.gcd(a, b)
        assert a * x + b * y == g


# ---------------------------------------------------------------------------
# mod_inverse
# ---------------------------------------------------------------------------

class TestModInverse:

    def test_lehrbespiel_23_inverse_mod_32(self):
        """Satz 2.2: 23^-1 = 7 mod 32."""
        assert mod_inverse(23, 32) == 7

    def test_lehrbespiel_7_inverse_mod_32(self):
        """Satz 2.2: 7^-1 = 23 mod 32."""
        assert mod_inverse(7, 32) == 23

    def test_product_is_one(self):
        """23 * 7 = 1 mod 32."""
        e, d, n = 23, mod_inverse(23, 32), 32
        assert (e * d) % n == 1

    def test_rsa_toy_example(self):
        """Lehrbeispiel n=55: e=23, phi=40, d=7."""
        d = mod_inverse(23, 40)
        assert d == 7
        assert (23 * d) % 40 == 1

    def test_inverse_of_inverse_is_original(self):
        for a, n in [(3, 11), (5, 17), (23, 32), (7, 40)]:
            inv = mod_inverse(a, n)
            assert mod_inverse(inv, n) == a % n

    def test_raises_when_not_coprime(self):
        with pytest.raises(ValueError, match="Kein Inverses"):
            mod_inverse(4, 8)  # gcd(4,8)=4

    def test_raises_gcd_not_one(self):
        with pytest.raises(ValueError):
            mod_inverse(6, 9)  # gcd(6,9)=3

    def test_result_in_range(self):
        for a, n in [(3, 7), (11, 100), (23, 32)]:
            inv = mod_inverse(a, n)
            assert 0 <= inv < n


# ---------------------------------------------------------------------------
# euler_phi
# ---------------------------------------------------------------------------

class TestEulerPhi:

    def test_phi_32_is_16(self):
        """Satz 2.1: phi(32) = 16 (2^5, Primzahlpotenz)."""
        assert euler_phi(32) == 16

    def test_phi_55_is_40(self):
        """phi(55) = phi(5)*phi(11) = 4*10 = 40."""
        assert euler_phi(55) == 40

    def test_phi_1(self):
        assert euler_phi(1) == 1

    def test_phi_prime(self):
        """phi(p) = p-1 fuer Primzahlen."""
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            assert euler_phi(p) == p - 1

    def test_phi_prime_squared(self):
        """phi(p^2) = p^2 - p."""
        assert euler_phi(9) == 6    # phi(3^2) = 9-3
        assert euler_phi(25) == 20  # phi(5^2) = 25-5
        assert euler_phi(49) == 42  # phi(7^2) = 49-7

    def test_phi_powers_of_two(self):
        """phi(2^k) = 2^(k-1)."""
        for k in range(2, 10):
            assert euler_phi(2**k) == 2**(k-1)

    def test_phi_invalid(self):
        with pytest.raises(ValueError):
            euler_phi(0)

    def test_phi_multiplicative(self):
        """phi(p*q) = (p-1)*(q-1) fuer verschiedene Primzahlen."""
        p, q = 5, 11
        assert euler_phi(p * q) == (p - 1) * (q - 1)


# ---------------------------------------------------------------------------
# euler_phi – Kapitel 9: Drei-Schichten-Architektur
# ---------------------------------------------------------------------------

class TestEulerPhiSchicht1Schnellpfad:
    """Schicht 1 (Proposition 9.1): Schnellpfad bei bekannten Primfaktoren.

    euler_phi(n, factors=(p, q)) muss phi(n) ohne Faktorisierung berechnen.
    """

    def test_schnellpfad_toy_example(self):
        """Korollar 9.1 (RSA): phi(55) = (5-1)*(11-1) = 40 via factors=(5,11)."""
        assert euler_phi(55, factors=(5, 11)) == 40

    def test_schnellpfad_stimmt_mit_allgemeinem_pfad_ueberein(self):
        """Schnellpfad und allgemeiner Pfad liefern dasselbe Ergebnis."""
        for p, q in [(5, 11), (7, 13), (11, 23), (17, 19), (3, 97)]:
            n = p * q
            assert euler_phi(n, factors=(p, q)) == euler_phi(n)

    def test_schnellpfad_duplikat_faktoren_ignoriert(self):
        """Bemerkung Kap. 9: set(factors) entfernt Dopplungen – factors=(p,p,q) ok."""
        assert euler_phi(55, factors=(5, 5, 11)) == 40

    def test_schnellpfad_drei_primfaktoren(self):
        """Produktformel phi(30) = 30*(1-1/2)*(1-1/3)*(1-1/5) = 8."""
        n = 2 * 3 * 5  # = 30
        assert euler_phi(n, factors=(2, 3, 5)) == 8

    def test_schnellpfad_primzahlpotenz(self):
        """phi(2^5) = 16; Schnellpfad mit factors=(2,) liefert dasselbe."""
        assert euler_phi(32, factors=(2,)) == 16

    def test_schnellpfad_grosses_rsa_modul(self):
        """Korollar 9.1 fuer groessere Primzahlen: phi(p*q) = (p-1)*(q-1)."""
        p, q = 1009, 1013   # beide prim, > 10^3
        n = p * q
        expected = (p - 1) * (q - 1)
        assert euler_phi(n, factors=(p, q)) == expected

    def test_schnellpfad_ist_konsistent_mit_phi_n_im_private_key(self):
        """Korollar Testinvariante (Satz 9.5): (e*d) mod phi_n == 1."""
        import math
        p, q, e = 5, 11, 23
        phi_n = euler_phi(p * q, factors=(p, q))
        d = mod_inverse(e, phi_n)
        assert (e * d) % phi_n == 1


class TestEulerPhiSchicht2Probedivision:
    """Schicht 2: Probedivision bis B = 10^6 (Lemma 9.2, Lemma 9.3)."""

    def test_kleine_primfaktoren_werden_gefunden(self):
        """Alle Primfaktoren <= 10^6 muessen durch Probedivision gefunden werden."""
        # n = 2^3 * 3 * 7 = 168; phi = 168*(1/2)*(2/3)*(6/7) = 48
        assert euler_phi(168) == 48

    def test_phi_n_gleich_eins(self):
        """phi(1) = 1 (Randbedingung, kein Primfaktor)."""
        assert euler_phi(1) == 1

    def test_phi_semiprime_kleine_faktoren(self):
        """Semiprimes mit Faktoren < 10^6 werden durch Schicht 2 abgedeckt."""
        # n = 997 * 991  (beide prim, < 10^6)
        p, q = 997, 991
        expected = (p - 1) * (q - 1)
        assert euler_phi(p * q) == expected

    def test_phi_produkt_kleiner_primzahlen(self):
        """phi(2*3*5*7*11*13) = 1440 (bekanntes Ergebnis aus Produktformel)."""
        n = 2 * 3 * 5 * 7 * 11 * 13   # = 30030
        expected = 1 * 2 * 4 * 6 * 10 * 12  # = 5760; aus (p-1) fuer jeden Faktor
        assert euler_phi(n) == expected

    def test_phi_primzahlpotenz_hoch_3(self):
        """phi(p^3) = p^3 - p^2 (Primzahlpotenzformel Satz 9.1)."""
        for p in [3, 5, 7, 11]:
            assert euler_phi(p**3) == p**3 - p**2


class TestEulerPhiSchicht3PollardRho:
    """Schicht 3: Pollard-Rho fuer Primfaktoren > 10^6 (Satz 9.3, 9.4)."""

    def test_semiprime_grosser_faktoren(self):
        """Satz 9.4: RSA-Modul mit p,q > 10^6 – Pollard-Rho uebernimmt.

        phi(p*q) wird korrekt berechnet, ohne factors-Schnellpfad.
        """
        p, q = 1_000_003, 1_000_033   # beide prim (Miller-Rabin geprueft)
        n = p * q
        expected = (p - 1) * (q - 1)
        assert euler_phi(n) == expected

    def test_allgemeiner_pfad_gleich_schnellpfad_fuer_grosse_n(self):
        """Korrektheit (Satz 9.2): Schicht 2+3 stimmt mit Schnellpfad ueberein."""
        p, q = 1_000_003, 1_000_033
        n = p * q
        assert euler_phi(n) == euler_phi(n, factors=(p, q))


# ---------------------------------------------------------------------------
# is_prime_miller_rabin
# ---------------------------------------------------------------------------

class TestIsPrimeMillerRabin:

    KNOWN_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
    KNOWN_COMPOSITES = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24,
                        25, 26, 27, 28, 50, 100, 1000]

    def test_known_primes(self):
        for p in self.KNOWN_PRIMES:
            assert is_prime_miller_rabin(p), f"{p} sollte prim sein"

    def test_known_composites(self):
        for c in self.KNOWN_COMPOSITES:
            assert not is_prime_miller_rabin(c), f"{c} sollte nicht prim sein"

    def test_returns_false_for_0_and_1(self):
        assert not is_prime_miller_rabin(0)
        assert not is_prime_miller_rabin(1)

    def test_large_prime(self):
        # Mersenne-Primzahl 2^31 - 1
        assert is_prime_miller_rabin(2**31 - 1)


# ---------------------------------------------------------------------------
# generate_prime
# ---------------------------------------------------------------------------

class TestGeneratePrime:

    def test_is_prime(self):
        for _ in range(5):
            p = generate_prime(16)
            assert is_prime_miller_rabin(p)

    def test_bit_length(self):
        for bits in [8, 16, 32]:
            p = generate_prime(bits)
            assert p.bit_length() == bits

    def test_is_odd(self):
        for _ in range(5):
            p = generate_prime(16)
            assert p % 2 == 1

    def test_raises_for_small_bits(self):
        with pytest.raises(ValueError):
            generate_prime(3)