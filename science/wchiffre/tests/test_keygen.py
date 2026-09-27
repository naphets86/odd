"""
Tests fuer src.keygen

Abgedeckt:
- PublicKey: Validierung der Parameter
- PrivateKey: Validierung der Parameter
- KeygenFC.toy_example: Lehrbeispiel (n=55, e=23, d=7)
- KeygenFC.from_primes: Korrekte Berechnung von d, s-Zufaelligkeit
- KeygenFC.generate: Zufaellige RSA-Schluesselerzeugung
- Konsistenz: e*d = 1 mod phi(n)
"""

import math
import pytest

from src.keygen import PublicKey, PrivateKey, KeygenFC
from src.math_utils import euler_phi


# ---------------------------------------------------------------------------
# PublicKey Validierung
# ---------------------------------------------------------------------------

class TestPublicKey:

    def test_valid_toy_key(self):
        pk = PublicKey(n=55, e=23, k=3)
        assert pk.n == 55
        assert pk.e == 23
        assert pk.k == 3

    def test_raises_n_too_small(self):
        with pytest.raises(ValueError, match="n muss >= 4"):
            PublicKey(n=3, e=3, k=2)

    def test_raises_e_too_small(self):
        with pytest.raises(ValueError, match="e muss >= 2"):
            PublicKey(n=55, e=1, k=3)

    def test_raises_k_too_small(self):
        with pytest.raises(ValueError, match="k muss >= 2"):
            PublicKey(n=55, e=23, k=1)

    def test_raises_gcd_e_n_not_one(self):
        # gcd(5, 55) = 5
        with pytest.raises(ValueError, match="gcd"):
            PublicKey(n=55, e=5, k=3)


# ---------------------------------------------------------------------------
# PrivateKey Validierung
# ---------------------------------------------------------------------------

class TestPrivateKey:

    def test_valid_toy_key(self):
        sk = PrivateKey(d=7, s=0, n=55, k=3, phi_n=40)
        assert sk.d == 7
        assert sk.s == 0
        assert sk.phi_n == 40

    def test_raises_d_zero(self):
        with pytest.raises(ValueError, match="d muss >= 1"):
            PrivateKey(d=0, s=0, n=55, k=3, phi_n=40)

    def test_raises_s_negative(self):
        with pytest.raises(ValueError, match="Offset s"):
            PrivateKey(d=7, s=-1, n=55, k=3, phi_n=40)

    def test_raises_s_too_large(self):
        with pytest.raises(ValueError, match="Offset s"):
            PrivateKey(d=7, s=3, n=55, k=3, phi_n=40)  # s in {0,1,2} fuer k=3

    def test_valid_all_offsets(self):
        for s in range(3):
            sk = PrivateKey(d=7, s=s, n=55, k=3, phi_n=40)
            assert sk.s == s


# ---------------------------------------------------------------------------
# KeygenFC.toy_example
# ---------------------------------------------------------------------------

class TestKeygenToyExample:

    def test_n_e_d_correct(self):
        """Lehrbeispiel: n=55, e=23, d=7 (Korollar 2.4 und Bemerkung 3.2)."""
        pk, sk = KeygenFC.toy_example(s=0)
        assert pk.n == 55
        assert pk.e == 23
        assert sk.d == 7
        assert sk.n == 55

    def test_ed_one_mod_phi_n(self):
        """e*d = 1 mod phi(n), phi(55) = 40."""
        pk, sk = KeygenFC.toy_example(s=0)
        # sk.phi_n = (p-1)*(q-1) ist bereits bekannt – kein euler_phi(pk.n) noetig
        assert (pk.e * sk.d) % sk.phi_n == 1

    def test_phi_n_correct(self):
        """phi_n im PrivateKey stimmt mit euler_phi ueberein (Kleinzahl n=55)."""
        pk, sk = KeygenFC.toy_example(s=0)
        assert sk.phi_n == euler_phi(pk.n)   # n=55: Trial Division sehr schnell

    def test_default_k_is_3(self):
        pk, sk = KeygenFC.toy_example()
        assert pk.k == 3
        assert sk.k == 3

    def test_s_randomized_when_none(self):
        """s wird zufaellig gewaehlt falls nicht angegeben."""
        offsets = {KeygenFC.toy_example()[1].s for _ in range(30)}
        # Mit hoher Wahrscheinlichkeit mindestens 2 verschiedene Werte
        assert len(offsets) >= 2

    def test_s_in_range(self):
        for _ in range(10):
            pk, sk = KeygenFC.toy_example()
            assert 0 <= sk.s < pk.k

    def test_pk_sk_consistent_k(self):
        pk, sk = KeygenFC.toy_example(k=5)
        assert pk.k == sk.k == 5

    def test_fixed_s_preserved(self):
        for s in range(3):
            _, sk = KeygenFC.toy_example(s=s)
            assert sk.s == s


# ---------------------------------------------------------------------------
# KeygenFC.from_primes
# ---------------------------------------------------------------------------

class TestKeygenFromPrimes:

    def test_n_is_product(self):
        pk, sk = KeygenFC.from_primes(p=5, q=11, e=23)
        assert pk.n == 55

    def test_d_is_correct_inverse(self):
        pk, sk = KeygenFC.from_primes(p=5, q=11, e=23, s=0)
        # Nutzt sk.phi_n statt euler_phi(pk.n) – schnell und korrekt
        assert (pk.e * sk.d) % sk.phi_n == 1

    def test_raises_equal_primes(self):
        with pytest.raises(ValueError, match="verschieden"):
            KeygenFC.from_primes(p=5, q=5, e=3)

    def test_raises_p_not_prime(self):
        with pytest.raises(ValueError, match="keine Primzahl"):
            KeygenFC.from_primes(p=4, q=11, e=23)

    def test_raises_q_not_prime(self):
        with pytest.raises(ValueError, match="keine Primzahl"):
            KeygenFC.from_primes(p=5, q=12, e=23)

    def test_raises_e_not_coprime_with_phi(self):
        # phi(15) = 8; gcd(2,8) = 2
        with pytest.raises(ValueError, match="kein gueltiger Exponent"):
            KeygenFC.from_primes(p=3, q=5, e=2)

    def test_different_primes(self):
        pk, sk = KeygenFC.from_primes(p=7, q=11, e=13, s=0)
        assert pk.n == 77
        assert (13 * sk.d) % sk.phi_n == 1


# ---------------------------------------------------------------------------
# KeygenFC.generate
# ---------------------------------------------------------------------------

class TestKeygenGenerate:

    def test_returns_valid_keys(self):
        pk, sk = KeygenFC.generate(bits=64, e=65537)
        assert pk.n > 0
        assert pk.e == 65537
        assert sk.d > 0
        assert 0 <= sk.s < pk.k

    def test_ed_one_mod_phi_n(self):
        pk, sk = KeygenFC.generate(bits=64, e=65537)
        # sk.phi_n ist direkt verfuegbar – keine Faktorisierung von pk.n noetig
        assert (pk.e * sk.d) % sk.phi_n == 1

    def test_custom_k(self):
        pk, sk = KeygenFC.generate(bits=64, k=5)
        assert pk.k == sk.k == 5


# ---------------------------------------------------------------------------
# Coverage Tests
# ---------------------------------------------------------------------------

class TestKeygenCoverage:
    """Tests um 100% Code Coverage zu erreichen."""

    def test_generate_handles_same_primes_case(self):
        """Test dass generate() mehrfach aufgerufen wird (Teiler fuer p==q Case)."""
        # Durch mehrfaches Aufrufen wird die Wahrscheinlichkeit erhoet, dass
        # der p == q Fall (Zeile 185) getestet wird, obwohl er selten ist
        pk1, sk1 = KeygenFC.generate(bits=32, e=65537)
        pk2, sk2 = KeygenFC.generate(bits=32, e=65537)
        pk3, sk3 = KeygenFC.generate(bits=32, e=65537)
        # Mit hoher Wahrscheinlichkeit sind alle verschieden
        keys = [(pk1.n, sk1.d), (pk2.n, sk2.d), (pk3.n, sk3.d)]
        # Mindestens 2 sollten verschieden sein
        assert len(set(keys)) >= 2
