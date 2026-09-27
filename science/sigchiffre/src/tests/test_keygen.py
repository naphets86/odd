"""Tests für sigchiffre.keygen – Primzahlfunktionen und Schlüsselgenerierung."""

import pytest

from sigchiffre.keygen import (
    effective_key_bits,
    generate_key,
    generate_prime,
    is_prime,
    key_space_size,
    next_prime,
    validate_key,
)


# ---------------------------------------------------------------------------
# is_prime
# ---------------------------------------------------------------------------

class TestIsPrime:
    def test_small_primes(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in primes:
            assert is_prime(p), f"{p} sollte prim sein"

    def test_composites(self):
        composites = [0, 1, 4, 6, 8, 9, 10, 12, 15, 25, 100]
        for n in composites:
            assert not is_prime(n), f"{n} sollte nicht prim sein"

    def test_known_large_prime(self):
        assert is_prime(1031)
        assert is_prime(257)  # Fermat-Primzahl aus Kapitel 7

    def test_fermat_prime_257(self):
        """257 = 2^8 + 1 ist eine Fermat-Primzahl."""
        assert is_prime(257)

    def test_large_composite(self):
        assert not is_prime(1033 * 1039)

    def test_edge_cases(self):
        assert not is_prime(0)
        assert not is_prime(1)
        assert is_prime(2)
        assert is_prime(3)
        assert not is_prime(4)


# ---------------------------------------------------------------------------
# next_prime
# ---------------------------------------------------------------------------

class TestNextPrime:
    def test_from_2(self):
        assert next_prime(2) == 2

    def test_from_composite(self):
        assert next_prime(4) == 5

    def test_from_prime(self):
        assert next_prime(17) == 17

    def test_after_256(self):
        p = next_prime(257)
        assert p == 257
        assert is_prime(p)

    def test_monotone(self):
        p = next_prime(100)
        assert p >= 100
        assert is_prime(p)


# ---------------------------------------------------------------------------
# generate_prime
# ---------------------------------------------------------------------------

class TestGeneratePrime:
    def test_result_is_prime(self):
        for _ in range(5):
            p = generate_prime(64)
            assert is_prime(p), f"{p} ist keine Primzahl"

    def test_min_value_respected(self):
        min_val = 1000
        for _ in range(3):
            p = generate_prime(16, min_value=min_val)
            assert p >= min_val

    def test_bit_length_approx(self):
        for _ in range(3):
            p = generate_prime(32)
            # Mindestens 31 Bits (oberstes Bit gesetzt)
            assert p.bit_length() >= 31


# ---------------------------------------------------------------------------
# validate_key
# ---------------------------------------------------------------------------

class TestValidateKey:
    def test_valid_key_ch6(self):
        # Beispiel Kapitel 6
        validate_key(4, 7, 3, 1031)  # kein Fehler erwartet

    def test_valid_key_ch7(self):
        # Beispiel Kapitel 7
        validate_key(4, 17, 16, 257)

    def test_p_not_prime(self):
        with pytest.raises(ValueError, match="Primzahl"):
            validate_key(4, 7, 3, 1000)  # 1000 nicht prim

    def test_p_too_small(self):
        # p=251 ist prim aber 251 < 256 = 2^(2*4)
        with pytest.raises(ValueError):
            validate_key(4, 7, 3, 251)

    def test_p_equals_bound(self):
        # p muss strikt > 2^(2n) sein
        with pytest.raises(ValueError):
            validate_key(4, 7, 3, 256)

    def test_a_out_of_range_low(self):
        with pytest.raises(ValueError):
            validate_key(4, 0, 3, 1031)

    def test_a_out_of_range_high(self):
        with pytest.raises(ValueError):
            validate_key(4, 1031, 3, 1031)

    def test_b_negative(self):
        with pytest.raises(ValueError):
            validate_key(4, 7, -1, 1031)

    def test_b_too_large(self):
        with pytest.raises(ValueError):
            validate_key(4, 7, 1031, 1031)

    def test_gcd_not_1(self):
        # a muss teilerfremd zu p sein; da p prim, ist gcd(a,p)=1 für a∈{1..p-1}
        # Dieser Fall kann nur bei nicht-prim p auftreten (wird vorher abgefangen)
        # Test: a=0 verletzt gcd-Bedingung (via Bereichsfehler)
        with pytest.raises(ValueError):
            validate_key(4, 0, 3, 1031)


# ---------------------------------------------------------------------------
# generate_key
# ---------------------------------------------------------------------------

class TestGenerateKey:
    def test_returns_triple(self):
        a, b, p = generate_key(4, bits=32)
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert isinstance(p, int)

    def test_p_is_prime(self):
        for _ in range(3):
            a, b, p = generate_key(4, bits=32)
            assert is_prime(p)

    def test_p_gt_2_2n(self):
        for _ in range(3):
            a, b, p = generate_key(4, bits=32)
            assert p > (1 << 8)

    def test_a_in_range(self):
        for _ in range(5):
            a, b, p = generate_key(4, bits=32)
            assert 1 <= a < p

    def test_b_in_range(self):
        for _ in range(5):
            a, b, p = generate_key(4, bits=32)
            assert 0 <= b < p

    def test_key_validates(self):
        for _ in range(3):
            a, b, p = generate_key(4, bits=32)
            validate_key(4, a, b, p)  # kein Fehler erwartet

    def test_bits_too_small_raises(self):
        with pytest.raises(ValueError, match="bits"):
            generate_key(10, bits=5)  # 5 < 2·10+1 = 21


# ---------------------------------------------------------------------------
# key_space_size / effective_key_bits
# ---------------------------------------------------------------------------

class TestKeySpace:
    def test_size_formula(self):
        p = 257
        assert key_space_size(p) == (p - 1) * p

    def test_size_1031(self):
        assert key_space_size(1031) == 1030 * 1031

    def test_effective_bits_positive(self):
        bits = effective_key_bits(1031)
        assert bits > 0

    def test_effective_bits_approx_2_log2p(self):
        from math import log2
        p = 1031
        expected = 2 * log2(p)
        actual = effective_key_bits(p)
        assert abs(actual - expected) < 1.0  # Toleranz: 1 Bit

    def test_256bit_prime_gives_512bit_keyspace(self):
        # Für p ≈ 2^256 soll |K| ≈ 2^512
        bits = effective_key_bits(2**256 - 189)  # bekannte 256-Bit-Primzahl-Näherung
        assert 510 < bits < 514
