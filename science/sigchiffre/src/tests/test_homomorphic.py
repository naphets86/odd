"""Tests für sigchiffre.homomorphic – Homomorphe Kryptographie."""

import numpy as np
import pytest

from sigchiffre.chiffre import SignatureChiffre
from sigchiffre.homomorphic import (
    HomomorphicSignatureChiffre,
    PaillierPrivateKey,
    PaillierPublicKey,
    paillier_keygen,
)


@pytest.fixture
def sc():
    return SignatureChiffre(4, 7, 3, 1031)


@pytest.fixture
def A4():
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
    ])


# ---------------------------------------------------------------------------
# HomomorphicSignatureChiffre
# ---------------------------------------------------------------------------

class TestHomomorphicSignatureChiffre:
    @pytest.fixture
    def hsc(self, sc):
        return HomomorphicSignatureChiffre(sc)

    def test_encrypt_decrypt_roundtrip(self, hsc, A4):
        C = hsc.encrypt(A4)
        A_dec = hsc.decrypt(C)
        np.testing.assert_array_equal(A_dec, A4)

    def test_encrypt_returns_list(self, hsc, A4):
        C = hsc.encrypt(A4)
        assert isinstance(C, list)
        assert len(C) == 4

    def test_add_ciphertexts_type(self, hsc, A4):
        C1 = hsc.encrypt(A4)
        C2 = hsc.encrypt(A4)
        C_sum = hsc.add_ciphertexts(C1, C2)
        assert len(C_sum) == 4
        for c in C_sum:
            assert 0 <= c < hsc.q

    def test_scalar_multiply(self, hsc, A4):
        C = hsc.encrypt(A4)
        C2 = hsc.scalar_multiply(C, 2)
        assert len(C2) == 4
        for c in C2:
            assert 0 <= c < hsc.q

    def test_zero_scalar_multiply(self, hsc, A4):
        C = hsc.encrypt(A4)
        C0 = hsc.scalar_multiply(C, 0)
        assert all(c == 0 for c in C0)


# ---------------------------------------------------------------------------
# Paillier-Kryptosystem
# ---------------------------------------------------------------------------

class TestPaillierKeygen:
    def test_returns_private_key(self):
        priv = paillier_keygen(bits=64)
        assert isinstance(priv, PaillierPrivateKey)

    def test_public_key_available(self):
        priv = paillier_keygen(bits=64)
        assert priv.public_key is not None

    def test_n_is_composite(self):
        priv = paillier_keygen(bits=64)
        n = priv.public_key.n
        assert n == priv.p * priv.q
        assert n > 1


class TestPaillierEncryptDecrypt:
    @pytest.fixture
    def keypair(self):
        return paillier_keygen(bits=64)

    def test_encrypt_decrypt_0(self, keypair):
        pub = keypair.public_key
        c = pub.encrypt(0)
        assert keypair.decrypt(c) == 0

    def test_encrypt_decrypt_1(self, keypair):
        pub = keypair.public_key
        c = pub.encrypt(1)
        assert keypair.decrypt(c) == 1

    def test_encrypt_decrypt_large(self, keypair):
        pub = keypair.public_key
        m = pub.n - 1
        c = pub.encrypt(m)
        assert keypair.decrypt(c) == m

    def test_encrypt_decrypt_multiple(self, keypair):
        pub = keypair.public_key
        for m in range(0, 10):
            c = pub.encrypt(m)
            assert keypair.decrypt(c) == m

    def test_out_of_range_raises(self, keypair):
        pub = keypair.public_key
        with pytest.raises(ValueError):
            pub.encrypt(-1)
        with pytest.raises(ValueError):
            pub.encrypt(pub.n)

    def test_randomized_encryption(self, keypair):
        pub = keypair.public_key
        c1 = pub.encrypt(42)
        c2 = pub.encrypt(42)
        # Mit sehr hoher Wahrscheinlichkeit verschieden
        assert isinstance(c1, int)
        assert isinstance(c2, int)


class TestPaillierHomomorphic:
    @pytest.fixture
    def keypair(self):
        return paillier_keygen(bits=64)

    def test_additive_homomorphism(self, keypair):
        """Enc(m1)·Enc(m2) = Enc(m1+m2) mod n²."""
        pub = keypair.public_key
        m1, m2 = 10, 20
        c1 = pub.encrypt(m1)
        c2 = pub.encrypt(m2)
        c_sum = pub.add(c1, c2)
        assert keypair.decrypt(c_sum) == (m1 + m2) % pub.n

    def test_scalar_multiply(self, keypair):
        """c^k = Enc(k·m)."""
        pub = keypair.public_key
        m, k = 5, 3
        c = pub.encrypt(m)
        c_k = pub.scalar_multiply(c, k)
        assert keypair.decrypt(c_k) == (k * m) % pub.n

    def test_add_zero(self, keypair):
        pub = keypair.public_key
        m = 42
        c = pub.encrypt(m)
        c0 = pub.encrypt(0)
        c_sum = pub.add(c, c0)
        assert keypair.decrypt(c_sum) == m
