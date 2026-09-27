"""Tests für sigchiffre.lwe – Learning With Errors."""

import numpy as np
import pytest

from sigchiffre.lwe import LWEParams, LWEPrivateKey, LWEPublicKey, lwe_keygen


class TestLWEKeygen:
    @pytest.fixture
    def params(self):
        return LWEParams(n=8, m=32, q=97)

    def test_returns_private_key(self, params):
        priv = lwe_keygen(params)
        assert isinstance(priv, LWEPrivateKey)

    def test_public_key_available(self, params):
        priv = lwe_keygen(params)
        assert priv.public_key is not None

    def test_matrix_shape(self, params):
        priv = lwe_keygen(params)
        pub = priv.public_key
        assert pub.A.shape == (params.m, params.n)

    def test_b_shape(self, params):
        priv = lwe_keygen(params)
        pub = priv.public_key
        assert pub.b.shape == (params.m,)


class TestLWEEncryptDecrypt:
    @pytest.fixture
    def keypair(self):
        params = LWEParams(n=8, m=32, q=97, sigma=1.0)
        return lwe_keygen(params)

    def test_encrypt_bit_0(self, keypair):
        pub = keypair.public_key
        c0, c1 = pub.encrypt(0)
        assert isinstance(c0, np.ndarray)
        assert isinstance(c1, int)

    def test_encrypt_bit_1(self, keypair):
        pub = keypair.public_key
        c0, c1 = pub.encrypt(1)
        assert isinstance(c0, np.ndarray)
        assert isinstance(c1, int)

    def test_invalid_bit_raises(self, keypair):
        pub = keypair.public_key
        with pytest.raises(ValueError):
            pub.encrypt(2)

    def test_decrypt_bit_0_multiple(self, keypair):
        pub = keypair.public_key
        successes = 0
        for _ in range(20):
            c0, c1 = pub.encrypt(0)
            if keypair.decrypt(c0, c1) == 0:
                successes += 1
        # Mindestens 80% korrekt (LWE mit kleinen Fehlern)
        assert successes >= 16

    def test_decrypt_bit_1_multiple(self, keypair):
        pub = keypair.public_key
        successes = 0
        for _ in range(20):
            c0, c1 = pub.encrypt(1)
            if keypair.decrypt(c0, c1) == 1:
                successes += 1
        assert successes >= 16

    def test_c0_shape(self, keypair):
        pub = keypair.public_key
        c0, _ = pub.encrypt(0)
        assert c0.shape == (pub.params.n,)

    def test_c1_in_range(self, keypair):
        pub = keypair.public_key
        _, c1 = pub.encrypt(0)
        assert 0 <= c1 < pub.params.q


class TestLWEParams:
    def test_default_sigma(self):
        p = LWEParams(n=4, m=16, q=37)
        assert p.sigma == 3.2

    def test_custom_sigma(self):
        p = LWEParams(n=4, m=16, q=37, sigma=1.0)
        assert p.sigma == 1.0


# ---------------------------------------------------------------------------
# LWESignatureChiffre (hybride Variante)
# ---------------------------------------------------------------------------

class TestLWESignatureChiffre:
    from sigchiffre.lwe import LWESignatureChiffre

    @pytest.fixture
    def lwe_sc(self):
        from sigchiffre.lwe import LWEParams, LWESignatureChiffre, lwe_keygen
        params = LWEParams(n=8, m=32, q=97, sigma=0.5)
        priv = lwe_keygen(params)
        return LWESignatureChiffre(n_block=4, lwe_priv=priv, sig_bits=8)

    @pytest.fixture
    def A4(self):
        return np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ])

    def test_encrypt_returns_list(self, lwe_sc, A4):
        C = lwe_sc.encrypt(A4)
        assert len(C) == 4

    def test_encrypt_each_column_has_bits(self, lwe_sc, A4):
        C = lwe_sc.encrypt(A4)
        for bit_enc in C:
            assert len(bit_enc) == 8  # sig_bits=8

    def test_decrypt_returns_matrix(self, lwe_sc, A4):
        C = lwe_sc.encrypt(A4)
        A_dec = lwe_sc.decrypt(C)
        assert A_dec.shape == (4, 4)

    def test_roundtrip_success_rate(self, A4):
        """LWE-Roundtrip mit sehr kleinen Fehlern (σ=0.1) → hohe Erfolgsrate."""
        from sigchiffre.lwe import LWEParams, LWESignatureChiffre, lwe_keygen
        successes = 0
        for _ in range(10):
            params = LWEParams(n=8, m=32, q=97, sigma=0.1)
            priv = lwe_keygen(params)
            sc = LWESignatureChiffre(n_block=4, lwe_priv=priv, sig_bits=8)
            C = sc.encrypt(A4)
            A_dec = sc.decrypt(C)
            if np.array_equal(A_dec, A4):
                successes += 1
        assert successes >= 5  # Mindestens 50% korrekt
