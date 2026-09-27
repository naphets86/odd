"""Tests für sigchiffre.mpc – Shamir Secret Sharing und ThresholdSignatureChiffre."""

import numpy as np
import pytest

from sigchiffre.mpc import ShamirSecretSharing, ThresholdSignatureChiffre


# ---------------------------------------------------------------------------
# ShamirSecretSharing
# ---------------------------------------------------------------------------

class TestShamirSecretSharing:
    @pytest.fixture
    def sss(self):
        return ShamirSecretSharing(t=3, n=5, p=1031)

    def test_split_returns_n_shares(self, sss):
        shares = sss.split(42)
        assert len(shares) == 5

    def test_shares_are_pairs(self, sss):
        shares = sss.split(42)
        for x, y in shares:
            assert isinstance(x, int)
            assert isinstance(y, int)

    def test_reconstruct_from_t_shares(self, sss):
        secret = 42
        shares = sss.split(secret)
        assert sss.reconstruct(shares[:3]) == secret

    def test_reconstruct_from_all_shares(self, sss):
        secret = 100
        shares = sss.split(secret)
        assert sss.reconstruct(shares) == secret

    def test_reconstruct_different_t_subsets(self, sss):
        secret = 77
        shares = sss.split(secret)
        # Verschiedene t-Teilmengen
        for start in range(3):
            subset = shares[start: start + 3]
            assert sss.reconstruct(subset) == secret

    def test_too_few_shares_raises(self, sss):
        shares = sss.split(42)
        with pytest.raises(ValueError, match="Mindestens"):
            sss.reconstruct(shares[:2])

    def test_secret_out_of_range_raises(self, sss):
        with pytest.raises(ValueError):
            sss.split(1031)  # ≥ p

    def test_secret_zero(self, sss):
        shares = sss.split(0)
        assert sss.reconstruct(shares[:3]) == 0

    def test_t_equals_n(self):
        sss = ShamirSecretSharing(t=3, n=3, p=1031)
        shares = sss.split(99)
        assert sss.reconstruct(shares) == 99

    def test_t_gt_n_raises(self):
        with pytest.raises(ValueError, match="Schwellenwert"):
            ShamirSecretSharing(t=4, n=3, p=1031)

    def test_non_prime_p_raises(self):
        with pytest.raises(ValueError, match="Primzahl"):
            ShamirSecretSharing(t=2, n=3, p=100)

    def test_2_3_threshold(self):
        sss = ShamirSecretSharing(t=2, n=3, p=257)
        secret = 123
        shares = sss.split(secret)
        # Jede 2er-Teilmenge
        for i in range(3):
            for j in range(i + 1, 3):
                assert sss.reconstruct([shares[i], shares[j]]) == secret


# ---------------------------------------------------------------------------
# ThresholdSignatureChiffre
# ---------------------------------------------------------------------------

class TestThresholdSignatureChiffre:
    @pytest.fixture
    def A4(self):
        return np.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ])

    @pytest.fixture
    def tsc(self):
        return ThresholdSignatureChiffre(
            n_block=4, a=7, b=3, p=1031, t=2, n_shares=3
        )

    def test_get_share_returns_pair(self, tsc):
        sh_a, sh_b = tsc.get_share(1)
        assert isinstance(sh_a, tuple) and len(sh_a) == 2
        assert isinstance(sh_b, tuple) and len(sh_b) == 2

    def test_decrypt_with_t_shares(self, tsc, A4):
        from sigchiffre.chiffre import SignatureChiffre
        sc = SignatureChiffre(4, 7, 3, 1031)
        C = sc.encrypt(A4)

        # 2 von 3 Shares
        shares_a = [tsc.shares_a[0], tsc.shares_a[1]]
        shares_b = [tsc.shares_b[0], tsc.shares_b[1]]
        A_dec = tsc.decrypt(C, shares_a, shares_b)
        np.testing.assert_array_equal(A_dec, A4)

    def test_decrypt_with_all_shares(self, tsc, A4):
        from sigchiffre.chiffre import SignatureChiffre
        sc = SignatureChiffre(4, 7, 3, 1031)
        C = sc.encrypt(A4)

        A_dec = tsc.decrypt(C, tsc.shares_a, tsc.shares_b)
        np.testing.assert_array_equal(A_dec, A4)

    def test_decrypt_with_different_t_subsets(self, tsc, A4):
        from sigchiffre.chiffre import SignatureChiffre
        sc = SignatureChiffre(4, 7, 3, 1031)
        C = sc.encrypt(A4)

        # Andere 2er-Teilmenge
        A_dec = tsc.decrypt(
            C,
            [tsc.shares_a[0], tsc.shares_a[2]],
            [tsc.shares_b[0], tsc.shares_b[2]],
        )
        np.testing.assert_array_equal(A_dec, A4)
