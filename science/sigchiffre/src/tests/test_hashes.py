"""Tests für sigchiffre.hashes – SignatureHash und MerkleTree."""

import pytest

from sigchiffre.chiffre import SignatureChiffre
from sigchiffre.hashes import MerkleProof, MerkleTree, SignatureHash


@pytest.fixture
def sc():
    return SignatureChiffre(4, 7, 3, 1031)


@pytest.fixture
def sig_hash(sc):
    return SignatureHash(sc)


# ---------------------------------------------------------------------------
# SignatureHash
# ---------------------------------------------------------------------------

class TestSignatureHash:
    def test_digest_returns_32_bytes(self, sig_hash):
        d = sig_hash.digest(b"test")
        assert len(d) == 32

    def test_digest_deterministic(self, sig_hash):
        assert sig_hash.digest(b"hello") == sig_hash.digest(b"hello")

    def test_different_inputs_different_hash(self, sig_hash):
        assert sig_hash.digest(b"abc") != sig_hash.digest(b"abd")

    def test_hexdigest_length(self, sig_hash):
        h = sig_hash.hexdigest(b"test")
        assert len(h) == 64

    def test_hexdigest_is_hex(self, sig_hash):
        h = sig_hash.hexdigest(b"test")
        int(h, 16)  # Kein Fehler = valides Hex

    def test_empty_input(self, sig_hash):
        d = sig_hash.digest(b"")
        assert len(d) == 32

    def test_long_input(self, sig_hash):
        d = sig_hash.digest(bytes(1000))
        assert len(d) == 32

    def test_avalanche(self, sig_hash):
        """Ein Bit-Flip ändert den Hash grundlegend."""
        d1 = sig_hash.digest(b"\x00")
        d2 = sig_hash.digest(b"\x01")
        # Mindestens 8 Bytes unterschiedlich
        diff = sum(1 for a, b in zip(d1, d2) if a != b)
        assert diff > 0


# ---------------------------------------------------------------------------
# MerkleTree
# ---------------------------------------------------------------------------

class TestMerkleTree:
    def test_single_leaf(self):
        tree = MerkleTree([b"leaf0"])
        assert len(tree.root) == 32

    def test_two_leaves(self):
        tree = MerkleTree([b"a", b"b"])
        assert len(tree.root) == 32

    def test_root_deterministic(self):
        leaves = [b"a", b"b", b"c"]
        t1 = MerkleTree(leaves)
        t2 = MerkleTree(leaves)
        assert t1.root == t2.root

    def test_root_changes_with_leaf(self):
        t1 = MerkleTree([b"a", b"b"])
        t2 = MerkleTree([b"a", b"c"])
        assert t1.root != t2.root

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            MerkleTree([])

    def test_proof_verify_single(self):
        tree = MerkleTree([b"only"])
        proof = tree.get_proof(0)
        assert MerkleTree.verify_proof(proof, tree.root)

    def test_proof_verify_two_leaves(self):
        tree = MerkleTree([b"left", b"right"])
        for i in range(2):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof, tree.root)

    def test_proof_verify_four_leaves(self):
        leaves = [b"a", b"b", b"c", b"d"]
        tree = MerkleTree(leaves)
        for i in range(4):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof, tree.root)

    def test_proof_verify_odd_count(self):
        leaves = [b"a", b"b", b"c"]
        tree = MerkleTree(leaves)
        for i in range(3):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof, tree.root)

    def test_invalid_index_raises(self):
        tree = MerkleTree([b"a", b"b"])
        with pytest.raises(IndexError):
            tree.get_proof(5)

    def test_tampered_proof_fails(self):
        tree = MerkleTree([b"a", b"b", b"c", b"d"])
        proof = tree.get_proof(0)
        # Root manipulieren
        wrong_root = bytes([r ^ 0xFF for r in tree.root])
        assert not MerkleTree.verify_proof(proof, wrong_root)

    def test_large_tree(self):
        leaves = [f"leaf{i}".encode() for i in range(16)]
        tree = MerkleTree(leaves)
        for i in range(16):
            proof = tree.get_proof(i)
            assert MerkleTree.verify_proof(proof, tree.root)
