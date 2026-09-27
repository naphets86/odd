"""
Tests fuer src.encrypt

Abgedeckt:
- message_to_blocks: Blockzerlegung, Padding, Randfaelle
- blocks_to_message: Inverse der Blockzerlegung
- encrypt: Paketlaenge, Dummy-Positionen, RSA-Verschluesselung
- count_dummy_positions: Indexberechnung fuer alle Offsets
- Invariante: echte Pakete in richtiger Reihenfolge
"""

import math
import pytest

from src.keygen import PublicKey, KeygenFC
from src.encrypt import (
    message_to_blocks,
    blocks_to_message,
    encrypt,
    count_dummy_positions,
    _block_byte_length,
)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def toy_pk(k: int = 3) -> PublicKey:
    pk, _ = KeygenFC.toy_example(k=k, s=0)
    return pk


# ---------------------------------------------------------------------------
# message_to_blocks
# ---------------------------------------------------------------------------

class TestMessageToBlocks:

    def test_single_byte(self):
        n = 55
        blocks = message_to_blocks(b"\x05", n)
        assert len(blocks) >= 1
        assert all(b < n for b in blocks)

    def test_all_blocks_lt_n(self):
        n = 55
        for msg in [b"A", b"Hello", b"\x00\x01\x02"]:
            blocks = message_to_blocks(msg, n)
            assert all(b < n for b in blocks), f"Block >= n fuer Nachricht {msg!r}"

    def test_empty_message(self):
        blocks = message_to_blocks(b"", 55)
        assert isinstance(blocks, list)
        # Auch leere Nachricht erzeugt mindestens einen (Padding-)Block
        assert len(blocks) >= 1

    def test_roundtrip_empty(self):
        n = 55
        original = b""
        blocks = message_to_blocks(original, n)
        recovered = blocks_to_message(blocks, n)
        assert recovered == original

    def test_roundtrip_single_byte(self):
        n = 55
        original = b"\x02"
        blocks = message_to_blocks(original, n)
        recovered = blocks_to_message(blocks, n)
        assert recovered == original

    def test_roundtrip_ascii(self):
        n = 55
        original = b"Hi"
        blocks = message_to_blocks(original, n)
        recovered = blocks_to_message(blocks, n)
        assert recovered == original

    def test_block_count_grows_with_message(self):
        n = 55
        short = message_to_blocks(b"A", n)
        long_ = message_to_blocks(b"A" * 20, n)
        assert len(long_) > len(short)

    def test_raises_n_too_small(self):
        with pytest.raises(ValueError):
            message_to_blocks(b"A", n=2)

    def test_roundtrip_longer_message(self):
        n = 55
        original = bytes(range(1, 10))  # \x01..\x09
        blocks = message_to_blocks(original, n)
        recovered = blocks_to_message(blocks, n)
        assert recovered == original

    def test_larger_n_allows_larger_blocks(self):
        """Mit groesserem n sind groessere Bloecke moeglich."""
        n_small = 55
        n_large = 2**16 + 1  # prim genug fuer Test
        msg = b"Test message for large n"
        bl_small = message_to_blocks(msg, n_small)
        bl_large = message_to_blocks(msg, n_large)
        assert len(bl_large) <= len(bl_small)


# ---------------------------------------------------------------------------
# blocks_to_message
# ---------------------------------------------------------------------------

class TestBlocksToMessage:

    def test_invalid_padding_raises(self):
        """Fehlendes 0x80-Byte muss ValueError ausloesen."""
        n = 55
        block_bytes = _block_byte_length(n) - 1
        # Block ohne 0x80-Padding
        bad_block = [0]  # nur Nullbyte
        with pytest.raises(ValueError, match="Padding"):
            blocks_to_message(bad_block, n)

    def test_known_roundtrip(self):
        n = 55
        messages = [b"A", b"BC", b"\x01\x02\x03", b""]
        for msg in messages:
            blocks = message_to_blocks(msg, n)
            assert blocks_to_message(blocks, n) == msg


# ---------------------------------------------------------------------------
# count_dummy_positions
# ---------------------------------------------------------------------------

class TestCountDummyPositions:

    def test_offset_0_k3_positions(self):
        """Bei s=0, k=3: Positionen 1, 4, 7, 10, ..."""
        dummies = count_dummy_positions(total=12, k=3, s=0)
        assert dummies == [1, 4, 7, 10]

    def test_offset_1_k3_positions(self):
        """Bei s=1, k=3: Positionen 3, 6, 9, 12."""
        dummies = count_dummy_positions(total=12, k=3, s=1)
        assert dummies == [3, 6, 9, 12]

    def test_offset_2_k3_positions(self):
        """Bei s=2, k=3: Positionen 2, 5, 8, 11."""
        dummies = count_dummy_positions(total=12, k=3, s=2)
        assert dummies == [2, 5, 8, 11]

    def test_different_offsets_produce_different_positions(self):
        total = 12
        pos = [set(count_dummy_positions(total, k=3, s=s)) for s in range(3)]
        assert pos[0] != pos[1]
        assert pos[1] != pos[2]
        assert pos[0] != pos[2]

    def test_dummy_count_approximately_1_over_k(self):
        """Dummy-Anteil soll nahe 1/k liegen."""
        for k in [2, 3, 4, 5]:
            total = k * 10
            dummies = count_dummy_positions(total, k=k, s=0)
            ratio = len(dummies) / total
            assert abs(ratio - 1 / k) < 0.15, f"k={k}: Anteil {ratio} weicht zu stark ab"

    def test_all_offsets_cover_all_positions(self):
        """Die k Offset-Muster ueberdecken zusammen alle Positionen."""
        total = 9
        k = 3
        all_covered = set()
        for s in range(k):
            all_covered |= set(count_dummy_positions(total, k=k, s=s))
        assert all_covered == set(range(1, total + 1))

    def test_empty_result_for_zero_total(self):
        assert count_dummy_positions(0, k=3, s=0) == []


# ---------------------------------------------------------------------------
# encrypt
# ---------------------------------------------------------------------------

class TestEncrypt:

    def test_packet_count_formula(self):
        """l = ceil(r * k / (k-1)), Schritt 2 des Algorithmus."""
        pk = toy_pk(k=3)
        msg = b"AB"  # ergibt r Bloecke
        r = len(message_to_blocks(msg, pk.n))
        expected_min = math.ceil(r * pk.k / (pk.k - 1))
        packets = encrypt(pk, s=0, message=msg)
        assert len(packets) >= expected_min

    def test_all_packets_in_range(self):
        """Alle Pakete muessen im Bereich [0, n) liegen."""
        pk = toy_pk()
        packets = encrypt(pk, s=0, message=b"Test")
        assert all(0 <= p < pk.n for p in packets), "Paket ausserhalb [0, n)"

    def test_raises_invalid_s(self):
        pk = toy_pk(k=3)
        with pytest.raises(ValueError, match="Offset"):
            encrypt(pk, s=3, message=b"Test")  # s muss in {0,1,2}

    def test_raises_negative_s(self):
        pk = toy_pk(k=3)
        with pytest.raises(ValueError, match="Offset"):
            encrypt(pk, s=-1, message=b"Test")

    def test_different_offsets_different_packets(self):
        """Verschiedene Offsets fuehren (mit sehr hoher Wahrscheinlichkeit)
        zu unterschiedlichen Paketfolgen."""
        pk = toy_pk(k=3)
        msg = b"Hello World from FC"
        results = [encrypt(pk, s=s, message=msg) for s in range(3)]
        # Mindestens zwei der drei Paketfolgen sollten verschieden sein
        assert results[0] != results[1] or results[1] != results[2]

    def test_encrypt_returns_list_of_ints(self):
        pk = toy_pk()
        packets = encrypt(pk, s=0, message=b"X")
        assert isinstance(packets, list)
        assert all(isinstance(p, int) for p in packets)

    def test_dummy_positions_match_expected(self):
        """Echte Pakete sind RSA-Verschluesselungen; alle anderen Dummies."""
        pk = toy_pk(k=3)
        s = 0
        msg = b"\x02"  # Einzelbyte, passt sicher in Block < 55
        blocks = message_to_blocks(msg, pk.n)
        packets = encrypt(pk, s=s, message=msg)

        # Pruefen: echte Positionen enthalten RSA-Verschluesselungen der Bloecke
        real_positions = [i for i in range(1, len(packets) + 1)
                          if (i - 1 + s) % pk.k != 0]
        real_packets = [packets[i - 1] for i in real_positions]
        expected_ciphers = [pow(b, pk.e, pk.n) for b in blocks]
        assert real_packets[:len(expected_ciphers)] == expected_ciphers

    def test_encrypt_various_k(self):
        """Verschluesselung funktioniert fuer verschiedene k-Werte."""
        msg = b"FC test"
        for k in [2, 3, 4, 5]:
            pk, _ = KeygenFC.toy_example(k=k, s=0)
            packets = encrypt(pk, s=0, message=msg)
            assert len(packets) >= 1
            assert all(0 <= p < pk.n for p in packets)

    def test_traffic_overhead_k3(self):
        """Traffic-Overhead bei k=3 betraegt ca. 50% (Proposition 5.2)."""
        pk = toy_pk(k=3)
        msg = bytes(range(1, 4))  # 3 Bytes -> mehrere Bloecke
        blocks = message_to_blocks(msg, pk.n)
        r = len(blocks)
        packets = encrypt(pk, s=0, message=msg)
        total = len(packets)
        # Overhead: total/r sollte nahe k/(k-1) = 1.5 liegen
        ratio = total / r
        assert 1.0 <= ratio <= 2.5, f"Unerwartetes Overhead-Verhaeltnis: {ratio}"

    def test_dummy_count_per_offset(self):
        """Fuer s=0,1,2 enthalten gleich viele Dummy-Positionen."""
        pk = toy_pk(k=3)
        msg = b"Test message"
        counts = []
        for s in range(3):
            packets = encrypt(pk, s=s, message=msg)
            dummy_count = sum(
                1 for i in range(1, len(packets) + 1) if (i - 1 + s) % pk.k == 0
            )
            counts.append(dummy_count)
        # Dummy-Anzahl sollte fuer alle Offsets gleich oder sehr aehnlich sein
        assert max(counts) - min(counts) <= 1


# ---------------------------------------------------------------------------
# Zusaetzliche Coverage Tests
# ---------------------------------------------------------------------------

class TestEncryptCoverage:
    """Tests um 100% Code Coverage zu erreichen."""

    def test_compute_packet_count_zero_blocks(self):
        """Test _compute_packet_count mit r=0."""
        from src.encrypt import _compute_packet_count
        result = _compute_packet_count(r=0, k=3, s=0)
        assert result == 0

    def test_compute_packet_count_increment(self):
        """Test _compute_packet_count wenn l inkrementiert werden muss."""
        from src.encrypt import _compute_packet_count
        # Mit verschiedenen Offsets kann die Approximation ungenauer sein
        # Dies sollte den l += 1 Pfad treffen
        result = _compute_packet_count(r=5, k=3, s=1)
        assert result > 0
        # Verifiziere dass genau r echte Slots entstehen
        real_slots = sum(1 for i in range(1, result + 1) if (i - 1 + 1) % 3 != 0)
        assert real_slots == 5

    def test_message_to_blocks_too_long(self):
        """Test message_to_blocks mit Nachricht > 65535 Bytes."""
        pk = toy_pk()
        long_message = b"X" * 65536  # 65536 Bytes (uber 65535)
        with pytest.raises(ValueError, match="Nachricht zu lang"):
            message_to_blocks(long_message, pk.n)

    def test_blocks_to_message_invalid_padding_various(self):
        """Test blocks_to_message mit ungueltigem Padding."""
        pk = toy_pk()
        # Leere Block-Liste
        with pytest.raises(ValueError, match="Ungueltiges Padding"):
            blocks_to_message([], pk.n)

    def test_message_to_blocks_padding_path(self):
        """Test der Padding-Pfad in message_to_blocks (Zeile 77)."""
        # Wenn remainder != 0, muss gepadded werden
        n = 55  # B = 5 bits pro Block
        # Nachricht die nicht auf Blockgrenze endet
        msg = b"test"
        blocks = message_to_blocks(msg, n)
        # Roundtrip verifizieren
        reconstructed = blocks_to_message(blocks, n)
        assert reconstructed == msg
