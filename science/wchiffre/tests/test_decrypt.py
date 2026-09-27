"""
Tests fuer src.decrypt

Abgedeckt:
- decrypt: Korrektheit der Entschluesselung
- Symmetrie mit encrypt: Dec(sk, Enc(pk, s, M)) == M
- Alle Offsets s in {0,...,k-1}
- Verschiedene Nachrichten
- Verschiedene k-Werte
- Dummy-Pakete werden korrekt ignoriert
"""

import pytest

from src.keygen import KeygenFC
from src.encrypt import encrypt, message_to_blocks
from src.decrypt import decrypt


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def toy_keys(k: int = 3, s: int = 0):
    return KeygenFC.toy_example(k=k, s=s)


# ---------------------------------------------------------------------------
# Grundlegende Korrektheit (Satz 4.1)
# ---------------------------------------------------------------------------

class TestDecryptCorrectness:

    def test_single_byte_roundtrip(self):
        """Dec(sk, Enc(pk, s, M)) = M fuer einzelnes Byte."""
        pk, sk = toy_keys(s=0)
        msg = b"\x02"
        assert decrypt(sk, encrypt(pk, sk.s, msg)) == msg

    def test_ascii_string_roundtrip(self):
        pk, sk = toy_keys(s=0)
        msg = b"Hello"
        assert decrypt(sk, encrypt(pk, sk.s, msg)) == msg

    def test_empty_message_roundtrip(self):
        pk, sk = toy_keys(s=0)
        msg = b""
        assert decrypt(sk, encrypt(pk, sk.s, msg)) == msg

    def test_binary_message_roundtrip(self):
        pk, sk = toy_keys(s=0)
        msg = bytes([1, 2, 3])
        assert decrypt(sk, encrypt(pk, sk.s, msg)) == msg

    def test_all_offsets_s_in_k3(self):
        """Korrektheit fuer alle Offsets s in {0, 1, 2}."""
        for s in range(3):
            pk, sk = toy_keys(k=3, s=s)
            msg = b"FC"
            packets = encrypt(pk, s, msg)
            recovered = decrypt(sk, packets)
            assert recovered == msg, f"Fehler bei s={s}"

    def test_correctness_korollar_2_4(self):
        """Korollar 2.4: M_j^{23*7} = M_j mod 55 fuer M_j in {1,...,54}."""
        pk, sk = toy_keys(s=0)
        for m in [2, 5, 8, 12, 17, 19]:
            c = pow(m, pk.e, pk.n)   # RSA-Enc
            m2 = pow(c, sk.d, sk.n)  # RSA-Dec
            assert m2 == m, f"RSA-Korrektheit verletzt fuer m={m}"

    def test_decrypt_returns_bytes(self):
        pk, sk = toy_keys(s=0)
        result = decrypt(sk, encrypt(pk, sk.s, b"Test"))
        assert isinstance(result, bytes)

    def test_decrypt_empty_packets(self):
        """Leere Paketliste: kein Padding-Bit vorhanden -> ValueError."""
        pk, sk = toy_keys(s=0)
        with pytest.raises(ValueError, match="Padding"):
            decrypt(sk, [])


# ---------------------------------------------------------------------------
# Verschiedene k-Werte
# ---------------------------------------------------------------------------

class TestDecryptVariousK:

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_roundtrip_various_k(self, k):
        pk, sk = KeygenFC.toy_example(k=k, s=0)
        msg = b"test"
        packets = encrypt(pk, sk.s, msg)
        assert decrypt(sk, packets) == msg

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_all_offsets_various_k(self, k):
        for s in range(k):
            pk, sk = KeygenFC.toy_example(k=k, s=s)
            msg = b"xy"
            packets = encrypt(pk, s, msg)
            assert decrypt(sk, packets) == msg, f"Fehler bei k={k}, s={s}"


# ---------------------------------------------------------------------------
# Symmetrie-Eigenschaft des Algorithmenpaares
# ---------------------------------------------------------------------------

class TestEncDecSymmetry:

    def test_same_index_filter_enc_dec(self):
        """Enc und Dec verwenden exakt dieselbe Fenster-Bedingung."""
        pk, sk = toy_keys(k=3, s=1)
        msg = b"\x03\x04"
        packets = encrypt(pk, sk.s, msg)

        # Manuell die echten Positionen aus Empfaenger-Sicht bestimmen
        real_positions_dec = [
            i for i in range(1, len(packets) + 1)
            if (i - 1 + sk.s) % pk.k != 0
        ]
        real_ciphers = [packets[i - 1] for i in real_positions_dec]

        # RSA-Entschluesselung
        real_blocks = [pow(c, sk.d, sk.n) for c in real_ciphers]

        # Erwartete Bloecke von Sender-Seite
        expected_blocks = message_to_blocks(msg, pk.n)
        assert real_blocks == expected_blocks

    def test_wrong_offset_gives_wrong_result(self):
        """Mit falschem Offset liefert Dec entweder falsche Nachricht oder Fehler."""
        pk, sk_correct = toy_keys(k=3, s=0)
        msg = b"\x05"
        packets = encrypt(pk, s=0, message=msg)

        from src.keygen import PrivateKey
        sk_wrong = PrivateKey(
            d=sk_correct.d,
            s=1,
            n=sk_correct.n,
            k=sk_correct.k
        )

        try:
            result_wrong = decrypt(sk_wrong, packets)
            assert result_wrong != msg
        except ValueError:
            pass

    def test_wrong_d_gives_wrong_result(self):
        """Mit falschem privatem Exponenten d wird falscher Klartext rekonstruiert."""
        pk, sk_correct = toy_keys(k=3, s=0)
        msg = b"\x03"
        packets = encrypt(pk, s=0, message=msg)

        from src.keygen import PrivateKey
        # d=1 bedeutet Paket unveraendert zurueckgeben
        sk_wrong = PrivateKey(d=1, s=sk_correct.s, n=sk_correct.n, k=sk_correct.k)
        result_wrong = decrypt(sk_wrong, packets)
        assert result_wrong != msg


# ---------------------------------------------------------------------------
# Groessere Nachrichten (mit zufaelligem RSA)
# ---------------------------------------------------------------------------

class TestDecryptLargerKeys:

    def test_roundtrip_larger_rsa(self):
        """Korrektheit mit groesseren RSA-Schluesseln."""
        pk, sk = KeygenFC.generate(bits=64, k=3)
        msg = b"Fenster-Chiffre Test mit groesseren Schluesseln"
        packets = encrypt(pk, sk.s, msg)
        recovered = decrypt(sk, packets)
        assert recovered == msg

    def test_roundtrip_multiple_messages(self):
        pk, sk = KeygenFC.generate(bits=64, k=3)
        messages = [b"", b"A", b"Hello World", bytes(range(10))]
        for msg in messages:
            packets = encrypt(pk, sk.s, msg)
            assert decrypt(sk, packets) == msg, f"Fehler fuer Nachricht {msg!r}"

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_roundtrip_generated_keys_various_k(self, k):
        pk, sk = KeygenFC.generate(bits=64, k=k)
        msg = b"Test FC k=" + str(k).encode()
        packets = encrypt(pk, sk.s, msg)
        assert decrypt(sk, packets) == msg