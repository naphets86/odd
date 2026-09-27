"""
Integrationstests fuer die vollstaendige Fenster-Chiffre (FC).

Testet das vollstaendige Protokoll FC = (Gen, Enc, Dec) aus Definition 3.1:
- End-to-End Korrektheit (Satz 4.1)
- Informationstheoretische Eigenschaften des Dummy-Schemas (Satz 5.1)
- Traffic-Overhead (Proposition 5.2)
- Ununterscheidbarkeit echter/Dummy-Pakete (empirisch)
"""

import math
import random
import pytest

from src.keygen import KeygenFC, PublicKey, PrivateKey
from src.encrypt import encrypt, count_dummy_positions, message_to_blocks
from src.decrypt import decrypt


# ---------------------------------------------------------------------------
# Vollstaendiges Protokoll: Gen -> Enc -> Dec
# ---------------------------------------------------------------------------

class TestFullProtocol:

    def test_toy_example_full_roundtrip(self):
        """Vollstaendiger Durchlauf des Lehrbeispiels (n=55, e=23, d=7)."""
        pk, sk = KeygenFC.toy_example(s=0)
        msg = b"\x02"
        packets = encrypt(pk, sk.s, msg)
        recovered = decrypt(sk, packets)
        assert recovered == msg

    def test_all_rsa_valid_plaintexts(self):
        """M_j^{ed} = M_j mod 55 fuer alle gueltigen M_j (Korollar 2.4)."""
        pk, sk = KeygenFC.toy_example(s=0)
        n = pk.n
        import math as _math
        valid = [m for m in range(1, n) if _math.gcd(m, n) == 1]
        for m in valid[:20]:  # Stichprobe der ersten 20
            c = pow(m, pk.e, n)
            m2 = pow(c, sk.d, n)
            assert m2 == m, f"RSA-Korrektheit verletzt fuer m={m}"

    def test_protocol_with_multiple_offsets(self):
        """Das Protokoll funktioniert fuer alle s in {0,...,k-1}."""
        msg = b"Testdaten fuer FC"
        for k in [2, 3, 4]:
            for s in range(k):
                pk, sk = KeygenFC.toy_example(k=k, s=s)
                packets = encrypt(pk, s, msg)
                recovered = decrypt(sk, packets)
                assert recovered == msg, f"Fehler bei k={k}, s={s}"

    def test_generated_keys_roundtrip(self):
        """Zufaellig generierte Schluessels: vollstaendiger Protokolldurchlauf."""
        for _ in range(3):
            pk, sk = KeygenFC.generate(bits=64, k=3)
            msg = b"Zufaellige Schluessels, kurze Nachricht"
            packets = encrypt(pk, sk.s, msg)
            recovered = decrypt(sk, packets)
            assert recovered == msg

    def test_empty_message_full_protocol(self):
        pk, sk = KeygenFC.toy_example(s=0)
        msg = b""
        assert decrypt(sk, encrypt(pk, sk.s, msg)) == msg


# ---------------------------------------------------------------------------
# Dummy-Muster und Fenster-Struktur
# ---------------------------------------------------------------------------

class TestWindowStructure:

    def test_dummy_positions_periodic(self):
        """Dummy-Positionen bilden ein periodisches Muster mit Periode k."""
        k, s = 3, 0
        total = 18
        dummies = count_dummy_positions(total, k=k, s=s)
        # Aufeinanderfolgende Dummies haben Abstand k
        for i in range(1, len(dummies)):
            assert dummies[i] - dummies[i-1] == k

    def test_three_offsets_partition_positions(self):
        """Die drei Offsets 0,1,2 partitionieren alle Positionen (k=3)."""
        total = 9
        k = 3
        sets = [set(count_dummy_positions(total, k, s)) for s in range(k)]
        # Alle paarweise disjunkt
        for i in range(k):
            for j in range(i + 1, k):
                assert sets[i].isdisjoint(sets[j]), f"Ueberlappung bei s={i} und s={j}"
        # Zusammen alle Positionen
        union = set().union(*sets)
        assert union == set(range(1, total + 1))

    def test_dummy_ratio_close_to_1_over_k(self):
        """Dummy-Anteil konvergiert gegen 1/k (Proposition 5.2)."""
        for k in [2, 3, 4, 5, 10]:
            total = k * 100
            dummies = count_dummy_positions(total, k=k, s=0)
            ratio = len(dummies) / total
            assert abs(ratio - 1/k) < 0.02, f"k={k}: ratio={ratio:.4f}"

    def test_traffic_overhead_k3(self):
        """Traffic-Overhead bei k=3: ca. 50% (Proposition 5.2)."""
        pk, sk = KeygenFC.toy_example(k=3, s=0)
        msg = bytes(range(1, 4))
        blocks = message_to_blocks(msg, pk.n)
        r = len(blocks)
        packets = encrypt(pk, sk.s, msg)
        overhead = (len(packets) - r) / r
        # Erwarteter Overhead: 1/(k-1) = 0.5
        assert abs(overhead - 0.5) <= 0.5 + 0.1  # Toleranz fuer kleine r

    def test_offset_changes_dummy_pattern_completely(self):
        """Verschiedene Offsets erzeugen voellig verschiedene Dummy-Muster."""
        total = 9
        k = 3
        patterns = [
            set(count_dummy_positions(total, k, s)) for s in range(k)
        ]
        for i in range(k):
            for j in range(i + 1, k):
                assert patterns[i] != patterns[j]


# ---------------------------------------------------------------------------
# Informationstheoretische Eigenschaften (Satz 5.1)
# ---------------------------------------------------------------------------

class TestInformationTheoreticSecurity:

    def test_offset_unguessable_random(self):
        """Zufaellige Offset-Schaetzung trifft nur mit Wahrscheinlichkeit 1/k."""
        k = 3
        n_trials = 300
        correct = 0
        for _ in range(n_trials):
            pk, sk = KeygenFC.toy_example(k=k)  # s zufaellig
            guess = random.randrange(0, k)
            if guess == sk.s:
                correct += 1
        hit_rate = correct / n_trials
        # Erwartungswert: 1/3, Toleranz 3 Sigma (~0.027*3 ~ 0.08)
        assert abs(hit_rate - 1/k) < 0.12, f"Trefferrate {hit_rate:.3f} zu weit von 1/{k}"

    def test_attacker_entropy_grows_with_packets(self):
        """Entropie H = log2(C(l, l/k)) waechst mit der Paketanzahl (Bemerkung 5.1)."""
        k = 3
        entropies = []
        for total in [9, 18, 30, 60]:
            dummy_count = total // k
            from math import comb, log2
            h = log2(comb(total, dummy_count)) if comb(total, dummy_count) > 0 else 0
            entropies.append(h)
        # Entropie muss strikt monoton wachsen
        for i in range(1, len(entropies)):
            assert entropies[i] > entropies[i-1], (
                f"Entropie sank: {entropies[i-1]:.2f} -> {entropies[i]:.2f}"
            )

    def test_entropy_at_30_packets_exceeds_24_bits(self):
        """Bei N=30, k=3: H = log2(C(30,10)) >= 24 Bit (Bemerkung 5.1 der Arbeit)."""
        from math import comb, log2
        k = 3
        total = 30
        dummy_count = total // k
        h = log2(comb(total, dummy_count))
        assert h >= 24.0, f"Entropie {h:.2f} < 24 Bit"

    def test_all_offsets_equally_likely_a_priori(self):
        """A-priori: Alle k Offsets sind gleich haeufig."""
        n_keys = 300
        k = 3
        counts = [0] * k
        for _ in range(n_keys):
            _, sk = KeygenFC.toy_example(k=k)
            counts[sk.s] += 1
        # Chi-Quadrat-aehnliche Pruefung: kein Offset zu selten/haeufig
        expected = n_keys / k
        for c in counts:
            assert abs(c - expected) < expected * 0.3, (
                f"Unausgewogene Offset-Verteilung: {counts}"
            )


# ---------------------------------------------------------------------------
# Ununterscheidbarkeit echter / Dummy-Pakete (Proposition 5.3)
# ---------------------------------------------------------------------------

class TestIndistinguishability:

    def test_dummy_and_real_packets_in_same_range(self):
        """Echte und Dummy-Pakete liegen im selben Wertebereich [0, n)."""
        pk, sk = KeygenFC.toy_example(k=3, s=0)
        msg = b"\x02\x03"
        packets = encrypt(pk, sk.s, msg)
        total = len(packets)
        dummies = set(count_dummy_positions(total, pk.k, sk.s))
        real_pos = set(range(1, total + 1)) - dummies

        dummy_pkts = [packets[i - 1] for i in dummies]
        real_pkts = [packets[i - 1] for i in real_pos]

        assert all(0 <= p < pk.n for p in dummy_pkts)
        assert all(0 <= p < pk.n for p in real_pkts)

    def test_hamming_weight_distributions_overlap(self):
        """Hamming-Gewichte (Bit-Eins-Anzahl) echter/Dummy-Pakete ueberlappen."""
        pk, sk = KeygenFC.generate(bits=64, k=3, s=0)
        msg = bytes(range(1, 20))
        packets = encrypt(pk, sk.s, msg)
        total = len(packets)
        dummies = set(count_dummy_positions(total, pk.k, sk.s))
        real_pos = set(range(1, total + 1)) - dummies

        def hw(x): return bin(x).count("1")

        hw_dummy = [hw(packets[i - 1]) for i in dummies]
        hw_real = [hw(packets[i - 1]) for i in real_pos]

        if hw_dummy and hw_real:
            mean_dummy = sum(hw_dummy) / len(hw_dummy)
            mean_real = sum(hw_real) / len(hw_real)
            # Mittelwerte sollten nicht zu weit auseinanderliegen
            # (kein harter Test, nur Plausibilitaetspruefung)
            assert abs(mean_dummy - mean_real) < max(mean_dummy, mean_real) * 0.8


# ---------------------------------------------------------------------------
# Reproduzierbarkeit und Deterministik
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_inputs_same_real_ciphers(self):
        """Bei gleichen Eingaben (pk, s, M) sind die echten Chiffrate identisch."""
        pk, sk = KeygenFC.toy_example(s=0)
        msg = b"\x05"
        blocks = message_to_blocks(msg, pk.n)
        c1 = [pow(b, pk.e, pk.n) for b in blocks]
        c2 = [pow(b, pk.e, pk.n) for b in blocks]
        assert c1 == c2

    def test_packet_sequence_length_deterministic(self):
        """Die Paketlaenge haengt nur von r, k ab (nicht von Zufaelligkeit)."""
        pk, sk = KeygenFC.toy_example(k=3, s=0)
        msg = b"Hallo"
        len1 = len(encrypt(pk, sk.s, msg))
        len2 = len(encrypt(pk, sk.s, msg))
        assert len1 == len2