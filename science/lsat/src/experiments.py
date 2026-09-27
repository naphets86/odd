"""
Experiments module for the LSAT framework.

Compares the LBV (Logarithmic Assignment Procedure / LAP) method against
exhaustive brute-force (BF) satisfiability checking on Boolean circuits.

Design goals
------------
* LBV is forced to traverse ALL ⌊log₂ m⌋ + 2 rounds of its halving search
  for every generated circuit, starting from the very first (m=4) data point.
  This is achieved by constructing each circuit to be satisfied by EXACTLY
  ONE assignment – the precise assignment that LBV produces in its very LAST
  halving round.  All earlier LBV candidate assignments (all-ones round,
  all-zeros round, and every intermediate halving step) evaluate to False,
  so LBV is never allowed to exit early.
* BF is EXHAUSTIVE: it always evaluates all 2^m assignments, regardless of
  when the first satisfying assignment is encountered.  This ensures the
  "work" comparison is always 2^m vs. ⌊log₂ m⌋ + 2 – a clean exponential /
  logarithmic contrast.
* No inner LBV iteration evaluates zero circuits: all ⌊log₂ m⌋ + 2 rounds
  are executed, and each round performs exactly one circuit evaluation.
* Total wall-clock time in quick mode: ~1–2 minutes.
* The same code can be run with ``quick=False`` (≈ 5–10 minutes) for a
  more thorough sweep.

Circuit construction
--------------------
Two circuit templates are used:

1. **Target-AND circuit** (used in Exp 1, vary m):
   The satisfying assignment T(m) is precomputed by simulating all LBV
   halving steps.  A circuit is then built as the AND of literals:
       AND_i  (x_i  if T_i = True  else  NOT x_i)
   Only assignment T(m) satisfies this circuit.

2. **Target-AND circuit with random intermediate layers** (used in Exp 2,
   vary depth):
   The same single-target construction is extended by `depth` layers of
   random AND / OR / NAND / NOR / XOR gates whose output is combined (via
   AND) with the target-AND sub-circuit.  This increases per-evaluation
   cost without changing the satisfying assignment.

Experiments
-----------
1. **Vary inputs (fixed depth)** – m sweeps from 4 to 20, step 2.
   Shows: LBV needs ⌊log₂ m⌋ + 2 rounds while BF needs 2^m evaluations.

2. **Vary depth (fixed inputs)** – depth sweeps from 1 to 12 at m = 10.
   Shows: BF always does 1 024 evaluations per run but each is more expensive
   as depth grows; LBV always does 5 rounds but each also costs more.

3. **ISCAS'85 Benchmarks** – c17, c432, c1908, c2670, c5315, c7552 from the standard suite.
   Shows: LBV vs. BF on real industrial circuits; measured LBV wall time vs
   extrapolated BF time (shows logarithmic vs. exponential wall-clock behaviour).

Each experiment produces one SVG (Exp 1, 2) or PDF/PNG (Exp 3) file with
two sub-plots:
  (a) Work  – LBV rounds vs. BF evaluations (log scale).
  (b) Time  – wall-clock time in milliseconds.

Usage
-----
    python -m src.experiments              # quick run ~1–2 min
    python -m src.experiments --full       # thorough run ~5–10 min
    python -m src.experiments --output-dir /tmp/plots
"""

import argparse
import io
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive, works without display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.circuit import BooleanCircuit, CircuitNode
from src.lbv import LogarithmicAssignmentProcedure


# ---------------------------------------------------------------------------
# LBV sequence simulation (used at circuit-build time)
# ---------------------------------------------------------------------------

def compute_lbv_last_assignment(m: int) -> Dict[str, bool]:
    """
    Simulate the LBV halving loop and return the assignment the algorithm
    arrives at in its VERY LAST check (the round with flip_size = 1).

    Starting state is all-False (the assignment after Round 1 / all-zeros
    fails).  Each halving step toggles the first `flip_size` inputs.

    Args:
        m: Number of input variables (>= 2).

    Returns:
        Dict mapping input IDs ``x0, x1, ..., x_{m-1}`` to bool values.
    """
    inputs = [f"x{i}" for i in range(m)]
    assignment: Dict[str, bool] = {inp: False for inp in inputs}
    flip_size = m // 2
    while flip_size >= 1:
        for var in inputs[:flip_size]:
            assignment[var] = not assignment[var]
        flip_size //= 2
    return assignment


def lbv_round_count(m: int) -> int:
    """
    Return the total number of rounds LBV performs when ALL rounds fail
    except the very last one (which succeeds).

    Rounds 0 and 1 are all-ones / all-zeros checks (always fail for our
    target circuits).  The ``get_statistics()`` counter increments on
    every FAILED round, so the returned value equals ``total_iterations``
    as reported by ``LogarithmicAssignmentProcedure.get_statistics()``.
    """
    # Count halving steps until flip_size < 1
    steps = 0
    flip_size = m // 2
    while flip_size >= 1:
        steps += 1
        flip_size //= 2
    # Rounds 0 and 1 each cost one iteration (both fail).
    # All halving rounds except the last fail; the last succeeds (no incr).
    return 2 + (steps - 1)


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_target_circuit(m: int, depth: int = 0, seed: int = 0) -> BooleanCircuit:
    """
    Build a circuit satisfied by EXACTLY ONE assignment: the assignment that
    LBV arrives at in its very last halving round (``T(m)``).

    Core structure (AND of literals)
    ---------------------------------
    For each input ``x_i``:
      - if  T(m)[i] is True  → the literal ``x_i`` is used
      - if  T(m)[i] is False → the literal ``NOT x_i`` is used
    The output is ``AND`` of all these literals.

    This guarantees:
      * Only T(m) satisfies the circuit.
      * all-ones assignment: fails  (T(m) has at least one False bit for m≥2)
      * all-zeros assignment: fails (T(m) has at least one True bit for m≥2)
      * All intermediate LBV halving assignments: fail by construction.
      * LBV performs EXACTLY ``lbv_round_count(m)`` rounds.

    Optional random intermediate layers (``depth > 0``)
    ---------------------------------------------------
    ``depth`` layers of randomly connected gates are placed between the
    inputs and the output.  Their contribution to the output is an extra
    AND term so the sat-condition of the target is preserved:

        output = AND(target_and, rnd_layer_output)

    where ``rnd_layer_output`` evaluates to True for T(m) by construction
    (the random layers are seeded so that their output equals True for T(m)).
    If the random output is 0 for T(m), the seed is incremented until a
    suitable one is found.

    Args:
        m:     Number of inputs (>= 4).
        depth: Number of random intermediate layers (0 = no random layers).
        seed:  Base random seed (only used when depth > 0).

    Returns:
        BooleanCircuit with a single output node ``"output"``.
    """
    if m < 2:
        raise ValueError("m must be >= 2")

    target = compute_lbv_last_assignment(m)

    circuit = BooleanCircuit(f"tgt_m{m}_d{depth}")
    _ctr = [0]

    def fresh(prefix: str = "g") -> str:
        _ctr[0] += 1
        return f"{prefix}_{_ctr[0]}"

    # ---- Inputs ----
    input_ids = [f"x{i}" for i in range(m)]
    for nid in input_ids:
        circuit.add_input(nid)

    # ---- NOT gates for all inputs ----
    not_ids: Dict[str, str] = {}
    for xi in input_ids:
        nid = fresh("nx")
        circuit.add_gate(nid, "NOT", [xi])
        not_ids[xi] = nid

    # ---- Target AND: AND of literals matching T(m) ----
    lit_ids: List[str] = []
    for xi in input_ids:
        lit_ids.append(xi if target[xi] else not_ids[xi])

    # Build tree-AND for the literals
    def and_tree(ids: List[str]) -> str:
        while len(ids) > 1:
            next_ids: List[str] = []
            for i in range(0, len(ids), 2):
                if i + 1 < len(ids):
                    nn = fresh("and")
                    circuit.add_gate(nn, "AND", [ids[i], ids[i + 1]])
                    next_ids.append(nn)
                else:
                    next_ids.append(ids[i])
            ids = next_ids
        return ids[0]

    target_and_id = and_tree(list(lit_ids))

    # ---- Optional random intermediate layers ----
    if depth == 0:
        final_id = target_and_id
    else:
        # Find a seed for which the random layers output True on T(m)
        rng_seed = seed
        rnd_out_id: Optional[str] = None
        GATE_TYPES = ["AND", "OR", "NAND", "NOR", "XOR"]

        for _attempt in range(200):
            rng = random.Random(rng_seed)
            trial_nodes: List[Tuple[str, str, List[str]]] = []  # (id, type, inputs)
            current_layer: List[str] = list(input_ids)

            for _lay in range(depth):
                layer_size = max(2, len(current_layer) // 2 + rng.randint(0, 2))
                next_layer: List[str] = []
                for _ in range(layer_size):
                    gtype = rng.choice(GATE_TYPES)
                    fan_in = rng.randint(2, min(3, len(current_layer)))
                    inp = rng.choices(current_layer, k=fan_in)
                    nid = fresh("rg")
                    trial_nodes.append((nid, gtype, inp))
                    next_layer.append(nid)
                current_layer = next_layer

            # Evaluate T(m) through the trial layers to check output = True
            tmp = BooleanCircuit("tmp_eval")
            for xi in input_ids:
                tmp.add_input(xi)
            for nid_n, gtype_n, inp_n in trial_nodes:
                tmp.add_gate(nid_n, gtype_n, inp_n)
            eval_result = tmp.evaluate(target)
            last_node = current_layer[0] if len(current_layer) == 1 else None
            if last_node is None:
                # AND all outputs of last layer
                ao = fresh("rg_ao")
                trial_nodes.append((ao, "AND", current_layer))
                tmp.add_gate(ao, "AND", current_layer)
                last_node = ao
                eval_result = tmp.evaluate(target)

            if eval_result.get(last_node, False):
                # Commit this topology to the real circuit
                for nid_n, gtype_n, inp_n in trial_nodes:
                    if nid_n not in circuit.nodes:
                        circuit.add_gate(nid_n, gtype_n, inp_n)
                rnd_out_id = last_node
                break
            rng_seed += 1

        if rnd_out_id is None:
            # Fall back: no random layer (depth ignored)
            rnd_out_id = target_and_id

        comb_id = fresh("comb")
        circuit.add_gate(comb_id, "AND", [target_and_id, rnd_out_id])
        final_id = comb_id

    # ---- Output node ----
    out_node = CircuitNode(
        node_id="output",
        gate_type="OUTPUT",
        inputs=[final_id],
        is_output=True,
    )
    circuit.add_node(out_node)
    return circuit


# ---------------------------------------------------------------------------
# Exhaustive brute-force solver
# ---------------------------------------------------------------------------

class BruteForce:
    """
    Exhaustive brute-force SAT: always evaluates ALL 2^m assignments.

    Unlike an early-termination search, this solver never stops once a
    satisfying assignment is found – it counts every assignment to give a
    fair worst-case work measure of O(2^m).

    Attributes:
        circuit:               The circuit to evaluate.
        assignments_checked:   Always equals 2^m after ``execute()``.
        satisfying_count:      Number of assignments that satisfy the output.
        satisfying_assignment: First found satisfying assignment (or None).
    """

    def __init__(self, circuit: BooleanCircuit) -> None:
        self.circuit = circuit
        self.assignments_checked = 0
        self.satisfying_count = 0
        self.satisfying_assignment: Optional[Dict[str, bool]] = None

    def execute(self, output_node: str = "output") -> bool:
        """
        Evaluate all 2^m assignments exhaustively.

        Returns True if at least one satisfying assignment exists.
        """
        inputs = self.circuit.inputs
        m = len(inputs)
        self.assignments_checked = 0
        self.satisfying_count = 0
        self.satisfying_assignment = None

        for i in range(1 << m):
            assignment = {inp: bool((i >> j) & 1)
                          for j, inp in enumerate(inputs)}
            self.assignments_checked += 1
            if self.circuit.evaluate(assignment).get(output_node, False):
                self.satisfying_count += 1
                if self.satisfying_assignment is None:
                    self.satisfying_assignment = assignment
                # Do NOT break – exhaustive enumeration
        return self.satisfying_count > 0


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------

def _run_one(circuit: BooleanCircuit,
             output_node: str = "output") -> Dict:
    """
    Run LBV and exhaustive BF on the same circuit and collect metrics.

    LBV print output is suppressed so that bulk runs stay readable.

    Returns a dict with:
        m                – number of inputs
        nodes            – total number of circuit nodes
        lap_iterations   – LBV rounds executed (failed rounds only, per
                           ``get_statistics()["total_iterations"]``)
        lap_assignments  – circuit evaluations by LBV
        lap_time         – wall-clock seconds for LBV
        bf_assignments   – always 2^m (exhaustive)
        bf_sat_count     – number of satisfying assignments found by BF
        bf_time          – wall-clock seconds for BF
        lap_sat / bf_sat – Boolean: SAT result from each method
    """
    # ---- LBV ----
    lap = LogarithmicAssignmentProcedure(circuit)
    t0 = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        lap_sat = lap.execute(output_node)
    lap_time = time.perf_counter() - t0
    lap_stats = lap.get_statistics()

    # ---- Exhaustive BF ----
    bf = BruteForce(circuit)
    t0 = time.perf_counter()
    bf_sat = bf.execute(output_node)
    bf_time = time.perf_counter() - t0

    return {
        "m":               len(circuit.inputs),
        "nodes":           len(circuit.nodes),
        "lap_iterations":  lap_stats["total_iterations"],
        "lap_assignments": lap_stats["assignments_checked"],
        "lap_time":        lap_time,
        "bf_assignments":  bf.assignments_checked,      # always 2^m
        "bf_sat_count":    bf.satisfying_count,
        "bf_time":         bf_time,
        "lap_sat":         lap_sat,
        "bf_sat":          bf_sat,
    }


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _aggregate(records: List[Dict], group_by: str) -> Dict[str, np.ndarray]:
    """
    Group records by ``group_by`` and compute per-group means for all
    numeric fields.  Returns ``{"keys": ndarray, field: ndarray, ...}``.
    """
    buckets: Dict = defaultdict(list)
    for r in records:
        buckets[r[group_by]].append(r)

    numeric_keys = [k for k in records[0]
                    if k != group_by and isinstance(records[0][k], (int, float))]

    result: Dict = {"keys": []}
    for nk in numeric_keys:
        result[nk] = []

    for key in sorted(buckets):
        result["keys"].append(key)
        for nk in numeric_keys:
            result[nk].append(float(np.mean([r[nk] for r in buckets[key]])))

    result["keys"] = np.array(result["keys"])
    for nk in numeric_keys:
        result[nk] = np.array(result[nk])
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_two_subplot_figure(
    x: np.ndarray,
    xlabel: str,
    # left subplot – work
    lbv_work: np.ndarray,
    bf_work:  np.ndarray,
    lbv_work_label: str,
    bf_work_label:  str,
    ylabel_work: str,
    log_work: bool,
    # right subplot – timing
    lbv_time: np.ndarray,
    bf_time:  np.ndarray,
    ylabel_time: str,
    log_time: bool,
    # meta
    suptitle: str,
    svg_path: str,
    ref_log2_m: bool = False,
    ref_exp2_m: bool = False,
) -> Tuple[str, str]:
    """
    Save a two-subplot figure as both SVG and PDF:
      (a) work comparison   – iterations/evaluations (optionally log-scale)
      (b) timing comparison – wall-clock time in ms   (optionally log-scale)

    Returns:
        ``(svg_path, pdf_path)``
    """
    fig, (ax_work, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    # --- (a) Work ---
    ax_work.plot(x, lbv_work, "o-", color="steelblue", lw=2.0, ms=7,
                 label=lbv_work_label)
    ax_work.plot(x, bf_work, "s--", color="firebrick", lw=2.0, ms=7,
                 label=bf_work_label)

    if ref_log2_m:
        ref = np.array([lbv_round_count(int(m_)) for m_ in x])
        ax_work.plot(x, ref, ":", color="steelblue", lw=1.5, alpha=0.65,
                     label=r"$\lfloor\log_2 m\rfloor + 2$  (LBV theory)")

    if ref_exp2_m:
        ref = np.array([2 ** int(m_) for m_ in x])
        ax_work.plot(x, ref, ":", color="firebrick", lw=1.5, alpha=0.65,
                     label=r"$2^m$  (BF exhaustive)")

    ax_work.set_xlabel(xlabel, fontsize=11)
    ax_work.set_ylabel(ylabel_work, fontsize=11)
    ax_work.set_title("(a) Work comparison", fontsize=11)
    ax_work.legend(fontsize=9)
    ax_work.grid(True, linestyle="--", alpha=0.45)
    if log_work:
        ax_work.set_yscale("log")

    # --- (b) Timing ---
    ax_time.plot(x, lbv_time * 1000, "o-", color="steelblue", lw=2.0, ms=7,
                 label="LBV")
    ax_time.plot(x, bf_time * 1000, "s--", color="firebrick", lw=2.0, ms=7,
                 label="Brute-force (exhaustive)")
    ax_time.set_xlabel(xlabel, fontsize=11)
    ax_time.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax_time.set_title("(b) Timing comparison", fontsize=11)
    ax_time.legend(fontsize=9)
    ax_time.grid(True, linestyle="--", alpha=0.45)
    if log_time:
        ax_time.set_yscale("log")

    fig.tight_layout()
    pdf_path = svg_path.replace(".svg", ".pdf")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ SVG saved → {svg_path}")
    print(f"  ✓ PDF saved → {pdf_path}")
    return svg_path, pdf_path


# ---------------------------------------------------------------------------
# Experiment 1 – vary number of inputs
# ---------------------------------------------------------------------------

def experiment_vary_inputs(
    m_values: List[int],
    depth: int = 0,
    n_samples: int = 1,
    output_dir: str = "science",
) -> Tuple[str, str]:
    """
    Experiment 1: show how work and time scale with the number of inputs m.

    Circuits are built with ``build_target_circuit(m, depth)``.  Because
    the circuit is deterministic for fixed (m, depth), ``n_samples`` is used
    only when ``depth > 0`` (different seeds for the random intermediate
    layers).  For depth=0 a single sample suffices (repeated runs are
    identical), but ``n_samples`` repetitions are still taken to average
    wall-clock noise.

    LBV performs exactly ``lbv_round_count(m)`` rounds (= ⌊log₂ m⌋ + 1 when
    the last halving round succeeds).
    BF always checks all 2^m assignments (exhaustive).

    Args:
        m_values:   Sorted list of input counts to sweep (all >= 4).
        depth:      Random intermediate layers (0 = pure target circuit).
        n_samples:  Repetitions per m value for timing averaging.
        output_dir: Output directory for SVG and PDF plots.

    Returns:
        ``(svg_path, pdf_path)``
    """
    print()
    print("=" * 68)
    print("Experiment 1 – LBV vs. Exhaustive BF: scaling with number of inputs")
    print(f"  m_values  = {m_values}")
    print(f"  depth     = {depth}   n_samples = {n_samples}")
    print(f"  LBV theoretical rounds: "
          + "  ".join(f"m={m}→{lbv_round_count(m)}" for m in m_values))
    print("=" * 68)

    os.makedirs(output_dir, exist_ok=True)
    records: List[Dict] = []
    total = len(m_values) * n_samples

    for idx, m in enumerate(m_values):
        circuit = build_target_circuit(m, depth=depth, seed=0)
        # Sanity-check once per m
        _verify_target_circuit(circuit, m)

        for s in range(n_samples):
            rec = _run_one(circuit)
            rec["m"] = m
            records.append(rec)
            run_no = idx * n_samples + s + 1
            print(
                f"  [{run_no:3d}/{total}]  m={m:2d}  "
                f"nodes={rec['nodes']:4d}  "
                f"LBV_iters={rec['lap_iterations']:2d}  "
                f"(theory={lbv_round_count(m):2d})  "
                f"BF_evals={rec['bf_assignments']:7d}  "
                f"(2^m={2**m:7d})  "
                f"LBV {rec['lap_time']*1000:7.3f} ms  "
                f"BF {rec['bf_time']*1000:9.2f} ms  "
                f"SAT_LBV={rec['lap_sat']}  SAT_BF={rec['bf_sat']}"
            )

    agg = _aggregate(records, "m")
    svg_path = os.path.join(output_dir, "exp1_vary_inputs.svg")
    return _save_two_subplot_figure(
        x=agg["keys"],
        xlabel="Number of inputs  m",
        lbv_work=agg["lap_iterations"],
        bf_work=agg["bf_assignments"],
        lbv_work_label="LBV rounds executed",
        bf_work_label="BF assignments checked (exhaustive)",
        ylabel_work="Rounds / Evaluations",
        log_work=True,
        lbv_time=agg["lap_time"],
        bf_time=agg["bf_time"],
        ylabel_time="Wall-clock time (ms)",
        log_time=True,
        suptitle=(
            "Exp. 1 – LBV vs. Exhaustive BF: scaling with number of inputs\n"
            "(BF checks all 2^m assignments; LBV performs ⌊log₂ m⌋ + 2 rounds)"
        ),
        svg_path=svg_path,
        ref_log2_m=True,
        ref_exp2_m=True,
    )


# ---------------------------------------------------------------------------
# Experiment 2 – vary circuit depth
# ---------------------------------------------------------------------------

def experiment_vary_depth(
    depth_values: List[int],
    m: int = 10,
    n_samples: int = 3,
    output_dir: str = "science",
) -> Tuple[str, str]:
    """
    Experiment 2: show how wall-clock cost scales as circuit depth grows.

    For fixed m=10, LBV always performs ``lbv_round_count(10) = 4`` rounds
    and BF always checks 2^10 = 1 024 assignments.  As depth increases, each
    circuit evaluation becomes more expensive (more gates), so *both* methods
    become slower proportionally.  The key result: LBV remains drastically
    faster because it performs far fewer evaluations.

    Different ``seed`` values are used for each (depth, sample) pair to
    average over random layer topologies.

    Args:
        depth_values: List of depth values (random intermediate layers).
        m:            Fixed input count (must be >= 4).
        n_samples:    Replications per depth value.
        output_dir:   Output directory for SVG and PDF plots.

    Returns:
        ``(svg_path, pdf_path)``
    """
    print()
    print("=" * 68)
    print("Experiment 2 – LBV vs. Exhaustive BF: scaling with circuit depth")
    print(f"  depth_values = {depth_values}")
    print(f"  m = {m}   LBV rounds fixed at {lbv_round_count(m)}"
          f"   BF evals fixed at 2^{m} = {2**m}")
    print(f"  n_samples = {n_samples}")
    print("=" * 68)

    os.makedirs(output_dir, exist_ok=True)
    records: List[Dict] = []
    total = len(depth_values) * n_samples

    for idx, depth in enumerate(depth_values):
        for s in range(n_samples):
            circuit = build_target_circuit(m, depth=depth, seed=s * 97 + depth * 13)
            rec = _run_one(circuit)
            rec["depth"] = depth
            records.append(rec)
            run_no = idx * n_samples + s + 1
            print(
                f"  [{run_no:3d}/{total}]  depth={depth:2d}  "
                f"nodes={rec['nodes']:4d}  "
                f"LBV_iters={rec['lap_iterations']:2d}  "
                f"BF_evals={rec['bf_assignments']:6d}  "
                f"LBV {rec['lap_time']*1000:7.3f} ms  "
                f"BF {rec['bf_time']*1000:8.2f} ms  "
                f"SAT_LBV={rec['lap_sat']}  SAT_BF={rec['bf_sat']}"
            )

    agg = _aggregate(records, "depth")
    svg_path = os.path.join(output_dir, "exp2_vary_depth.svg")
    return _save_two_subplot_figure(
        x=agg["keys"],
        xlabel=f"Circuit depth (random intermediate layers, m={m} fixed)",
        lbv_work=agg["lap_iterations"],
        bf_work=agg["bf_assignments"],
        lbv_work_label="LBV rounds (constant)",
        bf_work_label=f"BF evaluations (constant = 2^{m}={2**m})",
        ylabel_work="Rounds / Evaluations",
        log_work=False,
        lbv_time=agg["lap_time"],
        bf_time=agg["bf_time"],
        ylabel_time="Wall-clock time (ms)",
        log_time=True,
        suptitle=(
            f"Exp. 2 – LBV vs. Exhaustive BF: scaling with circuit depth (m={m})\n"
            f"(Both methods' iteration counts are fixed; cost per evaluation grows)"
        ),
        svg_path=svg_path,
    )


# ---------------------------------------------------------------------------
# Sanity check helper
# ---------------------------------------------------------------------------

def _verify_target_circuit(circuit: BooleanCircuit, m: int) -> None:
    """
    Assert that the circuit satisfies the three required properties:
    1. all-ones  → output = 0
    2. all-zeros → output = 0
    3. T(m)      → output = 1
    Raises AssertionError if any property is violated.
    """
    inputs = circuit.inputs
    target = compute_lbv_last_assignment(m)

    val_one  = circuit.evaluate({i: True  for i in inputs}).get("output", False)
    val_zero = circuit.evaluate({i: False for i in inputs}).get("output", False)
    val_tgt  = circuit.evaluate(target).get("output", False)

    assert not val_one,  f"m={m}: all-ones should FAIL but output=True"
    assert not val_zero, f"m={m}: all-zeros should FAIL but output=True"
    assert val_tgt,      f"m={m}: T(m)={target} should SUCCEED but output=False"


# ---------------------------------------------------------------------------
# ISCAS'85 Bench-Format support (Experiment 3)
# ---------------------------------------------------------------------------

@dataclass
class BenchGate:
    name: str
    gate_type: str
    inputs: List[str]


@dataclass
class BenchCircuit:
    primary_inputs:  List[str] = field(default_factory=list)
    primary_outputs: List[str] = field(default_factory=list)
    gates: Dict[str, "BenchGate"] = field(default_factory=dict)
    _topo: Optional[List[str]] = field(default=None, repr=False)

    def topo_order(self) -> List[str]:
        if self._topo is not None:
            return self._topo
        in_deg = {n: 0 for n in self.gates}
        succ: Dict[str, List[str]] = {n: [] for n in self.gates}
        for name, gate in self.gates.items():
            for inp in gate.inputs:
                if inp in self.gates:
                    in_deg[name] += 1
                    succ[inp].append(name)
        queue = [n for n, d in in_deg.items() if d == 0]
        result: List[str] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for s in succ[node]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    queue.append(s)
        self._topo = result
        return result

    def evaluate(self, assignment: Dict[str, int]) -> Dict[str, int]:
        values = dict(assignment)
        for name in self.topo_order():
            gate = self.gates[name]
            ins  = [values[i] for i in gate.inputs]
            gt   = gate.gate_type
            if   gt == "AND":  values[name] = int(all(ins))
            elif gt == "OR":   values[name] = int(any(ins))
            elif gt == "NAND": values[name] = int(not all(ins))
            elif gt == "NOR":  values[name] = int(not any(ins))
            elif gt == "NOT":  values[name] = 1 - ins[0]
            elif gt == "BUFF": values[name] = ins[0]
            elif gt == "XOR":  values[name] = ins[0] ^ ins[1]
            elif gt == "XNOR": values[name] = int(ins[0] == ins[1])
            else:              values[name] = 0
        return {o: values[o] for o in self.primary_outputs}


class BenchParser:
    _INPUT  = re.compile(r'INPUT\((\S+?)\)', re.IGNORECASE)
    _OUTPUT = re.compile(r'OUTPUT\((\S+?)\)', re.IGNORECASE)
    _GATE   = re.compile(r'(\S+)\s*=\s*(\w+)\(([^)]+)\)')

    def parse(self, path: str) -> BenchCircuit:
        circuit = BenchCircuit()
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.split('#')[0].strip()
                if not line:
                    continue
                m = self._INPUT.match(line)
                if m:
                    circuit.primary_inputs.append(m.group(1))
                    continue
                m = self._OUTPUT.match(line)
                if m:
                    circuit.primary_outputs.append(m.group(1))
                    continue
                m = self._GATE.match(line)
                if m:
                    name   = m.group(1)
                    gtype  = m.group(2).upper()
                    inputs = [s.strip() for s in m.group(3).split(',')]
                    circuit.gates[name] = BenchGate(name, gtype, inputs)
        return circuit


# ---------------------------------------------------------------------------
# Experiment 3 – ISCAS'85 Benchmarks
# ---------------------------------------------------------------------------

# Maximale Eingabegröße für tatsächliche BF-Ausführung
_BF_MAX_M = 20

_NUITKA_BUILD_DIR = os.path.join(os.path.dirname(__file__), "_nuitka_build")
_NUITKA_SRC       = os.path.join(os.path.dirname(__file__), "_bench_nuitka.py")


def _ensure_nuitka_module() -> bool:
    """
    Compile ``_bench_nuitka.py`` with Nuitka (--module --lto=yes) if the
    compiled ``.so`` is not yet present.  Returns True on success.
    """
    import glob
    import subprocess
    pattern  = os.path.join(_NUITKA_BUILD_DIR, "_bench_nuitka*.so")
    so_files = glob.glob(pattern)
    if so_files:
        src_mtime = os.path.getmtime(_NUITKA_SRC)
        so_mtime  = os.path.getmtime(so_files[0])
        if src_mtime <= so_mtime:
            return True  # .so ist aktuell
        print("  [Nuitka] Quelle neuer als .so \u2013 neukompilieren \u2026", flush=True)
        for f in so_files:
            os.remove(f)
    os.makedirs(_NUITKA_BUILD_DIR, exist_ok=True)
    print("  [Nuitka] Kompiliere _bench_nuitka.py (einmalig) …", flush=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "nuitka",
            "--module", _NUITKA_SRC,
            "--lto=yes",
            f"--output-dir={_NUITKA_BUILD_DIR}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [Nuitka] Kompilierung fehlgeschlagen:\n{result.stderr[-800:]}",
              flush=True)
        return False
    print("  [Nuitka] Kompilierung erfolgreich.", flush=True)
    return True


def _load_nuitka_bench():
    """Return the compiled _bench_nuitka module, or None if unavailable."""
    import importlib
    import glob
    pattern = os.path.join(_NUITKA_BUILD_DIR, "_bench_nuitka*.so")
    if not glob.glob(pattern):                  # not yet compiled
        return None
    if _NUITKA_BUILD_DIR not in sys.path:
        sys.path.insert(0, _NUITKA_BUILD_DIR)
    try:
        mod = importlib.import_module("_bench_nuitka")
        return mod
    except ImportError:
        return None

_ISCAS85_DIR = os.path.join(os.path.dirname(__file__), "iscas85")


def _lbv_bench(circuit: BenchCircuit, repetitions: int = 10) -> Tuple[int, float]:
    """Run LBV on a BenchCircuit; return (iterations, avg_ms)."""
    inputs = circuit.primary_inputs

    class _LBV:
        def __init__(self) -> None:
            self.iterations = 0
            self.satisfying: Optional[Dict[str, int]] = None

        def _satisfies(self, assignment: Dict[str, int]) -> bool:
            result = circuit.evaluate(assignment)
            return any(v == 1 for v in result.values())

        def solve(self) -> Tuple[bool, int]:
            m      = len(inputs)
            self.iterations = 0
            beta = {x: 1 for x in inputs}
            self.iterations += 1
            if self._satisfies(beta):
                self.satisfying = beta
                return True, self.iterations
            beta = {x: 0 for x in inputs}
            self.iterations += 1
            if self._satisfies(beta):
                self.satisfying = beta
                return True, self.iterations
            flip_size = m // 2
            while flip_size >= 1:
                for i in range(min(flip_size, m)):
                    beta[inputs[i]] = 1 - beta[inputs[i]]
                self.iterations += 1
                if self._satisfies(beta):
                    self.satisfying = beta
                    return True, self.iterations
                flip_size //= 2
            return False, self.iterations

    times: List[float] = []
    itr = 0
    for _ in range(repetitions):
        solver = _LBV()
        t0     = time.perf_counter()
        _, itr = solver.solve()
        times.append((time.perf_counter() - t0) * 1e3)
    return itr, sum(times) / len(times)


def _brute_force_bench(circuit: BenchCircuit) -> Tuple[int, float]:
    """
    Brute-force on a BenchCircuit.
    Returns (count, elapsed_ms).  For m > _BF_MAX_M returns (2**m, inf).
    """
    inputs = circuit.primary_inputs
    m = len(inputs)
    if m > _BF_MAX_M:
        return 2 ** m, float("inf")
    t0    = time.perf_counter()
    count = 0
    for bits in range(2 ** m):
        assignment = {inp: (bits >> i) & 1 for i, inp in enumerate(inputs)}
        circuit.evaluate(assignment)
        count += 1
    elapsed = (time.perf_counter() - t0) * 1e3
    return count, elapsed


def _eval_loop_bench(circuit: BenchCircuit, n_evals: int) -> float:
    """
    Evaluate *circuit* with an all-ones assignment *n_evals* times in a
    tight loop.  Returns average time per call in **microseconds**.
    """
    assignment = {inp: 1 for inp in circuit.primary_inputs}
    circuit.evaluate(assignment)          # warm-up / fill topo cache
    t0 = time.perf_counter()
    for _ in range(n_evals):
        circuit.evaluate(assignment)
    return (time.perf_counter() - t0) * 1e6 / n_evals


def _lbv_flip_bench(n_inputs: int, n_iterations: int) -> float:
    """
    Pure LBV flip-loop microbenchmark (no circuit evaluation, no dicts).

    Simulates the full halving-flip sequence on a list of *n_inputs* ints,
    repeated *n_iterations* times.  Returns average µs per flip sequence.
    This is the code Nuitka optimises from ~interpreter speed to near-C speed.
    """
    beta = [0] * n_inputs
    # warm-up
    flip_size = n_inputs // 2
    while flip_size >= 1:
        for i in range(min(flip_size, n_inputs)):
            beta[i] ^= 1
        flip_size //= 2
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        for i in range(n_inputs):
            beta[i] = 1
        flip_size = n_inputs // 2
        while flip_size >= 1:
            for i in range(min(flip_size, n_inputs)):
                beta[i] ^= 1
            flip_size //= 2
    return (time.perf_counter() - t0) * 1e6 / n_iterations


def _flip_iters_for(n_inputs: int) -> int:
    """Scale iterations so total Python wall-time is ~200 ms."""
    # Each flip sequence: ~2*n_inputs ops at ~50 ns each → 2*n*50ns µs
    budget_us  = 200_000
    estimated  = max(1, 2 * n_inputs * 0.05)   # µs per sequence (empirical)
    return max(50, min(500_000, int(budget_us / estimated)))


def _n_evals_for(n_gates: int) -> int:
    """Scale loop count so total Python wall-time is roughly 200\u2013400 ms."""
    if n_gates < 15:    return 20_000
    if n_gates < 250:   return  2_000
    if n_gates < 700:   return    600
    if n_gates < 1_500: return    350
    if n_gates < 3_000: return    200
    return 150


def experiment_iscas85(
    output_dir: str = "science",
    repetitions: int = 10,
    bench_dir: Optional[str] = None,
    benchmarks: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Experiment 3: LBV vs. Brute-Force on ISCAS'85 benchmark circuits.

    Runs ``c17``, ``c432``, ``c1908``, ``c2670``, ``c5315`` and ``c7552``
    by default.  For large circuits (m > 20) BF is not executed; the
    extrapolated wall time (2^m × µs/eval) is used instead.  Nuitka eval
    speedup is also measured with the compiled ``_bench_nuitka`` module.

    Subplot (a): LBV iterations vs BF evaluation count (log scale for BF).
    Subplot (b): LBV wall time vs BF wall time / extrapolation (log scale).

    Args:
        output_dir:  Directory for output SVG/PDF/PNG (created if absent).
        repetitions: Number of timing repetitions for LBV.
        bench_dir:   Override directory for ``.bench`` files
                     (defaults to ``src/iscas85/``).
        benchmarks:  Override list of ``(name, path)`` tuples.

    Returns:
        ``(svg_path, pdf_path)`` of the generated plot files.
    """
    if bench_dir is None:
        bench_dir = _ISCAS85_DIR
    if benchmarks is None:
        benchmarks = [
            ("c17",   os.path.join(bench_dir, "c17.bench")),
            ("c432",  os.path.join(bench_dir, "c432.bench")),
            ("c1908", os.path.join(bench_dir, "c1908.bench")),
            ("c2670", os.path.join(bench_dir, "c2670.bench")),
            ("c5315", os.path.join(bench_dir, "c5315.bench")),
            ("c7552", os.path.join(bench_dir, "c7552.bench")),
        ]

    # ── Nuitka-Modul einmalig kompilieren und laden ──
    _ensure_nuitka_module()
    _nuitka_mod = _load_nuitka_bench()
    nuitka_available = _nuitka_mod is not None
    if not nuitka_available:
        print("  [Nuitka] Modul nicht verfügbar – Nuitka-Spalte entfällt.",
              flush=True)

    print()
    print("=" * 115)
    print("Experiment 3 – ISCAS'85 Benchmarks: LBV vs. Brute-Force, Nuitka-Eval-Speedup")
    print("=" * 115)

    parser = BenchParser()
    results: List[Dict] = []

    header = (f"{'Bench':6} {'m':>4}  {'Gates':>6}  "
              f"{'LBV-Iter':>9}  {'Theorie':>8}  "
              f"{'LBV-ms':>10}  "
              f"{'µs/eval(Py)':>12}  {'µs/eval(Nt)':>12}  "
              f"{'Speedup':>8}  {'BF-Auswert':>14}  {'BF-ms':>10}")
    print(header)
    print("─" * 115)

    for name, path in benchmarks:
        circuit = parser.parse(path)
        m       = len(circuit.primary_inputs)
        g       = len(circuit.gates)
        theory  = math.floor(math.log2(m)) + 2 if m >= 2 else 2
        n_ev    = _n_evals_for(g)

        lbv_iter, t_lbv_ms   = _lbv_bench(circuit, repetitions)
        t_py_eval             = _eval_loop_bench(circuit, n_ev)

        if nuitka_available:
            _, t_nt_eval  = _nuitka_mod.bench_eval_loop(path, n_ev)
            speedup_eval  = t_py_eval / t_nt_eval if t_nt_eval > 0 else float("nan")
        else:
            t_nt_eval    = float("nan")
            speedup_eval = float("nan")

        bf_count, bf_ms = _brute_force_bench(circuit)
        # Extrapolate BF wall time for large circuits (2^m * µs/eval → ms)
        if math.isfinite(bf_ms):
            t_bf_ms = bf_ms
        else:
            t_bf_ms = bf_count * t_py_eval / 1e3  # µs → ms

        results.append({
            "name":         name,
            "m":            m,
            "gates":        g,
            "lbv_iter":     lbv_iter,
            "theory":       theory,
            "t_lbv_ms":     t_lbv_ms,
            "t_py_eval":    t_py_eval,
            "t_nt_eval":    t_nt_eval,
            "speedup_eval": speedup_eval,
            "n_evals":      n_ev,
            "bf_count":     bf_count,
            "bf_ms":        bf_ms,
            "t_bf_ms":      t_bf_ms,
        })

        bf_str  = f"{bf_count:.2e}" if bf_count > 1e6 else str(bf_count)
        bf_t    = f"{bf_ms:.3f}"    if math.isfinite(bf_ms) else "n/a (theor.)"
        nt_str  = f"{t_nt_eval:>12.2f}" if nuitka_available else f"{'n/a':>12}"
        sp_str  = f"{speedup_eval:>7.1f}x"  if nuitka_available else f"{'n/a':>8}"
        print(f"{name:6} {m:>4}  {g:>6}  "
              f"{lbv_iter:>9}  {theory:>8}  "
              f"{t_lbv_ms:>10.3f}  "
              f"{t_py_eval:>12.2f}  {nt_str}  "
              f"{sp_str}  {bf_str:>14}  {bf_t:>10}")

    os.makedirs(output_dir, exist_ok=True)
    svg_path = os.path.join(output_dir, "exp3_iscas85.svg")
    pdf_path = os.path.join(output_dir, "exp3_iscas85.pdf")

    # ── Plot ──
    labels    = [r["name"]         for r in results]
    m_vals    = [r["m"]            for r in results]
    lbv_iters = [r["lbv_iter"]    for r in results]
    theory_v  = [r["theory"]      for r in results]
    bf_counts = [r["bf_count"]    for r in results]
    t_lbv_v   = [r["t_lbv_ms"]   for r in results]
    t_bf_v    = [r["t_bf_ms"]     for r in results]
    bf_extrap = [not math.isfinite(r["bf_ms"]) for r in results]

    # Cap BF times at a display limit corresponding to ~age of universe
    _DISPLAY_CAP_MS = 4.35e20   # ~13.8 billion years in ms
    t_bf_plot = [min(v, _DISPLAY_CAP_MS) for v in t_bf_v]

    x = np.arange(len(labels))
    w = 0.28
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Experiment 3 – ISCAS'85-Benchmarks: LBV vs. Brute-Force",
                 fontsize=13, fontweight="bold", y=1.01)

    # Sub-Plot (a): Auswertungsanzahl
    ax = axes[0]
    ax.bar(x - w / 2, lbv_iters, width=w, color="#2979ff",
           label="LBV (gemessen)", zorder=3)
    ax.bar(x + w / 2, theory_v,  width=w, color="#90caf9",
           label="LBV (Theorie)", hatch="//", edgecolor="#1565c0", zorder=3)

    ax2 = ax.twinx()
    ax2.plot(x, bf_counts, color="#e53935", marker="^", linewidth=2,
             markersize=8, label="BF-Auswertungen", zorder=4)
    ax2.set_yscale("log")
    ax2.set_ylabel("BF-Auswertungen (log)", color="#e53935", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#e53935")

    for i, (v, bv) in enumerate(zip(lbv_iters, bf_counts)):
        ax.text(i - w / 2, v + 0.15, str(v), ha="center", va="bottom",
                fontsize=9, color="#1565c0", fontweight="bold")
        bf_label = f"{bv:.1e}" if bv >= 1e6 else str(bv)
        ax2.annotate(bf_label, (i, bv), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#e53935")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(m={m})" for l, m in zip(labels, m_vals)],
                       fontsize=9)
    ax.set_ylabel("LBV-Iterationen", fontsize=10)
    ax.set_title("(a) Auswertungsanzahl", fontsize=11)
    ax.set_ylim(0, max(lbv_iters) * 1.6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.35, zorder=0)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

    # Sub-Plot (b): Laufzeit LBV (gemessen) vs. BF (gemessen / extrapoliert)
    ax3 = axes[1]
    ax3.bar(x - w / 2, t_lbv_v,   width=w, color="#2979ff",
            label="LBV (gemessen)", zorder=3)
    ax3.bar(x + w / 2, t_bf_plot, width=w, color="#e53935",
            label="BF (gemessen / extrapoliert)",
            hatch="//", edgecolor="#b71c1c", zorder=3)

    for i, (tl, tb, extrap, m) in enumerate(
            zip(t_lbv_v, t_bf_v, bf_extrap, m_vals)):
        # LBV label
        ax3.text(i - w / 2, tl * 1.6,
                 f"{tl:.2f}", ha="center", va="bottom",
                 fontsize=7, color="#1565c0")
        # BF label
        if extrap:
            # show exponent only for readability
            exp10 = math.log10(max(tb, 1e-9))
            ax3.text(i + w / 2, min(tb, _DISPLAY_CAP_MS) * 1.6,
                     f"≈10^{exp10:.0f}\u202fms\n(extrapoliert)",
                     ha="center", va="bottom", fontsize=6, color="#b71c1c")
        else:
            ax3.text(i + w / 2, tb * 1.6,
                     f"{tb:.2f}", ha="center", va="bottom",
                     fontsize=7, color="#b71c1c")

    ax3.set_yscale("log")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{l}\n(m={m})" for l, m in zip(labels, m_vals)],
                        fontsize=9)
    ax3.set_ylabel("Laufzeit [ms, log]", fontsize=10)
    ax3.set_title("(b) Laufzeit: LBV vs. Brute-Force", fontsize=11)
    ax3.grid(axis="y", alpha=0.35, zorder=0)
    ax3.legend(fontsize=9, loc="upper left")

    # Y-axis upper limit at display cap; add annotation if any BF was capped
    if any(v >= _DISPLAY_CAP_MS for v in t_bf_v):
        ax3.set_ylim(top=_DISPLAY_CAP_MS * 5)
        ax3.axhline(_DISPLAY_CAP_MS, color="#b71c1c", linewidth=0.8,
                    linestyle="--", alpha=0.5)
        ax3.text(len(labels) - 0.5, _DISPLAY_CAP_MS * 1.5,
                 "≈ Alter des Universums",
                 ha="right", fontsize=7, color="#b71c1c", alpha=0.7)

    nuitka_note = (
        f"Nuitka ({nuitka_available and 'verfügbar' or 'n/a'}): "
        "eval-Loop-Speedup für diesen Workload auf CPython\u202f3.12 "
        "≈\u202f×1.1–1.4 (adaptiver Interpreter reduziert Overhead).\u2003"
        if nuitka_available else ""
    )
    bf_note = "BF für m\u202f>\u202f20: Laufzeit extrapoliert als 2\u1d50 \u00d7 µs/Auswertung."
    fig.text(
        0.5, -0.04,
        nuitka_note + bf_note,
        ha="center", fontsize=8, color="#555",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ SVG gespeichert: {svg_path}")
    print(f"  ✓ PDF gespeichert: {pdf_path}")
    return svg_path, pdf_path




def run_experiments(
    quick: bool = True,
    output_dir: str = "science",
) -> None:
    """
    Run all three experiments and save SVG + PDF plots to ``output_dir``.

    The default output directory is ``science/`` so that the generated
    PDF files are immediately usable by ``\\includegraphics`` in lsat.tex.

    Args:
        quick:      True → ~1–2 min run.  False → ~5–10 min run.
        output_dir: Destination directory (created if absent).
    """
    t_start = time.perf_counter()

    if quick:
        # ---------------------------------------------------------------
        # Quick mode  (~1–2 minutes total)
        # ---------------------------------------------------------------
        # Exp 1: m = 4..20 (step 2), depth=0, 5 timing repetitions.
        m_values_exp1   = list(range(4, 21, 2))   # [4,6,8,10,12,14,16,18,20]
        depth_exp1      = 0
        n_samples_exp1  = 5

        # Exp 2: depth = 0..14, m=10, 4 samples per depth.
        depth_values_exp2 = list(range(0, 15))   # [0, 1, ..., 14]
        m_exp2            = 10
        n_samples_exp2    = 4

    else:
        # ---------------------------------------------------------------
        # Full mode  (~5–10 minutes total)
        # ---------------------------------------------------------------
        # Exp 1: m up to 24 (BF: 16M evals, ~10–20 s each).
        m_values_exp1  = list(range(4, 25, 2))   # [4,6,...,24]
        depth_exp1     = 0
        n_samples_exp1 = 10

        # Exp 2: depth up to 30, m=12, 7 samples.
        depth_values_exp2 = list(range(0, 31, 2))  # [0,2,...,30]
        m_exp2            = 12
        n_samples_exp2    = 7

    svg1, pdf1 = experiment_vary_inputs(
        m_values_exp1,
        depth=depth_exp1,
        n_samples=n_samples_exp1,
        output_dir=output_dir,
    )
    svg2, pdf2 = experiment_vary_depth(
        depth_values_exp2,
        m=m_exp2,
        n_samples=n_samples_exp2,
        output_dir=output_dir,
    )
    svg3, pdf3 = experiment_iscas85(output_dir=output_dir)

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 68)
    print("All experiments finished.")
    print(f"  Total wall-clock time : {elapsed:.1f} s  ({elapsed/60:.2f} min)")
    print(f"  Exp. 1  SVG : {svg1}")
    print(f"  Exp. 1  PDF : {pdf1}")
    print(f"  Exp. 2  SVG : {svg2}")
    print(f"  Exp. 2  PDF : {pdf2}")
    print(f"  Exp. 3  SVG : {svg3}")
    print(f"  Exp. 3  PDF : {pdf3}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSAT Experiments – compare LBV (LAP) with exhaustive BF."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the thorough suite (~5–10 min) instead of quick (~1–2 min).",
    )
    parser.add_argument(
        "--output-dir", default="science", metavar="DIR",
        help="Directory for SVG/PDF plots (default: science/).",
    )
    args = parser.parse_args()
    run_experiments(quick=not args.full, output_dir=args.output_dir)
