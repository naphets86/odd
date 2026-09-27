"""
Standalone LBV benchmark module – compiled with Nuitka for real speedup measurement.

Usage (Python):
    python _bench_nuitka.py path/to/circuit.bench [repetitions]

Compiled usage:
    python -m nuitka --module _bench_nuitka.py --lto=yes --output-dir=<dir>
    → produces _bench_nuitka.<cpython-tag>.so  (importable)

The public API after compilation is identical to this source:
    bench_lbv(path: str, repetitions: int = 10) -> tuple[int, float]
    bench_lbv_npy(path: str, repetitions: int = 10) -> tuple[int, float]  # alias
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Circuit data model (self-contained, no external imports)
# ---------------------------------------------------------------------------

@dataclass
class BenchGate:
    name: str
    gate_type: str
    inputs: List[str]


@dataclass
class BenchCircuit:
    primary_inputs:  List[str]                  = field(default_factory=list)
    primary_outputs: List[str]                  = field(default_factory=list)
    gates:           Dict[str, BenchGate]        = field(default_factory=dict)
    _topo:           Optional[List[str]]         = field(default=None, repr=False)

    def topo_order(self) -> List[str]:
        if self._topo is not None:
            return self._topo
        in_deg: Dict[str, int]        = {n: 0 for n in self.gates}
        succ:   Dict[str, List[str]]  = {n: [] for n in self.gates}
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
    _INPUT  = re.compile(r'INPUT\((\S+?)\)',   re.IGNORECASE)
    _OUTPUT = re.compile(r'OUTPUT\((\S+?)\)',  re.IGNORECASE)
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
# Inline LBV (mirrors the implementation in experiment_iscas85)
# ---------------------------------------------------------------------------

def bench_lbv(path: str, repetitions: int = 10) -> tuple:
    """
    Parse *path*, run the inline LBV *repetitions* times and return
    ``(iterations, avg_ms)``.
    """
    circuit = BenchParser().parse(path)
    inputs  = circuit.primary_inputs

    class _LBV:
        def __init__(self) -> None:
            self.iterations = 0
            self.satisfying: Optional[Dict[str, int]] = None

        def _satisfies(self, assignment: Dict[str, int]) -> bool:
            result = circuit.evaluate(assignment)
            return any(v == 1 for v in result.values())

        def solve(self) -> tuple:
            m             = len(inputs)
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


def bench_eval_loop(path: str, n_evals: int = 1000) -> tuple:
    """
    Parse *path*, pre-build an all-ones assignment, then call
    ``circuit.evaluate()`` *n_evals* times in a tight loop.

    Returns ``(n_gates, avg_us_per_eval)`` where *avg_us_per_eval* is the
    average time per single evaluate() call in **microseconds**.
    """
    circuit    = BenchParser().parse(path)
    assignment = {inp: 1 for inp in circuit.primary_inputs}
    # warm-up (fills topo cache, pre-compiles hot path in Nuitka)
    circuit.evaluate(assignment)
    t0 = time.perf_counter()
    for _ in range(n_evals):
        circuit.evaluate(assignment)
    elapsed_us = (time.perf_counter() - t0) * 1e6
    return len(circuit.gates), elapsed_us / n_evals


def bench_lbv_flip(n_inputs: int, n_iterations: int = 10_000) -> float:
    """
    Pure LBV flip-loop microbenchmark (no circuit evaluation).

    Simulates the full halving-flip sequence of the LBV for a circuit with
    *n_inputs* primary inputs, repeated *n_iterations* times.
    Uses a Python list of ints (no dicts) -- the code Nuitka optimises best.

    Returns average time in **microseconds** per LBV flip sequence.
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
    elapsed_us = (time.perf_counter() - t0) * 1e6
    return elapsed_us / n_iterations


# ---------------------------------------------------------------------------
# CLI entry point (used when run as script, not needed by nuitka --module)
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: _bench_nuitka.py <path.bench> [repetitions]", file=sys.stderr)
        sys.exit(1)
    path        = sys.argv[1]
    repetitions = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    itr, avg_ms = bench_lbv(path, repetitions)
    print(json.dumps({"iterations": itr, "avg_ms": avg_ms}))


if __name__ == "__main__":
    main()
