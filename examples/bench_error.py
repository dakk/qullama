#!/usr/bin/env python3
"""Error benchmark: qullama at different quantizations vs qiskit Statevector.

Sweeps circuit size (n_qubits x depth) and gate-weight dtype, and reports the
max amplitude error of qullama's output state vs qiskit's Statevector.

Usage:
    python3 examples/bench_error.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qullama import simulate, statevector_reference
from qullama.qiskit_io import iter_gate_ops  # noqa: F401  (ensures qiskit path works)

DTYPES = ["F32", "F16", "BF16", "Q8_0", "Q4_0"]


def random_circuit(n: int, depth: int, seed: int):
    """Random circuit: depth rounds of random 1q rotations + random cx per round."""
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            g = rng.choice(["h", "s", "t", "rx", "ry", "rz"])
            if g in ("rx", "ry", "rz"):
                getattr(qc, g)(rng.uniform(0, 6.28), q)
            else:
                getattr(qc, g)(q)
        if n > 1:
            c, t = rng.randrange(n), rng.randrange(n)
            if c != t:
                qc.cx(c, t)
    return qc


def main() -> None:
    sizes = [(2, 2), (3, 4), (4, 8), (6, 8), (8, 16)]
    seed = 11

    print(f"max |amp| error vs qiskit Statevector   (rows: n_qubits x depth, cols: dtype)")
    header = f"{'n x depth':>10} {'gates':>5} | " + " | ".join(f"{d:>9}" for d in DTYPES)
    print(header)
    print("-" * len(header))
    for n, depth in sizes:
        qc = random_circuit(n, depth, seed)
        ref = statevector_reference(qc)
        ngates = len(qc.data)
        cells = []
        for dt in DTYPES:
            psi = simulate(qc, dtype=dt)
            err = float(np.max(np.abs(psi - ref)))
            cells.append(f"{err:9.2e}")
        print(f"{f'{n} x {depth}':>10} {ngates:>5} | " + " | ".join(cells))

    print("\nnote: the state embedding is always F32; only the gate weights are quantized.")
    print("F32/F16/BF16 are limited by fp32 storage + fp32 math; Q8_0/Q4_0 by 8-bit/4-bit weights.")


if __name__ == "__main__":
    main()
