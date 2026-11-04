#!/usr/bin/env python3
"""Deep-circuit check: many layers, few qubits. chunk_layers splits the gate
list into passes of <= 512 layers (LLAMA_MAX_LAYERS), so depth is unbounded -
it just costs one process spawn per chunk.

    ./venv/bin/python examples/deep_circuit.py [n_qubits] [depth]
"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.bench_quant import random_circuit
from qullama import simulate, statevector_reference


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    qc = random_circuit(n, depth, seed=3)
    ngates = len(qc.data)
    nchunks = (ngates + 511) // 512
    print(f"n={n} depth={depth} -> {ngates} gates, {nchunks} chunk(s) of <=512 layers")

    t0 = time.perf_counter()
    ref = statevector_reference(qc)
    tq = time.perf_counter() - t0
    t0 = time.perf_counter()
    psi = simulate(qc, chunk_layers=512)
    dt = time.perf_counter() - t0
    err = float(np.max(np.abs(psi - ref)))
    print(f"qiskit Statevector: {tq:.2f}s")
    print(f"qullama:            {dt:.2f}s  ({dt / tq:.2f}x qiskit)  max err vs qiskit = {err:.2e}  (sum p = {np.sum(np.abs(psi)**2):.8f})")


if __name__ == "__main__":
    main()
