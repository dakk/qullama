#!/usr/bin/env python3
"""Use qullama as a library: build the circuit directly in Python (qiskit),
simulate it with the patched llama.cpp backend, cross-check vs qiskit.

    python3 examples/library_demo.py
"""

import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qullama import simulate, QullamaBackend

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cp(0.7853981633974483, 1, 2)
qc.ry(0.3, 2)

# 1) plain function
psi = simulate(qc)
ref = Statevector(qc).data
print("simulate():   max err vs qiskit =", np.max(np.abs(psi - ref)))
print("probabilities:", dict(zip([f"{i:03b}" for i in range(8)], np.round(np.abs(psi) ** 2, 6))))

# 2) as a qiskit backend
backend = QullamaBackend()          # uses ./llama.cpp/build/bin/llama-embedding
job = backend.run(qc)
sv = job.result().get_statevector()
print("backend.run(): max err vs qiskit =", np.max(np.abs(np.asarray(sv) - ref)))
