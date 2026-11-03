"""Qiskit frontend - the only place qiskit is touched.

Everything (OpenQASM parsing, transpilation to a supported basis, gate matrices,
control-qubit metadata, reference statevectors) comes from qiskit itself:
  * qasm2 / qasm3 for OpenQASM 2/3 parsing
  * transpile() to BASIS_GATES (every op there has .to_matrix())
  * op.to_matrix() / op.base_gate + op.num_ctrl_qubits + op.ctrl_state for gates
  * quantum_info.Statevector for the reference output state

Qubit ordering matches the kernel: qiskit qubit i == bit i of the state index
(little-endian).
"""

from __future__ import annotations

import numpy as np

# gates the qgate kernel implements; qiskit transpiles everything into these
BASIS_GATES = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz",
               "p", "cp", "cx", "cz", "ccx", "swap"]


def from_qasm(qasm: str):
    """Parse an OpenQASM 2 or 3 string into a qiskit.QuantumCircuit."""
    from qiskit import qasm2, qasm3

    try:
        return qasm2.loads(qasm)
    except Exception:
        return qasm3.loads(qasm)


def load_circuit_file(path: str):
    """Load an OpenQASM 2/3 (.qasm) file into a qiskit.QuantumCircuit."""
    from pathlib import Path

    p = Path(path)
    if p.suffix != ".qasm":
        raise ValueError(f"only .qasm (OpenQASM 2/3) files are supported, got {p}")
    return from_qasm(p.read_text())


def iter_gate_ops(qc):
    """Yield (matrix, targets, controls) per gate of a QuantumCircuit.

    The circuit is first transpiled to BASIS_GATES, so every op is a standard
    gate with to_matrix(). Controlled gates arrive as ControlledGate: the base
    matrix comes from base_gate, controls from the leading qubits.
    """
    from qiskit import transpile

    qc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        if op.name in ("measure", "reset", "delay"):
            raise ValueError(f"unsupported non-unitary instruction: {op.name}")
        qargs = [qc.find_bit(q).index for q in inst.qubits]
        nc = int(getattr(op, "num_ctrl_qubits", 0))
        if nc:
            if int(op.ctrl_state) != (1 << nc) - 1:
                raise ValueError(f"open controls not supported: {op.name}")
            controls, targets = qargs[:nc], qargs[nc:]
            M = op.base_gate.to_matrix()
        else:
            controls, targets = [], qargs
            M = op.to_matrix()
        yield np.asarray(M, dtype=np.complex128), targets, controls


def statevector_reference(qc) -> np.ndarray:
    """Reference output state from qiskit (initial state |0...0>)."""
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(qc).data, dtype=np.complex128)
