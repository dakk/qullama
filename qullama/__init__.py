"""qullama: run OpenQASM quantum circuits inside patched llama.cpp via qgate GGUFs."""

from .qgate import build_gguf_gates, write_qgate_gguf, unitary_to_real_matrix, DEFAULT_PROMPT_TOKEN
from .qiskit_io import from_qasm, iter_gate_ops, load_circuit_file, statevector_reference
from .backend import simulate, run_gguf, QullamaBackend, QullamaJob

__all__ = [
    "build_gguf_gates",
    "write_qgate_gguf",
    "unitary_to_real_matrix",
    "DEFAULT_PROMPT_TOKEN",
    "from_qasm",
    "iter_gate_ops",
    "load_circuit_file",
    "statevector_reference",
    "simulate",
    "run_gguf",
    "QullamaBackend",
    "QullamaJob",
]
