#!/usr/bin/env python3
"""Tests. All quantum references come from qiskit (Operator / Statevector);
the emulator itself is checked end-to-end via scripts/eval_llama_cpp.py.

Run:  python3 tests/test_qgate.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qullama import build_gguf_gates, unitary_to_real_matrix
from qullama import from_qasm, iter_gate_ops, load_circuit_file, statevector_reference
from qullama import simulate, QullamaBackend
from gguf import GGUFReader  # official gguf-py from PyPI


def patched_binary_usable() -> bool:
    "True if the patched llama-embedding exists AND can actually execute"
    import subprocess

    binary = ROOT / "llama.cpp" / "build" / "bin" / "llama-embedding"
    if not binary.exists():
        return False
    try:
        return subprocess.run([str(binary), "-h"], capture_output=True).returncode == 0
    except OSError:
        return False


def test_gate_extraction():
    # bell.qasm -> transpiled to basis -> (matrix, targets, controls) per gate
    from qiskit.circuit.library import HGate, XGate

    qc = load_circuit_file(ROOT / "examples" / "bell.qasm")
    ops = list(iter_gate_ops(qc))
    assert len(ops) == 2
    (M0, t0, c0), (M1, t1, c1) = ops
    assert np.allclose(M0, HGate().to_matrix()) and t0 == [0] and c0 == []
    assert np.allclose(M1, XGate().to_matrix()) and t1 == [1] and c1 == [0]  # cx: control q0, target q1
    print("ok: gate extraction (matrix/targets/controls) from qiskit ops")


def test_qasm_frontend():
    # bell.qasm: qiskit Statevector gives the Bell state
    qc = load_circuit_file(ROOT / "examples" / "bell.qasm")
    ref = statevector_reference(qc)
    assert np.allclose(np.abs(ref) ** 2, [0.5, 0, 0, 0.5], atol=1e-12)

    # qft3.qasm (OpenQASM 3) on |001>: equals the DFT matrix column 1
    from qiskit import QuantumCircuit

    qc3 = load_circuit_file(ROOT / "examples" / "qft3.qasm")
    qc_in = QuantumCircuit(3)
    qc_in.x(0)
    qc_in = qc_in.compose(qc3)
    ref3 = statevector_reference(qc_in)
    d = 8
    qft = np.array([[np.exp(2j * math.pi * j * k / d) / math.sqrt(d) for k in range(d)] for j in range(d)])
    assert np.allclose(ref3, qft @ np.eye(d)[1], atol=1e-12)
    print("ok: qasm frontend (bell.qasm, qft3.qasm) via qiskit Statevector")


def test_qgate_gguf_structure():
    qc = load_circuit_file(ROOT / "examples" / "qft3.qasm")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.gguf"
        info = build_gguf_gates(qc, path)
        assert info["n_qubits"] == 3 and info["n_embd"] == 16
        gates = list(iter_gate_ops(qc))
        assert info["n_gates"] == len(gates)

        r = GGUFReader(str(path))
        T = {t.name: t for t in r.tensors}
        assert "token_embd.weight" in T and "output_norm.weight" in T
        for i in range(info["n_gates"]):
            assert f"blk.{i}.qgate.weight" in T
            assert f"blk.{i}.qgate_meta" in T

        def kv(name):
            return r.fields[name].contents()
        assert kv("general.architecture") == "qgate"
        assert int(kv("qgate.embedding_length")) == 16
        assert int(kv("qgate.block_count")) == info["n_gates"]
        assert int(kv("qgate.gate_dim")) == info["gate_dim"]
        assert str(kv("tokenizer.ggml.add_bos_token")) == "False"

        # per-gate contents match iter_gate_ops
        gate_dim = info["gate_dim"]
        for i, (G, targets, controls) in enumerate(gates):
            w = np.array(T[f"blk.{i}.qgate.weight"].data, dtype=np.float64).reshape(gate_dim, gate_dim)
            k = len(targets)
            g = 1 << (k + 1)
            M = unitary_to_real_matrix(G)
            assert np.allclose(w[:g, :g], M, atol=1e-12)
            assert np.allclose(w[g:, :], 0) and np.allclose(w[:, g:], 0)
            meta = np.array(T[f"blk.{i}.qgate_meta"].data, dtype=np.int32)
            assert meta[0] == k and meta[1] == len(controls)
            assert list(meta[2:2 + k]) == list(targets)
            assert list(meta[2 + k:2 + k + len(controls)]) == list(controls)

        # embedding row of the prompt token = sqrt(2d) * |0...0>
        n_embd = 16
        emb = np.array(T["token_embd.weight"].data, dtype=np.float64).reshape(2, n_embd)
        x0 = np.zeros(n_embd)
        x0[0] = math.sqrt(n_embd)
        assert np.allclose(emb[1], x0)  # "q" is token id 1
    print("ok: qgate gguf structure (tensors, meta, metadata, embedding row)")


def test_end_to_end_patched_binary():
    """Run bell.qasm through the patched llama-embedding and compare to qiskit."""
    import subprocess

    binary = ROOT / "llama.cpp" / "build" / "bin" / "llama-embedding"
    if not patched_binary_usable():
        print("skip: patched llama-embedding not runnable here "
              "(not built, or missing system libs like libgomp) - e2e test skipped")
        return
    qc = load_circuit_file(ROOT / "examples" / "bell.qasm")
    with tempfile.TemporaryDirectory() as td:
        gguf = Path(td) / "bell.gguf"
        info = build_gguf_gates(qc, gguf)
        d = info["d"]
        p = subprocess.run(
            [str(binary), "-m", str(gguf), "-p", "q", "--embd-normalize", "-1",
             "-n", "1", "--no-warmup", "--embd-output-format", "raw"],
            capture_output=True, text=True)
        assert p.returncode == 0, f"llama-embedding failed: {p.stderr[-500:]}"
        line = next(l for l in p.stdout.splitlines() if l.strip() and not l.startswith("{"))
        emb = np.array([float(x) for x in line.split()])
        psi_out = (emb[:d] + 1j * emb[d:]) / math.sqrt(2 * d)
        ref = statevector_reference(qc)
        err = np.max(np.abs(psi_out - ref))
        assert err < 1e-6, f"e2e error {err}"
        print(f"ok: e2e patched binary vs qiskit (bell.qasm, max err {err:.2e})")


def test_library_backend():
    """simulate() / QullamaBackend().run() vs qiskit Statevector on a 3-qubit circuit."""
    if not patched_binary_usable():
        print("skip: patched binary not runnable here - library backend test skipped")
        return
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cp(0.7853981633974483, 1, 2)
    qc.ry(0.3, 2)
    ref = statevector_reference(qc)

    psi = simulate(qc)
    assert np.max(np.abs(psi - ref)) < 1e-6

    # chunked execution (1 gate per pass) must give the same state
    psi_chunked = simulate(qc, chunk_layers=1)
    assert np.max(np.abs(psi_chunked - ref)) < 1e-6

    sv = QullamaBackend().run(qc).result().get_statevector()
    assert np.max(np.abs(np.asarray(sv) - ref)) < 1e-6
    print("ok: library API (simulate + chunked + QullamaBackend.run) vs qiskit Statevector")


def test_quantization_dtypes():
    """Gate-weight quantization: F32 exact, Q8_0/Q4_0/F16 close, via the real binary."""
    if not patched_binary_usable():
        print("skip: patched binary not runnable here - quantization test skipped")
        return
    qc = load_circuit_file(ROOT / "examples" / "qft3.qasm")
    ref = statevector_reference(qc)
    expect = {"F32": 1e-6, "F16": 1e-2, "BF16": 5e-2, "Q8_0": 1e-2, "Q4_0": 5e-2}
    for dtype, tol in expect.items():
        psi = simulate(qc, dtype=dtype)
        err = float(np.max(np.abs(psi - ref)))
        assert err < tol, f"{dtype}: err {err} >= {tol}"
        print(f"  ok: dtype={dtype:5s} max err {err:.2e}")
    print("ok: quantization dtypes (F32/F16/BF16/Q8_0/Q4_0) vs qiskit")


if __name__ == "__main__":
    test_gate_extraction()
    test_qasm_frontend()
    test_qgate_gguf_structure()
    test_end_to_end_patched_binary()
    test_library_backend()
    test_quantization_dtypes()
    print("\nall tests passed")
