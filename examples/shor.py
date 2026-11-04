#!/usr/bin/env python3
"""Shor's factoring algorithm on qullama (patched llama.cpp).

Layout: t=2n counting qubits + n work qubits = 3n qubits total, so at the
~22-qubit practical limit n=7 and we can factor N < 128 (largest with two
distinct prime factors: N=123 = 3*41).

The modular multiplier U_c: |x> -> |c*x mod N> is a permutation of Z_N,
extended to a bijection of the 2^n work-register basis states (identity on
x >= N), so it is a single n-qubit unitary gate - controlled on each counting
qubit for the phase estimation. That makes the whole circuit only
t (H) + 1 (X) + t (controlled permutations) + inverse-QFT gates.

Gates are emitted as a raw list [(matrix, targets, controls)] and run on
patched llama.cpp via qgate GGUFs; a mirror qiskit circuit provides the
Statevector cross-check.

Usage:
    ./venv/bin/python examples/shor.py                 # N=15, a=2 (12 qubits, full cross-check)
    ./venv/bin/python examples/shor.py 123 2           # N=123 = 3*41 (21 qubits - the largest at this size)
    ./venv/bin/python examples/shor.py --selftest-only
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, UnitaryGate
from qiskit.quantum_info import Statevector

from qullama import run_gguf
from qullama.qgate import write_qgate_gguf

MAX_QUBITS = 22
CHUNK_LAYERS = 512  # LLAMA_MAX_LAYERS

X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def add(qc, gates, M, targets, controls=()):
    M = np.asarray(M, dtype=np.complex128)
    targets, controls = list(targets), list(controls)
    if qc is None:
        gates.append((M, targets, controls))
        return
    if controls:
        # qiskit's UnitaryGate.control() re-synthesizes (slow for big M); build the
        # controlled matrix directly. Qubit order: controls first (low bits).
        nc, k = len(controls), len(targets)
        mask = (1 << nc) - 1  # controls are the low bits (qiskit little-endian)
        Mfull = np.eye(1 << (nc + k), dtype=np.complex128)
        for i in range(1 << k):
            for j in range(1 << k):
                Mfull[mask | (i << nc), mask | (j << nc)] = M[i, j]
        qc.append(UnitaryGate(Mfull), controls + targets)
    else:
        qc.append(UnitaryGate(M), targets)
    gates.append((M, targets, controls))


def add_qiskit_op(qc, gates, src_circ, inst, qubit_map):
    op = inst.operation
    if op.name == "barrier":
        return
    qargs = [qubit_map[src_circ.find_bit(q).index] for q in inst.qubits]
    nc = int(getattr(op, "num_ctrl_qubits", 0) or 0)
    M = op.base_gate.to_matrix() if nc else op.to_matrix()
    add(qc, gates, np.asarray(M, dtype=np.complex128), qargs[nc:], qargs[:nc])


def simulate_raw(n, gates, dtype="F32", chunk_layers=CHUNK_LAYERS, verbose=True, ctx=8):
    """Run a raw gate list [(M, targets, controls)] on patched llama.cpp."""
    d = 1 << n
    n_embd = 2 * d
    scale = math.sqrt(n_embd)
    x0 = np.zeros(n_embd)
    x0[0] = scale
    ngates = len(gates)
    with tempfile.TemporaryDirectory() as td:
        for ci in range(0, ngates, chunk_layers):
            chunk = gates[ci:ci + chunk_layers]
            p = Path(td) / f"chunk{ci}.gguf"
            write_qgate_gguf(n, chunk, x0, p, context_length=len(chunk), dtype=dtype)
            x0 = run_gguf(p, ctx=ctx)
            if verbose:
                print(f"  chunk {ci // chunk_layers + 1}/{(ngates + chunk_layers - 1) // chunk_layers} done", flush=True)
    return (x0[:d] + 1j * x0[d:]) / scale


def mul_perm(n, N, c):
    """Permutation matrix on n qubits: v -> c*v mod N for v < N, identity for v >= N.

    A bijection of the 2^n basis states because gcd(c, N) = 1 (c = a^(2^j)).
    """
    d = 1 << n
    M = np.zeros((d, d), dtype=complex)
    for v in range(d):
        w = (c * v) % N if v < N else v
        M[w, v] = 1.0
    return M


def multiplicative_order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = (x * a) % N
        r += 1
        if r > N:
            return None
    return r


def build_shor(N, a, max_qubits=MAX_QUBITS, mirror=True):
    if math.gcd(a, N) != 1:
        raise ValueError(f"gcd(a, N) = {math.gcd(a, N)} != 1 (that is already a factor)")
    n = max(2, (N - 1).bit_length())
    r = multiplicative_order(a, N)
    if r is None or r % 2:
        raise ValueError(f"order of {a} mod {N} is odd/unknown (r={r}); Shor needs even r")
    # continued fractions needs 2^(t+1) > 2 r^2; 2n is the textbook-safe choice
    t = min(2 * n, 2 * r.bit_length() + 2)
    if t + n > max_qubits:
        raise ValueError(f"N={N} needs {t + n} qubits (n={n}, t={t}) > {max_qubits}")
    count = list(range(t))
    work = list(range(t, t + n))
    qc = QuantumCircuit(t + n) if mirror else None
    gates = []
    for q in count:
        add(qc, gates, H, [q])
    add(qc, gates, X, [work[0]])  # work = 1
    for j in range(t):
        add(qc, gates, mul_perm(n, N, pow(a, 1 << j, N)), work, [count[j]])
    from qiskit import transpile
    qft = transpile(QFT(t).inverse(), basis_gates=["h", "x", "cp", "swap"], optimization_level=0)
    for inst in qft.data:
        add_qiskit_op(qc, gates, qft, inst, count)
    return qc, gates, n, t


def counting_marginal(psi, nq, t):
    # reshape([2]*nq): axis k = bit nq-1-k (little-endian). Count = qubits 0..t-1
    # = axes nq-t..nq-1, so sum over axes 0..nq-t-1 (the work register).
    p = (np.abs(psi) ** 2).reshape([2] * nq)
    return p.sum(axis=tuple(range(nq - t))).reshape(-1)


def _convergents(fr):
    a, b = fr.numerator, fr.denominator
    p0, p1, q0, q1 = 0, 1, 1, 0
    while b:
        k = a // b
        p0, p1 = p1, k * p1 + p0
        q0, q1 = q1, k * q1 + q0
        yield Fraction(p1, q1)
        a, b = b, a - k * b


def factors_from_outcome(m, t, a, N):
    # walk the continued-fraction convergents of m/2^t; the order r is one of the denominators
    for conv in _convergents(Fraction(m, 1 << t)):
        r = conv.denominator
        if r > N - 1:
            break
        if r == 0 or r % 2 or pow(a, r, N) != 1:
            continue
        f = math.gcd(pow(a, r // 2, N) - 1, N)
        if 1 < f < N:
            return f, N // f, r
    return None


def shor(N, a, dtype="F32", use_qullama=True, crosscheck=True, max_qubits=MAX_QUBITS, ctx=8):
    qc, gates, n, t = build_shor(N, a, max_qubits, mirror=crosscheck)
    nq = t + n
    print(f"N={N} a={a}: n={n} work, t={t} counting, {nq} qubits total, {len(gates)} gates")
    psi = None
    if use_qullama:
        import resource
        t0 = time.perf_counter()
        psi = simulate_raw(nq, gates, dtype=dtype, ctx=ctx)
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
        print(f"qullama: {time.perf_counter() - t0:.1f}s, sum|p| = {np.sum(np.abs(psi) ** 2):.8f}, "
              f"peak child RSS = {rss} MB")
    if crosscheck:
        t0 = time.perf_counter()
        ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
        print(f"qiskit Statevector: {time.perf_counter() - t0:.1f}s")
        if psi is not None:
            print(f"max err vs qiskit = {np.max(np.abs(psi - ref)):.2e}")
        psi = ref
    marg = counting_marginal(psi, nq, t)
    top = np.argsort(marg)[::-1][:8]
    print("top counting outcomes (m: prob):",
          ", ".join(f"{int(j)}:{marg[j]:.3f}" for j in top if marg[j] > 1e-3))
    seen = set()
    for j in np.argsort(marg)[::-1]:
        if marg[j] < 1e-3:
            break
        res = factors_from_outcome(int(j), t, a, N)
        if res and res not in seen:
            seen.add(res)
            print(f"m={int(j)} -> order r={res[2]} -> factors {res[0]} * {res[1]} = {N}")
            return res[:2]
    print("no nontrivial factor from top outcomes (try another a)")
    return None


def selftest():
    ok = True
    for N in (15, 21, 123):
        n = max(2, (N - 1).bit_length())
        for a in range(2, N):
            if math.gcd(a, N) != 1:
                continue
            M = mul_perm(n, N, a)
            d = 1 << n
            if not np.allclose(M.conj().T @ M, np.eye(d)):
                print(f"mul_perm N={N} a={a}: NOT UNITARY"); ok = False
            for v in range(N):
                if M[(a * v) % N, v] != 1.0:
                    print(f"mul_perm N={N} a={a}: wrong at v={v}"); ok = False
    print("mul_perm unitary + correct:", "OK" if ok else "FAILED")
    # end-to-end small run vs qiskit
    res = shor(15, 2, use_qullama=False, crosscheck=True)
    print("N=15 a=2 (qiskit only):", "OK" if res == (3, 5) or res == (5, 3) else f"FAILED {res}")
    return ok and res in ((3, 5), (5, 3))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--selftest-only" in sys.argv:
        sys.exit(0 if selftest() else 1)
    N = int(args[0]) if len(args) > 0 else 15
    a = int(args[1]) if len(args) > 1 else 2
    dtype = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--dtype=")), "F32")
    max_qubits = int(next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--max-qubits=")), MAX_QUBITS))
    crosscheck = "--no-crosscheck" not in sys.argv and max_qubits <= 24
    ctx = int(next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--ctx=")), 8))
    shor(N, a, dtype=dtype, crosscheck=crosscheck, max_qubits=max_qubits, ctx=ctx)


if __name__ == "__main__":
    main()
