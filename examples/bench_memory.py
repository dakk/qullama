#!/usr/bin/env python3
"""Benchmark: memory footprint of qullama (patched llama.cpp) vs qiskit's
Statevector simulator, on increasingly large random circuits.

For each n_qubits we build the same random circuit twice:
  * qiskit:  peak Python allocation (tracemalloc) around Statevector(qc)
             — theory: 16 * 2**n bytes for the complex128 statevector
  * qullama: GGUF file size + max RSS of the llama-embedding child process
             (measured in a fresh wrapper subprocess via resource.RUSAGE_CHILDREN)
and check both agree with each other.

Usage:
    python3 examples/bench_memory.py                # n = 4..24
    python3 examples/bench_memory.py --max-n 26     # push qullama further (qiskit skipped >24)
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def random_circuit(n: int, gates_per_qubit: int, seed: int):
    from qiskit import QuantumCircuit

    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(gates_per_qubit * n):
        t = rng.randrange(n)
        g = rng.choice(["h", "s", "t", "rx", "ry", "rz"])
        if g in ("rx", "ry", "rz"):
            getattr(qc, g)(rng.uniform(0, 6.28), t)
        else:
            getattr(qc, g)(t)
        if rng.random() < 0.5:
            c = rng.randrange(n)
            if c != t:
                qc.cx(c, t)
    return qc


def worker(n: int, gpq: int, seed: int) -> None:
    """Run one measurement in this fresh process; print JSON to stdout.

    Reports both backends from the same process:
      qiskit_self_maxrss_kb  — RUSAGE_SELF peak after Statevector(qc) (incl. interpreter baseline)
      child_maxrss_kb        — RUSAGE_CHILDREN peak of the llama-embedding run
    """
    from qiskit.quantum_info import Statevector

    from qullama import simulate

    qc = random_circuit(n, gpq, seed)
    with tempfile.TemporaryDirectory() as td:
        gguf = Path(td) / "bench.gguf"
        t0 = time.perf_counter()
        psi = simulate(qc, keep_gguf=gguf)
        dt = time.perf_counter() - t0
        gguf_bytes = sum(p.stat().st_size for p in Path(td).glob("bench*.gguf"))
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # Linux: KiB
    t1 = time.perf_counter()
    ref = np.asarray(Statevector(qc).data)
    qiskit_dt = time.perf_counter() - t1
    err = float(np.max(np.abs(psi - ref)))
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "n": n,
        "gguf_bytes": gguf_bytes,
        "child_maxrss_kb": child_kb,
        "qiskit_self_maxrss_kb": self_kb,
        "seconds": round(dt, 3),
        "qiskit_seconds": round(qiskit_dt, 3),
        "max_err_vs_qiskit": err,
    }))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-n", type=int, default=24)
    ap.add_argument("--gates-per-qubit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker is not None:
        worker(args.worker, args.gates_per_qubit, args.seed)
        return

    sizes = [n for n in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24] if n <= args.max_n]
    if args.max_n > 24:
        sizes += [26]

    print(f"{'n':>3} {'gates':>5} | {'qiskit MB':>9} {'qiskit sec':>10} | "
          f"{'qullama gguf MB':>15} {'qullama MB':>10} {'qullama sec':>11} | {'amp err':>9}")
    print("-" * 90)
    for n in sizes:
        qc = random_circuit(n, args.gates_per_qubit, args.seed)
        ngates = len(qc.data)
        qiskit_theory_mb = 16 * 2**n / 1e6  # complex128 statevector

        # both backends measured in a fresh wrapper process (per-run RSS is clean there)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(n),
             "--gates-per-qubit", str(args.gates_per_qubit), "--seed", str(args.seed)],
            capture_output=True, text=True, cwd=ROOT,
        )
        try:
            row = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            print(f"worker failed at n={n}: rc={proc.returncode}\nSTDERR tail:\n"
                  + "\n".join(proc.stderr.strip().splitlines()[-5:]))
            raise

        if n <= 24:
            qiskit_mb = f"{row['qiskit_self_maxrss_kb']/1e3:9.1f}"
            qiskit_sec = f"{row['qiskit_seconds']:10.2f}"
        else:
            qiskit_mb = f"{'(skip)':>9}"
            qiskit_sec = f"{'':>10}"
        print(f"{n:>3} {ngates:>5} | {qiskit_mb} {qiskit_sec} | "
              f"{row['gguf_bytes']/1e6:15.2f} {row['child_maxrss_kb']/1e3:10.1f} {row['seconds']:11.2f} | "
              f"{row['max_err_vs_qiskit']:9.2e}")

    print("\nnotes: qiskit = Statevector(qc) in-process (RSS peak, wall time; theory 16*2**n bytes for the")
    print("complex128 statevector; BasicSimulator caps at 24 qubits). qullama = GGUF on disk, child")
    print("process peak RSS and wall time (incl. ~0.1 s process spawn). Error = qullama vs qiskit.")


if __name__ == "__main__":
    main()
