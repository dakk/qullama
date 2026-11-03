#!/usr/bin/env python3
"""Combined benchmark: error + GGUF footprint + process memory across
circuit size x gate-weight quantization, all vs qiskit Statevector.

Prints three tables (max amplitude error, GGUF size MB, child process peak RSS MB),
rows = (n_qubits x depth), columns = dtype.

Usage:
    python3 examples/bench_quant.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DTYPES = ["F32", "F16", "BF16", "Q8_0", "Q4_0"]
DTYPES_BIG = ["F32", "Q8_0"]  # at large n, keep the sweep fast
# (n_qubits, depth); total gates must stay <= LLAMA_MAX_LAYERS (512)
SIZES = [(4, 8), (6, 16), (8, 32), (12, 32), (16, 16), (20, 6), (24, 4)]
SEED = 11


def dtypes_for(n: int):
    return DTYPES if n <= 12 else DTYPES_BIG


def random_circuit(n: int, depth: int, seed: int):
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


def worker(n: int, depth: int, dtype: str, seed: int) -> None:
    """One (size, dtype) measurement in a fresh process; print JSON line."""
    import resource
    import tempfile
    import time

    import numpy as np

    from qullama import simulate, statevector_reference

    qc = random_circuit(n, depth, seed)
    tq = time.perf_counter()
    ref = statevector_reference(qc)
    qiskit_dt = time.perf_counter() - tq
    with tempfile.TemporaryDirectory() as td:
        gguf = Path(td) / "m.gguf"
        t0 = time.perf_counter()
        psi = simulate(qc, dtype=dtype, keep_gguf=gguf)
        dt = time.perf_counter() - t0
        gguf_bytes = sum(p.stat().st_size for p in Path(td).glob("m*.gguf"))
    err = float(np.max(np.abs(psi - ref)))
    print(json.dumps({
        "err": err,
        "gguf_bytes": gguf_bytes,
        "child_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "sec": round(dt, 3),
        "qiskit_sec": round(qiskit_dt, 3),
    }))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", nargs=4, metavar=("N", "DEPTH", "DTYPE", "SEED"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(int(args.worker[0]), int(args.worker[1]), args.worker[2], int(args.worker[3]))
        return

    rows = []
    for n, depth in SIZES:
        dtypes = dtypes_for(n)
        cells = {}
        for dt in dtypes:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker",
                 str(n), str(depth), dt, str(SEED)],
                capture_output=True, text=True, cwd=ROOT,
            )
            cells[dt] = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(((n, depth), dtypes, cells))

    def table(title, fmt, with_qiskit_sec=False):
        dts = sorted({d for _, dts, _ in rows for d in dts}, key=DTYPES.index)
        cols = dts + (["qiskit"] if with_qiskit_sec else [])
        print(f"\n{title}")
        print(f"{'n x depth':>10} | " + " | ".join(f"{d:>10}" for d in cols))
        print("-" * (13 + 13 * len(cols)))
        for (n, depth), dtypes, cells in rows:
            vals = [fmt(cells[d]) for d in dtypes]
            if with_qiskit_sec:
                vals.append(f"{cells[dtypes[0]]['qiskit_sec']:10.2f}")
            print(f"{f'{n} x {depth}':>10} | " + " | ".join(vals))

    table("max |amp| error vs qiskit Statevector", lambda c: f"{c['err']:10.2e}")
    table("GGUF file size (MB)", lambda c: f"{c['gguf_bytes']/1e6:10.3f}")
    table("llama-embedding child peak RSS (MB)", lambda c: f"{c['child_kb']/1e3:10.1f}")
    table("wall time (s): qullama per dtype, last col = qiskit Statevector",
          lambda c: f"{c['sec']:10.2f}", with_qiskit_sec=True)


if __name__ == "__main__":
    main()
