#!/usr/bin/env python3
"""Benchmark qullama on the Vulkan GPU build vs the CPU build, across circuit
sizes, vs qiskit Statevector. Prints one table: time + max error per backend.

NVIDIA/CUDA is NOT tested here (no nvcc/NVIDIA card in this environment) -
the CUDA kernel exists in the patch but is compile/runtime-untested.

Usage:
    python3 examples/bench_gpu.py            # sweep n = 8..24
    python3 examples/bench_gpu.py 16 8       # single size: n_qubits depth
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.bench_quant import random_circuit
from qullama import build_gguf_gates, statevector_reference

VK = ROOT / "llama.cpp" / "build-vk" / "bin" / "llama-embedding"
CPU = ROOT / "llama.cpp" / "build" / "bin" / "llama-embedding"

# (n_qubits, depth); total gates kept <= ~512 (LLAMA_MAX_LAYERS).
# NOTE: 24q allocates ~400 MB GGUF uploaded to shared iGPU RAM + a 256 MB qiskit
# reference + per-node buffers - on shared-memory GPUs keep the default cap at 22
# and pass --max-n 24 explicitly only on a machine with headroom.
SWEEP = [(8, 8), (12, 8), (16, 8), (20, 4), (22, 2)]


def run(binary, gguf):
    t0 = time.perf_counter()
    p = subprocess.run([str(binary), "-m", str(gguf), "-p", "q", "--embd-normalize", "-1",
                        "-n", "1", "--no-warmup", "--embd-output-format", "raw", "-np", "2", "-c", "8"],
                       capture_output=True, text=True)
    dt = time.perf_counter() - t0
    line = next(l for l in p.stdout.splitlines() if l.strip() and not l.startswith("{"))
    return np.array([float(x) for x in line.split()]), dt


def bench_one(n, depth, seed=7):
    qc = random_circuit(n, depth, seed)
    ref = statevector_reference(qc)
    d = 1 << n
    scale = np.sqrt(2 * d)
    with tempfile.TemporaryDirectory() as td:
        gguf = Path(td) / "m.gguf"
        info = build_gguf_gates(qc, gguf)
        row = {"n": n, "depth": depth, "gates": info["n_gates"],
               "gguf_mb": info["size_bytes"] / 1e6, "backends": {}}
        for name, binary in [("cpu", CPU), ("vulkan", VK)]:
            if not binary.exists():
                continue
            emb, dt = run(binary, gguf)
            psi = (emb[:d] + 1j * emb[d:]) / scale
            row["backends"][name] = (dt, float(np.max(np.abs(psi - ref))))
        return row


def main():
    if len(sys.argv) > 2:
        sweep = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        max_n = 22
        for a in sys.argv[1:]:
            if a.startswith("--max-n="):
                max_n = int(a.split("=")[1])
        sweep = [s for s in SWEEP + ([(24, 2)] if max_n >= 24 else []) if s[0] <= max_n]

    print(f"{'n x depth':>10} {'gates':>5} {'gguf MB':>8} | {'cpu s':>7} {'vk s':>7} {'speedup':>7} | {'cpu err':>9} {'vk err':>9}")
    print("-" * 80)
    for n, depth in sweep:
        row = bench_one(n, depth)
        b = row["backends"]
        if "cpu" in b and "vulkan" in b:
            sp = f"{b['cpu'][0] / b['vulkan'][0]:6.2f}x"
            print(f"{f'{n} x {depth}':>10} {row['gates']:>5} {row['gguf_mb']:8.2f} | "
                  f"{b['cpu'][0]:7.2f} {b['vulkan'][0]:7.2f} {sp:>7} | "
                  f"{b['cpu'][1]:9.2e} {b['vulkan'][1]:9.2e}")
        else:
            name = next(iter(b))
            print(f"{f'{n} x {depth}':>10} {row['gates']:>5} {row['gguf_mb']:8.2f} | "
                  f"{b[name][0]:7.2f} ({name}) err {b[name][1]:.2e}")
    print("\nnote: CUDA/NVIDIA is NOT tested (no nvcc/NVIDIA GPU here); the CUDA kernel in")
    print("the patch is compile/runtime-untested. Vulkan tested on AMD Radeon 780M (RADV).")


if __name__ == "__main__":
    main()
