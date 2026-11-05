"""qullama command-line interface.

Subcommands:
  qullama to-gguf circuit.qasm -o model.gguf   write the qgate GGUF
  qullama eval circuit.qasm                    build + run + compare vs qiskit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .backend import run_gguf, simulate
from .qgate import build_gguf_gates
from .qiskit_io import load_circuit_file, statevector_reference


def cmd_to_gguf(args) -> None:
    qc = load_circuit_file(args.circuit)
    out = args.output or str(Path(args.circuit).with_suffix(".gguf"))
    info = build_gguf_gates(qc, out, prompt_token=args.prompt_token, dtype=args.dtype)
    print(json.dumps(info, indent=2))
    print(f"\nwrote {out}")
    print(f"run it:  qullama eval {args.circuit}   (or: llama-embedding -m {Path(out).resolve()} "
          f"-p {args.prompt_token} --embd-normalize -1 --embd-output-format raw)")


def _require_qgate_gguf(path: Path) -> None:
    """Exit with a clear message if path is not a readable qgate GGUF."""
    from gguf import GGUFReader

    try:
        reader = GGUFReader(str(path))
        field = reader.fields.get("general.architecture")
        arch = str(field.contents()) if field is not None else None
    except Exception as e:
        raise SystemExit(f"error: {path} is not a readable GGUF: {e}")
    if arch != "qgate":
        raise SystemExit(f"error: {path} is not a qgate GGUF (general.architecture={arch!r})")


def cmd_eval(args) -> None:
    inp = Path(args.input)
    qc = None
    if inp.suffix == ".gguf":
        _require_qgate_gguf(inp)
        emb = run_gguf(inp, llama_embedding=args.llama_embedding, prompt_token=args.prompt_token)
        d = len(emb) // 2
        if d < 2 or (d & (d - 1)):
            raise SystemExit(f"error: embedding length {len(emb)} is not 2 * 2^n for any n")
        scale = math.sqrt(2 * d)
        psi_out = (emb[:d] + 1j * emb[d:]) / scale
    else:
        qc = load_circuit_file(inp)
        psi_out = simulate(qc, llama_embedding=args.llama_embedding,
                           prompt_token=args.prompt_token, chunk_layers=args.chunk_layers)
        d = 1 << qc.num_qubits
    probs = np.abs(psi_out) ** 2
    n = int(math.log2(d))
    print(f"input: {inp}  n_qubits={n}")
    for i in np.argsort(probs)[::-1][:8]:
        if probs[i] > 1e-12:
            print(f"  |{i:0{n}b}>  p = {probs[i]:.8f}")
    print(f"sum p = {probs.sum():.10f}")
    if qc is not None:
        ref = statevector_reference(qc)  # qiskit Statevector is the only crosscheck
        print(f"max amp error vs qiskit Statevector: {np.max(np.abs(psi_out - ref)):.3e}")
        print(f"L2 error  vs qiskit Statevector:      {np.linalg.norm(psi_out - ref):.3e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qullama", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("to-gguf", help="write a qgate GGUF from an OpenQASM circuit")
    p.add_argument("circuit", help="OpenQASM 2/3 file (.qasm)")
    p.add_argument("-o", "--output", default=None, help="output .gguf path (default: <circuit>.gguf)")
    p.add_argument("--prompt-token", default="q")
    p.add_argument("--dtype", default="F32", choices=["F32", "F16", "BF16", "Q8_0", "Q4_0"],
                   help="quantization of the gate weights (state stays F32)")
    p.set_defaults(fn=cmd_to_gguf)

    p = sub.add_parser("eval", help="build the GGUF, run it in patched llama.cpp, compare vs qiskit")
    p.add_argument("input", help="OpenQASM circuit file (.qasm) or a .gguf")
    p.add_argument("--llama-embedding", default="llama.cpp/build/bin/llama-embedding")
    p.add_argument("--prompt-token", default="q")
    p.add_argument("--chunk-layers", type=int, default=None,
                   help="run gates in passes of N layers (opt-in; off by default)")
    p.add_argument("--dtype", default="F32", choices=["F32", "F16", "BF16", "Q8_0", "Q4_0"],
                   help="quantization of the gate weights (state stays F32)")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
