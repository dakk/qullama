"""Library API: use qullama as a simulation backend.

    from qiskit import QuantumCircuit
    from qullama import simulate, QullamaBackend

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    psi = simulate(qc)                          # -> complex statevector, len 2**n

    backend = QullamaBackend()                  # qiskit BackendV2
    job = backend.run(qc)
    psi = job.result().get_statevector()
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .qgate import build_gguf_gates

DEFAULT_BINARY = Path(__file__).resolve().parent.parent / "llama.cpp" / "build" / "bin" / "llama-embedding"


def parse_embedding_output(text: str) -> np.ndarray:
    """Parse llama-embedding's --embd-output-format raw (or json) output."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("{"):
            if line.startswith("{") and '"embedding"' in line:
                return np.array(json.loads(line)["embedding"], dtype=np.float64)
            continue
        try:
            vals = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(vals) >= 2:
            return np.array(vals, dtype=np.float64)
    raise RuntimeError(f"could not parse llama-embedding output:\n{text[-2000:]}")


def run_gguf(gguf: str | Path, *, llama_embedding: str | Path | None = None,
             prompt_token: str = "q", ctx: int = 8) -> np.ndarray:
    """Run a qgate GGUF in the patched llama.cpp; return the raw embedding vector.

    ctx: -c value. Keep small: ggml reserves ~5 x n_ctx x n_embd compute buffers
    (40 GB at 27 qubits with -c 8), so large n_embd needs a small ctx."""
    binary = Path(llama_embedding) if llama_embedding else DEFAULT_BINARY
    if not binary.exists():
        raise FileNotFoundError(
            f"patched llama-embedding not found at {binary} - run ./build_patched_llama.sh first"
        )
    proc = subprocess.run(
        [str(binary), "-m", str(gguf), "-p", prompt_token, "--embd-normalize", "-1",
         "-n", "1", "--no-warmup", "--embd-output-format", "raw",
         # -np 2: the embedding example bumps n_parallel==1 up to llama_max_parallel_sequences(),
         # which would reserve an n_embd x n_parallel output buffer (2 GB+ at 20 qubits);
         # -np 2 stays small. -c 8 keeps the context small - context compute buffers scale
         # with n_embd x n_ctx (40 GB at 24 qubits with -c 64).
         "-np", "2", "-c", str(ctx)],
        capture_output=True, text=True,
    )
    return parse_embedding_output(proc.stdout + proc.stderr)


def simulate(qc, *, llama_embedding: str | Path | None = None,
             prompt_token: str = "q", keep_gguf: str | Path | None = None,
             chunk_layers: int | None = None, dtype: str = "F32",
             chunk_budget_bytes: int = 1 << 30) -> np.ndarray:
    """Run a qiskit QuantumCircuit on patched llama.cpp; return the output statevector.

    The circuit starts from |0...0>. Returns a complex ndarray of length 2**qc.num_qubits.

    chunk_layers: if set, run the gate list in passes of this many layers (each pass
    is a separate GGUF inference; the output state feeds the next pass's embedding).
    Off by default; mainly useful with a custom chunk_budget_bytes for very large
    circuits on memory-constrained machines.
    """
    from .qiskit_io import iter_gate_ops
    from .qgate import write_qgate_gguf

    n = qc.num_qubits
    d = 1 << n
    n_embd = 2 * d
    scale = math.sqrt(n_embd)
    gates = list(iter_gate_ops(qc))
    state_emb = np.zeros(n_embd)  # |0...0>, pre-scaled
    state_emb[0] = scale

    if not gates:
        return (state_emb[:d] + 1j * state_emb[d:]) / scale

    if chunk_layers is None:
        chunks = [gates]
    else:
        chunks = [gates[i:i + chunk_layers] for i in range(0, len(gates), chunk_layers)]
    with tempfile.TemporaryDirectory() as td:
        for ci, chunk in enumerate(chunks):
            if keep_gguf:
                gguf = Path(keep_gguf)
                if len(chunks) > 1:
                    gguf = gguf.with_name(f"{gguf.stem}.chunk{ci}{gguf.suffix}")
            else:
                gguf = Path(td) / (f"chunk{ci}.gguf" if len(chunks) > 1 else "model.gguf")
            write_qgate_gguf(n, chunk, state_emb, gguf, prompt_token=prompt_token, dtype=dtype)
            # final RMSNorm is the identity (eps=0, unit norm), so the output
            # embedding is directly the next chunk's input embedding
            state_emb = run_gguf(gguf, llama_embedding=llama_embedding, prompt_token=prompt_token)
    return (state_emb[:d] + 1j * state_emb[d:]) / scale


class QullamaJob:
    """Synchronous job with the qiskit JobV1 duck-type (submit/result/status/backend)."""

    def __init__(self, backend, job_id, result):
        self._backend = backend
        self._job_id = job_id
        self._result = result

    def job_id(self):
        return self._job_id

    def backend(self):
        return self._backend

    def submit(self):
        return

    def result(self, timeout=None):
        return self._result

    def status(self):
        from qiskit.providers import JobStatus
        return JobStatus.DONE

    def cancel(self):
        pass


class QullamaBackend:
    """qiskit-compatible backend (duck-types BackendV2) simulating with patched llama.cpp.

        backend = QullamaBackend()
        job = backend.run(qc)
        sv = job.result().get_statevector()
    """

    def __init__(self, llama_embedding: str | Path | None = None, prompt_token: str = "q"):
        self.name = "qullama"
        self.backend_version = "0.1.0"
        self.description = "qullama: simulate circuits with patched llama.cpp (qgate GGUF)"
        self._llama_embedding = str(llama_embedding) if llama_embedding else None
        self._prompt_token = prompt_token

    def run(self, run_input, **run_options):
        from qiskit.result import Result

        circuits = run_input if isinstance(run_input, list) else [run_input]
        svs = [simulate(qc, llama_embedding=self._llama_embedding, prompt_token=self._prompt_token)
               for qc in circuits]
        result = Result.from_dict({
            "success": True,
            "results": [
                {"success": True, "shots": 1, "data": {"statevector": sv}, "metadata": {}}
                for sv in svs
            ],
        })
        return QullamaJob(self, uuid.uuid4().hex, result)
