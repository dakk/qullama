"""Write a "qgate" GGUF: a circuit stored as a list of gates, executed by the
patched llama.cpp (arch "qgate", one ggml_map_custom3 node per layer) without
ever materializing the dense 2^n x 2^n unitary.

State layout: the prompt token's embedding is sqrt(2d) * [Re psi; Im psi]
(d = 2^n), so n_embd = 2^(n+1) and the final RMSNorm (eps=0, weights=1) is a
no-op. Each layer i stores:
  blk.i.qgate.weight : F32 [gate_dim, gate_dim], top-left 2^(k+1) x 2^(k+1) block
                       = real form M = [[Ur,-Ui],[Ui,Ur]] of the complex gate
  blk.i.qgate_meta   : I32 [meta_dim] = [k, n_controls, targets..., controls..., 0...]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from gguf import GGUFWriter, GGMLQuantizationType  # official gguf-py (PyPI package "gguf")
from .qiskit_io import iter_gate_ops

DEFAULT_PROMPT_TOKEN = "q"
N_VOCAB = 2  # "<unk>" + the prompt token; only the prompt row is non-zero

# quantization for the gate weight tensors (state/embedding stays F32)
DTYPE_MAP = {
    "F32": GGMLQuantizationType.F32,
    "F16": GGMLQuantizationType.F16,
    "BF16": GGMLQuantizationType.BF16,
    "Q8_0": GGMLQuantizationType.Q8_0,
    "Q4_0": GGMLQuantizationType.Q4_0,
}
_QUANTIZED = {"Q8_0", "Q4_0"}  # block-quantized types need ne0 % 32 == 0


def unitary_to_real_matrix(U: np.ndarray) -> np.ndarray:
    """Complex d x d matrix -> real 2d x 2d matrix acting on [Re z; Im z]."""
    U = np.asarray(U, dtype=np.complex128)
    d = U.shape[0]
    if U.shape != (d, d):
        raise ValueError(f"U must be square, got {U.shape}")
    Ur, Ui = U.real, U.imag
    top = np.concatenate([Ur, -Ui], axis=1)
    bot = np.concatenate([Ui, Ur], axis=1)
    return np.concatenate([top, bot], axis=0)


def build_gguf_gates(
    qc,
    path: str | Path = "model-qgate.gguf",
    context_length: int = 512,
    prompt_token: str = DEFAULT_PROMPT_TOKEN,
    name: str = "qgate-circuit",
    dtype: str = "F32",
) -> dict:
    """qiskit.QuantumCircuit -> qgate GGUF. Initial state is always |0...0>.

    dtype: quantization for the gate weight tensors (F32/F16/BF16/Q8_0/Q4_0);
    the state embedding always stays F32.
    """
    n = qc.num_qubits
    if n < 1 or n > 30:
        raise ValueError("n_qubits must be in [1, 30]")
    gates = list(iter_gate_ops(qc))
    if not gates:
        raise ValueError("circuit has no gates")
    d = 1 << n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    x0 = math.sqrt(2 * d) * np.concatenate([psi.real, psi.imag])
    return write_qgate_gguf(n, gates, x0, path, context_length, prompt_token, name, dtype)


def write_qgate_gguf(
    n: int,
    gates: list,
    x0: np.ndarray,
    path: str | Path,
    context_length: int = 512,
    prompt_token: str = DEFAULT_PROMPT_TOKEN,
    name: str = "qgate-circuit",
    dtype: str = "F32",
) -> dict:
    """Write a qgate GGUF for n qubits, the given gate list [(M, targets, controls)],
    and the initial embedding row x0 (real vector of length 2**(n+1), pre-scaled)."""
    if dtype.upper() not in DTYPE_MAP:
        raise ValueError(f"dtype must be one of {list(DTYPE_MAP)}, got {dtype!r}")
    raw_dtype = DTYPE_MAP[dtype.upper()]
    d = 1 << n
    n_embd = 2 * d
    if len(x0) != n_embd:
        raise ValueError(f"x0 has length {len(x0)}, expected {n_embd}")
    k_max = max(len(t) for _, t, _ in gates)
    c_max = max(len(c) for _, _, c in gates)
    gate_dim = 1 << (k_max + 1)
    if dtype.upper() in _QUANTIZED:
        gate_dim = max(gate_dim, 32)
        gate_dim = (gate_dim + 31) // 32 * 32  # block-quantized types need ne0 % 32 == 0
    meta_dim = 2 + k_max + c_max

    scale = math.sqrt(n_embd)

    tokens = ["<unk>", prompt_token][:N_VOCAB]
    if prompt_token not in tokens:
        raise ValueError(f"prompt_token {prompt_token!r} not in vocab {tokens}")
    prompt_id = tokens.index(prompt_token)

    w = GGUFWriter(str(path), "qgate")  # sets general.architecture = "qgate"
    w.add_string("general.name", name)
    w.add_uint32("general.alignment", 32)
    w.add_file_type(0)  # all F32
    w.add_uint32("qgate.context_length", context_length)
    w.add_uint32("qgate.embedding_length", n_embd)
    w.add_uint32("qgate.block_count", len(gates))
    # keep the generic graph context happy (no attention is built, but hparams are
    # read): 1x1 KV heads so the (unused) KV cache stays tiny even for large n_embd
    w.add_uint32("qgate.attention.head_count", 1)
    w.add_uint32("qgate.attention.head_count_kv", 1)
    w.add_uint32("qgate.attention.key_length", 1)
    w.add_uint32("qgate.attention.value_length", 1)
    w.add_float32("qgate.attention.layer_norm_rms_epsilon", 0.0)
    w.add_uint32("qgate.gate_dim", gate_dim)
    w.add_uint32("qgate.meta_dim", meta_dim)
    # tokenizer: minimal gpt-2 BPE. llama.cpp requires the merges key to exist, but
    # gguf-py refuses to write empty arrays, so put one dummy merge that can never
    # fire (the prompt is the single char "q")
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "gpt-2")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.scores", [0.0] * N_VOCAB)
    w.add_array("tokenizer.ggml.token_type", [2] + [1] * (N_VOCAB - 1))
    w.add_array("tokenizer.ggml.merges", ["<unk> <unk>"])
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_bool("tokenizer.ggml.add_eos_token", False)

    embd = np.zeros((N_VOCAB, n_embd), dtype=np.float32)
    embd[prompt_id] = x0
    w.add_tensor("token_embd.weight", embd)
    w.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))

    for i, (G, targets, controls) in enumerate(gates):
        k = len(targets)
        g = 1 << (k + 1)
        M = unitary_to_real_matrix(G)  # (g, g)
        padded = np.zeros((gate_dim, gate_dim), dtype=np.float32)
        padded[:g, :g] = M
        if dtype.upper() in _QUANTIZED:
            from gguf.quants import quantize
            w.add_tensor(f"blk.{i}.qgate.weight", quantize(padded, raw_dtype), raw_dtype=raw_dtype)
        elif dtype.upper() == "BF16":
            u32 = padded.view(np.uint32) if padded.dtype == np.float32 else padded.astype(np.float32).view(np.uint32)
            bf16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)  # round-to-nearest-even
            w.add_tensor(f"blk.{i}.qgate.weight", bf16, raw_dtype=raw_dtype)
        else:
            w.add_tensor(f"blk.{i}.qgate.weight",
                         padded.astype(np.float32 if dtype.upper() == "F32" else np.float16),
                         raw_dtype=raw_dtype)
        meta = [k, len(controls)] + list(targets) + list(controls)
        meta += [0] * (meta_dim - len(meta))
        w.add_tensor(f"blk.{i}.qgate_meta", np.asarray(meta, dtype=np.int32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = path
    return {
        "path": str(out),
        "arch": "qgate",
        "n_qubits": n,
        "d": d,
        "n_embd": n_embd,
        "n_gates": len(gates),
        "gate_dim": gate_dim,
        "dtype": dtype.upper(),
        "scale": scale,
        "prompt_token": prompt_token,
        "prompt_id": prompt_id,
        "size_bytes": Path(out).stat().st_size,
    }
