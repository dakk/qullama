# qullama - run quantum circuits inside llama.cpp

`qullama` turns a quantum circuit into a `qgate` GGUF "model" whose layers are
the gates themselves; running it with `llama-embedding` on a one-token prompt
applies the circuit to |0…0⟩ and returns the output statevector. llama.cpp is
used with a small patch (a custom arch built on the stock `ggml_map_custom3` op
- no ggml core changes).


## Quick start

```bash
pip install .                # deps: numpy, gguf, qiskit, qiskit-qasm3-import
./build_patched_llama.sh     # -> llama.cpp/build/bin/llama-embedding

qullama eval examples/bell.qasm --llama-embedding llama.cpp/build/bin/llama-embedding
qullama eval examples/ghz4.qasm
qullama eval examples/qft3.qasm

# just write the GGUF
qullama to-gguf examples/bell.qasm -o bell.gguf

# tests
python3 tests/test_qgate.py
```

`qullama eval` prints the top measurement probabilities and the max/L2 amplitude
error against qiskit's `Statevector` (the only crosscheck).

## Library usage

Build the circuit directly in Python and use qullama as the simulator backend
(see `examples/library_demo.py`):

```python
from qiskit import QuantumCircuit
from qullama import simulate, QullamaBackend

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

psi = simulate(qc)                        # complex statevector, len 2**n

backend = QullamaBackend()                # qiskit-backend-style
sv = backend.run(qc).result().get_statevector()
```

`simulate(qc)` builds the qgate GGUF in a temp dir, runs it with
`llama.cpp/build/bin/llama-embedding`, and decodes the embedding back into the
statevector. `QullamaBackend(llama_embedding=...)` accepts a list of circuits too.

### Precision / quantization

Python-side math is float64, but the GGUF and the kernel run at the chosen dtype —
**F32 by default**. Gate-weight quantization is selectable in both the library and
the CLI (`--dtype`), the state embedding always stays F32:

```bash
qullama eval examples/qft3.qasm --dtype Q8_0
```

```python
simulate(qc, dtype="Q8_0")     # F32 | F16 | BF16 | Q8_0 | Q4_0
```

Measured via `python3 examples/bench_quant.py` (random circuits, rows = n_qubits ×
depth, columns = dtype; state embedding always F32, only gate weights quantized;
at n ≥ 16 only F32/Q8_0 are swept to keep runtime sane; total gates ≤ 512):

```
max |amp| error vs qiskit          GGUF size (MB)               child peak RSS (MB)
 n x depth |  F32  F16  BF16 Q8_0 Q4_0 |  F32  F16  BF16 Q8_0 Q4_0  |  F32  ... (Q8_0)
  4 x  8   | 1.5e-7 1.3e-4 1.7e-3 4.2e-3 6.1e-2 | 0.009 ... 0.029      | ~88
  8 x 32   | 7.9e-8 1.4e-4 1.2e-3 4.3e-3 1.1e-1 | 0.065 ... 0.210      | ~88
 12 x 32   | 4.5e-8 6.0e-5 8.0e-4 2.1e-3 6.8e-2 | 0.184 ... 0.396      | ~89
 16 x 16   | 1.8e-8 ...        7.0e-4        | 1.630  ... 1.908      | ~92
 20 x  6   | 1.1e-8 ...        5.7e-4        | 25.19  ... 25.32      | ~144
 24 x  4   | 3.5e-9 ...        8.4e-5        | 402.7  ... 402.8      | ~1065

wall time (s)                       |  qullama F32 | qullama Q8_0 | qiskit Statevector
 12 x 32   0.07 ...                  |  0.07       |  0.09        | 0.01
 16 x 16                            |  0.22       |  0.23        | 0.14
 20 x  6                            |  2.33       |  2.35        | 1.22
 24 x  4                            | 33.45       | 33.13        | 13.15
```

Findings:

* **Error**: F32 sits at the fp32 floor (~1e-7 and falling with n since the state
  stays normalized); each quantization step costs ~1–2 orders of magnitude
  (F16 ~1e-4, BF16 ~1e-3, Q8_0 ~3e-3, Q4_0 ~7e-2), growing only mildly with depth.
  **Q8_0 is the sweet spot** when quantizing; Q4_0 only for tiny circuits.
* **GGUF size** is dominated by the F32 state embedding (2·2ⁿ⁺¹·4 bytes — 256 MB of
  the 402 MB at 24 qubits), not the gates; Q8_0/Q4_0 are even *larger* for 1–2-qubit
  gates (block quantization pads each gate matrix to 32 rows). Quantized gate
  weights only pay off for ≥4-qubit gates.
* **Runtime RSS**: ~88 MB baseline until the state vector dominates (1.07 GB at
  24 qubits). dtype barely moves it. (The context is run with `-np 2 -c 8` —
  otherwise the tool reserves n_embd × n_ctx buffers: 40 GB at 24 qubits.)
* **Speed**: qiskit's `Statevector` stays ~2–3× faster than qullama on CPU at every
  size up to 24 qubits (33 s vs 13 s at 24q); qullama's edge is going past 24
  qubits and its memory/artifact profile, plus GPU (a CUDA kernel for the qgate op
  is the next step if speed on GPU matters).
* Error-only sweep at wider depths: `python3 examples/bench_error.py`;
  qullama-vs-qiskit memory/time scaling: `python3 examples/bench_memory.py`.

## Circuit format: OpenQASM

`.qasm` files (OpenQASM 2 or 3), parsed by qiskit (`qasm2`/`qasm3`) and
transpiled (`optimization_level=0`) to the gate basis the kernel implements:
`h x y z s sdg t tdg rx ry rz p cp cx cz ccx swap`. Qubit i = bit i of the state
index (little-endian, same as qiskit). See `examples/`.

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
```

## How it works

* The quantum state is the prompt token's embedding row: `sqrt(2d) * [Re psi;
  Im psi]` with d = 2ⁿ, so `n_embd = 2^(n+1)` and the final RMSNorm (eps=0,
  weights=1) is a no-op. The prompt is a single character, so it tokenizes to
  exactly one token (no BOS).
* Each gate becomes one model layer. The patch's graph builder emits one
  `ggml_map_custom3` node per layer; the kernel applies the gate's 2ᵏ×2ᵏ matrix
  - stored as its real block form `[[Ur,-Ui],[Ui,Ur]]` in `blk.i.qgate.weight` -
  to the target-qubit blocks of the state. `blk.i.qgate_meta` (I32:
  `[k, n_controls, targets..., controls...]`) selects targets and controls;
  control-off blocks pass through unchanged. Gate matrices and control metadata
  come straight from qiskit (`op.to_matrix()`, `num_ctrl_qubits`).
* No dense 2ⁿ×2ⁿ unitary is ever materialized: GGUF size is O(n_gates · 4ᵏ + 2ⁿ).
* Read back with `llama-embedding -m model.gguf -p q --embd-normalize -1
  --embd-output-format raw`: the printed vector is `M·x`, and the output state is
  `psi_out = (emb[:d] + 1j*emb[d:]) / sqrt(2d)`.

## The llama.cpp patch

`qullama-b12075.patch` (against tag **b12075**, the latest release tag) adds:

* `LLM_ARCH_QGATE` + tensor names in `src/llama-arch.{h,cpp}`
* `src/models/qgate.cpp`: model class + graph builder (new file)
* a `GGML_OP_MAP_CUSTOM3` case in `src/llama-model-loader.cpp`'s buffer-type probe
* the `llama_model_mapping` case in `src/llama-model.cpp`

`build_patched_llama.sh` automates clone → patch → cmake → `llama-embedding`.

## Measured (patched b12075 build, CPU)

| circuit | error vs qiskit Statevector |
|---|---|
| bell.qasm, ghz4.qasm | exact (0.0) |
| qft3.qasm | 1.6e-8 |
| random 8 / 10 qubits | 4.8e-8 / 2.8e-8 |

Scaling (random circuits, 4 gates/qubit): 12 qubits → 311 KB GGUF; 16 qubits →
1.6 MB / 0.27 s; 20 qubits → 25 MB GGUF / 3.9 s, with Σp = 1 to 1e-7. A dense
unitary at 20 qubits would be ~17 TB — the gate-list encoding is what makes
this feasible. Practical ceiling is the state vector in RAM (~25–28 qubits on
CPU, more on GPU).

Memory benchmark vs qiskit's Statevector (`python3 examples/bench_memory.py`):

```
  n gates | qiskit MB | qiskit sec | qullama gguf MB | qullama MB | qullama sec |  amp err
   16    91 |   108.4   |     0.03   |        1.59     |    88.3    |    0.16     | 1.4e-08
   18   105 |   169.6   |     0.15   |        6.31     |    92.2    |    0.56     | 1.3e-08
   20   120 |   428.3   |     0.83   |       25.19     |   104.2    |    2.16     | 5.7e-09
   22   130 |  1457.4   |     4.47   |      100.69     |   279.7    |    8.63     | 5.1e-09
```

qiskit = `Statevector(qc)` in-process (BasicSimulator caps at 24 qubits; theory
16·2ⁿ bytes). qullama = GGUF on disk; child-process peak RSS and wall time
(incl. ~0.1 s process spawn), run with `-np 2 -c 64` so the tool doesn't reserve
an `n_embd × n_parallel` output buffer (2 GB at 20 qubits otherwise).

Speed: qiskit's Statevector is ~2–3× faster than qullama on CPU up to 22 qubits
(optimized numpy per-gate ops vs our per-block kernel, which is multithreaded
via ggml but not vectorized); qullama's wall time also carries a fixed ~0.15 s
process-spawn cost. qullama wins where qiskit stops: beyond 24 qubits, memory
(104 MB vs 428 MB at 20q), and the GGUF artifact size. A CUDA kernel for the
qgate op would flip the speed ranking on GPU (not implemented — CPU only now).
