# qullama

[![CI](https://github.com/dakk/qullama/actions/workflows/ci.yml/badge.svg)](https://github.com/dakk/qullama/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`qullama` turns a quantum circuit into a `qgate` GGUF "model" whose layers are the gates themselves; running it with `llama-embedding` applies the circuit to |0...0> and returns the output statevector. llama.cpp is used with a small patch (a custom arch 
built on the stock `ggml_map_custom3` op).

The quantum circuit can be run on a quantized version, reducing the memory footprint. This implementation also contains *CUDA* (not tested) and *Vulkan* kernels. This is a proof of concepts, it needs further adjustment, testing and development in order to make it usable and useful. Feel free to submit PR.


## How it works

1. The quantum state is the prompt token's embedding row: `sqrt(2d) * [Re psi; Im psi]` with d = 2^n, so `n_embd = 2^(n+1)` and the final RMSNorm (eps=0, weights=1) is a no-op. The prompt is a single character, so it tokenizes to exactly one token (no BOS).

2. Each gate becomes one model layer. The patch's graph builder emits one `ggml_map_custom3` node per layer; the kernel applies the gate's 2^k x 2^k matrix - stored as its real block form `[[Ur,-Ui],[Ui,Ur]]` in `blk.i.qgate.weight` - to the target-qubit blocks of the state. `blk.i.qgate_meta` (I32: `[k, n_controls, targets..., controls...]`) selects targets and controls; control-off blocks pass through unchanged. Gate matrices and control metadata come straight from qiskit (`op.to_matrix()`, `num_ctrl_qubits`).

3. Read back with `llama-embedding -m model.gguf -p q --embd-normalize -1 --embd-output-format raw`: the printed vector is `M x x`, and the output state is `psi_out = (emb[:d] + 1j*emb[d:]) / sqrt(2d)`.


## Quick start

```bash
pip install .
./build_patched_llama.sh     # -> llama.cpp/build/bin/llama-embedding
# `VULKAN=1 ./build_patched_llama.sh` # Enable Vulkan
# `CUDA=1 ./build_patched_llama.sh` # Enable CUDA

qullama eval examples/bell.qasm --llama-embedding llama.cpp/build/bin/llama-embedding
qullama eval examples/ghz4.qasm
qullama eval examples/qft3.qasm

# just write the GGUF
qullama to-gguf examples/bell.qasm -o bell.gguf

# tests
python3 tests/test_qgate.py
```

`qullama eval` prints the top measurement probabilities and the max/L2 amplitude error against qiskit's `Statevector`.

## Docker usage

```bash
docker build -t qullama .
docker run --rm qullama eval examples/bell.qasm
```

## Library usage

Build the circuit directly in Python / Qiskit and use qullama as the simulator backend (see `examples/library_demo.py`):

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

`simulate(qc)` builds the qgate GGUF in a temp dir, runs it with `llama.cpp/build/bin/llama-embedding`, and decodes the embedding back into the statevector. `QullamaBackend(llama_embedding=...)` accepts a list of circuits too.

### Precision / quantization

Python-side math is float64, but the GGUF and the kernel run at the chosen dtype - **F32 by default**. Gate-weight quantization is selectable in both the library and the CLI (`--dtype`), the state embedding always stays F32:

```bash
qullama eval examples/qft3.qasm --dtype Q8_0
```

```python
simulate(qc, dtype="Q8_0")     # F32 | F16 | BF16 | Q8_0 | Q4_0
```

Measured via `python3 examples/bench_quant.py` (random circuits, rows = n_qubits x depth, columns = dtype; state embedding always F32, only gate weights quantized; at n >= 16 only F32/Q8_0 are swept to keep runtime sane; total gates <= 512):

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

## Shor's algorithm (`examples/shor.py`)

A full Shor factoring run on qullama. Layout: `t` counting qubits + `n` work qubits; `t` is auto-sized from the multiplicative order r of a mod N (`t = min(2n, 2*bit_length(r)+2)`, since continued fractions only needs 2^(t+1) > 2r^2). The modular multiplier `|x> -> |c*x mod N>` is a permutation of Z_N (extended to a bijection of the work register), emitted as a single n-qubit unitary gate controlled on each counting qubit.

```
./venv/bin/python examples/shor.py            # N=15,   a=2 -> 3 * 5    (12 qubits)
./venv/bin/python examples/shor.py 1023 2 --max-qubits=24
                                              # N=1023, a=2 -> 31 * 33  (20 qubits)
./venv/bin/python examples/shor.py 4095 4 --dtype=Q4_0 --max-qubits=24 --no-crosscheck
                                              # N=4095, a=4 -> 63 * 65  (20 qubits)
```

Measured on CPU (qullama vs qiskit Statevector cross-check where RAM allows):

```
N=15    (12 qubits):  err 4.7e-9  -> 3 * 5
N=123   (21 qubits):  5.0 s, err 1.5e-8 -> 3 * 41
N=255   (18 qubits):  err 2.2e-9  -> 15 * 17
N=511   (19 qubits):  err 1.6e-8  -> 7 * 73
N=1023  (20 qubits):  err 1.8e-8  -> 31 * 33
N=4095  (20 qubits, Q4_0): 54 s, ~4.5 GB peak RSS -> 63 * 65
```


## The llama.cpp patch

`qullama-b12075.patch` (against tag **b12075**, the latest release tag; verified to apply cleanly on a pristine checkout) adds a real ggml op, `GGML_OP_QGATE`:

* ggml core: enum + name/symbol tables + `ggml_qgate()` constructor (`ggml.h`, `ggml.c`)
* CPU kernel: `ggml_compute_forward_qgate` in `ggml-cpu/ops.cpp` (handles F32/F16/BF16/Q8_0/Q4_0 gate weights via ggml's own dequantizers), dispatch + task-count cases in `ggml-cpu.c`, split-state case in `ggml-backend-meta.cpp`
* **CUDA kernel**: `ggml-cuda/qgate.cu` (one thread per gate block; F32 gates - quantized gate weights automatically fall back to the CPU op), dispatch + `supports_op` in `ggml-cuda.cu`
* **Vulkan kernel**: GLSL compute shader `ggml-vulkan/vulkan-shaders/qgate.comp` + pipeline/dispatch/`supports_op` wiring in `ggml-vulkan.cpp`, push constants in `ggml-vulkan-push-constants.h`, registered in `vulkan-shaders-gen.cpp`. F32 gates only; quantized weights fall back to CPU per node.

GPU results (`examples/bench_gpu.py`, AMD Radeon 780M iGPU / RADV, vs qiskit Statevector; measured at 16–22 qubits):

```
 n x depth | gates |   cpu s |    vk s | cpu err |  vk err
 16 x  8   |   134 |    0.17 |    0.26 |  1.5e-8 |  2.0e-8
 20 x  4   |    84 |    1.58 |    1.90 |  1.5e-8 |  1.7e-8
 22 x  2   |    46 |    5.76 |    6.56 |  5.9e-9 |  1.3e-8
```


## Measured (patched b12075 build, CPU)

| circuit | error vs qiskit Statevector |
|---|---|
| bell.qasm, ghz4.qasm | exact (0.0) |
| qft3.qasm | 1.6e-8 |
| random 8 / 10 qubits | 4.8e-8 / 2.8e-8 |

Scaling (random circuits, 4 gates/qubit): 12 qubits -> 311 KB GGUF; 16 qubits -> 1.6 MB / 0.27 s; 20 qubits -> 25 MB GGUF / 3.9 s, with sum(p) = 1 to 1e-7. Practical ceiling is the state vector in RAM (~25–28 qubits on CPU, more on GPU).

Memory benchmark vs qiskit's Statevector (`python3 examples/bench_memory.py`):

```
  n gates | qiskit MB | qiskit sec | qullama gguf MB | qullama MB | qullama sec |  amp err
   16    91 |   108.4   |     0.03   |        1.59     |    88.3    |    0.16     | 1.4e-08
   18   105 |   169.6   |     0.15   |        6.31     |    92.2    |    0.56     | 1.3e-08
   20   120 |   428.3   |     0.83   |       25.19     |   104.2    |    2.16     | 5.7e-09
   22   130 |  1457.4   |     4.47   |      100.69     |   279.7    |    8.63     | 5.1e-09
```


## License

This software is licensed with [MIT License](LICENSE).


## Cite

```
@software{qullama2026,
  author = {Davide Gessa},
  title = {qullama: Run quantum circuits inside llama.cpp},
  url = {https://github.com/dakk/qullama},
  year = {2026},
}
```