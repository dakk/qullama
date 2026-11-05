# syntax=docker/dockerfile:1
#
# Build:  docker build -t qullama .
# Run:    docker run --rm qullama eval examples/bell.qasm
#         docker run --rm -v "$PWD:/work" qullama eval /work/my_circuit.qasm

# stage 1: build the patched llama.cpp + install qullama
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake g++ make libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/qullama
COPY . .
RUN pip install --no-cache-dir . && JOBS="$(nproc)" ./build_patched_llama.sh

# stage 2: qullama + the built binaries/libs
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/qullama
COPY pyproject.toml README.md LICENSE ./
COPY qullama/ qullama/
COPY examples/ examples/
COPY --from=builder /opt/qullama/llama.cpp/build/bin ./llama.cpp/build/bin
RUN pip install --no-cache-dir .
ENV QULLAMA_LLAMA_EMBEDDING=/opt/qullama/llama.cpp/build/bin/llama-embedding
ENTRYPOINT ["qullama"]
