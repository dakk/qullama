#!/usr/bin/env bash
# build_patched_llama.sh - download llama.cpp at the pinned tag, apply the qgate
# patch, compile, and produce the llama-embedding binary.
#
# Usage:
#   ./build_patched_llama.sh [SRC_DIR]        # SRC_DIR defaults to ./llama.cpp
#   TAG=b12075 JOBS=16 ./build_patched_llama.sh
#
# Env:
#   TAG        llama.cpp tag to download   (default: b12075 - the tag the patch was made against)
#   REPO_URL   git remote                 (default: https://github.com/ggml-org/llama.cpp.git)
#   BUILD_DIR  cmake build dir            (default: $SRC_DIR/build)
#   JOBS       parallel build jobs        (default: nproc)
set -euo pipefail

TAG="${TAG:-b12075}"
REPO_URL="${REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/qullama-b12075.patch"
SRC_DIR="${1:-$SCRIPT_DIR/llama.cpp}"
mkdir -p "$SRC_DIR"
SRC_DIR="$(cd "$SRC_DIR" && pwd)"   # absolutize before any cd
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build}"
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR="$PWD/$BUILD_DIR";; esac
JOBS="${JOBS:-$(nproc)}"

[ -f "$PATCH_FILE" ] || { echo "patch not found: $PATCH_FILE" >&2; exit 1; }

# 1. download the pinned llama.cpp version
if [ ! -d "$SRC_DIR/.git" ]; then
    echo "==> cloning llama.cpp tag $TAG into $SRC_DIR"
    git clone --depth 1 --branch "$TAG" "$REPO_URL" "$SRC_DIR"
else
    echo "==> reusing existing clone at $SRC_DIR"
fi
cd "$SRC_DIR"
echo "==> llama.cpp at $(git log --oneline -1) (tag: $(git describe --tags --exact-match HEAD 2>/dev/null || echo '?'))"

# 2. apply the qgate patch (idempotent)
if [ -f src/models/qgate.cpp ]; then
    echo "==> patch already applied (src/models/qgate.cpp exists)"
else
    echo "==> applying $PATCH_FILE"
    git apply --check "$PATCH_FILE"
    git apply "$PATCH_FILE"
    echo "==> patch applied"
fi

# 3. configure + build the tools we need
echo "==> configuring cmake build in $BUILD_DIR"
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON -DGGML_NATIVE=ON

echo "==> building llama-embedding + llama-tokenize with -j$JOBS"
cmake --build "$BUILD_DIR" --target llama-embedding llama-tokenize -j "$JOBS"

echo
echo "==> done. binaries:"
ls -la "$BUILD_DIR/bin/llama-embedding" "$BUILD_DIR/bin/llama-tokenize"
echo
echo "run a circuit:  python3 -m qullama eval $SCRIPT_DIR/examples/bell.qasm --llama-embedding $BUILD_DIR/bin/llama-embedding"
