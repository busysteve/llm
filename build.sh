#!/bin/sh
# Configure and build llm against either backend:
#   ./build.sh cuda     sibling ../nn    (CUDA nn, architecture 86 unless CMAKE_CUDA_ARCHITECTURES is set)
#   ./build.sh opencl   sibling ../nnCL  (OpenCL)
# With no argument, NN_SOURCE is used when set. Otherwise OpenCL is chosen when
# ../nnCL is present and nvcc is not, and CUDA is chosen otherwise.
set -eu

cd "$(dirname "$0")"

backend="${1:-}"
nn_source="${NN_SOURCE:-}"

usage() {
  echo "usage: ./build.sh [cuda|opencl]" >&2
  exit 1
}

case "$backend" in
  cuda|nn)
    nn_source="$(CDPATH= cd ../nn && pwd)"
    ;;
  opencl|nncl|nnCL)
    nn_source="$(CDPATH= cd ../nnCL && pwd)"
    ;;
  "")
    if [ -z "$nn_source" ]; then
      if [ -f ../nnCL/src/kernels.cl ] && ! command -v nvcc >/dev/null 2>&1; then
        nn_source="$(CDPATH= cd ../nnCL && pwd)"
      else
        nn_source="$(CDPATH= cd ../nn && pwd)"
      fi
    fi
    ;;
  *)
    usage
    ;;
esac

if [ -f "$nn_source/src/kernels.cl" ]; then
  echo "llm backend: OpenCL ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source"
else
  echo "llm backend: CUDA ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source" \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}"
fi

cmake --build build -j --target llm_tests
cmake --build build -j --target gguf
./build/llm_tests
cmake --build build -j --target train_tiny
./build/train_tiny
cmake --build build -j --target train_books
cmake --build build -j --target chat
#time ./build/train_books --files books.txt --out llama.mod --preset llama-small --vocab 2048 --seq-len 256 --batch 6 --steps 5000 --prompt "moral goodness is"
#./build/train_books --files ./conversation_data/dialogue.txt --out chat-test.mod --preset llama-small --vocab 2048 --seq-len 256 --batch 6 --steps 5000 --log-every 1 --prompt "How are you today?"
