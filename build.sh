#!/bin/sh
# Configure and build llm against an nn checkout.
# NN_SOURCE defaults to the sibling ../nn. That tree supplies the CMake target nn.
set -eu

cd "$(dirname "$0")"

if [ "$#" -ne 0 ]; then
  echo "usage: ./build.sh" >&2
  echo "Set NN_SOURCE to an nn checkout. The default is ../nn." >&2
  exit 1
fi

nn_source="${NN_SOURCE:-}"
if [ -z "$nn_source" ]; then
  nn_source="$(CDPATH= cd ../nn && pwd)"
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source"

cmake --build build -j --target llm_tests
cmake --build build -j --target gguf
./build/llm_tests
cmake --build build -j --target train_tiny
./build/train_tiny
cmake --build build -j --target train_books
cmake --build build -j --target chat
#time ./build/train_books --files books.txt --out llama.mod --preset llama-small --vocab 2048 --seq-len 256 --batch 6 --steps 5000 --prompt "moral goodness is"
#./build/train_books --files ./conversation_data/dialogue.txt --out chat-test.mod --preset llama-small --vocab 2048 --seq-len 256 --batch 6 --steps 5000 --log-every 1 --prompt "How are you today?"
