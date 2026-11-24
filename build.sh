#!/bin/sh
# Configure and build llm against one backend:
#   ./build.sh cuda     sibling ../nnCUDA  (CUDA nn, architecture 86 unless CMAKE_CUDA_ARCHITECTURES is set)
#   ./build.sh opencl   sibling ../nnCL    (OpenCL)
#   ./build.sh amd      sibling ../nnROCm  (ROCm, architecture gfx1152 unless CMAKE_HIP_ARCHITECTURES is set)
#   ./build.sh arc      sibling ../nnArc   (Intel Arc, SYCL via icpx, JIT SPIR-V)
#   ./build.sh agx      sibling ../nnAGX   (Apple Metal)
# With no argument, NN_SOURCE is used when set. On macOS, ../nnAGX is chosen when
# its Metal kernels are present. Otherwise OpenCL is chosen when ../nnCL is
# present and nvcc is not, and CUDA is chosen otherwise.
set -eu

cd "$(dirname "$0")"

backend="${1:-}"
nn_source="${NN_SOURCE:-}"

usage() {
  echo "usage: ./build.sh [cuda|opencl|amd|arc|agx]" >&2
  exit 1
}

case "$backend" in
  cuda|nn|nncuda|nnCUDA)
    nn_source="$(CDPATH= cd ../nnCUDA && pwd)"
    ;;
  opencl|nncl|nnCL)
    nn_source="$(CDPATH= cd ../nnCL && pwd)"
    ;;
  amd|rocm|nnrocm|nnROCm)
    nn_source="$(CDPATH= cd ../nnROCm && pwd)"
    ;;
  arc|intel|nnarc|nnArc)
    nn_source="$(CDPATH= cd ../nnArc && pwd)"
    ;;
  agx|apple|nnagx|nnAGX)
    nn_source="$(CDPATH= cd ../nnAGX && pwd)"
    ;;
  "")
    if [ -z "$nn_source" ]; then
      if [ "$(uname)" = "Darwin" ] && [ -f ../nnAGX/src/kernels.metal ]; then
        nn_source="$(CDPATH= cd ../nnAGX && pwd)"
      elif [ -f ../nnCL/src/kernels.cl ] && ! command -v nvcc >/dev/null 2>&1; then
        nn_source="$(CDPATH= cd ../nnCL && pwd)"
      else
        nn_source="$(CDPATH= cd ../nnCUDA && pwd)"
      fi
    fi
    ;;
  *)
    usage
    ;;
esac

# Resolve icpx before touching build/. A missing compiler must not clear a cache
# that was configured for another backend.
arc_cxx=""
if [ -f "$nn_source/src/elementwise.cpp" ] && [ ! -f "$nn_source/src/kernels.cl" ] &&
   [ ! -f "$nn_source/src/kernels.metal" ] && [ ! -f "$nn_source/src/elementwise.hip" ]; then
  if [ -n "${CMAKE_CXX_COMPILER:-}" ] && [ -x "${CMAKE_CXX_COMPILER}" ] &&
     [ "$(basename "${CMAKE_CXX_COMPILER}")" = "icpx" ]; then
    arc_cxx="${CMAKE_CXX_COMPILER}"
  fi
  if [ -z "$arc_cxx" ]; then
    for candidate in \
        /opt/intel/oneapi/compiler/latest/bin/icpx \
        /opt/intel/oneapi/compiler/2026.0/bin/icpx
    do
      if [ -x "$candidate" ]; then
        arc_cxx="$candidate"
        break
      fi
    done
  fi
  if [ -z "$arc_cxx" ] && command -v icpx >/dev/null 2>&1; then
    arc_cxx="$(command -v icpx)"
  fi
  if [ -z "$arc_cxx" ]; then
    echo "llm: icpx was not found. Install intel-oneapi-dpcpp-cpp." >&2
    exit 1
  fi
fi

if [ -f build/CMakeCache.txt ]; then
  cached="$(sed -n 's/^NN_SOURCE:[A-Z]*=//p' build/CMakeCache.txt)"
  if [ -n "$cached" ] && [ "$cached" != "$nn_source" ]; then
    echo "llm: clearing build/ configured with $cached" >&2
    rm -rf build
  fi
fi

if [ -f "$nn_source/src/kernels.metal" ]; then
  echo "llm backend: AGX ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source"
elif [ -f "$nn_source/src/kernels.cl" ]; then
  echo "llm backend: OpenCL ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source"
elif [ -f "$nn_source/src/elementwise.hip" ]; then
  echo "llm backend: ROCm ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source" \
    -DCMAKE_HIP_COMPILER="${CMAKE_HIP_COMPILER:-/opt/rocm/bin/amdclang++}" \
    -DCMAKE_HIP_ARCHITECTURES="${CMAKE_HIP_ARCHITECTURES:-gfx1152}"
elif [ -f "$nn_source/src/elementwise.cpp" ]; then
  echo "llm backend: Arc ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source" \
    -DCMAKE_CXX_COMPILER="$arc_cxx"
else
  # C++ sources in nn include CUDA headers. The HPC SDK nvcc rejects GCC later than 15.
  host_cxx="${CXX:-}"
  if [ -z "$host_cxx" ]; then
    host_cxx="$(command -v c++ || command -v g++ || true)"
  fi
  major=""
  if [ -n "$host_cxx" ]; then
    major="$("$host_cxx" -dumpversion 2>/dev/null | cut -d. -f1)"
  fi
  if [ -n "$major" ] && [ "$major" -gt 15 ] && command -v g++-15 >/dev/null 2>&1; then
    echo "llm host compiler: g++-15 (GCC $major is newer than this CUDA build allows)" >&2
    host_cxx="$(command -v g++-15)"
  fi
  if [ -n "$host_cxx" ] && [ -f build/CMakeCache.txt ]; then
    cached="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' build/CMakeCache.txt)"
    if [ -n "$cached" ] && [ "$cached" != "$host_cxx" ]; then
      echo "llm: clearing build/ configured with $cached" >&2
      rm -rf build
    fi
  fi
  echo "llm backend: CUDA ($nn_source)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNN_SOURCE="$nn_source" \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86}" \
    -DCMAKE_CXX_COMPILER="$host_cxx" -DCMAKE_CUDA_HOST_COMPILER="$host_cxx"
fi

cmake --build build -j --target llm_tests
cmake --build build -j --target gguf
./build/llm_tests
cmake --build build -j --target train_tiny
./build/train_tiny
cmake --build build -j --target train_data
cmake --build build -j --target chat
#time ./build/train_data --files books.txt --out llama.mod --preset llama-small --vocab 2048 --seq-len 256 --batch 6 --steps 5000 --prompt "moral goodness is"
#./build/train_data --files ./conversation_data/dialogue.txt --out chat-test.mod --preset llama-small --vocab 2048 --seq-len 256 --batch 6 --steps 5000 --log-every 1 --prompt "How are you today?"
