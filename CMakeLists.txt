cmake_minimum_required(VERSION 3.24)

project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sibling checkout of CUDA nnCUDA (../nnCUDA), OpenCL nnCL (../nnCL), or ROCm nnROCm (../nnROCm).
# ./build.sh cuda|opencl|amd sets this. The selected tree's own tests stay out of `all`.
set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../nnCUDA" CACHE PATH "path to CUDA nnCUDA, OpenCL nnCL, or ROCm nnROCm")
if(NOT EXISTS "${NN_SOURCE}/CMakeLists.txt")
  message(FATAL_ERROR "NN_SOURCE has no CMakeLists.txt: ${NN_SOURCE}")
endif()
# nnCL ships src/kernels.cl. nnROCm ships HIP kernels. The CUDA tree has neither.
set(NN_IS_OPENCL FALSE)
set(NN_IS_ROCM FALSE)
if(EXISTS "${NN_SOURCE}/src/kernels.cl")
  set(NN_IS_OPENCL TRUE)
elseif(EXISTS "${NN_SOURCE}/src/elementwise.hip")
  set(NN_IS_ROCM TRUE)
endif()
if(NN_IS_ROCM)
  if(NOT DEFINED CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER /opt/rocm/bin/amdclang++)
  endif()
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx1152)
  endif()
  # Enable HIP in this directory so executables can link nnROCm device code.
  # The nested project() call does not publish CMAKE_HIP_LINK_EXECUTABLE here.
  enable_language(HIP)
elseif(NOT NN_IS_OPENCL AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
add_subdirectory(${NN_SOURCE} nn_build EXCLUDE_FROM_ALL)
if(NN_IS_OPENCL)
  message(STATUS "llm backend: OpenCL (${NN_SOURCE})")
elseif(NN_IS_ROCM)
  message(STATUS "llm backend: ROCm (${NN_SOURCE})")
else()
  message(STATUS "llm backend: CUDA (${NN_SOURCE})")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(llm STATIC src/bpe.cpp src/data.cpp src/presets.cpp src/train.cpp src/generate.cpp)
target_link_libraries(llm PUBLIC nn Threads::Threads)
target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(llm_tests tests/test_bpe.cpp tests/test_data.cpp tests/test_train.cpp)
target_link_libraries(llm_tests PRIVATE llm)
# CUDA IndexTensor is read with cudaMemcpy. OpenCL and ROCm use IndexTensor::to_host().
if(NOT NN_IS_OPENCL AND NOT NN_IS_ROCM)
  target_compile_definitions(llm_tests PRIVATE LLM_NN_CUDA)
endif()

add_executable(train_tiny examples/train_tiny.cpp)
target_link_libraries(train_tiny PRIVATE llm)

add_executable(train_chat examples/train_chat.cpp)
target_link_libraries(train_chat PRIVATE llm)

add_executable(train_bert_chat examples/train_bert_chat.cpp)
target_link_libraries(train_bert_chat PRIVATE llm)

add_executable(train_books examples/train_books.cpp)
target_link_libraries(train_books PRIVATE llm)

add_executable(chat examples/chat.cpp)
target_link_libraries(chat PRIVATE llm)

add_executable(finetune examples/finetune.cpp)
target_link_libraries(finetune PRIVATE llm)

add_executable(gguf examples/gguf.cpp)

# nnROCm device code is linked by the HIP compiler. Host objects stay C++.
# gguf does not link nn.
if(NN_IS_ROCM)
  set_target_properties(llm_tests train_tiny train_chat train_bert_chat train_books chat finetune
    PROPERTIES LINKER_LANGUAGE HIP)
endif()

enable_testing()
add_test(NAME bpe COMMAND llm_tests)
