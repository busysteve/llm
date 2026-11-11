cmake_minimum_required(VERSION 3.24)

project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sibling checkout of CUDA nn (../nn) or OpenCL nnCL (../nnCL).
# ./build.sh cuda|opencl sets this. The selected tree's own tests stay out of `all`.
set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../nn" CACHE PATH "path to CUDA nn or OpenCL nnCL")
if(NOT EXISTS "${NN_SOURCE}/CMakeLists.txt")
  message(FATAL_ERROR "NN_SOURCE has no CMakeLists.txt: ${NN_SOURCE}")
endif()
# nnCL ships src/kernels.cl. The CUDA tree does not.
set(NN_IS_OPENCL FALSE)
if(EXISTS "${NN_SOURCE}/src/kernels.cl")
  set(NN_IS_OPENCL TRUE)
endif()
if(NOT NN_IS_OPENCL AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
add_subdirectory(${NN_SOURCE} nn_build EXCLUDE_FROM_ALL)
if(NN_IS_OPENCL)
  message(STATUS "llm backend: OpenCL (${NN_SOURCE})")
else()
  message(STATUS "llm backend: CUDA (${NN_SOURCE})")
endif()

add_library(llm STATIC src/bpe.cpp src/data.cpp src/presets.cpp src/train.cpp src/generate.cpp)
target_link_libraries(llm PUBLIC nn)
target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(llm_tests tests/test_bpe.cpp tests/test_data.cpp tests/test_train.cpp)
target_link_libraries(llm_tests PRIVATE llm)
# CUDA IndexTensor has no to_host(); the data test copies with cudaMemcpy.
if(NOT NN_IS_OPENCL)
  target_compile_definitions(llm_tests PRIVATE LLM_NN_CUDA)
endif()

add_executable(train_tiny examples/train_tiny.cpp)
target_link_libraries(train_tiny PRIVATE llm)

add_executable(train_books examples/train_books.cpp)
target_link_libraries(train_books PRIVATE llm)

add_executable(chat examples/chat.cpp)
target_link_libraries(chat PRIVATE llm)

add_executable(finetune examples/finetune.cpp)
target_link_libraries(finetune PRIVATE llm)

add_executable(gguf examples/gguf.cpp)

enable_testing()
add_test(NAME bpe COMMAND llm_tests)
