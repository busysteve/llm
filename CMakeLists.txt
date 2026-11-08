cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default is a sibling checkout. -DNN_SOURCE=... points at another tree.
# nn's own tests stay out of `all`; llm still links the nn library target.
set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../nn" CACHE PATH "path to nn")
add_subdirectory(${NN_SOURCE} nn_build EXCLUDE_FROM_ALL)

add_library(llm STATIC src/bpe.cpp src/data.cpp src/presets.cpp src/train.cpp src/generate.cpp)
target_link_libraries(llm PUBLIC nn)
target_include_directories(llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(llm_tests tests/test_bpe.cpp tests/test_data.cpp tests/test_train.cpp)
target_link_libraries(llm_tests PRIVATE llm)

add_executable(train_tiny examples/train_tiny.cpp)
target_link_libraries(train_tiny PRIVATE llm)

add_executable(train_books examples/train_books.cpp)
target_link_libraries(train_books PRIVATE llm)

add_executable(chat examples/chat.cpp)
target_link_libraries(chat PRIVATE llm)

add_executable(finetune examples/finetune.cpp)
target_link_libraries(finetune PRIVATE llm)

enable_testing()
add_test(NAME bpe COMMAND llm_tests)
