cmake_minimum_required(VERSION 3.24)

project(llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sibling checkout that provides the CMake target nn. ./build.sh sets NN_SOURCE.
# Device toolkits stay in that project. Its tests stay out of `all`.
set(NN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../nn" CACHE PATH "path to an nn checkout")
if(NOT EXISTS "${NN_SOURCE}/CMakeLists.txt")
  message(FATAL_ERROR "NN_SOURCE has no CMakeLists.txt: ${NN_SOURCE}")
endif()
add_subdirectory(${NN_SOURCE} nn_build EXCLUDE_FROM_ALL)
message(STATUS "llm nn: ${NN_SOURCE}")

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

add_executable(gguf examples/gguf.cpp)

enable_testing()
add_test(NAME bpe COMMAND llm_tests)
