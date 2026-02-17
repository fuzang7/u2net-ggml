cmake_minimum_required(VERSION 3.14)
project(u2net-ggml C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(GGML_NATIVE ON)
set(GGML_BUILD_TESTS OFF)

add_subdirectory(ggml)

add_library(u2net-core STATIC
    src/u2net_load.cpp
    src/u2net.h
    src/u2net_layers.h 
)

target_link_libraries(u2net-core PUBLIC ggml)

target_include_directories(u2net-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)

add_executable(u2net-cli src/main.cpp)
target_link_libraries(u2net-cli PRIVATE u2net-core)