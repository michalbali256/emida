cmake_minimum_required(VERSION 3.9)
project(EMIDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION On)


include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/option)

add_executable(emida
    src/main.cpp
    src/cross_corr.cu
)

