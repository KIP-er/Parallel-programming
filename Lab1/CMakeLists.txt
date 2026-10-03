cmake_minimum_required(VERSION 3.28)
project(matrix_lab CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_executable(matrix matrix.cpp Matrix.cppm)
target_compile_features(matrix PRIVATE cxx_std_23)