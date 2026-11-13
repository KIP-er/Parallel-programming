cmake_minimum_required(VERSION 3.16)
project(matrix_mpi_lab CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(matrix_mpi main.cpp)
target_link_libraries(matrix_mpi PRIVATE MPI::MPI_CXX)
