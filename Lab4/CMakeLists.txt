cmake_minimum_required(VERSION 4.3)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
    "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
set(CMAKE_CXX_MODULE_STD ON)

project(LabCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(LabCUDA
    main.cpp
    matrix_cuda.cu
)

target_sources(LabCUDA
    PRIVATE
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
            matrix_core_n_files.cppm
            matrix_ops.cppm
)

target_include_directories(LabCUDA PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(LabCUDA PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION OFF
)

if(MSVC)
    target_compile_options(LabCUDA PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
else()
    target_compile_options(LabCUDA PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    )
endif()
