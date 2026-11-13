#include "matrix_cuda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstddef>
#include <limits>

namespace
{
    void setErrorMessage(
        char* buffer,
        int bufferSize,
        const char* operation,
        cudaError_t error)
    {
        if (buffer != nullptr && bufferSize > 0)
        {
            std::snprintf(
                buffer,
                static_cast<std::size_t>(bufferSize),
                "%s failed: %s",
                operation,
                cudaGetErrorString(error));
        }
    }

    bool checkCuda(
        cudaError_t error,
        const char* operation,
        char* errorMessage,
        int errorMessageSize)
    {
        if (error == cudaSuccess)
        {
            return true;
        }

        setErrorMessage(errorMessage, errorMessageSize, operation, error);
        return false;
    }

    __global__ void matrixMultiplyKernel(
        const double* matrixA,
        const double* matrixB,
        double* matrixC,
        int matrixSize)
    {
        const std::size_t column =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const std::size_t row =
            static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
        const std::size_t size = static_cast<std::size_t>(matrixSize);

        if (row < size && column < size)
        {
            double sum = 0.0;

            for (int index = 0; index < matrixSize; ++index)
            {
                sum += matrixA[row * size + static_cast<std::size_t>(index)] *
                       matrixB[static_cast<std::size_t>(index) * size + column];
            }

            matrixC[row * size + column] = sum;
        }
    }
}

extern "C" int initializeCuda(char* errorMessage, int errorMessageSize)
{
    if (errorMessage != nullptr && errorMessageSize > 0)
    {
        errorMessage[0] = '\0';
    }

    int deviceCount = 0;

    if (!checkCuda(
            cudaGetDeviceCount(&deviceCount),
            "cudaGetDeviceCount",
            errorMessage,
            errorMessageSize))
    {
        return 1;
    }

    if (deviceCount == 0)
    {
        if (errorMessage != nullptr && errorMessageSize > 0)
        {
            std::snprintf(
                errorMessage,
                static_cast<std::size_t>(errorMessageSize),
                "No CUDA-capable NVIDIA GPU was found.");
        }
        return 1;
    }

    if (!checkCuda(
            cudaSetDevice(0),
            "cudaSetDevice",
            errorMessage,
            errorMessageSize))
    {
        return 1;
    }

    // Initializes the CUDA context before benchmark measurements begin.
    if (!checkCuda(
            cudaFree(nullptr),
            "CUDA context initialization",
            errorMessage,
            errorMessageSize))
    {
        return 1;
    }

    return 0;
}

extern "C" int multiplyCuda(
    const double* hostA,
    const double* hostB,
    double* hostC,
    int matrixSize,
    int blockX,
    int blockY,
    double* kernelTimeMs,
    double* totalTimeMs,
    char* errorMessage,
    int errorMessageSize)
{
    if (errorMessage != nullptr && errorMessageSize > 0)
    {
        errorMessage[0] = '\0';
    }

    if (hostA == nullptr || hostB == nullptr || hostC == nullptr ||
        kernelTimeMs == nullptr || totalTimeMs == nullptr)
    {
        if (errorMessage != nullptr && errorMessageSize > 0)
        {
            std::snprintf(
                errorMessage,
                static_cast<std::size_t>(errorMessageSize),
                "Null pointer was passed to multiplyCuda.");
        }
        return 1;
    }

    if (matrixSize <= 0 || blockX <= 0 || blockY <= 0)
    {
        if (errorMessage != nullptr && errorMessageSize > 0)
        {
            std::snprintf(
                errorMessage,
                static_cast<std::size_t>(errorMessageSize),
                "Matrix size and block dimensions must be positive.");
        }
        return 1;
    }

    if (static_cast<std::size_t>(matrixSize) >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(matrixSize) / sizeof(double))
    {
        if (errorMessage != nullptr && errorMessageSize > 0)
        {
            std::snprintf(
                errorMessage,
                static_cast<std::size_t>(errorMessageSize),
                "Matrix is too large for GPU memory allocation.");
        }
        return 1;
    }

    int device = 0;
    cudaDeviceProp properties{};

    if (!checkCuda(
            cudaGetDevice(&device),
            "cudaGetDevice",
            errorMessage,
            errorMessageSize) ||
        !checkCuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties",
            errorMessage,
            errorMessageSize))
    {
        return 1;
    }

    const long long threadsPerBlock =
        static_cast<long long>(blockX) * static_cast<long long>(blockY);

    if (blockX > properties.maxThreadsDim[0] ||
        blockY > properties.maxThreadsDim[1] ||
        threadsPerBlock > properties.maxThreadsPerBlock)
    {
        if (errorMessage != nullptr && errorMessageSize > 0)
        {
            std::snprintf(
                errorMessage,
                static_cast<std::size_t>(errorMessageSize),
                "Block %d x %d exceeds the device limits.",
                blockX,
                blockY);
        }
        return 1;
    }

    const int gridX = matrixSize / blockX + (matrixSize % blockX != 0);
    const int gridY = matrixSize / blockY + (matrixSize % blockY != 0);

    if (gridX > properties.maxGridSize[0] ||
        gridY > properties.maxGridSize[1])
    {
        if (errorMessage != nullptr && errorMessageSize > 0)
        {
            std::snprintf(
                errorMessage,
                static_cast<std::size_t>(errorMessageSize),
                "Grid dimensions exceed the device limits.");
        }
        return 1;
    }

    const std::size_t elementCount =
        static_cast<std::size_t>(matrixSize) *
        static_cast<std::size_t>(matrixSize);
    const std::size_t bytes = elementCount * sizeof(double);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    *kernelTimeMs = 0.0;
    *totalTimeMs = 0.0;

    const auto totalStart = std::chrono::steady_clock::now();

    auto execute = [&]() -> bool
    {
        if (!checkCuda(
                cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes),
                "cudaMalloc(matrix A)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        if (!checkCuda(
                cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes),
                "cudaMalloc(matrix B)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        if (!checkCuda(
                cudaMalloc(reinterpret_cast<void**>(&deviceC), bytes),
                "cudaMalloc(matrix C)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        if (!checkCuda(
                cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(A, host to device)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        if (!checkCuda(
                cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(B, host to device)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        if (!checkCuda(
                cudaEventCreate(&startEvent),
                "cudaEventCreate(start)",
                errorMessage,
                errorMessageSize) ||
            !checkCuda(
                cudaEventCreate(&stopEvent),
                "cudaEventCreate(stop)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        const dim3 block(
            static_cast<unsigned int>(blockX),
            static_cast<unsigned int>(blockY));

        const dim3 grid(
            static_cast<unsigned int>(gridX),
            static_cast<unsigned int>(gridY));

        if (!checkCuda(
                cudaEventRecord(startEvent),
                "cudaEventRecord(start)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        matrixMultiplyKernel<<<grid, block>>>(
            deviceA,
            deviceB,
            deviceC,
            matrixSize);

        if (!checkCuda(
                cudaGetLastError(),
                "matrixMultiplyKernel launch",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        if (!checkCuda(
                cudaEventRecord(stopEvent),
                "cudaEventRecord(stop)",
                errorMessage,
                errorMessageSize) ||
            !checkCuda(
                cudaEventSynchronize(stopEvent),
                "cudaEventSynchronize",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        float elapsedKernelMs = 0.0f;
        if (!checkCuda(
                cudaEventElapsedTime(
                    &elapsedKernelMs,
                    startEvent,
                    stopEvent),
                "cudaEventElapsedTime",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }
        *kernelTimeMs = static_cast<double>(elapsedKernelMs);

        if (!checkCuda(
                cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy(C, device to host)",
                errorMessage,
                errorMessageSize))
        {
            return false;
        }

        return true;
    };

    const bool success = execute();
    const auto totalFinish = std::chrono::steady_clock::now();

    *totalTimeMs = std::chrono::duration<double, std::milli>(
                       totalFinish - totalStart)
                       .count();

    if (startEvent != nullptr)
    {
        cudaEventDestroy(startEvent);
    }

    if (stopEvent != nullptr)
    {
        cudaEventDestroy(stopEvent);
    }

    if (deviceA != nullptr)
    {
        cudaFree(deviceA);
    }

    if (deviceB != nullptr)
    {
        cudaFree(deviceB);
    }

    if (deviceC != nullptr)
    {
        cudaFree(deviceC);
    }

    return success ? 0 : 1;
}
