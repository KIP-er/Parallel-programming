import std;

import matrix_core_n_files;
import matrix_ops;

#include "matrix_cuda.hpp"

namespace
{
    struct BlockConfig
    {
        int x;
        int y;
    };

    bool matricesEqual(
        const Mat& firstMatrix,
        const Mat& secondMatrix,
        double epsilon = 1e-9)
    {
        const std::size_t size = firstMatrix.dimension();

        if (size != secondMatrix.dimension())
        {
            return false;
        }

        for (std::size_t row = 0; row < size; ++row)
        {
            for (std::size_t column = 0; column < size; ++column)
            {
                const double difference = std::abs(
                    firstMatrix.at(row, column) -
                    secondMatrix.at(row, column));

                if (difference > epsilon)
                {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename Function>
    std::pair<Mat, double> measureTime(Function&& function)
    {
        const auto begin = std::chrono::steady_clock::now();
        Mat result = function();
        const auto finish = std::chrono::steady_clock::now();

        const double elapsedMs =
            std::chrono::duration<double, std::milli>(finish - begin).count();

        return { std::move(result), elapsedMs };
    }

    void checkCudaResult(int result, const char* errorMessage)
    {
        if (result != 0)
        {
            throw std::runtime_error(
                std::string("CUDA error: ") +
                (errorMessage[0] != '\0' ? errorMessage : "unknown error"));
        }
    }
}

int main(int argc, char* argv[])
{
    try
    {
        const std::string firstPath =
            argc >= 2 ? argv[1] : "matrix_a.txt";
        const std::string secondPath =
            argc >= 3 ? argv[2] : "matrix_b.txt";

        const Mat first = loadMat(firstPath);
        const Mat second = loadMat(secondPath);

        if (first.dimension() != second.dimension())
        {
            throw std::runtime_error("Matrix dimensions must be equal.");
        }

        const std::size_t matrixSize = first.dimension();

        if (matrixSize > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("Matrix dimension is too large for this CUDA implementation.");
        }

        constexpr std::array<BlockConfig, 4> configurations =
        {{
            { 8, 8 },
            { 16, 16 },
            { 32, 8 },
            { 32, 32 }
        }};

        constexpr int repetitions = 3;

        std::cout << "Parallel Programming - CUDA Laboratory\n";
        std::cout << "Matrix multiplication\n\n";
        std::cout << "Matrix size: "
                  << matrixSize << " x " << matrixSize << "\n\n";

        char cudaError[512]{};
        checkCudaResult(
            initializeCuda(cudaError, static_cast<int>(sizeof(cudaError))),
            cudaError);

        std::cout << "CUDA device initialized successfully.\n\n";

        const auto [sequentialResult, sequentialTime] =
            measureTime(
                [&]()
                {
                    return multiplySequential(first, second);
                });

        std::cout << "Sequential CPU time: "
                  << sequentialTime << " ms\n\n";

        double bestCudaTotalTime = std::numeric_limits<double>::max();
        BlockConfig bestConfiguration{};
        double bestKernelTime = 0.0;

        Mat cudaResult(matrixSize);

        for (const BlockConfig config : configurations)
        {
            const int gridX =
                static_cast<int>(matrixSize) / config.x +
                (static_cast<int>(matrixSize) % config.x != 0);
            const int gridY =
                static_cast<int>(matrixSize) / config.y +
                (static_cast<int>(matrixSize) % config.y != 0);

            // One untimed call warms up this particular launch configuration.
            double warmupKernelTime = 0.0;
            double warmupTotalTime = 0.0;

            checkCudaResult(
                multiplyCuda(
                    first.data(),
                    second.data(),
                    cudaResult.data(),
                    static_cast<int>(matrixSize),
                    config.x,
                    config.y,
                    &warmupKernelTime,
                    &warmupTotalTime,
                    cudaError,
                    static_cast<int>(sizeof(cudaError))),
                cudaError);

            double kernelTimeSum = 0.0;
            double totalTimeSum = 0.0;

            for (int repetition = 0; repetition < repetitions; ++repetition)
            {
                double kernelTime = 0.0;
                double totalTime = 0.0;

                checkCudaResult(
                    multiplyCuda(
                        first.data(),
                        second.data(),
                        cudaResult.data(),
                        static_cast<int>(matrixSize),
                        config.x,
                        config.y,
                        &kernelTime,
                        &totalTime,
                        cudaError,
                        static_cast<int>(sizeof(cudaError))),
                    cudaError);

                kernelTimeSum += kernelTime;
                totalTimeSum += totalTime;
            }

            const double averageKernelTime = kernelTimeSum / repetitions;
            const double averageTotalTime = totalTimeSum / repetitions;

            if (!matricesEqual(sequentialResult, cudaResult))
            {
                throw std::runtime_error(
                    "CPU and CUDA results differ for block " +
                    std::to_string(config.x) + "x" +
                    std::to_string(config.y) + ".");
            }

            logBenchmark(
                "benchmark.csv",
                matrixSize,
                config.x,
                config.y,
                gridX,
                gridY,
                sequentialTime,
                averageKernelTime,
                averageTotalTime);

            const double speedup =
                averageTotalTime > 0.0
                    ? sequentialTime / averageTotalTime
                    : 0.0;

            std::cout << "Block: " << config.x << " x " << config.y
                      << " | Grid: " << gridX << " x " << gridY << '\n';
            std::cout << "  CUDA kernel time: "
                      << averageKernelTime << " ms\n";
            std::cout << "  CUDA total time:  "
                      << averageTotalTime << " ms\n";
            std::cout << "  Speedup:          "
                      << speedup << "\n";
            std::cout << "  Result check:     OK\n\n";

            if (averageTotalTime < bestCudaTotalTime)
            {
                bestCudaTotalTime = averageTotalTime;
                bestConfiguration = config;
                bestKernelTime = averageKernelTime;
                saveMat(cudaResult, "result.txt");
            }
        }

        std::cout << "Best configuration: "
                  << bestConfiguration.x << " x "
                  << bestConfiguration.y << '\n';
        std::cout << "Best CUDA kernel time: "
                  << bestKernelTime << " ms\n";
        std::cout << "Best CUDA total time:  "
                  << bestCudaTotalTime << " ms\n";
        std::cout << "Result: result.txt\n";
        std::cout << "Benchmark: benchmark.csv\n";

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "ERROR: " << exception.what() << '\n';
        return 1;
    }
}
