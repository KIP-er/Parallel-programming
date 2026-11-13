export module matrix_core_n_files;

import std;

export class Mat
{
private:
    std::size_t _size = 0;
    std::vector<double> _data;

public:
    Mat() = default;

    explicit Mat(std::size_t size)
        : _size(size)
    {
        if (size == 0)
        {
            throw std::invalid_argument("Matrix dimension must be positive.");
        }

        if (size > std::numeric_limits<std::size_t>::max() / size)
        {
            throw std::length_error("Matrix is too large.");
        }

        _data.assign(size * size, 0.0);
    }

    [[nodiscard]]
    std::size_t dimension() const noexcept
    {
        return _size;
    }

    double& at(std::size_t row, std::size_t column) noexcept
    {
        return _data[row * _size + column];
    }

    const double& at(std::size_t row, std::size_t column) const noexcept
    {
        return _data[row * _size + column];
    }

    [[nodiscard]]
    double* data() noexcept
    {
        return _data.data();
    }

    [[nodiscard]]
    const double* data() const noexcept
    {
        return _data.data();
    }
};

export Mat loadMat(const std::string& filePath)
{
    std::ifstream file(filePath);

    if (!file)
    {
        throw std::runtime_error("Cannot open input file: " + filePath);
    }

    std::size_t size = 0;
    file >> size;

    if (!file || size == 0)
    {
        throw std::runtime_error("Invalid matrix dimension in: " + filePath);
    }

    Mat matrix(size);

    for (std::size_t row = 0; row < size; ++row)
    {
        for (std::size_t column = 0; column < size; ++column)
        {
            if (!(file >> matrix.at(row, column)))
            {
                throw std::runtime_error("Not enough values in: " + filePath);
            }
        }
    }

    return matrix;
}

export void saveMat(const Mat& matrix, const std::string& filePath)
{
    std::ofstream file(filePath);

    if (!file)
    {
        throw std::runtime_error("Cannot create output file: " + filePath);
    }

    const std::size_t size = matrix.dimension();

    file << size << '\n';
    file << std::setprecision(17);

    for (std::size_t row = 0; row < size; ++row)
    {
        for (std::size_t column = 0; column < size; ++column)
        {
            if (column > 0)
            {
                file << ' ';
            }

            file << matrix.at(row, column);
        }

        file << '\n';
    }
}

export void logBenchmark(
    const std::string& filePath,
    std::size_t matrixSize,
    int blockX,
    int blockY,
    int gridX,
    int gridY,
    double sequentialTime,
    double cudaKernelTime,
    double cudaTotalTime)
{
    const std::string expectedHeader =
        "matrix_size,block_x,block_y,grid_x,grid_y,"
        "sequential_ms,cuda_kernel_ms,cuda_total_ms,speedup";

    const bool fileExists = std::filesystem::exists(filePath);
    bool writeHeader = !fileExists || std::filesystem::file_size(filePath) == 0;

    if (fileExists && !writeHeader)
    {
        std::ifstream existingFile(filePath);
        std::string existingHeader;
        std::getline(existingFile, existingHeader);

        if (existingHeader != expectedHeader)
        {
            throw std::runtime_error(
                "benchmark.csv has an old or unexpected format. "
                "Delete it before starting a new CUDA experiment.");
        }
    }

    std::ofstream file(filePath, std::ios::app);

    if (!file)
    {
        throw std::runtime_error("Cannot open benchmark file: " + filePath);
    }

    if (writeHeader)
    {
        file << expectedHeader << '\n';
    }

    const double speedup =
        cudaTotalTime > 0.0 ? sequentialTime / cudaTotalTime : 0.0;

    file << matrixSize << ','
         << blockX << ','
         << blockY << ','
         << gridX << ','
         << gridY << ','
         << std::setprecision(10)
         << sequentialTime << ','
         << cudaKernelTime << ','
         << cudaTotalTime << ','
         << speedup << '\n';
}
