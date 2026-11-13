export module matrix_ops;

import std;
import matrix_core_n_files;

export Mat multiplySequential(const Mat& leftMatrix, const Mat& rightMatrix)
{
    const std::size_t size = leftMatrix.dimension();

    if (size != rightMatrix.dimension())
    {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    Mat product(size);

    for (std::size_t row = 0; row < size; ++row)
    {
        for (std::size_t column = 0; column < size; ++column)
        {
            double value = 0.0;

            for (std::size_t index = 0; index < size; ++index)
            {
                value += leftMatrix.at(row, index) *
                         rightMatrix.at(index, column);
            }

            product.at(row, column) = value;
        }
    }

    return product;
}

export Mat multiplyParallel(
    const Mat& leftMatrix,
    const Mat& rightMatrix,
    std::size_t workerCount)
{
    const std::size_t size = leftMatrix.dimension();

    if (size != rightMatrix.dimension())
    {
        throw std::invalid_argument("Matrix dimensions must match.");
    }

    if (workerCount == 0)
    {
        workerCount = 1;
    }

    workerCount = std::min(workerCount, size);

    Mat product(size);

    std::vector<std::thread> workers;
    workers.reserve(workerCount);

    const std::size_t blockSize =
        (size + workerCount - 1) / workerCount;

    for (std::size_t threadIndex = 0;
         threadIndex < workerCount;
         ++threadIndex)
    {
        const std::size_t rowBegin = threadIndex * blockSize;
        const std::size_t rowEnd = std::min(size, rowBegin + blockSize);

        if (rowBegin >= rowEnd)
        {
            break;
        }

        workers.emplace_back(
            [&, rowBegin, rowEnd]()
            {
                for (std::size_t row = rowBegin; row < rowEnd; ++row)
                {
                    for (std::size_t column = 0; column < size; ++column)
                    {
                        double value = 0.0;

                        for (std::size_t index = 0; index < size; ++index)
                        {
                            value += leftMatrix.at(row, index) *
                                     rightMatrix.at(index, column);
                        }

                        product.at(row, column) = value;
                    }
                }
            });
    }

    for (auto& worker : workers)
    {
        worker.join();
    }

    return product;
}
