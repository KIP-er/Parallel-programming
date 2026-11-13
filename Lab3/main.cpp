#include <mpi.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static bool readMatrix(const std::string& path, int& n, std::vector<double>& data, std::string& error) {
    std::ifstream in(path.c_str());
    if (!in) {
        error = "Cannot open " + path;
        return false;
    }

    if (!(in >> n) || n <= 0) {
        error = "Invalid matrix size in " + path;
        return false;
    }

    data.resize(static_cast<std::size_t>(n) * n);
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!(in >> data[i])) {
            error = "Not enough matrix values in " + path;
            return false;
        }
    }
    return true;
}

static bool writeMatrix(const std::string& path, const std::vector<double>& data, int n) {
    std::ofstream out(path.c_str());
    if (!out) return false;

    out << n << '\n' << std::setprecision(15);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j) out << ' ';
            out << data[static_cast<std::size_t>(i) * n + j];
        }
        out << '\n';
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int n = 0;
    int inputOk = 1;
    std::string error;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;

    if (rank == 0) {
        if (!readMatrix("matrix_a.txt", n, a, error)) {
            inputOk = 0;
        }
        else {
            int nB = 0;
            if (!readMatrix("matrix_b.txt", nB, b, error)) {
                inputOk = 0;
            }
            else if (n != nB) {
                error = "Matrix sizes do not match";
                inputOk = 0;
            }
        }
    }

    MPI_Bcast(&inputOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!inputOk) {
        if (rank == 0) std::cerr << "Error: " << error << '\n';
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) b.resize(static_cast<std::size_t>(n) * n);
    MPI_Bcast(b.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int baseRows = n / processes;
    const int extraRows = n % processes;
    const int localRows = baseRows + (rank < extraRows ? 1 : 0);
    std::vector<int> counts(processes);
    std::vector<int> displacements(processes);
    for (int i = 0; i < processes; ++i) {
        const int rows = baseRows + (i < extraRows ? 1 : 0);
        const int first = i * baseRows + std::min(i, extraRows);
        counts[i] = rows * n;
        displacements[i] = first * n;
    }

    std::vector<double> localA(static_cast<std::size_t>(localRows) * n);
    std::vector<double> localC(static_cast<std::size_t>(localRows) * n, 0.0);
    if (rank == 0) c.resize(static_cast<std::size_t>(n) * n);

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    MPI_Scatterv(a.data(), counts.data(), displacements.data(), MPI_DOUBLE,
        localA.data(), localRows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < localRows; ++i) {
        for (int k = 0; k < n; ++k) {
            const double aik = localA[static_cast<std::size_t>(i) * n + k];
            for (int j = 0; j < n; ++j) {
                localC[static_cast<std::size_t>(i) * n + j] += aik * b[static_cast<std::size_t>(k) * n + j];
            }
        }
    }

    MPI_Gatherv(localC.data(), localRows * n, MPI_DOUBLE,
        c.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double localTime = MPI_Wtime() - startTime;
    double totalTime = 0.0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double operations = 2.0 * n * n * n;
        const double memory = 3.0 * n * n * sizeof(double) / (1024.0 * 1024.0);

        std::cout << "Task size: " << n << " x " << n << " (" << n * n << " elements)\n";
        std::cout << "MPI processes: " << processes << "\n";
        std::cout << "Operations: " << operations << "\n";
        std::cout << "Memory (3 matrices): " << memory << " MB\n";
        std::cout << "Multiplication time: " << std::fixed << std::setprecision(3) << totalTime * 1000.0 << " ms\n";

        if (!writeMatrix("result.txt", c, n)) {
            std::cerr << "Error: cannot write result.txt\n";
            MPI_Finalize();
            return 1;
        }
        std::cout << "Result written to result.txt\n";
    }

    MPI_Finalize();
    return 0;
}
