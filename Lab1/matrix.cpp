import std;
import Matrix;

static void printMenu() {
	std::cout <<
		"\n=== Lab: square matrix multiplication ===\n"
		"1. Generate matrices (python gen.py)\n"
		"2. Multiply matrices (matrix_a.txt * matrix_b.txt -> result.txt)\n"
		"3. Show first 5x5 elements of the result\n"
		"4. Verify result (python verify.py)\n"
		"5. Exit\n"
		"Choice: ";
}

static int runPython(const std::string& script) {
	const std::string cmd = "python " + script;
	return std::system(cmd.c_str());
}

static void doGenerate() {
	if (runPython("gen.py") != 0)
		std::cerr << "Failed to run gen.py\n";
}

static void doMultiply() {
	try {
		const Matrix a = Matrix::read("matrix_a.txt");
		const Matrix b = Matrix::read("matrix_b.txt");
		if (a.size() != b.size())
			throw std::runtime_error("Matrix sizes do not match");

		const std::size_t n = a.size();
		const double ops = 2.0 * double(n) * double(n) * double(n);
		const double bytes = 3.0 * double(n) * double(n) * sizeof(double);

		std::cout << "Task size: " << n << " x " << n << " (" << n * n << " elements)\n";
		std::cout << "Operations: " << ops << "\n";
		std::cout << "Memory (3 matrices): " << bytes / (1024.0 * 1024.0) << " MB\n";

		const auto t0 = std::chrono::steady_clock::now();
		const Matrix c = Matrix::multiply(a, b);
		const auto t1 = std::chrono::steady_clock::now();

		const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		std::cout << "Multiplication time: " << ms << " ms\n";

		c.write("result.txt");
		std::cout << "Result written to result.txt\n";
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
	}
}

static void doShowResult() {
	try {
		const Matrix c = Matrix::read("result.txt");
		const std::size_t k = std::min<std::size_t>(c.size(), 5);
		std::cout << "First " << k << "x" << k << " elements:\n";
		for (std::size_t i = 0; i < k; ++i) {
			for (std::size_t j = 0; j < k; ++j)
				std::cout << c(i, j) << (j + 1 == k ? '\n' : ' ');
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << '\n';
	}
}

static void doVerify() {
	if (runPython("verify.py") != 0)
		std::cerr << "Verification failed\n";
}

int main() {
	int choice = -1;
	while (choice != 5) {
		printMenu();
		if (!(std::cin >> choice)) break;

		switch (choice) {
		case 1: doGenerate();   break;
		case 2: doMultiply();   break;
		case 3: doShowResult(); break;
		case 4: doVerify();     break;
		case 5: std::cout << "Exit.\n"; break;
		default: std::cout << "Invalid menu item\n";
		}
	}
	return 0;
}