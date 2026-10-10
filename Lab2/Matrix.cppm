export module Matrix;
import std;

export class Matrix {
public:
	Matrix() = default;
	explicit Matrix(std::size_t n) : n_(n), data_(n * n, 0.0) {}

	std::size_t size() const noexcept { return n_; }

	double  operator()(std::size_t i, std::size_t j) const { return data_[i * n_ + j]; }
	double& operator()(std::size_t i, std::size_t j) { return data_[i * n_ + j]; }

	static Matrix read(const std::filesystem::path& path);
	void          write(const std::filesystem::path& path) const;

	static Matrix multiply(const Matrix& a, const Matrix& b);

private:
	std::size_t n_ = 0;
	std::vector<double> data_;
};

Matrix Matrix::read(const std::filesystem::path& path) {
	std::ifstream in(path);
	if (!in)  throw std::runtime_error("Cannot open "  + path.string());

	std::size_t n = 0;
	in >> n;
	Matrix m(n);
	for (auto& v : m.data_) in >> v;
	return m;
}

void Matrix::write(const std::filesystem::path& path) const {
	std::ofstream out(path);
	if (!out) throw std::runtime_error("Cannot create " + path.string());

	out << n_ << '\n';
	for (std::size_t i = 0; i < n_; ++i) {
		for (std::size_t j = 0; j < n_; ++j) {
			if (j) out << ' ';
			out << data_[i * n_ + j];
		}
		out << '\n';
	}
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
	const std::size_t n = a.size();
	Matrix c(n);
#pragma omp parallel for
	for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
		for (std::size_t k = 0; k < n; ++k) {
			const double aik = a(i, k);
			for (std::size_t j = 0; j < n; ++j)
				c(i, j) += aik * b(k, j);
		}
	return c;
}