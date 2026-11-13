from pathlib import Path
import sys
import numpy as np


def load_matrix(filename):
    with filename.open("r", encoding="utf-8") as file:
        size = int(file.readline())
        values = []
        for line in file:
            values.extend(float(value) for value in line.split())

    if len(values) != size * size:
        raise ValueError(
            f"{filename}: expected {size * size} values, got {len(values)}"
        )

    return np.array(values, dtype=np.float64).reshape(size, size)


def main():
    root = Path(__file__).resolve().parent
    matrix_a_path = root / "matrix_a.txt"
    matrix_b_path = root / "matrix_b.txt"
    result_path = root / "result.txt"

    for path in (matrix_a_path, matrix_b_path, result_path):
        if not path.exists():
            print(f"ERROR: {path.name} not found.")
            return 1

    matrix_a = load_matrix(matrix_a_path)
    matrix_b = load_matrix(matrix_b_path)
    result = load_matrix(result_path)

    reference = matrix_a @ matrix_b
    max_difference = float(np.max(np.abs(result - reference)))

    print("\nPython / NumPy verification")
    print("---------------------------")
    print(f"Maximum absolute difference: {max_difference:.3e}")

    if np.allclose(result, reference, rtol=1e-9, atol=1e-9):
        print("RESULT: OK")
        return 0

    print("RESULT: FAILED")
    return 2


if __name__ == "__main__":
    sys.exit(main())
