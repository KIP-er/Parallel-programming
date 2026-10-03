from pathlib import Path
import sys
import numpy as np

def read_matrix(filename: Path) -> np.ndarray:
    with filename.open("r", encoding="utf-8") as file:
        size = int(file.readline())
        values = []
        for line in file:
            values.extend(float(v) for v in line.split())
    if len(values) != size * size:
        raise ValueError(f"{filename}: expected {size*size} values, got {len(values)}")
    return np.array(values, dtype=np.float64).reshape(size, size)

def main() -> int:
    root = Path(__file__).resolve().parent

    for name in ("matrix_a.txt", "matrix_b.txt", "result.txt"):
        if not (root / name).exists():
            print(f"ERROR: {name} not found.")
            return 1

    a = read_matrix(root / "matrix_a.txt")
    b = read_matrix(root / "matrix_b.txt")
    cpp = read_matrix(root / "result.txt")
    ref = a @ b

    diff = np.max(np.abs(cpp - ref))

    print()
    print("Python / NumPy verification")
    print("---------------------------")
    print(f"Matrix size: {a.shape[0]} x {a.shape[1]}")
    print(f"Maximum absolute difference: {diff:.3e}")

    if np.allclose(cpp, ref, rtol=1e-9, atol=1e-9):
        print("RESULT: OK")
        return 0

    print("RESULT: FAILED")
    return 2

if __name__ == "__main__":
    sys.exit(main())