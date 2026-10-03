from pathlib import Path
import sys
import numpy as np

def write_matrix(path: Path, m: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{m.shape[0]}\n")
        for row in m:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    a = rng.integers(0, 10, size=(n, n))
    b = rng.integers(0, 10, size=(n, n))
    write_matrix(root / "matrix_a.txt", a)
    write_matrix(root / "matrix_b.txt", b)
    print(f"Generated matrix_a.txt and matrix_b.txt ({n}x{n})")

if __name__ == "__main__":
    main()