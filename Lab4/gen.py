from pathlib import Path
import argparse
import numpy as np

DEFAULT_SIZE = 300
SEED = 20240117
MIN_VALUE = -3
MAX_VALUE = 6


def generate_matrix(size, rng):
    return rng.integers(
        MIN_VALUE,
        MAX_VALUE + 1,
        size=(size, size),
        dtype=np.int64,
    )


def save_matrix(matrix, filename):
    with filename.open("w", encoding="utf-8") as file:
        file.write(f"{matrix.shape[0]}\n")
        for row in matrix:
            file.write(" ".join(str(int(value)) for value in row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate two random square matrices for the CUDA laboratory."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"matrix dimension (default: {DEFAULT_SIZE})",
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be greater than zero")

    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    first = generate_matrix(args.size, rng)
    second = generate_matrix(args.size, rng)

    save_matrix(first, root / "matrix_a.txt")
    save_matrix(second, root / "matrix_b.txt")

    print(f"Generated two {args.size}x{args.size} matrices")
    print(f"  {root / 'matrix_a.txt'}")
    print(f"  {root / 'matrix_b.txt'}")


if __name__ == "__main__":
    main()
