from pathlib import Path
import sys
import numpy as np


def read_matrix(path: Path) -> np.ndarray:
    with path.open('r', encoding='utf-8') as file:
        n = int(file.readline())
        values = []
        for line in file:
            values.extend(float(value) for value in line.split())
    if len(values) != n * n:
        raise ValueError(f'{path}: expected {n * n} values, got {len(values)}')
    return np.array(values, dtype=np.float64).reshape(n, n)


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        a = read_matrix(root / 'matrix_a.txt')
        b = read_matrix(root / 'matrix_b.txt')
        result = read_matrix(root / 'result.txt')
        if a.shape != b.shape or a.shape != result.shape:
            raise ValueError('Matrix sizes do not match')
    except Exception as error:
        print('ERROR:', error)
        return 1

    reference = a @ b
    diff = float(np.max(np.abs(result - reference)))
    print()
    print('Python / NumPy verification')
    print('---------------------------')
    print(f'Matrix size: {a.shape[0]} x {a.shape[1]}')
    print(f'Maximum absolute difference: {diff:.3e}')
    if np.allclose(result, reference, rtol=1e-9, atol=1e-9):
        print('RESULT: OK')
        return 0
    print('RESULT: FAILED')
    return 2


if __name__ == '__main__':
    sys.exit(main())
