from pathlib import Path
import csv
import sys
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    csv_file = root / 'benchmark.csv'
    if not csv_file.exists():
        print('benchmark.csv not found. Run bench.py first.')
        return 1

    rows = []
    with csv_file.open('r', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        print('benchmark.csv is empty.')
        return 1

    processes = sorted(set(int(row['processes']) for row in rows))
    sizes = sorted(set(int(row['N']) for row in rows))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for count in processes:
        selected = [row for row in rows if int(row['processes']) == count]
        selected.sort(key=lambda row: int(row['N']))
        x = [int(row['N']) for row in selected]
        y = [float(row['time_ms']) for row in selected]
        ax[0].plot(x, y, 'o-', label=f'{count} processes')
    ax[0].set_title('MPI: time vs matrix size')
    ax[0].set_xlabel('Matrix size N')
    ax[0].set_ylabel('Time (ms)')
    ax[0].legend()
    ax[0].grid(True)

    for n in sizes:
        selected = [row for row in rows if int(row['N']) == n]
        selected.sort(key=lambda row: int(row['processes']))
        x = [int(row['processes']) for row in selected]
        y = [float(row['speedup']) for row in selected]
        ax[1].plot(x, y, 's-', label=f'N = {n}')
    ax[1].plot(processes, processes, '--', label='Ideal speedup')
    ax[1].set_title('MPI: speedup')
    ax[1].set_xlabel('MPI processes')
    ax[1].set_ylabel('Speedup')
    ax[1].set_xticks(processes)
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    out = root / 'performance.png'
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
