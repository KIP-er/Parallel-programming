from pathlib import Path
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    try:
        root = Path(__file__).resolve().parent
        csv_file = root / "benchmark.csv"
        if not csv_file.exists():
            print("benchmark.csv not found. Run bench.py first.")
            return 1

        data: dict[int, dict[int, float]] = {}
        with csv_file.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n = int(row["N"])
                t = int(row["threads"])
                ms = float(row["time_ms"])
                data.setdefault(t, {})[n] = ms
    except Exception as e:
        print("Error:", e)
        return 1

    threads = sorted(data.keys())
    sizes = sorted(next(iter(data.values())).keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for t in threads:
        ts = [data[t][n] for n in sizes]
        ax1.plot(sizes, ts, "o-", label=f"{t} thread(s)")

    ref_sizes = np.array(sizes, dtype=float)
    base = data[threads[0]][sizes[0]]
    ax1.plot(ref_sizes, base * (ref_sizes / sizes[0]) ** 3, "--",
             color="gray", label="N^3 reference")
    ax1.set_xlabel("Matrix size N")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Time vs N")
    ax1.legend()
    ax1.grid(True)

    base_t = threads[0]
    for t in threads:
        speedups = [data[base_t][n] / data[t][n] for n in sizes]
        ax2.plot(sizes, speedups, "s-", label=f"{t} threads")
        ax2.axhline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax2.set_xlabel("Matrix size N")
    ax2.set_ylabel("Speedup vs 1 thread")
    ax2.set_title("Speedup (dotted = ideal)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    out = root / "performance.png"
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())