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

        sizes, times = [], []
        with csv_file.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sizes.append(int(row["N"]))
                times.append(float(row["time_ms"]))
    except Exception as e:
        print("Error:", e)
        return 1

    sizes = np.array(sizes, dtype=float)
    times = np.array(times, dtype=float)
    reference = times[0] * (sizes / sizes[0]) ** 3
    normalized = times / sizes ** 3 * 1e9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sizes, times, "o-", label="Measured")
    ax1.plot(sizes, reference, "--", label="N^3 reference")
    ax1.set_xlabel("Matrix size N")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Time vs N")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(sizes, normalized, "s-", color="green")
    ax2.set_xlabel("Matrix size N")
    ax2.set_ylabel("time / N^3 (ns per op)")
    ax2.set_title("Normalized time")
    ax2.grid(True)

    plt.tight_layout()
    out = root / "performance.png"
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())