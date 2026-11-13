from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    csv_path = root / "benchmark.csv"

    if not csv_path.exists():
        print("benchmark.csv not found. Run the CUDA experiments first.")
        return 1

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows.append({
                "size": int(row["matrix_size"]),
                "block_x": int(row["block_x"]),
                "block_y": int(row["block_y"]),
                "sequential": float(row["sequential_ms"]),
                "kernel": float(row["cuda_kernel_ms"]),
                "total": float(row["cuda_total_ms"]),
                "speedup": float(row["speedup"]),
            })

    if not rows:
        print("benchmark.csv contains no experiment rows.")
        return 1

    by_config = defaultdict(list)
    sequential_by_size = {}

    for row in rows:
        config = (row["block_x"], row["block_y"])
        by_config[config].append(row)
        sequential_by_size[row["size"]] = row["sequential"]

    sizes = sorted(sequential_by_size)
    configurations = sorted(by_config)

    figure, (time_axis, kernel_axis, speed_axis) = plt.subplots(
        1, 3, figsize=(19, 5)
    )

    time_axis.plot(
        sizes,
        [sequential_by_size[size] for size in sizes],
        marker="o",
        label="Sequential CPU",
    )

    for config in configurations:
        config_rows = sorted(by_config[config], key=lambda item: item["size"])
        label = f"CUDA total {config[0]}x{config[1]}"
        time_axis.plot(
            [row["size"] for row in config_rows],
            [row["total"] for row in config_rows],
            marker="s",
            label=label,
        )

        kernel_axis.plot(
            [row["size"] for row in config_rows],
            [row["kernel"] for row in config_rows],
            marker="s",
            label=f"{config[0]}x{config[1]}",
        )

        speed_axis.plot(
            [row["size"] for row in config_rows],
            [row["speedup"] for row in config_rows],
            marker="^",
            label=f"{config[0]}x{config[1]}",
        )

    time_axis.set_title("CPU time vs CUDA total time")
    time_axis.set_xlabel("Matrix size N")
    time_axis.set_ylabel("Time (ms)")
    time_axis.grid(True)
    time_axis.legend(fontsize=8)

    kernel_axis.set_title("CUDA kernel execution time")
    kernel_axis.set_xlabel("Matrix size N")
    kernel_axis.set_ylabel("Time (ms)")
    kernel_axis.grid(True)
    kernel_axis.legend(fontsize=8)

    speed_axis.axhline(y=1.0, linestyle="--")
    speed_axis.set_title("Speedup using CUDA total time")
    speed_axis.set_xlabel("Matrix size N")
    speed_axis.set_ylabel("Speedup")
    speed_axis.grid(True)
    speed_axis.legend(fontsize=8)

    figure.tight_layout()
    output_path = root / "performance_cuda.png"
    figure.savefig(output_path, dpi=140)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
