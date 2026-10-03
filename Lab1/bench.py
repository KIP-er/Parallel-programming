import csv
import re
import subprocess
import sys
from pathlib import Path


EXE_CANDIDATES = (
    "build/Debug/matrix.exe",
    "build/Release/matrix.exe",
    "build/matrix.exe",
)


def find_exe(root: Path) -> Path:
    for rel in EXE_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError("matrix executable not found. Build it first.")


def do_generate(root: Path, n: int) -> None:
    subprocess.run([sys.executable, "gen.py", str(n)], cwd=root, check=True)


def do_multiply(root: Path, exe: Path) -> float:
    proc = subprocess.run(
        [str(exe)], cwd=root, input="2\n5\n",
        text=True, capture_output=True,
    )
    m = re.search(r"Multiplication time: ([0-9.]+)", proc.stdout)
    if not m:
        raise RuntimeError(f"cannot parse time.\nstdout:\n{proc.stdout}")
    return float(m.group(1))


def do_verify(root: Path) -> tuple[float, bool]:
    proc = subprocess.run(
        [sys.executable, "verify.py"], cwd=root,
        text=True, capture_output=True,
    )
    m = re.search(r"Maximum absolute difference: ([0-9.eE+-]+)", proc.stdout)
    diff = float(m.group(1)) if m else float("nan")
    return diff, "RESULT: OK" in proc.stdout


def main() -> int:
    try:
        root = Path(__file__).resolve().parent
        exe = find_exe(root)
    except Exception as e:
        print("Error:", e)
        return 1

    rows = []
    for n in (200, 400, 800, 1200, 1600, 2000):
        print(f"N = {n}")
        try:
            do_generate(root, n)
            t = do_multiply(root, exe)
            d, ok = do_verify(root)
        except Exception as e:
            print("Error:", e)
            return 1
        rows.append((n, t, d, ok))

    with (root / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "time_ms", "max_diff", "verified"])
        for n, t, d, ok in rows:
            w.writerow([n, f"{t:.3f}", f"{d:.3e}", ok])

    print()
    print("| N | Time (ms) | Max diff | Verified |")
    print("|---|---|---|---|")
    for n, t, d, ok in rows:
        print(f"| {n} | {t:.3f} | {d:.3e} | {'OK' if ok else 'FAIL'} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())