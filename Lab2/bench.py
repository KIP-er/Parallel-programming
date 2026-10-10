import csv
import os
import re
import subprocess
import sys
from pathlib import Path


EXE_CANDIDATES = (
    "build/Release/matrix.exe",
    "build/Debug/matrix.exe",
    "build/matrix.exe",
)

SIZES = (200, 400, 800, 1200, 1600, 2000)
MAX_THREADS = os.cpu_count() or 1
THREAD_COUNTS = tuple(t for t in (1, 2, 4, 8) if t <= MAX_THREADS) or (1,)


def find_exe(root: Path) -> Path:
    for rel in EXE_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError("matrix executable not found. Build it first.")


def do_generate(root: Path, n: int) -> None:
    subprocess.run([sys.executable, "gen.py", str(n)], cwd=root, check=True)


def do_multiply(root: Path, exe: Path, threads: int) -> float:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    proc = subprocess.run(
        [str(exe)], cwd=root, input="2\n5\n",
        text=True, capture_output=True, env=env,
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

    print(f"Logical CPUs: {MAX_THREADS}; thread sweep: {THREAD_COUNTS}")

    rows = []
    for n in SIZES:
        print(f"N = {n}")
        try:
            do_generate(root, n)
        except Exception as e:
            print("Error:", e)
            return 1

        diff, ok = float("nan"), False
        for t in THREAD_COUNTS:
            try:
                ms = do_multiply(root, exe, t)
            except Exception as e:
                print("Error:", e)
                return 1
            if t == THREAD_COUNTS[0]:
                diff, ok = do_verify(root)
            rows.append((n, t, ms, diff, ok))
            print(f"  threads={t:>2}: {ms:10.3f} ms")

    with (root / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "threads", "time_ms", "max_diff", "verified"])
        for n, t, ms, d, ok in rows:
            w.writerow([n, t, f"{ms:.3f}", f"{d:.3e}", ok])

    print()
    print("| N | Threads | Time (ms) | Max diff | Verified |")
    print("|---|---|---|---|---|")
    for n, t, ms, d, ok in rows:
        print(f"| {n} | {t} | {ms:.3f} | {d:.3e} | {'OK' if ok else 'FAIL'} |")

    return 0


if __name__ == "__main__":
    sys.exit(main())