import csv
import os
import re
import subprocess
import sys
from pathlib import Path


SIZES = (200, 400, 800, 1200, 1600, 2000)
PROCESSES = (1, 2, 4, 8)
EXE_CANDIDATES = (
    'build/Debug/matrix_mpi.exe',
    'build/Release/matrix_mpi.exe',
    'build/matrix_mpi.exe',
    'build/matrix_mpi',
)


def find_exe(root: Path) -> Path:
    for rel in EXE_CANDIDATES:
        path = root / rel
        if path.is_file():
            return path
    raise FileNotFoundError('matrix_mpi executable not found. Build it first.')


def run_command(command, root: Path):
    env = os.environ.copy()
    env.setdefault('OMPI_ALLOW_RUN_AS_ROOT', '1')
    env.setdefault('OMPI_ALLOW_RUN_AS_ROOT_CONFIRM', '1')
    return subprocess.run(command, cwd=root, text=True, capture_output=True, env=env)


def main() -> int:
    root = Path(__file__).resolve().parent
    try:
        exe = find_exe(root)
    except Exception as error:
        print('Error:', error)
        return 1

    process_values = tuple(int(x) for x in sys.argv[1].split(',')) if len(sys.argv) > 1 else PROCESSES
    rows = []

    for n in SIZES:
        print(f'\nN = {n}')
        gen = run_command([sys.executable, 'gen.py', str(n)], root)
        if gen.returncode != 0:
            print(gen.stdout, gen.stderr)
            return 1

        for process_count in process_values:
            print(f'  MPI processes = {process_count}')
            try:
                proc = run_command(['mpiexec', '-n', str(process_count), str(exe)], root)
            except FileNotFoundError:
                print('mpiexec not found. Install an MPI implementation and add mpiexec to PATH.')
                return 1
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                print('MPI run failed. Check mpiexec and MPI installation.')
                return 1

            match = re.search(r'Multiplication time: ([0-9.]+) ms', proc.stdout)
            if not match:
                print(proc.stdout)
                print('Cannot read execution time.')
                return 1
            time_ms = float(match.group(1))

            verify = run_command([sys.executable, 'verify.py'], root)
            diff_match = re.search(r'Maximum absolute difference: ([0-9.eE+-]+)', verify.stdout)
            diff = float(diff_match.group(1)) if diff_match else float('nan')
            verified = verify.returncode == 0 and 'RESULT: OK' in verify.stdout
            rows.append({
                'N': n,
                'processes': process_count,
                'time_ms': time_ms,
                'max_diff': diff,
                'verified': verified,
            })
            print(f'    {time_ms:.3f} ms, max diff = {diff:.3e}, {"OK" if verified else "FAIL"}')
            if not verified:
                print(verify.stdout)
                return 1

    for row in rows:
        baseline = next((x['time_ms'] for x in rows if x['N'] == row['N'] and x['processes'] == 1), None)
        row['speedup'] = baseline / row['time_ms'] if baseline and row['time_ms'] else 0.0
        row['efficiency'] = row['speedup'] / row['processes'] if row['processes'] else 0.0

    with (root / 'benchmark.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['N', 'processes', 'time_ms', 'speedup', 'efficiency', 'max_diff', 'verified'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f'{row[key]:.3f}' if key == 'time_ms' else
                                   f'{row[key]:.3f}' if key in ('speedup', 'efficiency') else
                                   f'{row[key]:.3e}' if key == 'max_diff' else row[key])
                             for key in writer.fieldnames})

    print('\n| N | MPI processes | Time (ms) | Speedup | Efficiency | Max diff | Verified |')
    print('|---:|---:|---:|---:|---:|---:|:---:|')
    for row in rows:
        print(f"| {row['N']} | {row['processes']} | {row['time_ms']:.3f} | {row['speedup']:.2f} | {row['efficiency']:.2f} | {row['max_diff']:.3e} | {'OK' if row['verified'] else 'FAIL'} |")

    plot = run_command([sys.executable, 'plot_perf.py'], root)
    print(plot.stdout)
    if plot.returncode != 0:
        print(plot.stderr)
        return 1
    update_readme(root, rows)
    return 0


def update_readme(root: Path, rows) -> None:
    readme = root / 'README.md'
    content = readme.read_text(encoding='utf-8')
    start = '<!-- BENCHMARK_RESULTS_START -->'
    end = '<!-- BENCHMARK_RESULTS_END -->'
    table = [
        '| N | Процессы MPI | Время, мс | Ускорение | Эффективность | Max diff | Проверка |',
        '|---:|---:|---:|---:|---:|---:|:---:|',
    ]
    for row in rows:
        table.append(f"| {row['N']} | {row['processes']} | {row['time_ms']:.3f} | {row['speedup']:.2f} | {row['efficiency']:.2f} | {row['max_diff']:.3e} | {'OK' if row['verified'] else 'FAIL'} |")
    table.extend(['', '![Графики эксперимента](performance.png)'])
    replacement = start + '\n' + '\n'.join(table) + '\n' + end
    left = content.find(start)
    right = content.find(end)
    if left >= 0 and right >= left:
        content = content[:left] + replacement + content[right + len(end):]
        readme.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
