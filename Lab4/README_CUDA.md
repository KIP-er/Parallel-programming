# Лабораторная работа по умножению матриц на CUDA

**Выполнили:** Эльменкин Виктор Хавьерович и Дарьин Иван Борисович

**Группа:** 6213-100503D

**Программное и аппаратное обеспечение**
- **RAM**: 16 GB DDR4
- **CPU**: Intel Core i5-12400F
- **GPU**: NVIDIA GeForce RTX 3080
- **OS**: Windows 11 Pro

## Файлы

- `main.cpp` — эталонное умножение на CPU, эксперименты с конфигурациями CUDA, проверка результатов и вывод результатов бенчмарка.
- `matrix_core_n_files.cppm` — хранение матриц, ввод/вывод и логирование в CSV.
- `matrix_ops.cppm` — неизмененные последовательные реализации на CPU и реализации с `std::thread` из Lab1.
- `matrix_cuda.hpp` — чистый C-интерфейс между единицами трансляции C++23 и CUDA.
- `matrix_cuda.cu` — ядро CUDA, передача памяти на устройство и замер времени событий CUDA.
- `CMakeLists.txt` — модули C++23 плюс CUDA C++20, ориентированные на RTX 3080 (`sm_86`).
- `gen.py` — создает две матрицы заданного размера.
- `check.py` — проверяет `result.txt` с помощью NumPy.
- `perf.py` — строит графики времени CPU/CUDA, времени работы ядра и ускорения.
- `run_all.ps1` — эксперименты для размеров 200, 400, 800, 1200, 1600 и 2000.

## Сборка

Выполните эти команды из этой папки в **Developer PowerShell for Visual Studio 2026**:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
Если Ninja не установлен, используйте вместо него генератор Visual Studio:
```
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```
Один тест
```
py .\gen.py --size 200
.\build\LabCUDA.exe
py .\check.py
```
Для генератора Visual Studio исполняемый файл обычно находится в build\Release\LabCUDA.exe.

# Все эксперименты
После сборки запустите:
```
.\run_all.ps1
py .\perf.py
```
