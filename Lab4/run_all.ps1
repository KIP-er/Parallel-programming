$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    $sizes = 200, 400, 800, 1200, 1600, 2000
    $buildDirectory = Join-Path $root "build"
    $preferredPaths = @(
        (Join-Path $buildDirectory "LabCUDA.exe"),
        (Join-Path $buildDirectory "Release\LabCUDA.exe")
    )

    $executablePath = $preferredPaths |
        Where-Object { Test-Path $_ -PathType Leaf } |
        Select-Object -First 1

    if ($null -eq $executablePath) {
        $executable = Get-ChildItem -Path $buildDirectory `
            -Filter "LabCUDA.exe" -File -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if ($null -ne $executable) {
            $executablePath = $executable.FullName
        }
    }

    if ($null -eq $executablePath) {
        throw "LabCUDA.exe not found. Configure and build the project first."
    }

    Remove-Item "benchmark.csv" -ErrorAction SilentlyContinue

    foreach ($size in $sizes) {
        Write-Host "`n===== Matrix size: $size x $size =====" -ForegroundColor Cyan

        py .\gen.py --size $size
        if ($LASTEXITCODE -ne 0) {
            throw "Matrix generation failed for size $size."
        }

        & $executablePath
        if ($LASTEXITCODE -ne 0) {
            throw "CUDA experiment failed for size $size."
        }
    }

    Write-Host "`nAll CUDA experiments completed." -ForegroundColor Green
    Write-Host "Benchmark: $root\benchmark.csv"
}
finally {
    Pop-Location
}
