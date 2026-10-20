$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$mamba = Join-Path $PSScriptRoot 'work\mamba-stable\Library\bin\micromamba.exe'
if (-not (Test-Path -LiteralPath $mamba)) {
    New-Item -ItemType Directory -Path 'work\mamba-stable' -Force | Out-Null
    Invoke-WebRequest -Uri 'https://micro.mamba.pm/api/micromamba/win-64/2.0.5' -OutFile 'work\micromamba-stable.tar.bz2'
    tar -xjf 'work\micromamba-stable.tar.bz2' -C 'work\mamba-stable' Library/bin/micromamba.exe
    if ($LASTEXITCODE -ne 0) { throw 'micromamba arsivi acilamadi.' }
}
if (-not (Test-Path -LiteralPath '.runtime\python.exe')) {
    & $mamba create -y -r (Join-Path $PSScriptRoot 'work\mamba') -p (Join-Path $PSScriptRoot '.runtime') -c conda-forge 'python=3.12' 'pybullet=3.25' pip
    if ($LASTEXITCODE -ne 0) { throw 'Windows fizik ortami kurulamadi.' }
}
$env:PATH = "$PSScriptRoot\.runtime;$PSScriptRoot\.runtime\Library\bin;$PSScriptRoot\.runtime\Scripts;$env:PATH"
& '.\.runtime\python.exe' -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch kurulumu basarisiz.' }
& '.\.runtime\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python bagimliliklari kurulamadi.' }
# The PyPI NumPy/OpenBLAS wheel avoids MKL/Qt/PyTorch OpenMP DLL conflicts.
& '.\.runtime\python.exe' -m pip install --force-reinstall --no-deps --only-binary=:all: numpy==2.5.4
if ($LASTEXITCODE -ne 0) { throw 'NumPy/OpenBLAS kurulumu basarisiz.' }
& '.\.runtime\python.exe' -c 'import numpy,pybullet,torch,gymnasium; from PySide6 import QtCore'
if ($LASTEXITCODE -ne 0) { throw 'Bagimlilik kontrolu basarisiz.' }
& '.\.runtime\python.exe' -m teslafliprl.dependencies
Write-Host 'Kurulum tamamlandi. Baslat: .\run.cmd watch.py --episodes 5'
