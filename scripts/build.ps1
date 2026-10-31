# Configures (first time) and builds celv2 into .\build with MSVC cl + nvcc.
$ErrorActionPreference = 'Stop'
. $PSScriptRoot\env.ps1
$root = Split-Path $PSScriptRoot -Parent
cmake -S $root -B "$root\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
if ($LASTEXITCODE) { throw 'configure failed' }
cmake --build "$root\build" --parallel $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE) { throw 'build failed' }
