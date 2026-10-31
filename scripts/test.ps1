# Runs CTest. Usage: scripts\test.ps1 [-R regex]
param([string]$R = '')
$ErrorActionPreference = 'Stop'
. $PSScriptRoot\env.ps1
$root = Split-Path $PSScriptRoot -Parent
$args = @('--test-dir', "$root\build", '--output-on-failure')
if ($R) { $args += @('-R', $R) }
ctest @args
exit $LASTEXITCODE
