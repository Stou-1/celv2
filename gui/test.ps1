# Builds the GUI and runs the test exe. Exit code 0 only if everything passes.
& (Join-Path $PSScriptRoot 'build.ps1')
if ($LASTEXITCODE -ne 0) { exit 1 }
& (Join-Path $PSScriptRoot 'bin\CelV2.Tests.exe')
exit $LASTEXITCODE
