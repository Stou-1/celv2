# Merges src\utf8.manifest (activeCodePage UTF-8) into the manifest of a COPY of av2_mux.exe, so its ANSI
# main(argc, argv) gets UTF-8 paths on Windows 10 1903+. Checks the merged manifest and that the exe still runs.
# Exit code 1 on any failure.
param([Parameter(Mandatory = $true)][string]$Exe)
$ErrorActionPreference = 'Stop'
function Fail($m) { Write-Host "utf8-manifest: $m"; exit 1 }

$kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
$mt = Get-ChildItem -Path $kits -Filter mt.exe -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -eq 'x64' } | Sort-Object { $_.Directory.Parent.Name -as [version] } | Select-Object -Last 1
if (-not $mt) { Fail "mt.exe not found under $kits (install the Windows SDK)" }
$manifest = Join-Path $PSScriptRoot '..\src\utf8.manifest'
if (-not (Test-Path -LiteralPath $Exe)) { Fail "missing $Exe" }

& $mt.FullName -nologo "-inputresource:$Exe;#1" -manifest $manifest "-outputresource:$Exe;#1"
if ($LASTEXITCODE -ne 0) { Fail "mt.exe merge failed for $Exe" }
$check = Join-Path $env:TEMP ('celv2-manifest-' + [guid]::NewGuid().ToString('N') + '.xml')
try {
    & $mt.FullName -nologo "-inputresource:$Exe;#1" "-out:$check"
    if ($LASTEXITCODE -ne 0) { Fail "mt.exe could not read back the manifest of $Exe" }
    $text = Get-Content -Raw -LiteralPath $check
} finally { Remove-Item -Force -ErrorAction SilentlyContinue $check }
if ($text -notmatch '<activeCodePage[^>]*>UTF-8</activeCodePage>' -or $text -notmatch 'requestedExecutionLevel') {
    Fail "merged manifest of $Exe lacks activeCodePage UTF-8 or the original trustInfo"
}
$ErrorActionPreference = 'Continue' # in 5.1 a native exe's stderr line would otherwise stop the script
& $Exe --help *> $null
if ($LASTEXITCODE -ne 0) { Fail "$Exe --help exited $LASTEXITCODE after the merge" }
