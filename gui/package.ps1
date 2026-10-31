# Builds gui\bin, then assembles dist\celv2-gui-<date>.zip (no tests, no test data).
# Needs build\celv2enc.exe and av2_mux.exe in -Av2MuxDir (default $env:AV2MUX_DIR), built from av2-tools with
# third_party\av2-tools\av2-tools-celv2.patch. This script only copies build\celv2enc.exe: rebuild it first
# (scripts\build.ps1). The SHA256 and time of both exes are printed at the end. Exit code 1 on any missing file.
param(
    [string]$Av2MuxDir = $env:AV2MUX_DIR,
    [string]$CrtDir = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Redist\MSVC\14.51.36231\x64\Microsoft.VC145.CRT'
)
$ErrorActionPreference = 'Stop'
$gui = $PSScriptRoot
$repo = Split-Path $gui
function Fail($m) { Write-Host "package: $m"; exit 1 }

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $gui 'build.ps1')
if ($LASTEXITCODE -ne 0) { Fail 'gui\build.ps1 failed' }

if (-not $Av2MuxDir) { Fail 'set -Av2MuxDir or $env:AV2MUX_DIR to the folder that holds av2_mux.exe' }
# The unpatched av2_mux links libisomediafile, whose license only covers MPEG-4 conformance products;
# its log message names the libisomedia call, the patched one has no such string.
$muxExe = Join-Path $Av2MuxDir 'av2_mux.exe'
if ((Test-Path -LiteralPath $muxExe) -and [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($muxExe)).Contains('MP4WriteMovieToFile')) {
    Fail "$muxExe uses libisomediafile; build av2_mux with third_party\av2-tools\av2-tools-celv2.patch"
}
# Needed by dumpbin /dependents of celv2enc.exe and av2_mux.exe; the api-ms-win-crt-* ones ship with Windows 10+.
$crt = 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'
$tp = Join-Path $gui 'third_party\av2-tools'
# source -> path inside the zip
$files = [ordered]@{}
foreach ($n in 'CelV2Gui.exe', 'CelV2Gui.exe.config', 'CelV2.Settings.dll', 'CelV2.Runner.dll', 'CelV2.Ui.dll') { $files[(Join-Path $gui "bin\$n")] = $n }
$files[(Join-Path $repo 'build\celv2enc.exe')] = 'celv2enc.exe'
$files[(Join-Path $Av2MuxDir 'av2_mux.exe')] = 'av2_mux.exe'
foreach ($n in $crt) { $files[(Join-Path $CrtDir $n)] = $n }
$files[(Join-Path $repo 'LICENSE')] = 'LICENSE'
$files[(Join-Path $repo 'third_party\avm\LICENSE')] = 'third_party\avm\LICENSE'
$files[(Join-Path $repo 'third_party\avm\PATENTS')] = 'third_party\avm\PATENTS'
foreach ($n in 'LICENSE', 'PATENTS', 'README.txt', 'av2-tools-celv2.patch') { $files[(Join-Path $tp $n)] = "third_party\av2-tools\$n" }
$files[(Join-Path $tp 'THIRD_PARTY_NOTICES.txt')] = 'third_party\av2-tools\THIRD_PARTY_NOTICES.txt'
$files[(Join-Path $gui 'third_party\celv2enc\NOTICE.txt')] = 'third_party\celv2enc\NOTICE.txt'
$files[(Join-Path $gui 'README_ja.txt')] = 'README_ja.txt'
foreach ($src in $files.Keys) { if (-not (Test-Path -LiteralPath $src)) { Fail "missing $src" } }

$stage = Join-Path $repo 'dist\stage'
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
foreach ($src in $files.Keys) {
    $dst = Join-Path $stage $files[$src]
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item -LiteralPath $src $dst
}
# the staged copy of av2_mux gets the UTF-8 active code page; the original in -Av2MuxDir is never changed
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $gui 'utf8-manifest.ps1') -Exe (Join-Path $stage 'av2_mux.exe')
if ($LASTEXITCODE -ne 0) { Fail 'utf8-manifest failed for av2_mux.exe' }
$zip = Join-Path $repo ("dist\celv2-gui-{0}.zip" -f (Get-Date -Format 'yyyyMMdd'))
if (Test-Path $zip) { Remove-Item -Force $zip }
# ZipFile.CreateFromDirectory in .NET Framework 4.x writes backslash separators; build entries by hand with '/'.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$za = [System.IO.Compression.ZipFile]::Open($zip, 'Create')
try {
    foreach ($f in Get-ChildItem -Recurse -File $stage) {
        $rel = $f.FullName.Substring($stage.Length + 1).Replace('\', '/')
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($za, $f.FullName, $rel)
    }
} finally { $za.Dispose() }
foreach ($n in 'celv2enc.exe', 'av2_mux.exe') {
    $f = Get-Item -LiteralPath (Join-Path $stage $n) # Copy-Item keeps LastWriteTime
    Write-Host ("package: {0}  {1}  SHA256 {2}" -f $n, $f.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'), (Get-FileHash -LiteralPath $f.FullName).Hash)
}
Remove-Item -Recurse -Force $stage
Write-Host "package: $zip"
