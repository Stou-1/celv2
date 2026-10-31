# Builds the GUI assemblies and the test exe into gui\bin. Exit code 1 on any csc failure.
$ErrorActionPreference = 'Stop'
$gui = $PSScriptRoot
$bin = Join-Path $gui 'bin'
New-Item -ItemType Directory -Force $bin | Out-Null
$csc = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\Roslyn\csc.exe'
$fx = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319'

function Build-Assembly($target, $out, $srcDir, $deps) {
    $refs = @('System', 'System.Core', 'System.Runtime.Serialization', 'System.Xml', 'System.Windows.Forms', 'System.Drawing') |
        ForEach-Object { "-reference:$fx\$_.dll" }
    $refs += $deps | ForEach-Object { "-reference:$bin\$_" }
    $srcs = @(Get-ChildItem -Recurse -Filter *.cs (Join-Path $gui $srcDir) | ForEach-Object { $_.FullName }) # @() keeps a single file an array
    & $csc -nologo -noconfig -langversion:7.3 -codepage:65001 -platform:anycpu "-target:$target" "-out:$bin\$out" @refs @srcs
    if ($LASTEXITCODE -ne 0) { Write-Host "csc failed for $out"; exit 1 }
}

Build-Assembly 'library' 'CelV2.Settings.dll' 'CelV2.Settings' @()
Build-Assembly 'library' 'CelV2.Runner.dll' 'CelV2.Runner' @('CelV2.Settings.dll')
Build-Assembly 'library' 'CelV2.Ui.dll' 'CelV2.Ui' @('CelV2.Settings.dll', 'CelV2.Runner.dll')
Build-Assembly 'winexe' 'CelV2Gui.exe' 'CelV2Gui' @('CelV2.Settings.dll', 'CelV2.Runner.dll', 'CelV2.Ui.dll')
Build-Assembly 'exe' 'CelV2.Tests.exe' 'tests' @('CelV2.Settings.dll', 'CelV2.Runner.dll', 'CelV2.Ui.dll')
Copy-Item -Recurse -Force (Join-Path $gui 'tests\data') $bin
Copy-Item -Force (Join-Path $gui 'CelV2Gui\App.config') (Join-Path $bin 'CelV2Gui.exe.config')

# So that gui/bin/CelV2Gui.exe runs on a dev PC: the engine from the repo build and av2_mux from AV2MUX_DIR (the
# av2-tools build with third_party\av2-tools\av2-tools-celv2.patch), else from AVM_BIN.
# The release zip (Task 6) assembles its own folder; these copies stay in the gitignored bin folder.
$engine = Join-Path $gui '..\build\celv2enc.exe'
if (Test-Path $engine) { Copy-Item -Force $engine $bin }
$muxDir = if ($env:AV2MUX_DIR) { $env:AV2MUX_DIR } else { $env:AVM_BIN }
# The copy gets the UTF-8 active code page (the original is never changed); the tests use this copy.
if ($muxDir -and (Test-Path (Join-Path $muxDir 'av2_mux.exe'))) {
    Copy-Item -Force (Join-Path $muxDir 'av2_mux.exe') $bin
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $gui 'utf8-manifest.ps1') -Exe (Join-Path $bin 'av2_mux.exe')
    if ($LASTEXITCODE -ne 0) { Write-Host 'utf8-manifest failed for av2_mux.exe'; exit 1 }
}
# fdkaac (nu774/fdkaac built with libfdk-aac) for 「HE-AACv2 に変換する」, from FDKAAC_DIR; the zip does not ship it.
if ($env:FDKAAC_DIR -and (Test-Path (Join-Path $env:FDKAAC_DIR 'fdkaac.exe'))) { Copy-Item -Force (Join-Path $env:FDKAAC_DIR 'fdkaac.exe') $bin }
