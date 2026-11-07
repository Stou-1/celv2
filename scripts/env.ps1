# Imports the MSVC x64 toolchain environment plus NASM, LLVM and Ninja into the current PowerShell.
# Usage: . $PSScriptRoot\env.ps1
# Visual Studio: $env:CELV2_VS, else the newest install with the C++ x64 tools (vswhere).
$env:PATH = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"
$vs = if ($env:CELV2_VS) { $env:CELV2_VS } else {
  vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vs) { throw 'Visual Studio with the C++ x64 tools not found (set CELV2_VS to its folder)' }
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
}
# Git's usr\bin provides patch.exe, which AVM's TFLite dependency fetch needs.
# NASM: $env:CELV2_NASM, else %LOCALAPPDATA%\bin\NASM (or already on the PATH).
$nasm = if ($env:CELV2_NASM) { $env:CELV2_NASM } else { "$env:LOCALAPPDATA\bin\NASM" }
$env:PATH = "$nasm;$env:ProgramFiles\LLVM\bin;$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:ProgramFiles\Git\usr\bin;$env:PATH"
# AVM sources contain UTF-8 comments; without this, cl/clang-cl on a Japanese (CP932) system can drop code lines.
$env:CL = '/utf-8'
# cl prints its include notes in the console's code page, and ninja's msvc_deps_prefix was recorded
# in the system's: on a UTF-8 console a Japanese Windows's notes match nothing, ninja records no
# header dependencies, and a header edit leaves stale objects (mismatched structs, crashes).
$acp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Nls\CodePage').ACP
if ([Console]::OutputEncoding.CodePage -ne [int]$acp) {
  chcp $acp | Out-Null
  [Console]::OutputEncoding = [Text.Encoding]::GetEncoding([int]$acp)
}
# AVM v1.0.0 source tree: $env:AVM_ROOT, else ~\avm. Exported so CMake sees it too.
$global:AVM_ROOT = if ($env:AVM_ROOT) { $env:AVM_ROOT } else { "$HOME\avm" }
$global:AVM_LIB_BUILD = "$AVM_ROOT\_build_lib"
$env:AVM_ROOT = $AVM_ROOT
