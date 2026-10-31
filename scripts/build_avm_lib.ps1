# Builds AVM v1.0.0 as a plain (non-LTO) static library that MSVC/nvcc can link against.
$ErrorActionPreference = 'Stop'
. $PSScriptRoot\env.ps1
$nasm = (Get-Command nasm.exe -ErrorAction Stop).Source  # env.ps1 puts NASM on the PATH
# __FILE__ in AVM and its TFLite dependencies says avm/... instead of the full path (and the user name in it).
$map = '/clang:-ffile-prefix-map={0}=avm' -f $AVM_ROOT.Replace('\', '/')
cmake -S $AVM_ROOT -B $AVM_LIB_BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl `
  "-DCMAKE_C_FLAGS=/DWIN32 /D_WINDOWS $map" "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc $map" `
  -DENABLE_TESTS=0 -DENABLE_DOCS=0 "-DCMAKE_ASM_NASM_COMPILER=$nasm"
if ($LASTEXITCODE) { throw 'AVM configure failed' }
# avmenc pulls in every third-party library (TFLite, absl, ...) that avm.lib's objects reference.
cmake --build $AVM_LIB_BUILD --target avmenc --parallel $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE) { throw 'AVM build failed' }
Get-Item "$AVM_LIB_BUILD\avm.lib" | Select-Object FullName, Length
