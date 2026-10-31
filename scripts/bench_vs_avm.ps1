# Speed and rate/PSNR comparison of celv2enc and AVM avmenc (M4).
# Usage: scripts\bench_vs_avm.ps1 -Input clip.y4m [-AvmBin dir] [-Frames 30] [-Clip name] [-Out results.csv]
# -AvmBin (default $env:AVM_BIN) is the folder holding avmenc.exe and avmdec.exe.
# Appends one CSV row per (encoder, qindex): clip,encoder,qindex,frames,bytes,kbps,psnr_y,seconds,
# cpu_seconds (wall time and the encoder process's CPU time; celv2's GPU work is not CPU time).
# PSNR-Y: avmdec.exe decodes each stream, ffmpeg's psnr filter compares it with the source.
# No native-command pipes: Windows PowerShell 5.1 corrupts binary data in them.
# Saved as UTF-8 with BOM: PowerShell 5.1 reads BOM-less scripts in the system code page.
param(
  [Parameter(Mandatory = $true)][Alias('Input')][string]$Source,
  [string]$AvmBin = $env:AVM_BIN,
  [int]$Frames = 30,
  [string]$Clip = '',
  [string]$Out = "$PSScriptRoot\..\docs\reports\bench.csv",
  [int[]]$Qs = @(120, 150, 180, 210),
  [string]$Work = "$env:TEMP\celv2_bench"
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $AvmBin) { throw 'pass -AvmBin or set AVM_BIN to the folder with avmenc.exe and avmdec.exe' }
$bin = $AvmBin
$celv2 = "$root\build\celv2enc.exe"
foreach ($exe in @($celv2, "$bin\avmenc.exe", "$bin\avmdec.exe")) {
  if (-not (Test-Path $exe)) { throw "not found: $exe" }
}
New-Item -ItemType Directory -Force $Work | Out-Null
if (-not $Clip) { $Clip = [IO.Path]::GetFileNameWithoutExtension($Source) }

# Size and frame rate from the y4m header.
$fs = [IO.File]::OpenRead($Source)
$hdr = New-Object byte[] 200
[void]$fs.Read($hdr, 0, 200)
$fs.Close()
$line = ([Text.Encoding]::ASCII.GetString($hdr) -split "`n")[0]
$w = [int]([regex]::Match($line, ' W(\d+)').Groups[1].Value)
$h = [int]([regex]::Match($line, ' H(\d+)').Groups[1].Value)
$fm = [regex]::Match($line, ' F(\d+):(\d+)')
$fps = [double]$fm.Groups[1].Value / [double]$fm.Groups[2].Value
$rate = "$($fm.Groups[1].Value)/$($fm.Groups[2].Value)"

# Runs a native program and returns its exit code. Its output is text only; under 'Stop',
# PowerShell 5.1 would turn stderr lines into terminating errors.
function Invoke-Native([string]$Exe, [string[]]$ArgList) {
  $ErrorActionPreference = 'Continue'
  & $Exe @ArgList 2>&1 | Out-Null
  return $LASTEXITCODE
}

function Measure-Psnr([string]$ivf) {
  $yuv = "$Work\dec.yuv"
  if (Invoke-Native "$bin\avmdec.exe" @('--i420', '-o', $yuv, $ivf)) { throw "avmdec failed on $ivf" }
  $ErrorActionPreference = 'Continue'
  # The psnr filter pairs frames by timestamp: rawvideo must get the y4m's frame rate (it
  # defaults to 25 fps; setpts=N does not help, the two inputs keep different time bases).
  $log = & ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s "${w}x${h}" -framerate $rate `
    -i $yuv -i $Source -frames:v $Frames -lavfi '[0:v][1:v]psnr' -f null - 2>&1 |
    ForEach-Object { "$_" }
  $m = [regex]::Match(($log -join "`n"), 'PSNR y:([0-9.]+)')
  if (-not $m.Success) { throw "no PSNR in ffmpeg output for $ivf" }
  return [double]$m.Groups[1].Value
}

if (-not (Test-Path $Out)) {
  'clip,encoder,qindex,frames,bytes,kbps,psnr_y,seconds,cpu_seconds' | Out-File -Encoding ascii $Out
}
foreach ($q in $Qs) {
  foreach ($enc in @('celv2', 'avm')) {
    $ivf = "$Work\${Clip}_${enc}_q$q.ivf"
    if ($enc -eq 'celv2') {
      $exe = $celv2
      # quoted: Start-Process joins the arguments with spaces as they are
      $al = @('-i', "`"$Source`"", '-o', "`"$ivf`"", '--qindex', "$q", '--frames', "$Frames")
    } else {
      $exe = "$bin\avmenc.exe"
      $al = @('--ivf', '--good', '--cpu-used=6', '--end-usage=q', "--qp=$q", '--threads=1',
              '--lag-in-frames=0', "--limit=$Frames", '-o', "`"$ivf`"", "`"$Source`"")
    }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $exe -ArgumentList $al -NoNewWindow -Wait -PassThru `
      -RedirectStandardError "$Work\stderr.txt" -RedirectStandardOutput "$Work\stdout.txt"
    $sw.Stop()
    if ($p.ExitCode) { throw "$enc failed at q$q (exit $($p.ExitCode))" }
    $cpu = $p.TotalProcessorTime.TotalSeconds
    $bytes = (Get-Item $ivf).Length
    $kbps = $bytes * 8 / ($Frames / $fps) / 1000
    $psnr = Measure-Psnr $ivf
    $row = '{0},{1},{2},{3},{4},{5:F1},{6:F3},{7:F2},{8:F2}' -f $Clip, $enc, $q, $Frames, $bytes, $kbps, $psnr, $sw.Elapsed.TotalSeconds, $cpu
    $row | Out-File -Encoding ascii -Append $Out
    Write-Host $row
  }
}
