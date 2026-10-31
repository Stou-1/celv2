# AVM tool ablation: encodes with avmenc with one coding tool switched off per variant.
# Usage: scripts\ablate_avm.ps1 -Input clip.y4m [-AvmBin dir] [-Frames 10] [-Clip name] [-Out ablation.csv]
# Starts one process per (variant, qindex) in parallel, then writes the CSV in bench_vs_avm.ps1's
# format with the variant name in the encoder column (for scripts/bdrate.py --anchor base).
# Same PSNR method as bench_vs_avm.ps1. Saved as UTF-8 with BOM (PowerShell 5.1 reads BOM-less scripts in the system code page).
param(
  [Parameter(Mandatory = $true)][Alias('Input')][string]$Source,
  [int]$Frames = 10,
  [string]$Clip = '',
  [string]$Out = "$PSScriptRoot\..\docs\reports\ablation.csv",
  [int[]]$Qs = @(120, 150, 180, 210),
  [string]$Work = "$env:TEMP\celv2_ablation",
  [string]$CelV2 = "$PSScriptRoot\..\build\celv2enc.exe",
  [string]$AvmBin = $env:AVM_BIN,
  [string]$Avmenc = '',
  [string[]]$Only = @(),
  # Needs an avmenc built with the AVM_ABL_INTMV patch (see the tool-ablation report): 1 = screen
  # content tools on in inter frames (control), 2 = also integer MVs. Suffixes the encoder name.
  [int]$IntMv = 0,
  [string]$Variant = '',
  [int]$Q = 0
)
$ErrorActionPreference = 'Stop'
if (-not $AvmBin) { throw 'pass -AvmBin or set AVM_BIN to the folder with avmenc.exe and avmdec.exe' }
$bin = $AvmBin
if (-not $Avmenc) { $Avmenc = "$bin\avmenc.exe" }
$suffix = @('', '_sct', '_intmv')[$IntMv]
if (-not $Clip) { $Clip = [IO.Path]::GetFileNameWithoutExtension($Source) }
$Source = (Resolve-Path $Source).Path

$off = [ordered]@{
  no_deblock = @('--enable-deblocking=0')
  no_cdef    = @('--enable-cdef=0')
  no_lr      = @('--enable-restoration=0')
  no_ccso    = @('--enable-ccso=0')
  no_gdf     = @('--enable-gdf=0')
  dct_only   = @('--use-intra-dct-only=1', '--use-inter-dct-only=1')
  # Rect partitions stay on: --enable-rect-partitions=0 caps AVM's own recon at ~38 dB (key frame,
  # q120, encoder-reported PSNR) whatever the pruning settings; celv2 has HORZ/VERT anyway.
  no_ext     = @('--enable-ext-partitions=0', '--enable-uneven-4way-partitions=0')
  one_ref    = @('--max-reference-frames=1')
}
$all = @($off.Values | ForEach-Object { $_ })
# Tools celv2 also lacks beyond all_off, in groups: intra is DC/V/H/Smooth only, chroma DCT only.
$groups = [ordered]@{
  quant_off = @('--enable-trellis-quant=0', '--enable-tcq=0')
  sectx_off = @('--enable-ist=0', '--enable-inter-ist=0', '--enable-inter-ddt=0', '--enable-cctx=0',
    '--enable-fsc=0', '--enable-chroma-dctonly=1')
  intra_off = @('--enable-intrabc=0', '--enable-palette=0', '--enable-cfl-intra=0', '--enable-mrls=0',
    '--enable-ibp=0', '--enable-intra-dip=0', '--enable-mhccp=0', '--enable-paeth-intra=0',
    '--enable-angle-delta=0', '--enable-intra-edge-filter=0', '--enable-sdp=0', '--enable-extended-sdp=0')
  inter_off = @('--enable-warped-motion=0', '--enable-warp-causal=0', '--enable-warp-delta=0',
    '--enable-refinemv=0', '--enable-opfl-refine=0', '--enable-tip=0', '--enable-bawp=0', '--enable-skip-mode=0')
}
$variants = [ordered]@{ base = @() }
foreach ($k in $off.Keys) { $variants[$k] = $off[$k] }
$variants['lf_off'] = $off['no_deblock'] + $off['no_cdef'] + $off['no_lr'] + $off['no_ccso'] + $off['no_gdf']
foreach ($k in $groups.Keys) { $variants[$k] = $groups[$k] }
$variants['all_off'] = $all
$variants['celv2like'] = $all + @($groups.Values | ForEach-Object { $_ })
if (Test-Path $CelV2) { $variants['celv2'] = $null }  # celv2enc itself, for reference
$Only = @($Only | ForEach-Object { $_ -split ',' })  # -File passes "a,b" as one string
if ($Only) { foreach ($k in @($variants.Keys)) { if ($Only -notcontains $k) { $variants.Remove($k) } } }

$rows = "$Work\rows"
if (-not $Variant) {
  New-Item -ItemType Directory -Force $rows | Out-Null
  Remove-Item "$rows\*" -ErrorAction SilentlyContinue
  $procs = foreach ($v in $variants.Keys) {
    foreach ($q in $Qs) {
      Start-Process powershell -PassThru -NoNewWindow -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        "`"$PSCommandPath`"", '-Input', "`"$Source`"", '-Frames', $Frames, '-Clip', $Clip, '-Work',
        "`"$Work`"", '-CelV2', "`"$CelV2`"", '-Avmenc', "`"$Avmenc`"", '-IntMv', $IntMv, '-Variant', $v, '-Q', $q) -RedirectStandardError "$rows\${v}_q$q.err"
    }
  }
  $procs | Wait-Process
  'clip,encoder,qindex,frames,bytes,kbps,psnr_y,seconds,cpu_seconds' | Out-File -Encoding ascii $Out
  foreach ($v in $variants.Keys) {
    foreach ($q in $Qs) {
      $f = "$rows\${v}_q$q.csv"
      if (Test-Path $f) { Get-Content $f | Out-File -Encoding ascii -Append $Out }
      else { Write-Warning "missing $v q$q, see $rows\${v}_q$q.err" }
    }
  }
  Get-Content $Out
  return
}

# One (variant, qindex) job.
# shortcut: with 64 jobs on 48 threads, 2 of 64 in-run PSNRs came out ~3.4 dB low (cause unknown;
# the same .ivf re-measured alone was right). Re-measure the kept .ivf files before trusting a row.
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

$tag = "${Clip}_${Variant}${suffix}_q$Q"
if ($IntMv) { $env:AVM_ABL_INTMV = "$IntMv" }
$ivf = "$Work\$tag.ivf"
if ($Variant -eq 'celv2') {
  $exe = $CelV2
  $al = @('-i', "`"$Source`"", '-o', "`"$ivf`"", '--qindex', "$Q", '--frames', "$Frames")
} else {
  $exe = $Avmenc
  $al = @('--ivf', '--good', '--cpu-used=6', '--end-usage=q', "--qp=$Q", '--threads=1',
          '--lag-in-frames=0', "--limit=$Frames") + $variants[$Variant] + @('-o', "`"$ivf`"", "`"$Source`"")
}
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $exe -ArgumentList $al -NoNewWindow -Wait -PassThru `
  -RedirectStandardError "$Work\$tag.stderr.txt" -RedirectStandardOutput "$Work\$tag.stdout.txt"
$sw.Stop()
if ($p.ExitCode) { throw "$Variant failed at q$Q (exit $($p.ExitCode))" }
$cpu = $p.TotalProcessorTime.TotalSeconds
$bytes = (Get-Item $ivf).Length
$kbps = $bytes * 8 / ($Frames / $fps) / 1000

$yuv = "$Work\$tag.yuv"
$ErrorActionPreference = 'Continue'
& "$bin\avmdec.exe" --i420 -o $yuv $ivf 2>&1 | Out-Null
if ($LASTEXITCODE) { throw "avmdec failed on $ivf" }
# rawvideo needs the y4m's frame rate, or the psnr filter pairs the wrong frames (see bench_vs_avm.ps1).
$log = & ffmpeg -hide_banner -f rawvideo -pix_fmt yuv420p -s "${w}x${h}" -framerate $rate `
  -i $yuv -i $Source -frames:v $Frames -lavfi '[0:v][1:v]psnr' -f null - 2>&1 | ForEach-Object { "$_" }
$m = [regex]::Match(($log -join "`n"), 'PSNR y:([0-9.]+)')
if (-not $m.Success) { throw "no PSNR in ffmpeg output for $ivf" }
Remove-Item $yuv
'{0},{1},{2},{3},{4},{5:F1},{6:F3},{7:F2},{8:F2}' -f $Clip, "$Variant$suffix", $Q, $Frames, $bytes, $kbps,
  [double]$m.Groups[1].Value, $sw.Elapsed.TotalSeconds, $cpu | Out-File -Encoding ascii "$rows\${Variant}_q$Q.csv"
