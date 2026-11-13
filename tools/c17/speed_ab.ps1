# C17 step 1's speed A/B: two celv2enc builds alternating on the ledger's runs (docs/design/speed-budget.md), CPU a
# frame and fps from CELV2_PROFILE=1. S1, S2 and 吉村 3 rounds, S3 and S4 one.
# usage: speed_ab.ps1 <old.exe> <new.exe> <sources dir> <out.csv> [-NewEnv NAME=VALUE] [-NewArgs "--opt v"] (the new build's runs only)
param([string]$Old, [string]$New, [string]$Src, [string]$Out, [string]$NewEnv = '', [string]$NewArgs = '')
$runs = @(
  @{ n = 'S1'; i = 'mxam600.y4m'; a = @('--tune', 'animation'); r = 3 },
  @{ n = 'S2air'; i = 's2_air600.y4m'; a = @('--tune', 'animation'); r = 3 },
  @{ n = 'S2md'; i = 's2_md600.y4m'; a = @('--tune', 'animation'); r = 3 },
  @{ n = 'yo'; i = 'g_yo.y4m'; a = @('--tune', 'film'); r = 3 },
  @{ n = 'S3'; i = 'g_am30_4k.y4m'; a = @('--tune', 'animation'); r = 1 },
  @{ n = 'S4'; i = 'mxam600.y4m'; a = @('--tune', 'animation', '--preset', 'max'); r = 1 }
)
'run,build,round,cpu_ms,fps' | Set-Content -Encoding utf8 $Out
$env:CELV2_PROFILE = '1'
foreach ($run in $runs) {
  for ($k = 0; $k -lt $run.r; $k++) {
    foreach ($b in @(@('old', $Old), @('new', $New))) {
      $args2 = @('-i', (Join-Path $Src $run.i), '-o', "$env:TEMP\speed_ab.ivf", '--workers', 'auto', '--refs', '7', '--qindex', '130') + $run.a
      if ($NewArgs -and $b[0] -eq 'new') { $args2 += $NewArgs.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) }
      # The variable's previous value comes back after the run, so a NAME already set in the shell survives.
      if ($NewEnv -and $b[0] -eq 'new') { $kv = $NewEnv.Split('=', 2); $prev = [Environment]::GetEnvironmentVariable($kv[0]); Set-Item "env:$($kv[0])" $kv[1] }
      $log = & $b[1] @args2 2>&1 | Out-String
      $code = $LASTEXITCODE
      if ($NewEnv -and $b[0] -eq 'new') { [Environment]::SetEnvironmentVariable($kv[0], $prev) }
      if ($code -ne 0) { throw "$($run.n) $($b[0]) round ${k} exited $code`n$log" }
      $cpu = [regex]::Match($log, 'CPU [\d.]+ s \(([\d.]+) ms a frame').Groups[1].Value
      $fps = [regex]::Match($log, '([\d.]+) fps').Groups[1].Value
      if (-not $cpu -or -not $fps) { throw "$($run.n) $($b[0]) round ${k}: no CPU or fps line`n$log" }
      "$($run.n),$($b[0]),$k,$cpu,$fps" | Add-Content -Encoding utf8 $Out
    }
  }
}
Remove-Item env:CELV2_PROFILE
