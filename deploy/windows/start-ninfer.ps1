# Starts ninfer-serve with the settings in ninfer.config.ps1 and waits until /health answers.
# Exit code 0 = READY, 1 = refused or failed (the reason is printed; details in logs\ninfer.err).

$Root = $PSScriptRoot
. (Join-Path $Root "ninfer.config.ps1")
$c = $NInferConfig
function Resolve-Rooted($p) { if ([IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $Root $p } }
$exe   = Resolve-Rooted $c.Exe
$model = Resolve-Rooted $c.Model
$logs  = Resolve-Rooted $c.Logs

foreach ($f in $exe, $model) {
    if (-not (Test-Path $f)) { Write-Output "ABORT: missing $f (run install.ps1)"; exit 1 }
}
New-Item -ItemType Directory -Force $logs | Out-Null

# A VPN or LAN address may not exist yet right after boot, and binding a missing address fails.
$loopback = $c.BindAddress -in "127.0.0.1", "0.0.0.0", "localhost"
if (-not $loopback) {
    for ($i = 0; $i -lt 60 -and -not (Get-NetIPAddress -IPAddress $c.BindAddress -ErrorAction SilentlyContinue); $i++) {
        Start-Sleep -Seconds 2
    }
    if (-not (Get-NetIPAddress -IPAddress $c.BindAddress -ErrorAction SilentlyContinue)) {
        Write-Output "ABORT: $($c.BindAddress) is not assigned on this PC (network or VPN down?)"
        exit 1
    }
}

# Replace a previous ninfer-serve on this port, but never stop an unrelated program.
$holders = Get-NetTCPConnection -LocalPort $c.Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($holder in $holders) {
    $proc = Get-Process -Id $holder -ErrorAction SilentlyContinue
    if (-not $proc) { continue }
    if ($proc.ProcessName -ne "ninfer-serve") {
        Write-Output "ABORT: port $($c.Port) is in use by $($proc.ProcessName) (pid $holder). Stop it or set Port in ninfer.config.ps1."
        exit 1
    }
    Stop-Process -Id $holder -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3   # let the driver release its VRAM before the check below
}

# Advisory only: the engine sizes its KV pool from the VRAM actually free and refuses with an
# exact "minimum Engine runtime reservation" message if the profile does not fit, so it never
# starts in a spilling state. nvidia-smi lists GPU 0 first, which is normally the engine's
# device on a single-GPU system.
$freeGiB = [double](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1) / 1024
Write-Output ("VRAM free {0:N1} GiB (this profile typically uses ~{1:N1} GiB)" -f $freeGiB, $c.NeedVramGiB)
if ($c.NeedVramGiB -gt $freeGiB) {
    Write-Output "WARN: less VRAM free than this profile typically uses. If startup fails, close other GPU programs or lower MaxContext / Concurrency / HostKvMiB."
}

# Keep the previous run's log instead of overwriting it.
foreach ($name in "ninfer.err", "ninfer.out") {
    $current = Join-Path $logs $name
    if ((Test-Path $current) -and (Get-Item $current).Length -gt 0) {
        $item  = Get-Item $current
        $stamp = $item.LastWriteTime.ToString("yyyyMMdd-HHmmss")
        Move-Item $current (Join-Path $logs ("{0}-{1}{2}" -f $item.BaseName, $stamp, $item.Extension)) -Force
    }
    $base = [IO.Path]::GetFileNameWithoutExtension($name)
    $ext  = [IO.Path]::GetExtension($name)
    Get-ChildItem $logs -Filter "$base-*$ext" | Sort-Object LastWriteTime -Descending |
        Select-Object -Skip $c.KeepLogs | Remove-Item -Force
}

$serveArgs = @($model,
    "--spec", $c.Spec, "--draft-tokens", "$($c.DraftTokens)", "--lm-head-draft",
    "--host", $c.BindAddress, "--port", "$($c.Port)", "--model-id", $c.ModelId,
    "--max-context", "$($c.MaxContext)", "--kv-capacity", "auto", "--kv-dtype", $c.KvDtype,
    "--device-state-slots", "1", "--prefill-chunk", "8192",
    "--max-concurrency", "$($c.Concurrency)",
    "--host-state-slots", "$($c.HostStateSlots)", "--host-kv-mib", "$($c.HostKvMiB)",
    # Measured context-cache bounds from upstream's launchers: the defaults gave 1/5 round-2
    # cache hits on resent prompts, these give 5/5.
    "--max-shared-prefixes", "7", "--max-private-continuations", "8",
    "--max-long-anchors-per-continuation", "4",
    "--preserve-thinking", "--default-thinking-budget", "$($c.ThinkingBudget)",
    "--pending-timeout-ms", "600000")

# Launch through cmd.exe, not Start-Process -RedirectStandard*: that form makes the server
# inherit this script's own stdout, so a caller reading this script through a pipe
# (Tee-Object, a service wrapper, another program) would wait until the server exits.
# Start-Process without redirection passes no handles down; cmd writes the server's output to
# the log files and stays alive exactly as long as the server, so $p.HasExited still works.
$quoted = $serveArgs | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { "$_" } }
$line = '"{0}" {1} > "{2}" 2> "{3}"' -f $exe, ($quoted -join ' '),
        (Join-Path $logs "ninfer.out"), (Join-Path $logs "ninfer.err")
$p = Start-Process -FilePath "$env:WINDIR\System32\cmd.exe" -ArgumentList "/d /s /c `"$line`"" `
    -PassThru -WindowStyle Hidden
Write-Output "ninfer starting (launcher pid $($p.Id))"

$probe = if ($c.BindAddress -eq "0.0.0.0") { "127.0.0.1" } else { $c.BindAddress }
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { Write-Output "process exited early - see $(Join-Path $logs 'ninfer.err')"; break }
    try { $null = Invoke-WebRequest "http://${probe}:$($c.Port)/health" -TimeoutSec 3 -UseBasicParsing; $ok = $true; break } catch {}
}
Write-Output "ninfer on $($c.BindAddress):$($c.Port) -> $(if ($ok) { 'READY' } else { 'FAILED' })"
if (-not $ok) { exit 1 }
