# Watchdog run by the NInferServer scheduled task every 5 minutes (and at boot). Does nothing
# while the server answers /health; otherwise runs start-ninfer.ps1, so the server returns
# after a reboot or crash. Each action is noted in logs\watchdog.log.
#
# To use the GPU for something else, create a file named ninfer.disabled next to this script
# before stopping the server; delete it to let the watchdog start NInfer again.

$Root = $PSScriptRoot
. (Join-Path $Root "ninfer.config.ps1")
$c = $NInferConfig
$logs = if ([IO.Path]::IsPathRooted($c.Logs)) { $c.Logs } else { Join-Path $Root $c.Logs }
New-Item -ItemType Directory -Force $logs | Out-Null
$log   = Join-Path $logs "watchdog.log"
$probe = if ($c.BindAddress -eq "0.0.0.0") { "127.0.0.1" } else { $c.BindAddress }
$url   = "http://${probe}:$($c.Port)/health"

function Note($msg) {
    Add-Content -Path $log -Value ("{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg)
    $lines = Get-Content $log -ErrorAction SilentlyContinue
    if ($lines.Count -gt 600) { $lines | Select-Object -Last 500 | Set-Content $log }
}

if (Test-Path (Join-Path $Root "ninfer.disabled")) { exit 0 }

try { $null = Invoke-WebRequest $url -TimeoutSec 5 -UseBasicParsing; exit 0 } catch {}

# /health fails for ~15 s while a server loads; give an in-progress start time to finish.
if (Get-NetTCPConnection -LocalPort $c.Port -State Listen -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
    try { $null = Invoke-WebRequest $url -TimeoutSec 5 -UseBasicParsing; exit 0 } catch {}
}

Note "health check failed; running start-ninfer.ps1"
$out  = & (Join-Path $Root "start-ninfer.ps1") 2>&1
$code = $LASTEXITCODE
Note ("start-ninfer.ps1 exit {0}: {1}" -f $code, (($out | Select-Object -Last 2) -join " | "))
exit $code
