<#
.SYNOPSIS
    Setup for NInfer 512K on Windows: prerequisite checks and model download with SHA-256
    verification. Firewall rules and a boot/watchdog task are optional.

.DESCRIPTION
    By default this only checks prerequisites and downloads/verifies the model into the release
    folder: no administrator rights, and nothing on the system is changed. Safe to re-run.
    Settings come from ninfer.config.ps1.

    The switches below change system settings and therefore need an elevated PowerShell. Use them
    only if you want them; the server runs fine without any of them.

.PARAMETER BlockOutbound
    Add Windows Firewall rules blocking outbound connections from the engine executables. The
    server does not need outbound access for text use (it would only fetch image/video URLs).

.PARAMETER AllowFrom
    Add an inbound Windows Firewall rule for the server port from this address range, e.g.
    192.168.1.0/24 for a LAN or 100.64.0.0/10 for Tailscale. Only needed to serve other devices,
    together with a non-loopback BindAddress in ninfer.config.ps1.

.PARAMETER RegisterTask
    Register the NInferServer scheduled task (runs ensure-ninfer.ps1 at boot and every 5
    minutes, whether or not anyone is signed in). For your own account no password is stored
    (an S4U task, which only needs local resources - all this server uses). Use -TaskUser to run
    it as another local account, or -StorePassword to store your password instead of S4U; both
    prompt for the password.

.PARAMETER TaskUser
    Account the task runs as (default: you). Not elevated either way.

.PARAMETER StorePassword
    Register with a stored password instead of S4U, e.g. if your environment refuses S4U tasks.

.PARAMETER SkipModel
    Do not download or verify the model (for example when copying it from another machine).
#>
param(
    [switch]$BlockOutbound,
    [string]$AllowFrom,
    [switch]$RegisterTask,
    [string]$TaskUser = "$env:USERDOMAIN\$env:USERNAME",
    [switch]$StorePassword,
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
. (Join-Path $Root "ninfer.config.ps1")
$c = $NInferConfig
function Resolve-Rooted($p) { if ([IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $Root $p } }
$exe   = Resolve-Rooted $c.Exe
$model = Resolve-Rooted $c.Model

# Pinned artifact: cometkim/Qwen3.8-27B-nvfp4full-NInfer (v3 container, Apache-2.0), the same
# pin as upstream's download_model.py.
$ModelUrl    = "https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer/resolve/main/qwen3_8_27b_nvfp4full.ninfer"
$ModelSize   = 19407229188
$ModelSha256 = "ac98cd392c84a04b2a21c2f5c3988dece88d20a697ba1de663fb32d5998b8ee9"

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "   WARN: $msg" -ForegroundColor Yellow }
function Ok($msg)   { Write-Host "   ok: $msg" -ForegroundColor Green }

# Only the optional system changes need elevation; check before the long download, not after.
$systemChanges = @()
if ($BlockOutbound) { $systemChanges += "-BlockOutbound" }
if ($AllowFrom)     { $systemChanges += "-AllowFrom" }
if ($RegisterTask)  { $systemChanges += "-RegisterTask" }
if ($systemChanges) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) {
        throw "These options change system settings and need an elevated (Administrator) PowerShell: $($systemChanges -join ', '). Without them, install.ps1 only checks prerequisites and downloads the model, which needs no elevation."
    }
}

Step "Prerequisites"
$gpu = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Select-Object -First 1)
if (-not $gpu) { throw "nvidia-smi not found: install the NVIDIA driver first." }
$name, $driver, $vram = $gpu -split ',\s*'
if ($name -notmatch 'RTX 5090') {
    throw "Found '$name'. This build is compiled only for the RTX 5090 (sm_120a) and will not run on other GPUs."
}
Ok "$name, driver $driver, $vram"
if ([version]($driver -replace '[^\d.]', '') -lt [version]"617.14") {
    Warn "driver $driver is older than 617.14, the oldest tested; update if startup fails with a CUDA error"
}
if (-not (Test-Path "$env:WINDIR\System32\vcruntime140_1.dll")) {
    throw "Microsoft Visual C++ 2015-2022 x64 runtime is missing. Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe and re-run."
}
Ok "Visual C++ runtime present"
$ramGiB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
$pinGiB = [math]::Round(($c.HostKvMiB + 150 * $c.HostStateSlots) / 1024, 1)
if ($pinGiB -gt $ramGiB / 2 - 0.5) {
    Warn "config pins ~$pinGiB GiB but Windows allows about $([math]::Round($ramGiB / 2, 1)) GiB on $ramGiB GiB of RAM; lower HostKvMiB in ninfer.config.ps1"
} else { Ok "$ramGiB GiB RAM (~$pinGiB GiB will be pinned)" }
if (-not (Test-Path $exe)) { throw "engine not found at $exe (set Exe in ninfer.config.ps1)" }
Ok "engine $exe"

Step "Model"
if ($SkipModel) {
    Write-Host "   skipped (-SkipModel)"
} elseif ((Test-Path $model) -and (Get-Item $model).Length -eq $ModelSize) {
    Ok "already present ($model)"
} else {
    New-Item -ItemType Directory -Force (Split-Path $model) | Out-Null
    $freeGB = (Get-PSDrive ((Resolve-Path (Split-Path $model)).Drive.Name)).Free / 1GB
    if ($freeGB -lt 20) { throw ("need ~19.4 GB free for the model, have {0:N1} GB" -f $freeGB) }
    Write-Host "   downloading 19.4 GB from Hugging Face (resumable; re-run to continue)..."
    # curl.exe ships with Windows 10+; -C - resumes a partial download.
    & curl.exe -L --fail -C - --retry 5 -o "$model.part" $ModelUrl
    if ($LASTEXITCODE -ne 0) { throw "download failed (curl exit $LASTEXITCODE); re-run to resume" }
    Move-Item "$model.part" $model -Force
}
if (-not $SkipModel) {
    if ((Get-Item $model).Length -ne $ModelSize) { throw "model size mismatch: expected $ModelSize bytes" }
    Write-Host "   verifying SHA-256 (takes a minute)..."
    $hash = (Get-FileHash $model -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $ModelSha256) {
        throw "SHA-256 mismatch ($hash). Delete $model and re-run; if it persists the publisher replaced the file."
    }
    Ok "SHA-256 verified"
}

if ($BlockOutbound -or $AllowFrom) { Step "Firewall" }
if ($BlockOutbound) {
    $bin = Split-Path $exe
    foreach ($f in Get-ChildItem $bin -Filter "ninfer*.exe") {
        $rule = "NInfer 512K - block outbound ($($f.Name))"
        if (-not (Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue)) {
            # The server never needs to open outbound connections for text use; this also
            # disables fetching image/video URLs. Replies to inbound requests are unaffected.
            New-NetFirewallRule -DisplayName $rule -Direction Outbound -Action Block -Program $f.FullName -Profile Any | Out-Null
        }
        Ok $rule
    }
}
if ($AllowFrom) {
    $rule = "NInfer 512K - allow inbound $($c.Port)"
    Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow -Protocol TCP -LocalPort $c.Port `
        -RemoteAddress $AllowFrom -Program $exe -Profile Any | Out-Null
    Ok "$rule from $AllowFrom"
    if ($c.BindAddress -eq "127.0.0.1") {
        Warn "BindAddress is 127.0.0.1, so other devices still cannot connect; set it in ninfer.config.ps1"
    }
}

if ($RegisterTask) {
    Step "Scheduled task NInferServer (as $TaskUser)"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $Root 'ensure-ninfer.ps1')`""
    $boot = New-ScheduledTaskTrigger -AtStartup
    $boot.Delay = "PT45S"
    $watchdog = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5)
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew
    $task = @{
        TaskName    = "NInferServer"
        Force       = $true
        Action      = $action
        Trigger     = @($boot, $watchdog)
        Settings    = $settings
        Description = "Keeps NInfer 512K up on $($c.BindAddress):$($c.Port) ($Root\ensure-ninfer.ps1)"
    }
    $self = $TaskUser -in "$env:USERDOMAIN\$env:USERNAME", $env:USERNAME
    if ($self -and -not $StorePassword) {
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
        try {
            Register-ScheduledTask @task -Principal $principal -ErrorAction Stop | Out-Null
            Ok "registered for $TaskUser without a stored password (S4U)"
        } catch {
            throw "Registering the S4U task failed ($($_.Exception.Message.Trim())). Re-run with -StorePassword."
        }
    } else {
        $cred = Get-Credential -UserName $TaskUser -Message "Password for $TaskUser (stored by Task Scheduler for the NInfer task)"
        Register-ScheduledTask @task -RunLevel Limited -User $cred.UserName -Password $cred.GetNetworkCredential().Password | Out-Null
        Ok "registered for $TaskUser with a stored password; re-run this if that password changes"
    }
    Write-Host "   a non-administrator task account needs the 'Log on as a batch job' right (secpol.msc)"
}

Step "Done"
if ($RegisterTask) {
    Write-Host "   start now: Start-ScheduledTask NInferServer"
} else {
    Write-Host "   start the server: powershell -ExecutionPolicy Bypass -File `"$(Join-Path $Root 'start-ninfer.ps1')`""
}
if (-not $systemChanges) {
    Write-Host "   no system settings were changed. Optional (elevated PowerShell): -BlockOutbound,"
    Write-Host "   -AllowFrom <range> to serve other devices, -RegisterTask to start at boot."
}
