<#
.SYNOPSIS
    Setup for NInfer 512K on Windows: prerequisite checks and model download with SHA-256
    verification.

.DESCRIPTION
    Checks the GPU, driver, Visual C++ runtime and RAM, then downloads the model into the release
    folder and verifies it. Needs no administrator rights and changes nothing else on the system;
    firewall rules, services and scheduled tasks are left to you (see README.md). Safe to re-run.
    Settings come from ninfer.config.ps1.

.PARAMETER SkipModel
    Do not download or verify the model (for example when copying it from another machine).
#>
param(
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

Step "Done"
Write-Host "   start the server: powershell -ExecutionPolicy Bypass -File `"$(Join-Path $Root 'start-ninfer.ps1')`""
