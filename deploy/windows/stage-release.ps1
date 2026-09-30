# Maintainer tool: assemble the release zip from a source build.
#   stage-release.ps1 -Build C:\src\ninfer\build -Version v1.1.0-512k.1
# Produces dist\ninfer-512k-<Version>-win64-rtx5090.zip containing bin\ (engine + FFmpeg DLLs),
# the deploy scripts, docs, licenses and SHA256SUMS. The model is never packaged.
param(
    [Parameter(Mandatory)] [string]$Build,
    [Parameter(Mandatory)] [string]$Version
)
$ErrorActionPreference = "Stop"
$repo  = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$name  = "ninfer-512k-$Version-win64-rtx5090"
$stage = Join-Path $repo "dist\$name"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force "$stage\bin", "$stage\licenses" | Out-Null

foreach ($f in "ninfer-serve.exe", "ninfer-perplexity.exe", "ninfer.exe") {
    $src = Join-Path $Build "apps\$f"
    if (Test-Path $src) { Copy-Item $src "$stage\bin\" }
}
if (-not (Test-Path "$stage\bin\ninfer-serve.exe")) { throw "no ninfer-serve.exe under $Build\apps" }
foreach ($d in "avcodec", "avformat", "avutil", "swscale", "swresample") {
    Copy-Item (Join-Path $repo "ffmpeg\bin\$d-*.dll") "$stage\bin\"
}

foreach ($f in "ninfer.config.ps1", "start-ninfer.ps1", "ensure-ninfer.ps1", "install.ps1", "README.md") {
    Copy-Item (Join-Path $PSScriptRoot $f) $stage
}
Copy-Item (Join-Path $repo "LICENSE") "$stage\licenses\LICENSE-ninfer-Apache-2.0.txt"
Copy-Item (Join-Path $repo "NOTICE") $stage
Copy-Item (Join-Path $PSScriptRoot "THIRD_PARTY_NOTICES.md") $stage
foreach ($f in Get-ChildItem (Join-Path $repo "ffmpeg") -Filter "LICENSE*") {
    Copy-Item $f.FullName "$stage\licenses\FFmpeg-$($f.Name)"
}
New-Item -ItemType Directory -Force "$stage\models", "$stage\logs" | Out-Null

$sums = Get-ChildItem $stage -Recurse -File | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(),
                  $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
}
$sums | Set-Content "$stage\SHA256SUMS" -Encoding ascii

$zip = Join-Path $repo "dist\$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip
"{0}  {1}" -f (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower(), (Split-Path $zip -Leaf)
