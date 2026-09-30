# NInfer 512K deployment settings. Every other script in this folder reads this file.
# Relative paths resolve against the folder this file is in (the release root).

$NInferConfig = @{
    # Engine and model. A release zip ships bin\; from a source checkout use an absolute path
    # such as C:\src\ninfer\build\apps\ninfer-serve.exe.
    Exe         = "bin\ninfer-serve.exe"
    Model       = "models\qwen3_8_27b_nvfp4full.v3.ninfer"
    Logs        = "logs"
    KeepLogs    = 20            # rotated ninfer-<timestamp>.err/.out files to keep

    # Network. 127.0.0.1 serves this PC only. To serve other devices, set this PC's LAN or VPN
    # address and allow inbound connections to Port in your firewall (see README.md).
    # There is no API key; anything that can reach the port can use the model.
    BindAddress = "127.0.0.1"
    Port        = 8088
    ModelId     = "qwen3.8-27b-nvfp4-yarn512k"   # clients send this as "model"

    # Context. Above 262,144 (the model's native length) YaRN is enabled with
    # factor = MaxContext / 262144 (at most 2). 524,288 needs nvfp4 KV to fit on 32 GB.
    # Set 262144 with KvDtype fp8 and Spec dflash2 for the upstream native profile.
    MaxContext  = 524288
    KvDtype     = "nvfp4"
    Spec        = "mtp"         # DFlash/DFlash2 are refused when MaxContext > 262144
    DraftTokens = 5             # MTP depth 5 is the measured best for this artifact

    # Concurrency: requests generated at once, sharing one KV pool. 3 was measured to fit at
    # 524,288 on an RTX 5090 (1.23 GiB VRAM spare) and helps parallel agents with contexts up to
    # ~90k each; with 140k+ contexts each, concurrent requests evict each other's cache and run
    # slower than serial. 1 is the safe default.
    Concurrency = 1

    # Pinned host cache (system RAM) for conversations evicted from VRAM. Windows caps pinned
    # memory at about half of physical RAM; state slots cost ~150 MiB each. On 32 GB of RAM,
    # 8 slots + 8192 MiB leaves about 13 GB free. Lower HostKvMiB if startup fails pinning it.
    HostStateSlots = 8
    HostKvMiB      = 8192

    # Reasoning cap per request (tokens); clients can override per request.
    ThinkingBudget = 4096

    # Typical VRAM use of this profile (GiB), for start-ninfer.ps1's advisory warning only; the
    # engine itself decides whether the profile fits. ~29.5 GiB at Concurrency 1, ~30 at 3.
    NeedVramGiB = 29.5
}
