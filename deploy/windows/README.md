# NInfer 512K for Windows (RTX 5090)

Qwen3.8-27B on one RTX 5090 with a **524,288-token context**, served through OpenAI- and
Anthropic-compatible HTTP APIs. This is [NInfer](https://github.com/Neroued/ninfer)'s native
Windows port with two additions: YaRN context extension beyond the model's native 262,144 tokens,
and a fix that lets Windows pin the engine's host cache reliably.

## Requirements

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 5090** (32 GB). The engine is compiled only for `sm_120a` and will not run on any other GPU. |
| Driver | NVIDIA driver 617.14 or newer (the oldest tested). No CUDA toolkit is needed; the CUDA runtime is linked in. |
| OS | Windows 11 x64 |
| RAM | 32 GB or more. The default settings pin ~9 GiB; Windows allows pinning about half of RAM. |
| Disk | ~20 GB for the model |
| Runtime | [Microsoft Visual C++ 2015-2022 x64 runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) (most PCs have it) |

## Install

1. Download `ninfer-512k-<version>-win64-rtx5090.zip` from Releases and check it against the
   SHA-256 in the release notes. Extract it anywhere without spaces in the path, e.g. `C:\ninfer`.
2. Optionally edit `ninfer.config.ps1` (network address, context, concurrency, cache sizes).
3. From a PowerShell in that folder (no administrator rights needed):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```
   It checks the GPU, driver, runtime and RAM, then downloads the 19.4 GB model (resumable) and
   verifies its SHA-256. It changes nothing else on the system.
4. Start it:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start-ninfer.ps1
   ```
   It waits for `/health` and prints `READY` (about 15 s after the first start). Stop it by
   ending `ninfer-serve.exe` (Task Manager, or `Stop-Process -Name ninfer-serve`).

### Optional system integration

None of these are needed to run the server. Each changes system settings, so run
`install.ps1` with it from an **elevated** PowerShell:

| Switch | Effect |
|---|---|
| `-BlockOutbound` | Windows Firewall rules blocking the engine's outbound connections. It never needs them for text; only image/video URLs in requests would be fetched. |
| `-AllowFrom <range>` | Inbound firewall rule for the port, to serve other devices (see *Serving other devices*). |
| `-RegisterTask` | Scheduled task that runs `ensure-ninfer.ps1` at boot and every 5 minutes, whether or not anyone is signed in, restarting the server after a reboot or crash. Asks for the account's password, which Task Scheduler stores. |

## Use it

Base URL `http://127.0.0.1:8088/v1`, model `qwen3.8-27b-nvfp4-yarn512k`, no API key.

```powershell
$body = @{ model = "qwen3.8-27b-nvfp4-yarn512k"; messages = @(@{ role = "user"; content = "Hello" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8088/v1/chat/completions -Method Post -Body $body -ContentType "application/json"
```

Routes: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/models`, `/health`.
Requests are validated strictly: unsupported fields (`top_k`, `min_p`, `response_format` other
than text, non-empty `include`, `n > 1`, ...) are refused with a named error instead of ignored.
See the upstream README's "Client compatibility" section.

**Serving other devices.** Set `BindAddress` in `ninfer.config.ps1` to this PC's LAN or VPN
(e.g. Tailscale) address. Windows Firewall normally blocks inbound connections to it, so either
allow them your own way or run `install.ps1 -AllowFrom 192.168.1.0/24` (or `100.64.0.0/10` for
Tailscale) from an elevated PowerShell. There is no authentication, so anything that can reach
the port can use the model.

## What to expect

Measured on the reference machine (Windows 11, RTX 5090, 31.5 GiB RAM):

| | |
|---|---|
| Fit at 524,288 | KV pool 524,288 tokens (nvfp4), 2.34 GiB VRAM spare; 567,552 tokens and 1.23 GiB spare at `Concurrency = 3` |
| Full-context check | 510,057-token prompt: VRAM and shared memory constant for the whole request (no spill to system RAM); 4/4 planted facts recalled from ~102k to ~500k |
| Long-context quality | perplexity over the same 424,258 tokens: 1.57881 in 262k windows, 1.57655 as one 524k window |
| Short-prompt quality | GSM8K (50 questions): 98% at the native 262k profile and 98% at 524k |
| Speed | short prompts ~200-245 tok/s; at 510k depth, prefill 1.68k tok/s (a full 510k prompt takes ~5 min once; follow-up turns reuse the cache) and decode ~160 tok/s |

**Trade-offs of the 524K profile** compared with upstream's native 262K launchers:

- MTP speculative decoding instead of DFlash2 (DFlash drafts are not trained past 262K and are
  refused with YaRN).
- nvfp4 KV cache instead of fp8, which is what makes 524K fit.
- No image/video input (vision is not enabled or tested with YaRN).
- Static YaRN applies to every request, including short ones. Qwen notes this can affect short
  texts; no loss was measurable on the GSM8K sample above.

To run upstream's native profile instead, set `MaxContext = 262144`, `KvDtype = "fp8"`,
`Spec = "dflash2"`, `DraftTokens = 7`. The engine is byte-identical to upstream at 262,144 or less.

**Concurrency.** `Concurrency = 3` lets up to three requests generate at once from one shared KV
pool. On the reference machine (262K profile) it raised total throughput 35-53% for parallel
agents with contexts up to ~90k each, but with three contexts of 140k+ they evicted each other's
cache and ran about 2× slower than one at a time. Default: 1.

## Operating it

- Logs: `logs\ninfer.err` (per-request speed, cache hits, errors). Each start keeps the previous
  log as `ninfer-<timestamp>.err`; the newest 20 are kept.
- Watch live: `Get-Content logs\ninfer.err -Wait -Tail 0 | Select-String 'req#\d+ done|WARN|ERROR'`
- With `-RegisterTask`: the watchdog writes `logs\watchdog.log`. To pause it (to use the GPU
  for something else), create an empty `ninfer.disabled` file next to the scripts, then stop
  `ninfer-serve`. Delete the file to hand the GPU back.
- Optional hardening: in NVIDIA Control Panel, set *CUDA - Sysmem Fallback Policy* to
  *Prefer No Sysmem Fallback* for `ninfer-serve.exe`, so running out of VRAM fails loudly instead
  of spilling to system RAM.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cudaMallocHost failed ... out of memory` at startup | Lower `HostKvMiB` (and/or `HostStateSlots`). Windows caps pinned memory at ~50% of RAM. |
| `minimum Engine runtime reservation requires ...` | Not enough VRAM for the profile: close other GPU programs, or lower `Concurrency` / `MaxContext`. |
| `ABORT: not enough free VRAM` | Another model server holds the GPU. |
| Process exits immediately, `0xC0000135` | A DLL is missing: keep `bin\` intact, and install the VC++ runtime. |
| HTTP 400 `context length exceeded` | Prompt plus `max_tokens` is above `MaxContext`. |
| Scheduled task result `0x8007052E` | The stored password changed: re-run `install.ps1 -RegisterTask`. |

## Building from source

Build with the repository's `build_windows.bat` (CUDA 13.3, Visual Studio 2022 Build Tools or
2026 with the C++ workload), then point `Exe` in `ninfer.config.ps1` at `build\apps\ninfer-serve.exe`,
or assemble a release with `deploy\windows\stage-release.ps1 -Build <build dir> -Version <tag>`.
