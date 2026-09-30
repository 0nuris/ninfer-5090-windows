# NInfer 512K for Linux (RTX 5090)

Qwen3.8-27B on one RTX 5090 with a **524,288-token context**, served through OpenAI- and
Anthropic-compatible HTTP APIs: [NInfer](https://github.com/Neroued/ninfer) with YaRN context
extension beyond the model's native 262,144 tokens. The Windows kit is in
[`deploy/windows`](../windows/README.md).

> **Status: built, not yet run on Linux hardware.** The Linux tarball is compiled on Ubuntu 24.04
> with CUDA 13.3 by this repository's GitHub Actions workflow. The engine, YaRN extension and
> profile were validated on Windows 11 with an RTX 5090 (figures below); the Linux build and
> these scripts have not yet been run on a Linux machine with a GPU. Reports are welcome.

## Requirements

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 5090** (32 GB). The engine is compiled only for `sm_120a` and will not run on any other GPU. |
| Driver | An NVIDIA driver supporting **CUDA 13.3 or newer** (`nvidia-smi` shows the supported CUDA version). The CUDA runtime is bundled; no toolkit is needed. |
| OS | Ubuntu 24.04 x86_64 (the tarball links Ubuntu 24.04's FFmpeg and libcurl; other distributions: build from source) |
| Packages | `sudo apt install libavcodec60 libavformat60 libavutil58 libswscale7 libswresample4 libcurl4t64 curl iproute2` (most are present on desktop installs) |
| RAM | 32 GB or more. The default settings pin ~9 GiB. |
| Disk | ~20 GB for the model |

## Install

1. Download `ninfer-512k-<version>-linux-x86_64-rtx5090.tar.gz` from Releases, check it against
   the SHA-256 in the release notes, and extract it anywhere:
   ```bash
   tar -xzf ninfer-512k-<version>-linux-x86_64-rtx5090.tar.gz && cd ninfer-512k-<version>-linux-x86_64-rtx5090
   ```
2. Optionally edit `ninfer.conf` (network address, context, concurrency, cache sizes).
3. `./install.sh`: checks the GPU, driver, shared libraries and RAM, then downloads the 19.4 GB
   model (resumable) and verifies its SHA-256. It needs no root and changes nothing else on the
   system; a missing package is reported with the `apt` command to install it.
4. `./start-ninfer.sh`: waits for `/health` and prints `READY`. Stop it with
   `pkill -x ninfer-serve`.

## Use it

Base URL `http://127.0.0.1:8088/v1`, model `qwen3.8-27b-nvfp4-yarn512k`, no API key.

```bash
curl -s http://127.0.0.1:8088/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4-yarn512k","messages":[{"role":"user","content":"Hello"}]}'
```

Routes: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/models`, `/health`.
Requests are validated strictly: unsupported fields (`top_k`, `min_p`, `response_format` other
than text, non-empty `include`, `n > 1`, ...) are refused with a named error instead of ignored.
See the repository README's "Client compatibility" section.

**Serving other devices.** Set `BIND_ADDRESS` in `ninfer.conf` to this machine's LAN or VPN
address, and allow inbound TCP to `PORT` in your firewall (e.g. ufw), ideally only from the range
that should reach it. There is no authentication, so anything that can reach the port can use the
model.

## Running it unattended (your setup)

The kit does not install services, timers or firewall rules; how the server runs on your system is
your choice. What it provides:

- `start-ninfer.sh` starts the server detached and exits 0 once `/health` answers (1 on failure).
- `ensure-ninfer.sh` is a watchdog: it does nothing while the server is healthy and runs
  `start-ninfer.sh` otherwise, logging to `logs/watchdog.log`. Run it every few minutes from any
  scheduler (a systemd timer, cron) to start the server at boot and restart it after a crash. A
  `ninfer.disabled` file next to the scripts pauses it.

Things to know when you set that up: the server needs no root; it pins host memory, so a service
manager's memory-lock or memory limits must allow it; and it needs the GPU device nodes the NVIDIA
driver creates.

**Firewall.** As configured by this kit (vision off), the server never opens outbound
connections: requests containing images or video are refused before anything is fetched. Serving
other devices needs an inbound rule for `PORT`.

## What to expect

Measured on the reference machine (Windows 11, RTX 5090, 32 GB RAM); Linux figures may differ:

| | |
|---|---|
| Fit at 524,288 | KV pool 524,288 tokens (nvfp4), 2.34 GiB VRAM spare; 567,552 tokens and 1.23 GiB spare at `CONCURRENCY=3` |
| Full-context check | 510,057-token prompt: VRAM constant for the whole request; 4/4 planted facts recalled from ~102k to ~500k |
| Long-context quality | perplexity over the same 424,258 tokens: 1.57881 in 262k windows, 1.57655 as one 524k window |
| Short-prompt quality | GSM8K (50 questions): 98% at the native 262k profile and 98% at 524k |
| Speed | short prompts ~200-245 tok/s; at 510k depth, prefill 1.68k tok/s (~5 min for a full prompt, once) and decode ~160 tok/s |

**Trade-offs of the 524K profile:** MTP speculative decoding instead of DFlash2 (DFlash drafts are
refused with YaRN), nvfp4 KV instead of fp8 (what makes 524K fit), no image/video input, and static
YaRN on every request. To run upstream's native profile, set `MAX_CONTEXT=262144`, `KV_DTYPE=fp8`,
`SPEC=dflash2`, `DRAFT_TOKENS=7`.

## Operating it

- Logs: `logs/ninfer.err` (per-request speed, cache hits, errors). Each start keeps the previous
  log as `ninfer-<timestamp>.err`; the newest 20 are kept.
- Watch live: `tail -f logs/ninfer.err | grep --line-buffered -E 'req#[0-9]+ done|WARN|ERROR'`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `install.sh`: shared libraries missing | Install the packages it lists (Ubuntu 24.04 names). |
| `cudaMallocHost failed ... out of memory` at startup | Lower `HOST_KV_MIB` (and/or `HOST_STATE_SLOTS`), or raise the service's memory-lock limit. |
| `minimum Engine runtime reservation requires ...` | Not enough VRAM: close other GPU programs, or lower `CONCURRENCY` / `MAX_CONTEXT`. |
| `ABORT: port 8088 is in use by ...` | Another program listens on the port; the script never stops it. Stop it or change `PORT`. |
| CUDA error at startup | Check `nvidia-smi` reports CUDA 13.3 or newer; update the driver. |
| HTTP 400 `context length exceeded` | Prompt plus `max_tokens` is above `MAX_CONTEXT`. |

## Building from source

Ubuntu 24.04: install CUDA 13.3 (`nvcc`), `cmake`, `ninja-build`, `pkg-config`,
`libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev`, then

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --target ninfer-serve ninfer-perplexity
```

and set `EXE` in `ninfer.conf` to `build/apps/ninfer-serve`, or package a tarball with
`deploy/linux/stage-release.sh build <version>`. `.github/workflows/linux-build.yml` does exactly
this on GitHub Actions.

## Versioning

Releases share the repository's version (see [the Windows README](../windows/README.md#versioning)).
