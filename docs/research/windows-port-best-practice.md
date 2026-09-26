# Windows-port best practice: CUDA/MSVC, platform branches, ABI, fail-stop, and the nine-file surface

Research against primary sources (NVIDIA, Microsoft, CMake, ISO C++/cppreference) into whether the
Windows port of NInfer follows the best available practice, and where a specific better option
exists. Verified on 2026-09-26 against CUDA 13.3, MSVC 14.51, and the current working tree.

The port's platform-divergent surface is the nine files named in the brief plus the launchers and
build scripts. Everything below stays inside that surface: the port deliberately follows upstream's
architecture and diverges only where Windows requires it.

## Verdicts at a glance

| Topic | Verdict | One-line reason |
|---|---|---|
| TDR (timeouts) | **Already correct, one gap** | RTX 5090 is GeForce → WDDM only → default 2 s TDR applies; no measured launch exceeds it, but no handling/guidance for `cudaErrorLaunchTimeout` (702) exists |
| WDDM vs TCC | **Already correct** | GeForce cannot use TCC; the port correctly runs as a foreground process |
| Pinned memory | **Already correct** | `cudaMallocHost` with error checks; the large pin is measured, and Windows has no unified-memory mitigation anyway |
| `__grid_constant__` / param limits | **Already correct** | Four `CUtensorMap` = 512 B, far under the 32,764 B CUDA 12.1 limit |
| `alignas` / kernel-param ABI | **Already correct** | Inheriting `TENSOR_MAP_ALIGN` (64 under MSVC) is the right fix; the pointer workaround was the bug |
| `#ifdef _WIN32` branches | **Correct approach; one real defect** | No portable standard API covers the categories, so branches are right — but the MSVC libcurl path is missing, leaving the Winsock URL code dead |
| Artifact file I/O | **Already correct** | `CreateFileW` + `FILE_FLAG_NO_BUFFERING` is the documented `O_DIRECT` equivalent; 4096-byte payload alignment covers Advanced Format |
| Build-file README claim | **Improvable (documentation)** | README says "memory-mapped"; the code uses `ReadFile`, not `CreateFileMapping` |
| Fail-stop / supervision | **Improvable, with a Windows trap** | Latching `503` forever is a readiness signal with no liveness action; a Windows Service is *not* the fix on a GeForce card |
| Media acquisition | **Improvable (real gap)** | Remote HTTP(S) media is compiled out on MSVC; `find_package(CURL)` fixes it |
| HTTP transport (cpp-httplib) | **Correct; one small hardening** | Windows is supported and Win10+ is enforced; the port does not set `SO_EXCLUSIVEADDRUSE` |

---

## 1. CUDA + MSVC on Windows, 2025–2026 practice

### 1.1 Timeout Detection and Recovery (TDR)

**Already correct.** Microsoft documents the WDDM TDR mechanism and the default delay directly:
"the default timeout period in Windows is two seconds. If the GPU can't complete or preempt the
current task within the TDR timeout period, the OS diagnoses that the GPU is frozen."
([WDDM Support for Timeout Detection and Recovery](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery))

TDR is not optional on this hardware. NVIDIA's Windows installation guide states under *Use a
Suitable Driver Model* that TCC is available only on non-display devices such as Tesla, and
"NVIDIA GeForce GPUs (excluding GeForce GTX Titan GPUs) do not support TCC mode."
([CUDA Installation Guide for Microsoft Windows, §2.5](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-microsoft-windows/index.html))
An RTX 5090 is a GeForce, so it runs WDDM and TDR is active. NVIDIA's older getting-started guide
is blunter about what TCC buys: it "eliminates the timeouts that can occur when running under WDDM
due to the Windows Timeout Detection and Recovery mechanism," "reduces the latency of CUDA kernel
launches," and is required to use CUDA from a Windows service — none of which a GeForce card can
enable ([CUDA Getting Started Guide for Microsoft Windows 6.0](https://developer.download.nvidia.com/compute/cuda/6_0/rel/docs/CUDA_Getting_Started_Windows.pdf);
note the document is from 2014 but is still NVIDIA's clearest statement of the tradeoff).

What matters for this engine is what a TDR does to a running context. NVIDIA's Nsight documentation
records that after a TDR the application "will receive a grid launch failure, and the CUcontext
will begin to report errors" ([Timeout Detection & Recovery](https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm)).
In other words a TDR is a device-lost event, not a transient error.

The port is structurally safe against TDR today: prefill is chunked (`--prefill-chunk 8192` in the
launchers), decode uses CUDA graphs with a handful of tokens per round,
and a multi-GB weight upload is DMA, not a kernel. No single GPU operation in the measured profiles
is anywhere near 2 seconds. Nothing in the tree measures a launch duration against the TDR budget,
so this is an inference from the workload shape, not a measurement.

**Improvable (small):** two concrete gaps, both documentation/robustness rather than architecture.

- No profile or document mentions the Windows TDR budget or `TdrDelay`. Microsoft documents the
  registry keys (`TdrDelay`, `TdrLimitCount`, `TdrLimitTime`) and their meanings
  ([Testing and debugging TDR](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys)).
  If a future profile ever contains a single launch that could exceed 2 s, the failure mode is a
  silent device reset, and the README's Windows section does not say so.
- `grep` over `src/` finds no handling of `cudaErrorLaunchTimeout` (error 702). A TDR surfaces as an
  exception from the engine's synchronize/commit path and is caught by the generic latch in
  `engine_core.h` `fail_all_locked` (see §4), so it is not silently ignored — but it is not
  distinguished from a programming error either. If the fail-stop design in §4 changes, this is the
  error that should take the "device lost, must restart" branch.

### 1.2 WDDM vs TCC

Covered in §1.1. The port's choice to ship `.bat` foreground launchers rather than a Windows
Service is not merely acceptable; on a GeForce card it is the only sane choice for CUDA. See §4.

### 1.3 `cudaMallocHost` / pinned memory on Windows vs Linux

**Already correct.** The engine pins host state, host KV, and staging through `cudaMallocHost`, with
a checked failure path (`src/core/arena.cu:248–250` throws `cudaMallocHost failed`). NVIDIA's runtime
documentation describes `cudaMallocHost` as allocating "page-locked memory accessible to the device"
and warns that "allocating excessive amounts of memory ... may degrade system performance, since it
reduces the amount of memory available to the system for paging"
([CUDA Runtime API, Host Memory](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-runtime-api/group__CUDART__MEMORY.html)).
The startup log reports the pinned amounts per profile (1.46 / 2.19 / 2.92 GiB at 8 / 12 / 16 slots),
so the cost is measured rather than assumed.

There is a genuine Windows/Linux difference, but it is not a correctness problem this port can
engineer around. NVIDIA's own programming guide states that Windows is in the limited unified-memory
paradigm, and lists the Windows/WSL/Tegra restrictions explicitly
([Unified Memory on Windows, WSL, and Tegra](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)).
The precise claim that WDDM pinned transfers overlap worse than Linux is documented only in
developer-forum posts, not in an NVIDIA manual — that is a forum-level source, and I am labelling it
as such ([NVIDIA developer forum thread on pinned throughput across OSes](https://forums.developer.nvidia.com/t/pinned-memory-throughput-significantly-lower-on-ubuntu-than-on-windows/352580)).
The engine uses explicit `cudaMallocHost` staging plus `cudaMemcpyAsync`, which is the paradigm the
guide recommends for Windows, so there is no better documented alternative to adopt.

### 1.4 Kernel parameter space and `__grid_constant__` under MSVC

**Already correct.** CUDA 12.1 raised the kernel-parameter limit from 4,096 to 32,764 bytes on
Volta and later, and NVIDIA explicitly recommends `__grid_constant__` for large parameters: "Using
the `__grid_constant__` qualifier with kernel parameters is necessary to ..." (the same post)
([CUDA 12.1 Supports Large Kernel Parameters](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters)).
The port's descriptor blocks are four `CUtensorMap` (4 × 128 = 512 bytes), which is two orders of
magnitude under the limit, so the parameter size was never the problem; only the alignment was
(§3). The kernels pass the descriptor block by value under `__grid_constant__`
(`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:78`,
`src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:24–31`), which is the documented fast path.

### 1.5 Stream / event synchronisation differences

**Undetermined, and not actionable from primary sources.** The only NVIDIA statement I found that
touches Windows synchronisation cost is the 2014 getting-started guide's "TCC ... reduces the
latency of CUDA kernel launches" ([link above](https://developer.download.nvidia.com/compute/cuda/6_0/rel/docs/CUDA_Getting_Started_Windows.pdf)),
which is a WDDM-vs-TCC claim, not a portability bug. I found no NVIDIA or Microsoft document that
states a stream/event *semantic* difference between Windows and Linux. The engine's synchronisation
is event/graph based and was not changed for the port beyond what compiles; I would not assert a
difference exists without a primary source, and I did not find one.

---

## 2. Is `#ifdef _WIN32` the right approach, or is something better now standard?

**Correct approach.** For every category present in the nine files, C++20 (or any current standard)
has no portable API, so a branch is not a legacy habit — it is the only option. The important
question is whether each branch is minimized to the thing that actually differs, and they are.

Category by category:

- **Unbuffered positional file I/O** (`src/artifact/file_io.cpp`). The branch is the I/O *mode*,
  `O_DIRECT`/`pread` versus `CreateFileW` with `FILE_FLAG_NO_BUFFERING` + `OVERLAPPED`, not path
  handling. `std::filesystem::path` is already used on both sides (`file_io.h:26`,
  `CreateFileW(path.c_str(), …)`), so the portable path type is in place. The standard filesystem
  library has no unbuffered-I/O facility; Microsoft's own guidance documents `FILE_FLAG_NO_BUFFERING`
  as the mechanism ([File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)).
  There is nothing to delete here.
- **TTY detection** (`src/product/logging/logging.cpp:30–36`, `_isatty` vs `isatty`). `std::filesystem`
  and the C++ standard library have no terminal-is-a-tty query. No replacement exists.
- **Local wall-clock formatting** (`logging.cpp:119–123`, `localtime_s` vs `localtime_r`). The C++
  library never provided `localtime` at all. C++20 `<chrono>` time zones (`std::chrono::current_zone`,
  `zoned_time`) are a real standard alternative
  ([cppreference: `std::chrono::current_zone`](https://en.cppreference.com/w/cpp/chrono/current_zone);
  [MSVC `<chrono>`](https://learn.microsoft.com/en-us/cpp/standard-library/chrono?view=msvc-170)).
  For a per-message log formatter I do **not** recommend it: it pulls the time-zone database and
  allocation into a hot path to replace two lines, and upstream does not use it. The branch is the
  better engineering choice; this is a case where "more standard" is not "better".
- **Terminal width** (`startup_log.cpp:85–98`, `GetConsoleScreenBufferInfo` vs `ioctl(TIOCGWINSZ)`).
  No portable API. The Win32 call is the documented one.
- **Process id** (`request_log.cpp:45–49`, `context_cost.cpp:303–308`, `_getpid`). The standard library
  has `std::this_thread::get_id`, not a process id. `_getpid` is the MSVC CRT spelling of the same
  thing; `GetCurrentProcessId()` would also be valid but is not better.
- **File rename for atomic publish** (`context_cost.cpp:296–324`). `std::filesystem::rename` is
  already portable and is used directly. cppreference specifies the required POSIX-like
  delete-then-link-without-observing-deletion behaviour
  ([`std::filesystem::rename`](https://en.cppreference.com/w/cpp/filesystem/rename)). This is the
  standard correct pattern.

CMake's platform handling is likewise correct and idiomatic: `if(MSVC)` for compiler flags
(`CMakeLists.txt:31–42`) and `if(WIN32)` for link libraries (`src/product/CMakeLists.txt:9–11`), using
the documented `MSVC` and `WIN32` variables
([CMake MSVC](https://cmake.org/cmake/help/latest/variable/MSVC.html),
[CMake WIN32](https://cmake.org/cmake/help/latest/variable/WIN32.html)).

**One real defect in this category: the MSVC libcurl path is missing.**
`cmake/Dependencies.cmake:30–31` calls `pkg_check_modules(LIBCURL …)` only under `if(NOT MSVC)`, and
`src/product/CMakeLists.txt:5–8` defines `NINFER_HAVE_LIBCURL` only when `TARGET PkgConfig::LIBCURL`
exists. On MSVC that target is never created, so the entire URL branch of
`src/product/media_acquire/acquire.cpp` (lines 46–304) is compiled out — including the Winsock
`resolve_public()` SSRF guard (lines 155–200) — while `acquire.cpp:7–14` still includes
`<winsock2.h>` and `src/product/CMakeLists.txt:9–11` still links `ws2_32`. At runtime,
`acquire.cpp:367` throws `ninfer compiled without curl. Remote URLs are unsupported.`

So on Windows the port: (a) advertises vision, (b) links a socket library, (c) contains SSRF
protection that never compiles, and (d) rejects every remote `http(s)` media URL. That is the
clearest correctness/UX gap in the nine files. The fix is upstream-compatible: use CMake's own
`FindCURL` in module mode and link its imported target,
`target_link_libraries(ninfer_media_acquire PRIVATE CURL::libcurl)`
([CMake FindCURL](https://cmake.org/cmake/help/latest/module/FindCURL.html)), which supplies a
Windows libcurl the same way upstream's pkg-config does on Linux. The alternative — fetching URLs
through FFmpeg's libavformat protocol layer, which the port already links — would centralize the
transport but would lose the `CURLOPT_RESOLVE`-based SSRF controls unless they were reimplemented,
so it is not the better change.

---

## 3. `alignas` / ABI portability for CUDA kernel parameters

**The fix the port already made is the recommended one.** The history is in the port's own report
([`docs/upstream-reports/ninfer-tma-descriptor-graph-capture.md`](../upstream-reports/ninfer-tma-descriptor-graph-capture.md)):
the pointer-based workaround copied the descriptor into a device buffer with `cudaMemcpyAsync`, which
under stream capture becomes a graph *node* whose source is a dead stack frame, producing an illegal
instruction on replay. The by-value `__grid_constant__` path carries the bytes in the kernel node and
is what upstream uses on other platforms.

The ABI facts that make inheriting `TENSOR_MAP_ALIGN` the right fix, each from a primary source:

- `CUtensorMap` is defined by NVIDIA as an over-aligned opaque type. The shipped CUDA 13.3 header
  (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include\cuda.h`, lines 3749–3762) reads:

  ```c
  #if defined(_MSC_VER)
    #define TENSOR_MAP_ALIGN 64
  #else
    #define TENSOR_MAP_ALIGN 128
  #endif
  typedef struct CUtensorMap_st {
      alignas(TENSOR_MAP_ALIGN) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
  } CUtensorMap;
  ```

  The header is part of the CUDA Toolkit and is not served at a stable URL; the corresponding
  reference page states the general requirement as "Requires compiler support for aligning to 128
  bytes" ([CUDA Driver API: `CUtensorMap`](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html)).
  The MSVC-specific lowering to 64 in the shipped header is what lets the port pass a four-descriptor
  struct by value to an MSVC-compiled kernel; a local `alignas(128)` override on the wrapper defeats
  exactly that lowering.
- MSVC rejects an explicitly over-aligned *parameter*: "'parameter': formal parameter with
  `__declspec(align('#'))` won't be aligned ... Function parameter alignment is controlled by the
  calling convention used" ([Compiler Error C2719](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c2719)).
  C2719 is the error the port hit; removing the override is the documented remedy, because the type's
  own alignment then falls to the value MSVC's x64 convention can honour.
- `__grid_constant__` applies to "const qualified non-reference type kernel parameters" and exists to
  avoid a per-thread stack copy of the parameter; by-value is therefore the intended shape
  ([LLVM RFC on `__grid_constant__`](https://discourse.llvm.org/t/rfc-hip-grid-constant-support/67759);
  the CUDA C++ Programming Guide documents the specifier at §10.2.4,
  [contents](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)). A pointer would
  have side-stepped the specifier's benefit and reintroduced the capture bug.

**Better alternative? No.** The two candidates are worse:

- Keep the pointer but give it a persistent, replay-stable source. This is what the upstream PR
  suggested, and the port's report already proved it insufficient: the memcpy node's *source* is
  still the caller's frame, so a persistent destination does not fix capture. Rejected on evidence,
  not preference.
- Copy the descriptor to a `__constant__` symbol with `cudaMemcpyToSymbolAsync` per launch. This is
  the pre-12.1 idiom the CUDA 12.1 post exists to replace; it adds a host-side copy and a symbol per
  launch and, being an async copy, has the same capture-node hazard. Rejected.

The port already removed the override on both affected routes
(`src/ops/linear/bf16/bf16_a16_tma_mma.cuh:14–22`, `src/ops/linear/nvfp4/nvfp4_a4_tma.cuh:24–31`)
with a comment recording why. **Already correct; no change.**

---

## 4. Windows process / watchdog / service practice for the fail-stop

### What the port does

`src/runtime/engine/engine_core.h` `fail_all_locked()` sets `failed_ = true` under the queue lock
(lines 1912–1918), reports the error to every live and pending request, and clears physical state.
`is_available()` then returns `!stopping_ && !failed_` forever (lines 250–253). The serve layer maps
that to `GET /health → 503 {"status":"unavailable"}` (`src/serve/http_server.cpp:452–457`) and to
`Unavailable`/503 on generation (`src/serve/generation_service.cpp:81–86`). The process never exits,
and the `.bat` launchers run the server in the foreground with no restart loop
(`start_ninfer_v3_dflash2_vision.bat:135`).

### What the standard says

**If it were a Windows service**, the documented contract is an exit code a supervisor can act on.
`SERVICE_STATUS` carries `dwWin32ExitCode`, and "to return an error code specific to the service,
the service must set this value to `ERROR_SERVICE_SPECIFIC_ERROR` to indicate that the
`dwServiceSpecificExitCode` member contains the error code"; "the service should set this value to
`NO_ERROR` when it is running and on normal termination"
([`SERVICE_STATUS`](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/ns-winsvc-service_status)).
The SCM can then restart the service through configured recovery actions — `sc failure … actions=
restart/…` ([Sc failure](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/cc742019(v=ws.11))),
programmatically via `SERVICE_FAILURE_ACTIONS`
([`SERVICE_FAILURE_ACTIONS`](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/ns-winsvc-service_failure_actionsa)).
A service is considered failed "when it terminates without reporting a status of `SERVICE_STOPPED`"
(same page).

**It is not a service, it is a console process**, so the supervisor contract is the process exit
code. Task Scheduler can be configured to restart on failure
([RestartOnFailure settings element](https://learn.microsoft.com/en-us/windows/win32/taskschd/taskschedulerschema-restartonfailure-settingstype-element),
[TaskSettings.RestartCount](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasksettings-restartcount)).
Note the disagreement about exactly when Task Scheduler restarts: the documented wording is
"attempt to restart the task if the task fails for any reason," while a widely-cited Stack Overflow /
Server Fault answer claims it only covers the scheduler failing to *start* the action. That is a
hosted-community source, not a Microsoft statement, and I am flagging it as such — the Microsoft
wording is the one to rely on.

**The liveness/readiness split is the convention the current behaviour half-satisfies.** Kubernetes
documents the pattern directly: a readiness probe returning non-2xx/3xx removes the instance from the
endpoint list, while "liveness probes determine when to restart a container ... liveness probes could
catch a deadlock, where an application is running, but unable to make progress. Restarting a
container in such a state can help to make the application more available despite bugs"
([Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)).
The port's `/health` returning 503 is a textbook readiness signal; there is no liveness action.

### The Windows trap that rules out the obvious "fix"

The obvious upgrade — wrap the engine as a Windows Service so the SCM can restart it — does not work
here. NVIDIA's getting-started guide lists, as a TCC-only benefit, that "TCC allows the use of CUDA
from within processes running as Windows services, which is not possible for WDDM devices"
([CUDA Getting Started Guide 6.0](https://developer.download.nvidia.com/compute/cuda/6_0/rel/docs/CUDA_Getting_Started_Windows.pdf)),
and GeForce does not support TCC
([CUDA Installation Guide §2.5](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-microsoft-windows/index.html)).
So a Windows Service on an RTX 5090 would start and then fail to create a CUDA context. I could not
find a CUDA 13.x restatement of this service-session restriction; the 6.0 guide is the primary
source I have, and I am labelling it as such. The port's foreground `.bat` model is therefore
correct, and it is the model to keep.

### Concrete improvement

Give the fatal latch an exit code the existing launcher can consume:

- In `src/product/logging/startup_log.cpp` the failure path already logs a single line; the change
  belongs in the serve application around `engine_core.h` `fail_all_locked`, not in the engine.
  Once `failed_` is latched, the serve loop should flush logs, stop accepting, and `std::exit` with a
  documented, specific non-zero code (for example `3` for "engine fault, restart required", distinct
  from `1` for a configuration error and `2` for a bind failure). Microsoft documents flush-on-exit
  ordering through `std::exit`/`ExitProcess`; keeping the constant documented in `README.md` makes it
  a contract.
- Then wrap the launcher: either add a `:restart` loop around the `"%SERVE%"` line in the
  `start_*_vision.bat` files, or register the command as a Task Scheduler task with
  `RestartOnFailure` ([link above](https://learn.microsoft.com/en-us/windows/win32/taskschd/taskschedulerschema-restartonfailure-settingstype-element)).
  Today a latched process is indistinguishable from a healthy one at the process level.
- Keep `/health` returning 503 during the grace period before exit, so in-flight readiness checks
  drain.

The tradeoff to record: latching lets an operator attach a debugger and inspect a live process;
exiting lets a supervisor recover automatically. The standard native pattern is the latter, and the
port currently has no supervisor at all, so the latch produces no recovery. This is the decision
under review, and the research supports exit-with-code once the exit code has a consumer.

---

## 5. Category alternatives: file I/O, media, HTTP

### 5.1 Windows file I/O for multi-GB artifacts

**Already correct — the Win32 equivalent of `O_DIRECT`.** `src/artifact/file_io.cpp` opens with
`CreateFileW` and `FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN`
(lines 36–45, 158–175), issues positional `ReadFile` calls via `OVERLAPPED.Offset/OffsetHigh`
(lines 59–67), and chunks `read_exact` at 64 MiB so a request never exceeds `DWORD`
(lines 126–134). Microsoft documents the unbuffered requirements: access sizes and the `OVERLAPPED`
offset "must be … an integer multiple of the volume sector size," buffer addresses "should be
physical sector-aligned," and it "strongly recommends that developers align unbuffered I/O to the
physical sector size" reported by `IOCTL_STORAGE_QUERY_PROPERTY`
([File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)). The port's
`kPayloadAlignment = 4096` (`src/artifact/framing.h:18`) is a multiple of both the 512-byte logical
sector and the 4096-byte physical sector, so it is safe on Advanced Format and NVMe devices without
querying the device. Using `OVERLAPPED` on a synchronous handle to select an offset is also
documented behaviour ([`ReadFile`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile)).
Reading only with positional 64 MiB chunks is correct and needs no change.

**Improvable (documentation only):** `README.md:120` says artifact I/O "uses Win32 memory-mapped and
unbuffered positional reads." The code does not memory-map; `grep` finds no `CreateFileMapping` or
`MapViewOfFile` in `src/` (the only hits are inside cpp-httplib's own static-file code). The README
should drop "memory-mapped."

**Genuinely better option online, but not recommended here:** DirectStorage is Microsoft's API for
exactly this workload — getting NVMe data to a GPU at high bandwidth with low CPU cost, with the
destination buffer mapped directly into the pipeline and hardware decompression
([DirectStorage Overview](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview);
desktop details at [aka.ms/directstorage](https://aka.ms/directstorage)). It would, however, replace
the artifact reader's entire I/O path with a DirectX-12/DirectStorage dependency and diverge from
upstream's architecture, which the brief forbids. Worth recording as the ceiling for a future
platform-specific fast path; not a change for this port.

### 5.2 Media acquisition

Covered in §2: the port already depends on FFmpeg for decode (`src/media/decode/decode.cpp` includes
`libavcodec`/`libavformat`) and on libcurl for URL acquisition. The single concrete fix is linking
libcurl under MSVC via CMake's `FindCURL` / `CURL::libcurl`
([CMake FindCURL](https://cmake.org/cmake/help/latest/module/FindCURL.html)), which restores the
existing SSRF-guarded fetch path rather than writing a new one. Until then, remote HTTP(S) media is
unsupported on Windows and the Winsock code is dead — that should at minimum be stated in the README
alongside the FFmpeg-DLL note, so a user is not left to discover the runtime throw.

### 5.3 HTTP transport (cpp-httplib)

**Correct, and Windows-aware.** The vendored version is 0.54.1 and the library explicitly rejects
Windows 8 and earlier at compile time (`_WIN32_WINNT < 0x0A00` → `#error`), which matches the Win11
target ([cpp-httplib README](https://github.com/yhirose/cpp-httplib)). The SSE path, keepalive
tuning, and connection-close detection the port uses are all first-class in the library's API
(same README: `is_connection_closed`, `set_keep_alive_timeout`, the Stream/SSE documentation).

**Improvable (small, security-relevant hardening):** cpp-httplib's default server socket options
enable `SO_REUSEADDR` on Windows, and the library's own README warns that on Windows "`SO_REUSEADDR`
allows two sockets that both set it to bind to the same port, so use `SO_EXCLUSIVEADDRUSE` instead"
([cpp-httplib README, "Port sharing and exclusive binding"](https://github.com/yhirose/cpp-httplib)).
Microsoft's Winsock guidance is stronger: "All server applications must set `SO_EXCLUSIVEADDRUSE` for
a strong level of socket security"; without it a second process can hijack the port and "the behavior
for all sockets bound to that port is indeterminate"
([Using SO_REUSEADDR and SO_EXCLUSIVEADDRUSE](https://learn.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse)).
`src/serve/http_transport.cpp` `configure_http_server_socket()` (lines 96–120) calls
`httplib::default_socket_options(socket)` and then only adds TCP keepalive; it does not replace the
default with `SO_EXCLUSIVEADDRUSE`. The port currently compensates with a `netstat` port check in
every `start_*_vision.bat`, but that is a TOCTOU check on the launcher, not protection for the
listener. The concrete change is to use `svr.set_socket_options(...)` to set
`SO_EXCLUSIVEADDRUSE` on Windows, as the library README demonstrates.

**Not recommended:** Microsoft's HTTP Server API (`http.sys`) would be the Windows-native transport,
but it is a full rewrite of the serving layer and Windows-only, so it fails the "do not fork away
from upstream" constraint.

---

## 6. What I could not determine

- **Whether any current profile contains a single GPU operation near the 2 s TDR budget.** This
  needs a measurement (CUDA-graph replay and prefill-chunk timings on this card), not a document.
  The workload shape strongly suggests not, but that is an inference.
- **A current (CUDA 13.x) statement of the WDDM/service-session CUDA restriction.** The clearest
  primary source is the CUDA 6.0 getting-started guide; the modern installation guide documents TCC's
  GeForce exclusion but not the service-session consequence. If the port ever considers a service
  wrapper, this should be re-verified against current NVIDIA guidance or by experiment.
- **A Microsoft guarantee that `std::filesystem::rename` is atomic on all Windows filesystems.**
  cppreference specifies the required POSIX-like semantics
  ([`std::filesystem::rename`](https://en.cppreference.com/w/cpp/filesystem/rename)) and the port's
  temp-file + rename in `context_cost.cpp:296–324` is the standard pattern, but Microsoft's
  `MoveFileEx` documentation does not use the word "atomic"; community sources disagree about whether
  `MOVEFILE_REPLACE_EXISTING` or `ReplaceFile` is the stronger primitive
  ([MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw),
  [LWN's summary of the ambiguity](https://lwn.net/Articles/682988)). The preset file this guards is
  regenerable, so the risk is low; I am recording the uncertainty rather than asserting atomicity.
- **Whether NInfer's pinned-memory overlap is measurably worse on WDDM than on Linux.** The claim
  exists only at forum level (§1.3). Measuring it is out of scope for this document.

## Sources

Primary (NVIDIA/Microsoft/CMake/ISO C++):

- Microsoft, [WDDM Support for Timeout Detection and Recovery (TDR)](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery)
- Microsoft, [Testing and debugging TDR](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys)
- NVIDIA, [CUDA Installation Guide for Microsoft Windows, §2.5](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-microsoft-windows/index.html)
- NVIDIA, [CUDA Getting Started Guide for Microsoft Windows 6.0 (PDF)](https://developer.download.nvidia.com/compute/cuda/6_0/rel/docs/CUDA_Getting_Started_Windows.pdf)
- NVIDIA, [Timeout Detection & Recovery (Nsight)](https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm)
- NVIDIA, [CUDA Programming Guide: Unified Memory on Windows, WSL, and Tegra](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- NVIDIA, [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- NVIDIA, [CUDA 12.1 Supports Large Kernel Parameters](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters)
- NVIDIA, [CUDA Driver API: CUtensorMap](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html)
- NVIDIA, [CUDA Driver API: Tensor Map Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)
- NVIDIA, [CUDA Runtime API: Host Memory](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-runtime-api/group__CUDART__MEMORY.html)
- NVIDIA, [CUDA C++ Programming Guide contents (§10.2.4)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html)
- Microsoft, [Compiler Error C2719](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-2/compiler-error-c2719)
- Microsoft, [File Buffering](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- Microsoft, [ReadFile](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-readfile)
- Microsoft, [MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)
- Microsoft, [File System Navigation (C++)](https://learn.microsoft.com/en-us/cpp/standard-library/file-system-navigation)
- Microsoft, [path class](https://learn.microsoft.com/en-us/cpp/standard-library/path-class)
- Microsoft, [`SERVICE_STATUS`](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/ns-winsvc-service_status)
- Microsoft, [`SERVICE_FAILURE_ACTIONS`](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/ns-winsvc-service_failure_actionsa)
- Microsoft, [Sc failure](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/cc742019(v=ws.11))
- Microsoft, [RestartOnFailure settings element](https://learn.microsoft.com/en-us/windows/win32/taskschd/taskschedulerschema-restartonfailure-settingstype-element)
- Microsoft, [TaskSettings.RestartCount](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasksettings-restartcount)
- Microsoft, [Service Programs](https://learn.microsoft.com/en-us/windows/win32/services/service-programs)
- Microsoft, [Using SO_REUSEADDR and SO_EXCLUSIVEADDRUSE](https://learn.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse)
- Microsoft, [DirectStorage Overview](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview) ([desktop](https://aka.ms/directstorage))
- CMake, [FindCURL](https://cmake.org/cmake/help/latest/module/FindCURL.html), [MSVC](https://cmake.org/cmake/help/latest/variable/MSVC.html), [WIN32](https://cmake.org/cmake/help/latest/variable/WIN32.html)
- cppreference, [`std::filesystem::rename`](https://en.cppreference.com/w/cpp/filesystem/rename), [`std::filesystem::path`](https://en.cppreference.com/w/cpp/filesystem/path), [`std::chrono::current_zone`](https://en.cppreference.com/w/cpp/chrono/current_zone)
- Kubernetes, [Liveness, Readiness, and Startup Probes](https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/)
- cpp-httplib, [README (v0.54.1 vendored)](https://github.com/yhirose/cpp-httplib)

Non-primary (labelled where used): the NVIDIA developer-forum thread on pinned throughput, the LLVM
RFC for `__grid_constant__` semantics, the LWN article and Stack Overflow answers on Windows rename
atomicity, and the CUDA 13.3 `cuda.h` shipped header (read locally, quoted above).
