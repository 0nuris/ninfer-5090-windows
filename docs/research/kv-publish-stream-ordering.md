# The KV block-table publish has no ordering edge against the compute stream

Status: **latent defect, confirmed present in this tree by inspection. Never observed on this
platform.** Recorded 2026-09-29. No code changed.

## The mechanism

Three facts about this tree, each read from source:

| Fact | Where |
|---|---|
| The compute and load streams are created **non-blocking** | `src/core/device.cu:92` `cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking)`, `:98` for `load` |
| `activate()` takes **no** `cudaStream_t` | `src/models/qwen3_5/program/storage/kv_store.h:895` |
| The publish defaults to the **legacy** stream | `kv_store.h:954` `commit_activation(..., cudaStream_t stream = nullptr)`, and `kv_store.h:1135` `commit_prefix_fork(...)` |

A stream created with `cudaStreamNonBlocking` has, by definition, **no implicit ordering
relationship with the legacy default stream**. So a KV block-table copy issued on stream 0 while
compute runs on the non-blocking compute stream is unordered with respect to the kernel that reads
it. That is a race by construction.

This needs no reproduction to state. It is a property of the code, and it would hold identically on
any platform — CUDA's stream semantics do not differ between Windows and Linux.

The path, from `codegraph`:

```
create_active (kv_store.h:863)  <- program_impl.cpp:376, graphs.cpp:136, 2 test sites
  -> activate  (kv_store.h:895) <- context.cpp, bind_sequence_kv
       -> commit_activation (kv_store.h:954)  <- prefill.cpp, 3 test sites
commit_active_snapshot (kv_store.h:1331)     <- transactions/capture.cpp
```

## What is not established

**No occurrence of this fault has been observed on the native Windows port.** Every report is from
another environment:

| Issue | Environment | Symptom |
|---|---|---|
| #208 | WSL2 / Ubuntu 24.04, driver 595.71.05, GCC 13.3, CUDA 13.1 | intermittent `cudaErrorIllegalAddress` (700), NVFP4 + MTP, fails in 10–15 min |
| #210 (closed) | Linux | device-side assert / full GPU lockup — same mechanism, different symptom |
| #320 | Linux, driver 595.91.07 | `Xid 79` "GPU has fallen off the bus", 8 times in 4.2 days |
| #329 | — | `Program::abort` releases the lane with no device sync, so a cancelled prefill's queued table copies can read the next request's page indices |
| #333 | RTX 5090 | `cudaErrorLaunchTimeout` immediately after a ~98 %-cache-replay prefill |

The #208 evidence is what makes the mechanism hard to dismiss and hard to confirm: it reproduces
intermittently under a multi-hour agentic workload, `CUDA_LAUNCH_BLOCKING=1` hides it (187 min clean),
and a captured request replayed against a fresh engine succeeds. Compute-sanitizer was measured as
too slow to use for it — prefill fell to ~200 tok/s and generation to a few tokens/sec — so **there
is no sanitizer report of the actual fault, here or upstream.**

Our driver is **617.14**, far above the 595.71.05 that carries NVIDIA's documented TMA-descriptor
bug. That separate candidate is moot for us, and it is not this defect: the TMA descriptor is built
once at load, so a misconfigured one would fail immediately rather than after ten minutes.

## Why the fix is not landing here yet

Upstream #320 carries a candidate patch — thread `device.stream` through `activate()` /
`create_active()` and drop the `= nullptr` default on the publish path — across 6 files
(`kv_store.h`, `paged_kv_cache.h`, `context.cpp`, `graphs.cpp`, `program_impl.cpp`,
`test_context_store.cpp`). All six exist here at the corresponding paths, and the patch's verified
base `d44ab584` is already in this tree's history via `4b3acfc2`.

It is not being applied here on speculation:

1. **No proving test exists.** The fault is sporadic and hours-long, so a 28-site change across the
   KV/context path would land unmeasured, on the hot path, with nothing able to show it helped.
   `CUDA_LAUNCH_BLOCKING` and the upstream author's 47-hour run are the only evidence available, and
   the author calls it "mechanism removed + 47 h clean under a workload that previously killed it in
   4.2 days", explicitly **not** a randomised trial, with a hardware caveat (card pinned at 600 W,
   `Xid 79` is a link-level death).
2. **Scope is upstream's call, not ours.** The patch author asked upstream whether threading the
   stream is the wanted direction, versus resolving `nullptr` inside the store, versus keeping the
   publish synchronous. That is their architectural decision. This port has no standing to answer it,
   and a PR from here would assert a Windows-port position on a Linux-observed fault with no
   measurement of our own.

So: land it when upstream lands it, through the normal merge, carrying their measurements.

## Meanwhile

`tools/release/check_production_stream_defaults.py` is a **ratchet, not a detector**. It pins the
exact set of defaulted-stream sites and fails if that set changes in **either** direction — growth is
a regression, shrinkage means the upstream fix landed and the pin needs a deliberate, recorded
update. It cannot tell whether the race fires — only whether the surface is widening.

The pinned set is wider than the KV path: **36 sites across 10 production files**, including
`src/core/device.cu:90-91` where the `compute` and `load` stream members themselves are declared
with `= nullptr`, `src/core/pdl.cuh` (programmatic dependent launch, where ordering is subtle by
construction), and one site in the FP8 TMA launcher. Two counts in this area have already been wrong
in this session's own notes: a survey matching only the parameter name `stream` reported 32 and
missed `device.h`'s `transfer_stream`, `device.h`'s `stream_`, and the two `device.cu` members. Any
future count here should come from the gate, not from a fresh grep.

The honest limit, stated because it is easy to overclaim: this catches one reintroduction vector —
someone re-adding a default. It does **not** catch a wrong stream being passed, nor a new publish
path added somewhere else, nor the race itself.
