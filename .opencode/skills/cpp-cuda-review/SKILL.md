---
name: cpp-cuda-review
description: >
  Review C++/CUDA changes in NInfer for correctness, memory safety, concurrency, and
  contract compliance. Use when asked to "review", "code review", "review this change/kernel/Op",
  or before landing an engine, Op, planner, artifact, or serving change. This is the review skill
  for this repo; the project-local `code-review` skill is a .NET/Roslyn import and does not apply.
---

# C++/CUDA Review (NInfer)

Review native C++/CUDA changes for what tooling cannot certify: numerical correctness under the
oracle contract, memory safety, concurrency, and the architectural ownership boundaries this repo
declares in `AGENTS.md`. It complements the release gate; it does not replace it.

## 1. Set depth by blast radius, not line count

| Blast radius | Examples | Depth |
|---|---|---|
| Critical | kernel math, Op numerics, state/KV lifetime, context admission, artifact framing/binding, CUDA graph capture | every path, against an independent oracle |
| High | planner/execution composition, KV/context capacity, serving schema, model binding | consumers + boundary behavior |
| Medium | new Op following an existing pattern, bug fix with a regression test | checklist pass + oracle |
| Low | docs, comments, formatting, logging | glance; build + affected checks |

A one-line change in admission or state release outranks a 300-line rename.

## 2. NInfer contract checks (highest priority)

Review these before style. Each is a real project invariant, not a preference.

1. **Numerics.** Every floating-point Op has a naive FP32/FP64 oracle; packed inputs are decoded
   independently with their stored scales; the production route is qualified directly against the
   oracle at relevant real shapes, never against another kernel or "plausible model output". Exact
   transforms use an exact oracle. `docs/maintainer/op-development.md` is the full contract.
2. **Ownership boundaries** (`AGENTS.md`, `docs/maintainer/engine-architecture.md`): Core owns
   physical primitives; artifact owns framing/materialization; Ops own closed math/state
   transitions; Models own fixed math, binding, frontend semantics; Runtime owns execution
   contracts. Reject hidden device allocation, runtime weight repacking, generic model graphs,
   family base classes, plugin discovery, string-driven execution, or a Python inference route.
3. **State lifetime.** Programs share no mutable state or device allocation. `release()` is gated
   on references; capacity returns to the global pool only on terminal release or an explicit
   entitlement shrink (resource-scheduling design doc §6.1). A change that frees state must name
   the reference it drops.
4. **Execution contract.** One GPU, one resident model, startup-fixed concurrency 1–8, bounded
   FIFO, no active-request preemption, one compact decode batch per round.
5. **Artifact/format.** Framing, layout, codec, conversion, or binding changes are validated with
   the affected contract tests and a real artifact when semantics require it.
6. **External contracts.** OpenAI/Anthropic protocol behavior is external: update the schema tests
   and `docs/serving.md` together.
7. **Sampling and speculative semantics.** A distribution claim is not established by reading the
   arithmetic. Two specific traps, both hit in one session:
   - **Grade the severity, do not assume it.** The accept test is `min(1, p/q)` with a
     `normalize(max(0, p - q))` correction, which holds for *any* normalised `q`. A drafter that
     ignores `presence_penalty` therefore costs acceptance and cannot change what is sampled
     (measured 4.18 vs 2.45 tokens per round; `P(accept) = 1 - TV(p,q)`). Calling that a correctness
     defect was wrong. When `p` and `q` disagree, say which of the two facts you are reporting.
   - **A control that passes for a structural reason is not a control.** At `top_k = 1` the support
     is one token, so `p` is a point mass, so `p >= q` holds for every draft whatever `q` is and
     accept-always is correct. Agreement at `top_k = 1` says nothing about the accept rule, and the
     multi-token case is the only place its arithmetic can be wrong. Before reporting a control,
     state what would have to be true for it to fail.
   A per-decision test pins one seed and cannot see a distributional error at all. The accept rule is
   covered by `accept_distribution_isolation_case`, which splits `extent=0` (samples from `p`, no
   accept test) from `extent=1` (accept plus residual) and requires both to reproduce `p`; keep that
   split when touching it.

## 3. C/C++ memory safety (Trail of Bits `c-review` taxonomy)

- **Corruption:** buffer over/underflow, use-after-free, double free, uninitialized reads, invalid
  downcasts, type confusion, aliasing violations.
- **Integer:** overflow/underflow/truncation in size and index arithmetic; signed/unsigned
  comparison and wraparound; unchecked `size_t` multiplication.
- **Concurrency:** data races, missing/incorrect locking, lock ordering, atomics and memory
  ordering, TOCTOU, lifetime across threads and streams.
- **Resource:** leaks on error paths, missing RAII, non-owning views outliving their owner.
- **Platform (MSVC/Windows):** exe-lock on rebuild, `/W4` warning regressions, DLL staging for the
  app to start at all.

## 4. CUDA-specific

- Stream and event correctness; work enqueued on the wrong stream; missing sync points.
- Error checking on every launch, and never a host synchronisation introduced inside CUDA graph
  capture that breaks capture safety.
- Memory ordering between dependent kernels; pinned-host buffer lifetime across async copies.
- Workspace aliasing between Ops in the same Program; per-Program workspace must not be shared.
- For performance claims: measure at the claimed scope; profile only for an unresolved
  attribution (`ncu-report` skill covers Nsight Compute analysis).

## 5. How to review (opencode-native)

1. Get the change: `git diff`, `git show`, or the named files.
2. Reach for `codegraph_explore` before grep/read to see the symbols, the call path, and the
   blast radius in one call.
3. For numerical changes, require the oracle comparison evidence at real shapes. "It matches the
   reference kernel" is not evidence.
4. Delegate depth with the `task` tool (`explore`, `general`) across genuine seams, then verify
   their claims against the artifact yourself.
5. Run the affected checks (`tests/README.md`, `tools/scripts/test_v3.cmd`), and the release gate
   `tools/release/check_test_baseline.py` for a core change.

## 6. Red flags earned by this repo

- A change carried out but not verified; **verification is the deliverable**.
- A test changed to make it pass, or a test that mirrors implementation / freezes private layout.
- Silent failure modes: "Build Complete!" on a failed link, missing FFmpeg DLLs, launcher
  preflight gaps.
- Claiming an end-to-end improvement from an Op microbenchmark.
- Rationalising a reading of the code instead of measuring it.
- **A harness error reported as a product defect.** One session reported a disagreement between a
  device and the suite's own oracle as a kernel-or-oracle divergence; it was the throwaway harness,
  and the cause was never identified. Four more harness errors that session were caught before they
  reached a report, which is the only reason the first one is the lesson: a draft that was
  deterministically accepted so the branch under test never ran, Op-maintained token counts drifting
  between trials, a probe that passed the wrong draft and seed to the oracle, and a scratch script
  whose `cd` walked out of the repo so the suite's exit code came from a failed `call`. When a device
  and an oracle disagree, attribute it before reporting it, and say "unattributed" rather than picking
  the side that is easier to describe.
- A failing test committed "because it documents a finding". A test that fails for an unattributed
  reason is worse than no test; the finding belongs in `docs/active-work.md` with the reproduction.
- A correction to a claim that has already been pushed. Do not rewrite published history — add a
  commit that records the correction, and correct the comments and gates that carry the claim, since
  those are what the next session reads.

## 7. Output

```
## Review: [scope]

### Summary
[1-3 sentences: scope, risk, recommendation]

### Critical (must fix)
- **[title]** — [file:line] what is wrong, why it matters, how to fix

### Warnings
- **[title]** — [file:line] ...

### Suggestions
- **[title]** — [file:line] ...

### Contract compliance
Numerics / ownership / state lifetime / execution / artifact / external: PASS or the violation.

### What's good
- [always include]
```

Every finding states what is wrong, why it matters, and how to fix it. Never bury a numerical or
memory-safety bug under style notes.
