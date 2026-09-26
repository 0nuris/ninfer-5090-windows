# DFlash2 NVFP4 drafter cluster — what the merge actually revealed

**Status:** open decision. Written 2026-09-26 after the upstream `e31bc99b` merge.
**Authority:** this file records measurements and comparisons; it proposes, it does not decide.

## The question

The merge with upstream `e31bc99b` (15 commits, NVFP4/FP8/Q4-Q8/BF16 template unification) left the
port's DFlash2 NVFP4 drafter cluster in conflict. The cluster is five files:

| file | what it is |
|---|---|
| `src/ops/linear/nvfp4/nvfp4_gemv.cuh` | port's A16 GEMV kernel |
| `src/ops/linear/nvfp4/nvfp4_output.cuh` | `Nvfp4SplitOutput3`, `Nvfp4IdentityEpilogue` |
| `src/ops/attn_input_proj/nvfp4/nvfp4_dflash2_attn_input.cu` | fused q/k/v attention projection |
| `src/ops/linear/nvfp4/shapes/dflash2.cu` | five NVFP4 shape registrations |
| `src/ops/linear/nvfp4/nvfp4_dflash2_geometry.h` | five geometry aliases + small-T bounds |

plus the port's `nvfp4_simt.cuh` (360 lines) and `nvfp4_config.h` (which upstream deleted).

## What was measured, not inferred

### 1. The five shapes are required by our shipped artifact

`out/qwen3_8_27b_nvfp4full.v3.ninfer.conversion.json`, `methods` array (1082 entries), filtered to
the `dflash2` draft:

| shape | entries | format |
|---|---|---|
| `[5120, 25600]` | 1 | nvfp4 |
| `[6144, 5120]` | 5 | nvfp4 |
| `[5120, 4096]` | 5 | nvfp4 |
| `[1280, 5120]` | 10 | nvfp4 |
| `[256, 5120]` | 1 | nvfp4 |

These are `kNvfp4DFlash2{Feature,Qkv,AttnOut,ConvProj,Selector}`. Upstream's registry
(`nvfp4_geometry.h`) has exactly five geometries and **none of them is a drafter shape**. So the
five shapes are a real requirement of the port's artifacts, not an older-format relic.

### 2. Every part except the shapes already exists upstream

| port's cluster | upstream |
|---|---|
| `nvfp4_gemv.cuh`: `Nvfp4CodePack` L19 | `nvfp4_a16_gemv.cuh`: `Nvfp4CodePack` L20 |
| `nvfp4_gemv.cuh`: `load_nvfp4_codes` L29 | `nvfp4_a16_gemv.cuh`: `load_nvfp4_codes` L30 |
| `nvfp4_gemv.cuh`: `Nvfp4GemvSharedStorage` L53 | `nvfp4_a16_gemv.cuh`: `Nvfp4A16GemvSharedStorage` L54 |
| `nvfp4_gemv.cuh`: `nvfp4_scale_offset` L88 | `nvfp4_a16_gemv.cuh`: `nvfp4_scale_offset` L90 |
| `nvfp4_gemv.cuh`: `load_nvfp4_coefficients` L116 | `nvfp4_a16_gemv.cuh`: `load_nvfp4_coefficients` L118 |
| `nvfp4_gemv.cuh`: `nvfp4_gemv_kernel` L205 | `nvfp4_a16_gemv.cuh`: `nvfp4_a16_gemv_kernel` L250 |
| `nvfp4_simt.cuh` (360 lines) | `nvfp4_a16_simt.cuh` |
| `nvfp4_output.cuh`: `Nvfp4IdentityEpilogue` | `linear/common/epilogue.cuh`: `LinearIdentityEpilogue` |
| `nvfp4_output.cuh`: `Nvfp4SplitOutput3<4096,1024>` | see below — the mechanism exists, the *route* does not |
| `nvfp4_config.h`: scale enums, schedule templates | `nvfp4_layout.h`, `nvfp4_schedule.cuh` (`Nvfp4A16*` names) |

Upstream also has the structural model for the fused route: `q8/q8_dflash2_attn_input.cu` is the
same design — `LinearBf16SegmentedOutput<kQueryRows, kKvRows, kKvRows>`, exact-launcher array for
small T, tile launchers above.

**Verdict: the cluster is a pre-upstream duplicate of routes upstream has since written.** It is not
"our implementation" in the sense the rule means — upstream's base covers it.

### 3. The one genuine difference: template parameterisation

Upstream's kernels already accept an output policy and epilogue:

```cpp
template <class Schedule, class Output, class Epilogue, class Rows>
__global__ void nvfp4_a16_gemv_kernel(x, codes, scales, alpha, Output output,
                                      Epilogue epilogue, Rows policy, int rows);
```

The port's kernel instead takes `<Geometry, ActiveTokens, Schedule, Epilogue, OutputPolicy, ...>`
and `(x, codes, scales, divisor, epilogue, output, columns)`.

Differences that make this a port-forward, not a rename:

- geometry is a template parameter in the port, derived from `Schedule::kStaticK` upstream;
- upstream adds a `Rows policy` argument;
- argument order differs (`output`/`epilogue` swapped);
- upstream's launch wrappers (`nvfp4_linear_a16_gemv/_simt`) hardcode `LinearBf16Output` +
  `LinearIdentityEpilogue`, so a three-output route must call the kernel-level template directly,
  exactly as `q8_dflash2_attn_input.cu` does.

## The route is genuinely absent upstream — verified

The port's NVFP4 drafter route is the port's own addition. The decisive evidence is the public
header diff, where the `-` lines are upstream's and the `+` lines are ours:

```
- * Three-output Q8 specialization. The Q8_G32_FP16 RowSplit parent stores rows in order
- * [query 4096, key 1024, value 1024]. Registered parent forms are [6144,2048] with BF16
- * x [2048,T] for the Qwen3.6 companion and [6144,5120] with BF16 x [5120,T] for DFlash2.
+ * Three-output specialization. The parent stores rows in physical order [query, key, value].
+ * Registered parent forms are the Q8_G32_FP16 RowSplit matrices [6144,2048] ... [6144,5120]
+ * ... for DFlash2, and the weight-only NVFP4 BlockScaleK16M128x4 matrix [6144,5120] with BF16
+ * x [5120,T] for the NVFP4-encoded DFlash2 module. ... (the NVFP4 route serves extents above
+ * its fused small-T family in 32-token chunks).
```

Upstream's three-output drafter route is **Q8 only** (`q8/q8_dflash2_attn_input.cu`). Upstream has:

- no NVFP4 drafter attn-input route — `git grep 4096 upstream/dev -- src/ops/attn_input_proj/nvfp4`
  returns nothing;
- no `Nvfp4Geometry<6144,5120>` — its five registered shapes are `[14336,5120]`, `[16384,5120]`,
  `[34816,5120]`, `[5120,6144]`, `[5120,17408]`. Its `6144` hits are
  `Nvfp4Activation6144Geometry` (rows to quantise) and `N5120K6144` (5120x6144, the other
  orientation).

### Why upstream does not need it

Upstream's published nvfp4 artifact does ship `dflash2` — `artifact-manifest.json` for
`neroued/Qwen3.8-27B-nvfp4-NInfer` lists components `text, vision, mtp, dflash2`, 112 nvfp4
tensors, bindings 1513, uses 844. Its drafter simply is not NVFP4-encoded. This repo's own loader
says so:

```cpp
// src/models/qwen3_5/load/dflash2.cpp:25
// Bound without an exact format: the official artifact stores BF16
// codebooks while the QUASAR checkpoint stores NVFP4 ones, and the
// selector dispatches on the resolved weight's qtype.
```

So the split is by checkpoint, not by engine:

| artifact | drafter format | served by |
|---|---|---|
| upstream's official nvfp4 | BF16 drafter | upstream's generic BF16 attn_input |
| this port's QUASAR nvfp4 | **NVFP4 drafter** | the port's cluster, exclusively |

The port's converter is behaviourally identical to upstream's — the 7-line diff in
`tools/convert/qwen3_5.py` is only `dict[str, Any]` annotations required by the port's `mypy`, with
no change to any format mapping — so this is a property of the source checkpoints, not of two
different conversion policies.

**Conclusion: the cluster is a genuine port requirement with no upstream counterpart.** It is the
"our implementation" case, not a divergence from an upstream design that already covers it.

## What upstream does share

| port's cluster | upstream |
|---|---|
| `nvfp4_gemv.cuh`: `Nvfp4CodePack` L19 | `nvfp4_a16_gemv.cuh`: `Nvfp4CodePack` L20 |
| `nvfp4_gemv.cuh`: `load_nvfp4_codes` L29 | `nvfp4_a16_gemv.cuh`: `load_nvfp4_codes` L30 |
| `nvfp4_gemv.cuh`: `Nvfp4GemvSharedStorage` L53 | `nvfp4_a16_gemv.cuh`: `nvfp4A16GemvSharedStorage` L54 |
| `nvfp4_gemv.cuh`: `nvfp4_scale_offset` L88 | `nvfp4_a16_gemv.cuh`: `nvfp4_scale_offset` L90 |
| `nvfp4_gemv.cuh`: `load_nvfp4_coefficients` L116 | `nvfp4_a16_gemv.cuh`: `load_nvfp4_coefficients` L118 |
| `nvfp4_gemv.cuh`: `nvfp4_gemv_kernel` L205 | `nvfp4_a16_gemv.cuh`: `nvfp4_a16_gemv_kernel` L250 |
| `nvfp4_simt.cuh` (360 lines) | `nvfp4_a16_simt.cuh` |
| `nvfp4_output.cuh`: `Nvfp4IdentityEpilogue` | `linear/common/epilogue.cuh`: `LinearIdentityEpilogue` |
| `nvfp4_config.h`: scale enums, schedule templates | `nvfp4_layout.h`, `nvfp4_schedule.cuh` (`Nvfp4A16*` names) |

So the port's cluster re-implements kernels upstream now has, in order to serve a route upstream
does not have. That is duplication worth removing on its own terms, and it is also the current
build break: the port's `nvfp4_gemv.cuh` and upstream's `nvfp4_a16_gemv.cuh` both define
`Nvfp4CodePack` and `load_nvfp4_codes`, and both headers are now included in one translation unit.

`ops/linear/nvfp4/nvfp4_launch.cuh` is the file the merge replaced, and it held the two port-only
helpers the drafter shapes need — `select_nvfp4_exact` and `launch_nvfp4_a16_chunks` — neither of
which upstream has.

## A merge loss, not a merge conflict

`src/ops/wrapper/attn_input_proj.cpp` is byte-identical to `upstream/dev` after the merge, which
initially read as "the port never had a change here". It did. The diff against the pre-merge state
shows the port's NVFP4 three-output dispatch being removed with no conflict raised:

```
-#include "ops/linear/nvfp4/nvfp4_config.h"
+#include "ops/linear/nvfp4/nvfp4_layout.h"
-    if (query_key_value_weight.qtype == QType::NVFP4) {
-        if (hidden != 5120 || query_key_value_weight.n != kRows ||
-            query_key_value_weight.k != hidden) {
-            throw std::invalid_argument("attn_input_proj: unsupported NVFP4 Q/K/V profile");
-        (void)detail::validate_nvfp4_weight(query_key_value_weight, "attn_input_proj");
-        detail::nvfp4_dflash2_attn_input(x, query_key_value_weight, q, k, v, stream);
```

Upstream's version of that file won the region silently, leaving `nvfp4_dflash2_attn_input` defined,
declared and registered but unreachable from the public Op. The symptom was misleading: the route
oracle failed with `attn_input_proj: invalid query/key/value weight`, which reads like a malformed
test weight rather than a missing dispatch branch — the Q8 `require_q8_rowsplit` check was rejecting
a perfectly valid NVFP4 weight because nothing branched on `qtype` first.

**Rule this earns:** after merging a large upstream refactor, a file being *identical to upstream* is
not evidence that the port had no changes there. `git diff <pre-merge-tag> -- <path>` is the check
that distinguishes "never differed" from "differed and lost", and it is cheap. Comparing only
against `upstream/dev` cannot tell those two apart.

## The second gap: the drafter's MLP — resolved

With the attention route restored, `ninfer_qwen3_5_dflash2_real_test` advanced past the shapes and
the projection and then failed at:

```
nvfp4 linear_swiglu A16 is registered only through T=16
```

`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp` is upstream's and byte-identical before
and after the merge. Its A16 policy serves `T == 1` (decode) and `T <= 16` (small-T fused). The
drafter calls SwiGLU with `T = (k+1) * batch`, which at the test's defaults (`k = 15`, `batch = 8`)
is **128**.

The same asymmetry as the attention route, one Op over: upstream has
`src/ops/linear_swiglu/q8/q8_dflash2_linear_swiglu.cu`, bounded at `T <= 40` with no chunker, and
no NVFP4 equivalent. Upstream never reaches the limit because its drafter is BF16 and goes through
the BF16 SwiGLU, which has no such bound.

This was the same merge loss as the dispatch branch, found the same way. Pre-merge:

```cpp
if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
return Nvfp4LinearSwiGluRoute::LinearA16Post;      // wide-T A16
```

Upstream's version ends that branch with the `throw`. Restored as `LinearA16Post`, leaning on
upstream for the heavy part: the gate/up projection runs through the registered `[34816,5120]`
NVFP4 shape — one of upstream's five — and only the elementwise `silu_mul` over the projected
halves is composition. `nvfp4_linear_swiglu_workspace_capacity_bytes` also had to change: it
returned 0 for A16, which would have under-reported the `[34816,T]` BF16 buffer the route
materializes (8.9 MB at T=128).

**Cost, stated rather than hidden:** this route is not fused. It writes and re-reads `[34816,T]`
BF16 where the small-T route stays in registers. That is real memory traffic on a path that runs
every decode step, and it has not been measured. A fused wide-T A16 SwiGLU is the obvious
follow-up; it was not written because restoring the proven path is the lower-risk change and the
alternative is new numerics with no oracle of its own.

**Coverage gap, stated rather than implied:** there is no dedicated NVFP4 SwiGLU oracle. This route
is covered only by `ninfer_qwen3_5_dflash2_real_test` reaching T=128 on the real artifact.

## The options

**A. Port-forward the cluster onto upstream's kernels.** Keep the five shapes, the geometry aliases
and the chunk/select helpers; delete `nvfp4_gemv.cuh`, `nvfp4_output.cuh`, `nvfp4_simt.cuh`,
`nvfp4_config.h`; rewrite `nvfp4_dflash2_attn_input.cu` against `nvfp4_a16_gemv_kernel` /
`nvfp4_a16_simt_kernel` with `LinearBf16SegmentedOutput` + `LinearIdentityEpilogue` + a row policy.
*Cost:* one careful rewrite of a numerical kernel passage where a mistake is silent. *Result:* one
implementation, upstream's, no duplication, and the five shapes as the port's only addition.

**B. Keep the cluster as-is, namespaced.** Restore the port's headers under names upstream does not
use, so no duplicate definition. *Cost:* the port maintains ~500 lines of kernels upstream has.
*Result:* divergence from upstream's base on a route upstream also serves, which the standing rule
says to avoid unless ours is better — and nothing measured shows it is.

## Evidence available to decide

- `ninfer_qwen3_5_dflash2_real_test` — real artifact, exercises the drafter end to end.
- `ninfer_dflash2_nvfp4_routes_test` — the port's oracle over the NVFP4 routes.
- `ninfer_linear_nvfp4_a16_test` — upstream's A16 oracle, now carrying the port's drafter section.
- QUASAR corpus perplexity, recorded with a ±1% band in `docs/perplexity-baseline.md`.

## Recommendation

Option A. The rule is upstream's base unless ours is measurably better, and every piece of the
cluster has an upstream counterpart except the five shapes — which are then the entire port
contribution, small and testable. Option B keeps duplicated kernels for no measured gain.

## Corrections to earlier claims in this session

Five confident statements were each falsified by a single check. They are recorded because the
method error, not the individual claim, is what recurred.

| # | claim | falsified by | error |
|---|---|---|---|
| 1 | "upstream's own test requires these five shapes" | `git diff upstream/dev -- tests/ops/linear/test_nvfp4_a16.cpp` | the drafter block is a `+` side — the port's addition |
| 2 | "upstream's test fails on their HEAD" | `git show upstream/dev:tests/...` | their test never requests the shapes |
| 3 | "the cluster implements an upstream requirement" | `nvfp4_geometry.h` + `linear.h:77` | it implements a port requirement; upstream has the kernels, not the shapes |
| 4 | "the cluster is port-only, no upstream counterpart" | `git grep Nvfp4CodePack upstream/dev` | upstream's is `nvfp4_a16_gemv.cuh`; I searched for the port's *name*, not the content |
| 5 | "the segmented output has no upstream counterpart" | `nvfp4_attn_input_output.cuh` | the mechanism is upstream's; the *route* is genuinely absent, which is the real finding |
| 6 | "the wrapper is identical to upstream, so the port never touched it" | `git diff pre-merge-e31bc99b -- src/ops/wrapper/attn_input_proj.cpp` | the port's dispatch branch was there and the merge removed it silently |

Common cause in 4, 5 and 6: a name search where a content search was needed; a pathspec that reached
the working tree instead of the named commit; and comparing against `upstream/dev` where the
question was about the port's own history. `git grep <pattern> <rev>` **without** a pathspec,
`git show <rev>:<path>`, and `git diff <pre-merge-tag> -- <path>` are the forms that settled every
question here in one call.

Two further process errors this session, both costly:

- Reasoning about an *older* artifact format (the port's fused 6144×5120 qkv parent) as though it
  described the current one. The conversion report is the authority for what an artifact contains.
- Several claims were stated before a single check, then retracted. The required test
  (`ninfer_qwen3_5_dflash2_real_test`) would have adjudicated from the first hour, at the cost of
  one build.

## What is still open

- Whether upstream's nvfp4 drafter reaches its projection through a generic single-parent linear
  call rather than a three-output attn-input. If it does, the port's fused route is an optimisation
  of a path upstream also serves, and option A below becomes a straight port-forward with no
  numerical risk.
- Which of the five shapes the real-artifact drafter test actually requests. The build currently
  fails on the port's duplicated `Nvfp4CodePack`, so this has never been measured. It is one
  counter or one restored build away.

