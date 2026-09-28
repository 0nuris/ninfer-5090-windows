# DFlash / DFlash2 acceptance baseline: what upstream publishes, and where our 7-22% sits

Research note. Written 2026-09-28. Every number carries its provenance: **[paper]**,
**[model card]**, **[author blog]**, **[read from source]**, **[community]**, or
**[no evidence found]**. Claims about our own tree are **[read from source]** with file:line, or
**[measured here]** for the sweep the note was asked about.

**Headline:** upstream publishes acceptance numbers for exactly this model at exactly the
recommended width, and ours is **3.2x below them**. The discontinuity at one added column is a
*second, separate* problem: the route predicate the sweep attributes it to does not switch at that
column in either our tree or upstream `master` (§7). So the discontinuity is not yet explained, and
the first measurement to make is the per-position acceptance breakdown the bench already emits (§8).

---

## 1. DFlash (v1) published acceptance, with conditions

Source: Chen, Liang & Liu, *DFlash: Block Diffusion for Flash Speculative Decoding*,
arXiv:2602.06036v2 (ICML 2026), read in full at `arxiv.org/html/2602.06036v2`. **[paper]**

DFlash reports **acceptance length τ** — Eq. 1 defines τ ∈ [1, γ+1] as "the expected number of
accepted tokens per cycle, **including the bonus token** produced by the target model". That is the
same convention as our `acceptance_length` (§8), so the numbers are directly comparable.

| model / setting | condition | τ |
|---|---|---|
| Qwen3-8B, DFlash block 16, T=0 | Math/Code/Chat avg, Transformers backend, ≤2048 new tokens, H200 | 6.49 |
| Qwen3-4B, DFlash block 16, T=0 | same | 6.54 |
| Qwen3-8B, DFlash block 16, T=1 | same | 5.48 |
| LLaMA-3.1-8B-Instruct, DFlash block 10, T=0 | SGLang, FlashInfer, single B200; GSM8K / HumanEval / Alpaca | 4.32 / 4.91 / 3.73 |
| **Qwen3.5-27B, DFlash block 16, LongBench 8K** | hotpotqa / qasper / gov_report | **4.46 / 4.17 / 3.32** (base drafter) |
| Qwen3.5-27B, same, long-context fine-tuned | 8K | 5.76 / 5.62 / 4.04 |

The Qwen3.5-27B row (Table 4) is the closest published analogue to our lane: same target family,
8K context, and the paper notes the base drafter degrades past 4K context, which is consistent with
4.46 at 8K. Temperature is not stated for that table. **[paper]**

EAGLE-3 in the same tables sits at τ ≈ 2.8-3.7 (tree 16-60), so **τ ≈ 3 is roughly the
"speculation is working" floor** for a 1-layer autoregressive drafter at these model sizes.

### Per-position acceptance, the more diagnostic number

DFlash2's blog publishes conditional per-position acceptance for DFlash2 on Qwen3.5-4B, MATH-500,
T=0, block 16: **88.3% at position 0 decaying only to 86.5% at position 14** — essentially flat. For
DFlash 1 on Qwen3-4B, GSM8K, T=0, Recall@1 by position is 85.4% → 72.9% over 7 positions, and the
*oracle* over the top-16 candidates only falls 99.5% → 87.8%. **[author blog]**

The authors name the decline "suffix decay" and treat it as a backbone-capacity problem, fixed by
the two-tap dynamic convolution (position 6 Recall@1: 72.86% → 77.61%). **[author blog]**

This is the number our sweep should be compared against position by position, and the one we do not
currently report (§8).

---

## 2. DFlash2 published acceptance, with conditions

**There is no DFlash2 paper.** DFlash2 exists as (a) an Inco AI blog post and (b) a Hugging Face
model card. The vLLM/SGLang implementations cite the blog as the DFlash2 reference and pin Z Lab's
inference code as the architecture authority. I treat the model card and blog as primary *because
they are the authors' own artefacts*, not because they are peer-reviewed. **[no evidence found]** for
an arXiv paper or OpenReview entry for DFlash2.

Best-conditioned source: model card for `incoai/Qwen3.8-27B-DFlash2`, read in full. **[model card]**

> Runtime: SGLang on one NVIDIA H200, with FlashAttention 3 for target and draft attention.
> Speculation block size: 8 (7 draft tokens per verification step).
> Sampling: Qwen3.8's officially recommended parameters (temperature 1.0, top-p 0.95, top-k 20),
> with `xhigh` reasoning effort. Maximum new tokens: 4096.

> Acceptance length is the per-request mean of completion tokens divided by verification steps.

| task | native MTP (7 draft) | DSpark (7 draft) | **DFlash2 (7 draft)** |
|---|---:|---:|---:|
| GSM8K | 5.02 | 4.36 | **5.46** |
| MATH-500 | 4.72 | 3.92 | **5.28** |
| HumanEval | 3.91 | 3.30 | **4.39** |
| MBPP | 3.99 | 3.51 | **4.79** |
| MT-Bench | 3.74 | 3.01 | **4.10** |
| mean | 4.28 | 3.62 | **4.80** |

Concurrence-1 output throughput in the same run: autoregressive 68.9-69.0 tok/s, MTP 134.9-178.5,
**DFlash2 184.0-236.1**. **[model card]** The blog's Table 4 carries the same acceptance numbers and
adds the framing "over 20% more output from every verification pass, for around 1% added cycle
latency" (DFlash → DFlash2 = 4.92 → 5.97 mean on Qwen3.5-4B, +21%). **[author blog]**

**[no evidence found]** for DFlash2-on-Qwen3.8-27B acceptance at **more than 7 draft tokens**. The
27B drafter is published and evaluated at block size 8 only. The closest wider evidence is DFlash 1
at block 16 in §1.

---

## 3. Recommended / default draft width

Unambiguous, and all four sources agree on **block size 8 = 7 draft tokens** for this model:

| source | statement |
|---|---|
| Inco model card + blog | SGLang `--speculative-num-draft-tokens 8`; vLLM `"num_speculative_tokens": 7`; llama.cpp `--spec-draft-n-max 7`; oMLX runtime block size 5 |
| vLLM Speculators docs, `dflash2` page | "DFlash2 defaults to five draft layers, **block size 8**, `sample_from_anchor: False`" |
| our own upstream tracker, issue #188 | maintainer: `--spec dflash2 --draft-tokens 7 --lm-head-draft`; asked whether 7 or 15, answered "**7, 15 is only for fun**" (the 15 was a concurrency-8 demo) |

Two traps in this evidence:

- The `block size 5` in Z Lab's own README is **not** a quality recommendation. It is attached to the
  MLX 4-bit example and justified as "MLX's current quantized matmul kernel becomes less efficient at
  larger verify widths" — a latency reason. **[read from source]**
- `block_size 8` produces **7** speculative tokens, because `sample_from_anchor: False` means slot 0
  is the anchor and is not trained. So "block size 8" and "7 draft tokens" are the same
  configuration, and the *verify width* is 8 query columns. **[read from source, vLLM Speculators
  docs]**

DFlash 1 adds a relevant constraint: inference at or below the training block size generalises, the
reverse does not (§5.5.4 — a block-16 model evaluated at block 8 lands close to a natively trained
block-8 model, and block-8 models "fully accept entire blocks 35.7% of the time, suggesting that
block size 8 is often underutilized"). **[paper]**

---

## 4. Our measurement against that baseline

**[measured here]** — DFlash2 lane, NVFP4 27B, bf16 KV, 8192 context, 64 generated tokens,
optimised proposal head, CUDA graphs, batch 1:

| draft window | verify columns | route label in the sweep | round ms | acceptance | tokens/round | tok/s |
|---:|---:|---|---:|---:|---:|---:|
| 4 | 5 | non-chunked small-T | 17.727 | 22.06% | 1.882 | 106.19 |
| 5 | 6 | non-chunked small-T | 17.843 | 16.28% | 1.778 | 99.63 |
| 6 | 7 | chunked small-T | 18.636 | 7.45% | 1.455 | 78.05 |
| 7 | 8 | chunked small-T | 18.432 | 7.27% | 1.488 | 80.75 |
| 9 | 10 | chunked small-T | 18.977 | 4.06% | 1.362 | 71.76 |

**At the recommended width (7 draft tokens / 8 columns), against the recommended width upstream:**

| | tokens/round | vs ours |
|---|---:|---:|
| DFlash2, Qwen3.8-27B, H200/SGLang/FA3, block 8 **[model card]** | 4.10-5.46 (mean 4.80) | **2.8x - 3.7x higher** (3.2x vs mean) |
| native MTP, same run, same 7 draft tokens **[model card]** | 3.74-5.02 (mean 4.28) | 2.5x - 3.4x higher |
| DFlash 1, Qwen3.5-27B, 8K context, LongBench **[paper]** | 4.17-4.46 | 2.8x - 3.0x higher |
| ours, window 7 **[measured here]** | 1.488 | — |

Per-position, the aggregate 7.27% at window 7 decomposes to a marginal per-draft-position acceptance
of 0.488/7. Upstream's *conditional* per-position figure for DFlash2 is 85-88% (Qwen3.5-4B, MATH-500,
T=0) and 73-85% (Qwen3-4B, GSM8K, T=0). **[author blog]** Ours is 4x to 20x lower in kind, on a
different target, so it is not a like-for-like ratio — but nothing in the published range is within an
order of magnitude of 7%.

**The round time is not the problem, and neither is the GPU.** Decomposing the window-7 row against
the model card's GSM8K cell (236.1 tok/s, 5.46 tokens/round → 43.2 ms/round): our round is **2.35x
faster** (18.4 ms) while our tokens/round is **3.67x worse**. A 5090 beating an H200 on a short-context
4-bit verify is unremarkable — that round is weight-bound, not KV-bandwidth-bound at 8K. The two
errors therefore nearly cancel in tok/s (our 80.75 sits within 1.26x of the 64.3 tok/s we would post
at H200's round time, and is still 2.9x under upstream's actual 236.1) and the quantity that is
wrong is tokens committed per round, not round cost.

**Caveats I am obliged to state.** (a) 64 generated tokens on one workload is a small sample of a
different distribution from upstream's 4096-token benchmark runs at `xhigh` reasoning effort; our
7-22% should not be treated as a settled baseline. (b) Our target is **NVFP4**; upstream's is
**BF16**. A 4-bit target's argmax differs from its BF16 self, and a BF16-trained drafter tracks the
BF16 self, so some loss is expected. §6 bounds how much that can plausibly account for. (c) Upstream's
window is 7; we have no published acceptance for 9 draft tokens on this target.

---

## 5. Is a chunked / split-KV verification path a known source of accuracy or acceptance problems?

**Mechanism: yes, it is a known and named class. Reported acceptance cliffs at a token-count
threshold: no.**

Sources for the mechanism, in order of how directly they bear on us:

1. **He & Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (10 Sep 2025)**,
   §"Batch-invariant attention", read in full. The split-reduction strategies used for attention
   "unfortunately also breaks batch invariance, as our precise reduction strategy depends on how many
   query tokens from the sequence we're processing in any given request"; specifically, "FlashInfer's
   'balanced scheduling algorithm' chooses the largest split-size that can still saturate all the
   GPU's cores, thus making the reduction strategy not 'batch-invariant'". The stated fix is a
   **fixed split-size** strategy — fix the size of each split and let the count vary — so the
   reduction order is identical regardless of query width. **[read from source]**
2. **FlashInfer attention API docs**: the `disable_split_kv` argument is documented as "Whether to
   disable the split-kv **for determinism** in CUDA Graph, defaults to `False`." The library ships a
   determinism switch for split-KV. **[read from source]**
3. **SGLang official docs, "Attention Backend"**, §"Hybrid attention": "The backend used for **draft
   decoding and target verification** depends on `--speculative-attention-mode`" — `decode`
   (recommended) or `prefill` (default). Shipped engines treat the verify pass's attention kernel as a
   separate decision from the decode pass's. **[read from source]**
4. **FlashInfer issue #3420**: "`qo_len=2` (MTP / spec-decode draft=1) dispatches to prefill kernel
   instead of decode kernel, 10x slower than FA on H20" — fixed by PR #3859, which routes spec-decode
   chunks with `q_len_per_req > 1` through the XQA decode path. A **query-width threshold selecting a
   different attention kernel inside a spec-decode verify step** is a real, reported defect class. It
   is a *performance* defect. **[community]**
5. **vLLM issue #49547**: with spec-decode on, vLLM silently downgrades `cudagraph_mode` to
   `PIECEWISE` (−16% measured) when the FlashInfer native decode path is selected, because it "is
   capped at `UNIFORM_SINGLE_TOKEN_DECODE`, which is below the `UNIFORM_BATCH` level required to keep
   FULL decode graphs under spec-decode". Again a width/level-dependent kernel-capability boundary in
   the spec path; again performance. **[community]**
6. **SGLang PR #32288**, "Fix stale flashinfer-MLA fallback poisoning spec verify capture": a
   mis-scoped mitigation had silently disabled a target attention path in the spec-verify CUDA-graph
   capture. Wrong kernel selection leaking into spec verify is a recurring failure mode. **[community]**
7. **Counter-evidence, and it matters.** The FlashInfer paper (arXiv 2502.04334) §D.2 argues that
   requests with short KV lengths bypass the split partial-output workspace entirely, "without
   significantly compromising numerical accuracy". **[read in a search-engine rendering of the PDF;
   re-verify from the PDF before relying on it]** Taken at face value, this says a *different number
   of splits* should not move results much. So a pure reduction-order change is a weak explanation
   for a 2x acceptance change, and a strong explanation requires that the chunked path computes
   something *different*, not merely a differently-ordered sum.
8. **[no evidence found]** for a reported speculative-**acceptance** cliff at a token-count threshold
   in FlashInfer, vLLM, SGLang or TensorRT-LLM. Every report above is performance, cudagraph
   capability, or a crash. The nearest thing is vLLM issue #47602, "Native MTP draft acceptance rate
   decays with total context" — a smooth dependence on context length, not a threshold. **[community]**
9. One report inside our own tracker does touch the *consequence*: in issue #53, a user sweeping
   DFlash1 against MTP3 on `qwen3.6-35b-a3b` on a 5090 across k=7..15 recorded 5 distinct
   completions across 10 configs on structured output and 10 on code and prose, attributed to
   "floating-point non-associativity across different draft batch shapes". So a draft-width change
   perturbing completions is already observed on our engine. Their tok/round at k=7 was **6.86**,
   climbing monotonically to 10.81 at k=15 with no discontinuity. **[community]**

---

## 6. Does verification-path precision matter at the 2x-acceptance magnitude?

**[no evidence found]** for any paper or issue quantifying a change of that size in acceptance rate
attributable to verification-path precision. What exists bounds it in two directions.

**Bounding consequence, upward.** Thinking Machines' own experiment, read in the same post: 1000
greedy completions from `Qwen3-235B-A22B-Instruct-2507` on one prompt produced **80 distinct
completions**, all identical for the first 102 tokens, with 992 runs taking "Queens, New York" and 8
taking "New York City". That is ~0.8% of greedy decisions flipped by a reduction-order violation at
one model and one prompt — a data point on consequence size, not a bound. It is far too small to
produce a 2x move in an aggregate acceptance rate *unless* the accept/reject decision sits near the
boundary for a large fraction of positions.

**Bounding the quantized-target excuse.** QSpec (arXiv:2410.11305v3, EMNLP 2025) reports that even at
γ=6 "the token acceptance rate remains relatively high, approximately **74%**, compared to
28-58% ... in conventional speculative decoding" pairs — that is, a **4-bit** target with a drafter
aligned to it still accepts three quarters of its proposals. **[read in a search-engine rendering of
the HTML; re-verify from the PDF]** So 4-bit target weights are not by themselves an acceptance sink,
and NVFP4 cannot plausibly explain a 3.2x gap on its own.

**Reading.** The three candidate explanations, ranked by what the evidence supports:

1. **Something is actually wrong** in the lane — either a target verification path or a drafter path
   or the target artifact's own quality (§7, `Skylux70`). The published per-position profile for
   DFlash2 is *flat*; ours falls off a cliff at one added column. A flat-vs-cliff shape is not what a
   precision or quantization effect produces.
2. **A missing drafter-side feature.** Not sufficient on its own, but it bounds the space: the
   DFlash2 selector is worth +0.34 τ on a 7-position block and the convolution moves position-6
   Recall@1 from 72.86% to 77.61%; the whole DFlash→DFlash2 delta is +1.05 τ (21%) on Qwen3.5-4B.
   **[author blog]** A lane missing both would be at DFlash-1 level, i.e. ~4.9, not 1.5.
3. **NVFP4 target / drafter mismatch.** Real, unquantified here, and bounded by QSpec above.

---

## 7. Upstream findings on DFlash2 specifically, in our own tracker

`Neroued/ninfer` issue **#188** "DFlash2 support is now available for Qwen3.8-27B" (OPEN, 24 comments)
is the primary DFlash2 channel. The maintainer's measurements are throughput at `--draft-tokens 7`
with int8 KV, not acceptance. What the thread does contain:

- **`Skylux70`: output-quality regression on the v2 NVFP4 artifact, unresolved.** "The new model
  completely took a massive dump on my tests, where the output quality has effectively degraded back
  to Qwen 3.5 levels. **Even when disabling DFlash and falling back to MTP, the intelligence drop
  persists.**" The maintainer replies "I only added dflash2 weights, nothing else changed :(". The
  reporter retests on `latest main` and confirms, and stays on v1. Last comment on the thread; no fix
  or explanation recorded. This is a report about the **target**, on the exact artifact family our
  lane uses. **[community, unresolved]**
- **`young-developer`: 197.2 tok/s mean decode** over 104 completed requests, `qwen3.8-27b/nvfp4`,
  `--spec dflash2 --draft-tokens 7`, `kv-dtype fp8`, single-request decode (batch 1), range
  150.0-297.1, median 190.6 — parsed from `req#N done` lines. Same artifact family, same draft width,
  same engine, batch 1. **Against our 80.75 tok/s at window 7, that is 2.4x.** Sampling mode and
  thinking setting are not stated. **[community]** This is the closest apples-to-apples external
  reference I found, and it corroborates that our lane is slow on this path, not that the method is.
- **`knoopx`: "z-lab 3.8 dflash2 weights seem to work on 3.6 too, acceptance rates are not great but
  still outperforming mtp3: 3.8 dlash2 on ThinkingCap-Qwen3.6-27B @ 11 draft tokens"**, with a
  screenshot. **I did not read the numbers in the image**, so the magnitude is unverified; the
  qualitative claim that acceptance is "not great" while still beating MTP is the only part I can
  carry. **[community, magnitude unread]**
- `potatohog`, `Doelfke`, `cometkim` (power-limit caveat) report it working well; `pkochubey`'s
  tool-loop failures were traced to the chat template, not the drafter.

Issue **#279** "ChunkedSmallT re-streams full KV per query chunk (DFlash K≥8 verify at long context)"
(OPEN) is the only upstream discussion of the chunked path in a DFlash context. It is a **throughput**
report: `launch_chunked_small_t` re-sweeps the full visible key range once per query chunk, so W=9-12
costs two KV sweeps where W≤8 costs one; measured 1.75x-1.96x on the attention microbenchmark. It
states: "**Current published configs (DFlash/DFlash2 K=7 → W=8) never trigger it**", and that
acceptance-vs-K at long context was never measured because no local artifact carried a DFlash draft
component. The proposed actions are a guardrail note and an adaptive K-cap; **no accuracy concern is
raised anywhere in it**. **[read from source, issue text]**

---

## 8. The predicate the discontinuity is attributed to does not switch at that column

**[read from source]** — the sweep attributes the 5→6 column step to "our code selects the chunked
path when `columns > 6` at batch size 1". That does not match the code, in our tree or in upstream
`master`.

`causal_attention_resolve_route` is the **only** SmallT/ChunkedSmallT/Prompt resolver in the tree
(`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:345`, declared in
`launch.h:15,31`, called from `:422`, `:471`, `:501`). Its batch-1 boundaries are:

- `q_heads == 24` (line 348-371): prompt-route short-circuit, then
  `return width <= 8 ? SmallT : ChunkedSmallT;`  ← **:370**
- `q_heads == 16` (line 372 onward): `if (width <= 6) return SmallT;`  ← **:372**, then
  `if (batch_size > 1) return ChunkedSmallT;`, then a `q_heads == 16`-gated chunked branch, else Prompt.

`out/qwen3_8_27b_nvfp4full.v3.ninfer.conversion.json`, `components.text.config`, gives
`num_attention_heads: 24`, `num_key_value_heads: 4`, `head_dim: 256` — and `require_causal_geometry`
(`:37-43`) admits exactly `(24,4)` and `(16,2)`. **So this artifact takes the `q_heads == 24`
branch**, where the batch-1 boundary is **width 8**, not 6. Widths 5, 6, 7 and 8 are all `SmallT`;
`ChunkedSmallT` first becomes reachable at **width 9** (= 8 draft tokens). Widths 7-8 at batch > 1
fall through to `Prompt`, not `ChunkedSmallT`.

`git show upstream/master:src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`
is byte-identical in this region — same line 370, same line 372. (Upstream `dev` has since
restructured the file and no longer contains the route enum at all, so `master` is the right
comparator.) The `width <= 6` predicate is upstream's own q16 threshold, and note that
`causal_attention_chunk_tokens` at `:28` likewise returns 6 only for `q_heads == 16`.

**Consequence.** Three of the five sweep rows (7, 8, 10 columns) are labelled with a route that the
resolver does not select at those widths, and only the width-10 row can be `ChunkedSmallT` on this
artifact. So the discontinuity at verify width 7 has **no established route change behind it**, and
treating it as a chunked-path defect is currently unsupported. Either the sweep's route label came
from a different revision, a different op, or `columns` is not the attention query width — and the
engine emits the route name itself via `causal_attention_route_name` (`:382`), so the run's own log
settles it. Do not instrument a branch before confirming it ran.

**Where a width-7 cliff could instead come from, from reading the code:**

- The 27B is **hybrid**: 48 `linear_attention` (GDN) and 16 `full_attention` layers. The GDN layers
  process the same multi-column verify width through different machinery, including the three-way
  row partition `CausalConvSplitOutput3<Rows0,Rows1,Rows2>` in
  `src/ops/kernel/causal_conv1d.cuh:54-69`, whose comment notes "Callers writing pairs pass halved
  row counts, which requires every boundary to be even" — a shape-keyed row split is exactly the kind
  of thing that mis-slices at an unexpected width.
- The DFlash2 drafter is a different geometry again (32 q heads / 8 kv heads, head_dim 128, five
  `sliding_attention` layers, `sliding_window: 2048`, `conv_kernel_size: 2`,
  `selector_top_k: 16`) and is **not** admitted by `require_causal_geometry`, so the drafter has its
  own width-dependent dispatch that this route resolver says nothing about.

Both are hypotheses from reading structure, not diagnosed mechanisms. Neither has been tested.

---

## 9. The measurement to make next, and the instrument already in the tree

Upstream's decisive artefact is the **per-position conditional acceptance profile** (blog Figure 5
table; vLLM publishes the same as `accepted_tokens_per_pos_lists` / per-position `acceptance_rates` in
its spec-decode metrics, `mean_acceptance_length = 1 + num_accepted_tokens / num_drafts`).

**We already emit it.** `accepted_per_position` is a `std::vector<std::uint64_t>` in
`include/ninfer/types.h:760`, is populated in `bench/models/qwen3_5/dflash_round_bench.cpp:308-314`,
is printed by the round bench at `:354-356`, by `ninfer_bench` as
`speculative.accepted_per_position` (`bench/inference/ninfer_bench_support.cpp:249-253`), and by the
CLI at `apps/cli/main.cpp:218-222`. Our `acceptance_rate` is `accepted_tokens / drafted_tokens` and
`acceptance_length` is `1 + accepted_tokens / rounds` (`:235-248`) — the same convention as vLLM's
and as DFlash's Eq. 1, so the cross-source comparison in §4 is like-for-like.

The sweep in §4 did not report it. At the recommended width 7, the profile decides the diagnosis:

- positions 0..6 all low, roughly flat → a drafter or target-wide problem (missing selector/conv,
  wrong drafter weights, or the `Skylux70` target-quality report). Nothing to do with a kernel path.
- positions 0..5 normal and position 6 collapsing → a genuine width-boundary defect, and the width-7
  verify pass is where to look.
- a cliff that moves when `columns` is held fixed and only the *drafter* width changes → the drafter's
  dispatch, not the target's.

Pair it with the route name the engine already prints for each round, so the branch question is
answered by observation rather than by reading the predicate back.

---

## 10. Sources

| source | what was taken from it |
|---|---|
| `arxiv.org/html/2602.06036v2` (DFlash, ICML 2026) | τ tables 1-5, Table 4 (Qwen3.5-27B @ 8K), §3.1 τ definition, §5.5.4 block-size generalisation |
| `huggingface.co/incoai/Qwen3.8-27B-DFlash2` (model card) | DFlash2 τ by task, MTP/DSpark baselines, all evaluation conditions, conc-1/8/32 throughput |
| `inco.ai/blog/dflash2/` (authors' blog, 18 Aug 2026) | Table 1-5, Figure 2 and Figure 5 per-position tables, selector/conv deltas, "suffix decay" |
| `docs.vllm.ai/projects/speculators/.../dflash` and `/dflash2/` | `sample_from_anchor` semantics, block size 8 default, `dflash/model.py` pin, DFlash2 has no paper |
| `github.com/z-lab/dflash` README | supported checkpoints, launch configs, the block-size-5-for-MLX caveat |
| `thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference` | split-KV / batch-invariance mechanism, FlashInfer balanced scheduling, fixed-split-size fix, 80/1000 completions |
| `docs.flashinfer.ai/api/attention.html` | `disable_split_kv` documented "for determinism" |
| `docs.sglang.ai` advanced_features/attention_backend.html | `--speculative-attention-mode` selects the draft/verify attention backend |
| FlashInfer #3420 / PR #3859; vLLM #49547, #47602; SGLang PR #32288 | width/level-dependent kernel selection in spec-decode; no acceptance-cliff report |
| FlashInfer paper arXiv:2502.04334 §D.2 | split-K partials "without significantly compromising numerical accuracy" (via search rendering — re-verify) |
| QSpec arXiv:2410.11305v3 | ~74% acceptance at γ=6 with a 4-bit target (via search rendering — re-verify) |
| `Neroued/ninfer` #188, #53, #279 | recommended width 7; 197.2 tok/s external reference; unresolved NVFP4 target-quality report; the chunked-path throughput issue |
| our tree | `causal_softmax_attention.cpp:25-43,345-389,422,471,501`; `launch.h:15,31`; `causal_conv1d.cuh:54-69`; `types.h:760`; `ninfer_bench_support.cpp:235-253`; `dflash_round_bench.cpp:308-314,354-356`; `apps/cli/main.cpp:218-222`; the artifact's `components.text.config` and `components.dflash2.config`; `git show upstream/master:…causal_softmax_attention.cpp` |
