# Active work — 2026-09-28

**This file is temporary.** It is the single current record of work in progress. Per `AGENTS.md`,
remove it when the list is empty; do not grow it into a roadmap and do not add a parallel `v2`.

Everything here is measured on this product or read from source in this repository. Claims carried
over from other projects are marked, and several were checked against this tree and found not to
apply.

Context for all of it: `upstream/dev` was merged through `a012e2bc` (`beb46d83`), rewriting 88 files
of `src/ops/softmax_attention`. The merge touched **nothing** under `src/artifact`, `types.h`,
`round_buffers.h/.cpp` or `tools/convert`, and no container version changed, so **the four shipping
artifacts do not need rebuilding**. What the merge invalidated is the recorded performance table.

---

## Open, in order

### 1. Perplexity regression on a real artifact, post-merge — **DONE 2026-09-28**
**Why:** 88 attention files were rewritten wholesale. The suite passes, but the suite does not check
output *quality* on a real model, and nothing had confirmed quality survived the rewrite.
**Result:** all four shipping artifacts re-measured, full corpus, fp8, 1,044,876 tokens. QUASAR
**4.997441** against a recorded **4.99744** — identical to every digit. The reorganisation is
output-neutral, which is a stronger result than the +/-1% band required. The other three lanes moved
+0.15 % to +0.31 %, but their baselines predate the earlier `e31bc99b` merge and QUASAR's does not,
so that drift belongs to the older baseline and must not be cited as an attention regression without a
same-day control. Recorded in `docs/perplexity-baseline.md`.

### 1b. Can a full-NVFP4-coverage artifact be built from NVIDIA's source?
**Why:** the `nvfp4full` lane is built from `Qwen3.8-27B-NVFP4-unsloth`, a community quantization,
and scores **5.002854**. The `nvfp4nvidia` lane is built from NVIDIA's ModelOpt output — which is the
official stock, its 4.90168 sitting against the official 4.90169 on the same protocol — and scores
**4.915181**. That is a **1.75 % quality gap** on the same protocol, on the same nominal model.
The lane's name appears to describe a format property rather than a source, so the community
checkpoint was probably chosen deliberately as the one permitting complete NVFP4 coverage. That
reasoning is written down nowhere and has not been tested.
**Done when:** we know whether NVIDIA's checkpoint admits the same complete coverage. If it does, a
lane built from it is strictly better than the one we ship under that name. If it does not, the
current split is correct and the reasoning gets written down so the next reader does not have to
rediscover it.

### 2. Give the local NVFP4 encoder a scale search instead of max scaling - **ANSWERED 2026-09-28, negative**
Built as `nvfp4_mse` (`900a0f78`, reverted `b891e1e9`), wired into the 128 object groups
`qwen3_8_27b_nvfp4_nvidia` re-encodes locally, and measured paired against the shipping artifact in one
window on the full corpus. **Max-abs 4.915181334, searched 4.925917194 — +0.218 %, the wrong way.** The
two builds share a `prefill_signature`, a format set, an `execution` block and a corpus, so only weight
values moved, and the max-abs run reproduces the recorded 4.915181 to every digit. Weight
reconstruction error fell 40-66 % while perplexity rose, which is the finding: per-weight MSE and
output perplexity are decoupled, because the blocks the search improves are the ones it clips, and the
clipped value is carrying signal.

The premise that motivated this was misattributed, and that is the more useful half. The 193
`calibrator=NVFP4MSECalibrator` sites in the source's `.quant_summary.txt` are 192 `mlp.*` plus
`lm_head` — which `qwen3_8_27b_nvfp4_nvidia` already **imports verbatim**, producer codes and scales
included. The 128 object groups re-encoded here are the `self_attn` (128) and `linear_attn` (288)
sites, which the producer quantized as **FP8** with plain `MaxCalibrator`. The two sets are disjoint:
there was never a searched-scale artifact to match for the weights this experiment changed.
`hessian` and `local_hessian` appear zero times, so the producer did use the plain squared-error path
and the objective was the right one to test. What survives is that **max-abs is the right default for
weights with no producer evidence behind them.** Full write-up, the grid difference from ModelOpt's 126
log-spaced candidates, and what the result does and does not scope, in
`docs/perplexity-baseline.md`. The instrument is recoverable from `900a0f78`; the tree carries no
unused method.

**The activation scale is the better-motivated axis, and it is untouched — item 10.**

**Carry forward:** any converter change justified by lower weight error is measured on perplexity
before it is believed. This is the same rule as "a cost you can compute is not a cost you have
measured", pointed the other way.

### 3. The W8 endpoints are 2.52 GiB, 14.3 % of everything bound
**Why:** both W8 endpoints are Q8, which is the right default and the only choice with a measurement
behind it. An NVFP4-endpoint build would return roughly 1.26 GiB — the same order as the whole spread
between lanes that reach 262144 context and one that does not.
**Done when:** an NVFP4-endpoint variant of one lane is built and measured, and the capacity difference
is recorded against the lane's current free memory.

### 4. The vision tower is quantized in all four artifacts and BF16 in all three sources
**Why:** ours alone; no source checkpoint quantizes it, and no source measures doing so. Estimated
around 600 MiB. **Perplexity cannot see it**, so it needs its own check.
**Done when:** the decision is measured rather than assumed -- either a quality check that shows
quantizing the tower is safe, or the tower is left BF16.

### 5. Recall@1 / Recall@16 / path-acceptance split
**Why:** the only diagnostic that discriminates three different root causes, and it needs no new
kernel. The drafter emits `frame.candidate_ids` and `scores` at `draft.cpp:351-355`, shape `[16,K,B]`.
Decompose per position:
- Recall@1 — the drafter's unary top pick
- Recall@16 — the target argmax anywhere in the 16
- path acceptance — what the selector actually commits

**Done when:** all three are reported per position on a shipping lane, and they point at one of:
healthy Recall@16 with collapsing path acceptance (selector); Recall@16 itself collapsing after
position 1 (backbone or conditioning); or Recall@1 low at position 1 (head or conditioning weak from
the first column).

### 6. Interleaved re-measurement of all eight profiles - **DONE 2026-09-28**
All eight re-measured in one interleaved window, three rounds, each round visiting every lane in a
rotated order. Every lane's three rounds span **1.5 % or less**, each returns **one digest** across all
three, and there is **no position effect** — position 0 and position 7 read the same. `profiles.py`,
both doc tables and the eight launchers' `REM` lines now carry the new figures.

**The cause was a harness defect, not the lanes, and not a contaminated card.** The 52 % spread that
invalidated the NVIDIA MTP5 row was reproducible to within 1 % across four independent server starts,
which is the opposite of contamination. The first full-length decode after a start is a transient: it
returns faster than every later identical request and **different, shorter text**, while requests 2..n
are byte-identical. On that lane it read 261.7 tok/s against 169.8 for requests 2-7. `measure_decode`
warmed up with a **16-token probe**, which does not reach the state the transient affects, so the
transient was averaged into the first measured run. Acceptance was contaminated the same way, and
because the transient accepts *better* (64.6 % against 32.7 %) it inflated that column too: the same
log reads 35.3 % over all records and 32.7 % without the first.

Fixed by discarding one full-length warmup, and by skipping it in `parse_spec_jsonl` because the engine
appends to its request log for the whole session. Both are in this commit. An earlier revision of the
table was **inflated on all eight lanes** — the largest correction is NVIDIA MTP5 at 228.3 → 165.7
tok/s and 56.4 % → 38.2 %, which is the row that was already suspected.

**What was not changed:** no lane, flag, context, draft depth or artifact. Only the four measured
fields moved in `profiles.py`, and the launchers were regenerated rather than hand-edited.

**Still open, and it is a real one:** the transient is a genuine first-request behaviour, not a
measurement artefact — a client's *first* request after a lane starts gets different, shorter text
from the same seed. It is invisible at temperature 0 (the greedy digest is stable from request 1) and
visible at the sampling temperature the bench uses. That is a serving behaviour question, not a
benchmark one, and it is not diagnosed.

### 7. k8v4 KV against the shipped fp8
**Why:** every profile ships `--kv-dtype fp8`. K8V4 is available as a third option behind one flag.
**No published evidence exists for k8v4.** The "within 0.08 % of BF16" figure that first prompted this
was traced to NVFP4-KV against FP8-KV on Qwen3.5-397B-A17B — a different model, a different baseline,
and not k8v4 at all. It is withdrawn. The test stands only as our own measurement to be made.
**Done when:** perplexity and decode acceptance are compared across bf16 / fp8 / k8v4 on one shipping
lane, interleaved.

### 8. MTP draft window 5 to 10
**Why:** our MTP lanes use 4 and 5. A NInfer fork raised MTP to 10 on the claim that longer proposals
emit more per round, and **published no tok/s gain for it.**
**Done when:** windows 4/5/8/10 are compared on one MTP lane, interleaved, with tok/s and acceptance
reported together.

### 9. ngram: the verify-tree integration
**Why:** `PromptLookup` is ported and tested (`c0da270e`); the integration is not built. It needs
`candidate_selector_tree`, `speculative_accept_tree_drafts`, `speculative_compact_columns`, tree-aware
GDN replay and tree-aware target attention, each with a host oracle.
**Expected value is low, and that is the finding, not a reason to skip it:** the selector's value is
inversely proportional to drafter strength, and our shipping DFlash2 lanes accept 52-67%. The +55%
reference was measured on a lane accepting 27%.
**Done when:** one DFlash2 lane is measured with and without the copy grafted, on copy-heavy traffic,
against our own baseline. **"Built, measured, and not shipped" is an acceptable outcome** and should be
reported as one.

### 10. The A4 activation divisor at the re-encoded attention sites
**Why:** this is the axis item 2 was aimed at by mistake, and it has a documented failure mode rather
than a plausible one. The 128 groups `qwen3_8_27b_nvfp4_nvidia` re-encodes are assigned
`activation_policy="AllowA4"`, and their `activation_input_divisor` is recovered from a source
`input_scale` the producer calibrated for **FP8** — `6 / input_scale` at an FP8 site against
`1 / input_scale` at an already-NVFP4 one (`official_recipes.py`, the `_activation_divisor` probe). A
divisor sized for 8-bit activations is not obviously roomy enough for 4-bit ones, and nothing here has
measured it.
**The producer's own source names this failure mode.** ModelOpt's `NVFP4ActHeadroomCalibrator` exists
because plain max calibration of the activation global scale "would drag the global scale up so far
that every other block's FP8 block scale falls below subnormal and flushes to zero — losing the whole
tensor to protect one value"; its default anchors to the 99.99th percentile and clips the rare blocks
deliberately instead. The source checkpoint shows exactly that shape: `mlp.gate_proj` records
`amax=[0.0047, 0.4219]`, a 90x spread between the smallest block and the tensor maximum.
**Expected gain:** unknown, and that is the point — it is the one quantization scale in these
artifacts with a documented way to be badly wrong and no measurement behind it.
**Done when:** the A4 divisors at these sites are compared against headroom-anchored ones on one
shipping lane, measuring perplexity and decode acceptance. Unlike the weight scale, the comparison
needs a calibration corpus, so it is a real cost and not a free-at-runtime change.

### 11. The sparse accept path is only oracle-checked at `top_k=1`, and it disagrees with the oracle above that
**Found 2026-09-28, while auditing a claim of mine that turned out to be false. Unresolved.**

`tests/ops/test_speculative_round.cpp` compares the sparse accept device path against a host oracle
(`sparse_accept_oracle` / `sparse_target_distribution`, FP64). **Every sparse case in the file sets
`top_k = 1`** — `generated_general_case` and `sparse_general_mixed_case` both do — so the whole
multi-token branch of the accept rule is unverified by the suite's own oracle. The production dflash2
lane sends `top_k = 20`, which is the uncovered case.

A throwaway case built to measure the emitted-token distribution (a real gap: the existing cases pin
one seed and compare one decision, so they cannot see a distributional error) put the two sides side by
side and they disagree:

| configuration | device | host oracle |
|---|---|---|
| `top_k=1`, `top_p=1.0`, no penalty | agrees on every trial | agrees |
| `top_k=2`, `top_p=1.0`, **no penalty at all** | accepted the draft on 16384 of 16384 trials | accepted on ~47 % |

The `top_k=1` row is the control that says the harness is sound: the histogram, the chi-square and the
oracle all agree exactly there (`chi2 = 0.0`). So this is not a broken harness. It appears as soon as
the target's support has more than one token.

**Which side is wrong is not established.** Two candidates, and I have no evidence favouring either:
the kernel's accept/correction arithmetic for a multi-token support, or the host oracle's residual and
uniform arithmetic in the same regime. I am deliberately not calling it a production defect — the last
time I asserted a severity for this code without deriving it, I was wrong.

**Done when:** the divergence is attributed to one side and fixed, or the oracle is corrected and the
`top_k=1` restriction on the sparse cases is lifted so that `top_k=20` is covered by the suite. Either
way the case has to be re-added, because a distributional check on the accept rule does not exist in the
tree today and the penalty-asymmetry discussion in `tools/release/v3_profile_matrix.py` rests on the
identity that this case was written to test.

---

## Closed — do not reopen

Each of these was investigated and settled. They look like open work and are not.

| Item | Why it is closed |
|---|---|
| **Rebuild the four models** | The merge touched no artifact, layout, binding or converter file, and no container version. Weights, layout and bindings are byte-compatible. What was stale is the *measurements* — item 3. |
| **Store the DFlash2 drafter at Q4** | Already measured here, per target. `tools/convert/official_recipes.py:303-312` records that the NVFP4 draft rule "was tried here and *lost* 3.2 acceptance points on the DFlash2 lane (57.7% against 60.9%)", which is why the Swift line keeps its draft at Q8, and states the rule: "a draft encoding is measured per target, and this target's hidden states are not the stock ones." Lines 40-51 and 57-58 already assign `Q8` to drafter parameters. The published Q4 result is on a different model. |
| **Fix C2719 by passing the descriptor by reference** | It compiles and then faults: nvcc's host stub passes the host address as a device pointer. A compile-only check would pass it. The correct fix is to drop `alignas(128)`, which this port already did in `1218d574`. |
| **The drafter's block uses a causal-over-block mask** | Refuted. The reference is non-causal (DFlash paper section 4.2; the published checkpoint sets `is_causal: false`), and `context_query.cuh:275-283` already gives every query row `valid_keys = valid` with no causal predicate. |
| **Re-measure at T=1.0 to match the model card** | Backwards. DFlash2's selector measures 4.61 at T=0 against 4.25 at T=1; greedy is the *favourable* side. Re-measuring at T=1 would widen the gap. |
| **The planner's `chunked_target` topology class causes the acceptance cliff** | Refuted by measurement. Acceptance is bit-identical before and after `a012e2bc` removed the predicate. The class selected which shared `cudaGraphExec_t` a profile reused, not which kernel ran. |
| **Acceptance is low because our drafter is mismatched to the target** | True of `qwen3_8_27b_nvfp4.v3.ninfer` (unsloth quantization plus official drafter — upstream issue 298 section 3 documents 3.3-5.1% for exactly that pairing), but that artifact **is not a shipping lane**. The four shipping lanes accept 52-69%. |
| **Take the full 16-commit upstream merge** | The 7 fp8 TMA commits fail on MSVC (`error C2719`) and are deferred, not forgotten. Three one-line `alignas` reapplications, already validated here. |
| **The NVFP4 block scale should be searched, not taken from the block max** | Measured on the weights that are actually re-encoded, and it loses: 4.925917 against 4.915181 paired in one window, while weight reconstruction error fell 40-66 %. It cannot be said about `NVFP4MSECalibrator` at all: that calibrator's 193 sites are the MLP and `lm_head`, which this recipe imports unencoded, while the re-encoded attention sites are FP8 `MaxCalibrator` in the source. See item 2 and `docs/perplexity-baseline.md`. Lower weight error is not better output quality. |
| **Lower quantization error implies better perplexity** | The same measurement, stated as the general form. A searched scale that clips a block's largest value reduces squared error on that block and costs output quality, because the large value is carrying signal. Qualify a converter change on perplexity, never on reconstruction error. |

## Also worth doing, small

- `--ngram chain` is accepted, validated against the backend, carried into `Program`, and produces
  nothing: `ngram_drafted_tokens` and `ngram_accepted_tokens` are declared at `types.h:761-762` and
  written nowhere, and `impl->ngram` (`startup.cpp:819`) is never read. **It does not reserve
  288 MiB** — that allowance went with `ngram_policy.h` in `8c7242e6`, and `startup.cpp:884` now says a
  copy round runs at the round's own width and provisions nothing extra. So the fix is item 9, not
  withdrawal: the flag is the switch `PromptLookup` needs, and the requirement is to add copy
  drafting. Recorded here only because an earlier revision of this file said to withdraw it, on the
  strength of the withdrawn allowance.
- ~~`tools/release/check_doc_links.py` does not skip fenced code blocks~~ — done in `46ec0c5b`, with
  seven tests, six of which fail against the previous body.
- DFlash **v1**'s published config has no `is_causal` key, so the reference gives it five causal and
  one non-causal draft layer. Our converter's `_fixed` check only raises when the key is *present*
  and the runtime never reads it, so a converted v1 drafter silently gets a uniformly non-causal
  block. Not a shipping lane; a real defect.
- `C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer` sits beside the four shipping artifacts, is not a
  shipping lane, and reads acceptably by filename. It cost a full session of benchmarking before
  `profiles.py` was checked.
