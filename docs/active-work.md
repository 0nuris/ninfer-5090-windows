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

### 1. Perplexity regression on a real artifact, post-merge
**Why:** 88 attention files were rewritten wholesale. The suite passes, but the suite does not check
output *quality* on a real model, and nothing has confirmed quality survived the rewrite.
**Done when:** `ninfer-perplexity` runs on a shipping artifact at the same corpus and context as
`docs/perplexity-baseline.md`, and the result is within the recorded band or the regression is
explained. This gates everything below.

### 2. Recall@1 / Recall@16 / path-acceptance split
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

### 3. Interleaved re-measurement of all eight profiles
**Why:** the published table is wrong on our own numbers. `start_ninfer_v3_dflash2_vision` records
340.4 tok/s / 68.8% and measured 291.0 / 53.4% — off by 14.5% and 15.4 points.
`start_nvidia_v3_mtp5_vision` measured `[254.9, 167.7, 168.6]`, a **52% spread inside one profile**,
which is not a measurement at all; something else was on the card.
**Done when:** every row is re-measured by alternating lanes rather than sequentially, and
`tools/release/profiles.py` carries numbers that reproduce. `profiles.py`'s own docstring is explicit
that interleaving is not optional on this machine, because decode drifts up to ~9% between windows and
two earlier "improvements" were really that drift.

### 4. k8v4 KV against the shipped fp8
**Why:** every profile ships `--kv-dtype fp8`. K8V4 is available as a third option behind one flag.
**Marked as third-party:** measured within 0.08% of BF16 on perplexity and no worse than fp8 — but on
a different model, so it needs our own number.
**Done when:** perplexity and decode acceptance are compared across bf16 / fp8 / k8v4 on one shipping
lane, interleaved.

### 5. MTP draft window 5 to 10
**Why:** our MTP lanes use 4 and 5. A NInfer fork raised MTP to 10 on the claim that longer proposals
emit more per round, and **published no tok/s gain for it.**
**Done when:** windows 4/5/8/10 are compared on one MTP lane, interleaved, with tok/s and acceptance
reported together.

### 6. ngram: the verify-tree integration
**Why:** `PromptLookup` is ported and tested (`c0da270e`); the integration is not built. It needs
`candidate_selector_tree`, `speculative_accept_tree_drafts`, `speculative_compact_columns`, tree-aware
GDN replay and tree-aware target attention, each with a host oracle.
**Expected value is low, and that is the finding, not a reason to skip it:** the selector's value is
inversely proportional to drafter strength, and our shipping DFlash2 lanes accept 52-67%. The +55%
reference was measured on a lane accepting 27%.
**Done when:** one DFlash2 lane is measured with and without the copy grafted, on copy-heavy traffic,
against our own baseline. **"Built, measured, and not shipped" is an acceptable outcome** and should be
reported as one.

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

## Also worth doing, small

- `--ngram` is accepted, does nothing, and reserves 288 MiB. Withdraw it until item 6 lands.
- `tools/release/check_doc_links.py` does not skip fenced code blocks, so a C++ lambda in a snippet
  reads as a markdown link. It blocked a commit twice.
- DFlash **v1**'s published config has no `is_causal` key, so the reference gives it five causal and
  one non-causal draft layer. Our converter's `_fixed` check only raises when the key is *present*
  and the runtime never reads it, so a converted v1 drafter silently gets a uniformly non-causal
  block. Not a shipping lane; a real defect.
- `C:\AI\models\qwen3_8_27b_nvfp4.v3.ninfer` sits beside the four shipping artifacts, is not a
  shipping lane, and reads acceptably by filename. It cost a full session of benchmarking before
  `profiles.py` was checked.
