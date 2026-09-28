# DFlash2's block mask, and what a steep per-position decay does and does not mean

Research note. Written 2026-09-28. Every claim carries its provenance: **[paper]**,
**[model card]**, **[author blog]**, **[read from source]** (file and line, or upstream URL),
**[community]**, **[measured here]**, or **[no evidence found]**. Measurements described as *ours*
are the submitting session's and are taken as given; I did not re-derive them.

**Headline: the causal-mask hypothesis is dead, and it is dead in both directions.** DFlash2's
published checkpoint sets `"is_causal": false`, the reference implementation's default for a
`sliding_attention` layer is *causal*, and our tree's mask is already non-causal. Our measured
profile is therefore **not** a causal-mask bug, and the literature does not support reading it as
one — the steepest decay any published drafter shows is roughly 20 points over seven positions,
against our 44 points over four. The decay is real and unexplained; the mask is not the cause.

**Second finding, and it is the actionable one: our greedy measurement is the *favourable*
setting, not the explanation.** Every published DFlash number at temperature 0 is *higher* than the
same drafter at temperature 1. Ours cannot be explained by the model card's having been measured
with sampling.

---

## 1. Q1 — What mask does DFlash's drafter use over the K+1 block?

### 1.1 The answer: non-causal (bidirectional within the block), for DFlash2

Three independent sources agree, and the third settles it against the reference's own default.

**1. The paper, §4.2 "Training".** Chen, Liang & Liu, *DFlash: Block Diffusion for Flash
Speculative Decoding*, arXiv:2602.06036v2 (ICML 2026), read in full at
`arxiv.org/html/2602.06036v2`. **[paper]**

> During training, all blocks are concatenated into a single sequence and processed jointly using
> a sparse attention mask as shown in Figure 4. **Tokens attend bidirectionally within the same
> block** and to the corresponding injected target context features, while attention across
> different blocks is disallowed.

Figure 4's caption adds that the invisible (white) tokens "enforce causal consistency and
prevent inter-block information leakage" — i.e. causality is enforced *between* blocks, not within
one. Every column of the K+1 block attends to the anchor **and to every other column of the same
block**, which is what makes the single forward pass a genuine parallel block prediction.

**2. The reference implementation, `dflash/model.py` in `z-lab/dflash`.** **[read from source]**

```python
def _attention_mask(query, key, *, is_causal, sliding_window):
    query_position = key.shape[-2] - query.shape[-2] + torch.arange(query.shape[-2], device=query.device)[:, None]
    key_position = torch.arange(key.shape[-2], device=query.device)[None, :]
    visible = torch.ones((query.shape[-2], key.shape[-2]), dtype=torch.bool, device=query.device)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    return visible[None, None]
```

The mask starts as `torch.ones` — everything visible — and the sliding window is applied
**symmetrically** when not causal (`key_position - query_position < sliding_window` as well as
`query_position - key_position < sliding_window`). This matches upstream NInfer's own documented
rule to the letter: *"allowed(p_query, p_key) = abs(p_key - p_query) < S … With a full left
context and `p_query=F+i`, the row sees at most `S-1-i` context positions, plus all W query
rows."*

**3. The published checkpoint's config — this is the decisive artifact.**
`huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/config.json` **[read from source]**:

```json
"is_causal": false,
"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention"],
"sliding_window": 2048,
"dflash_config": { "block_size": 8, "mask_token_id": 248070, "conv_group_size": 16,
                   "conv_kernel_size": 2, "selector_rank": 256, "selector_top_k": 16,
                   "target_layer_ids": [5, 19, 33, 47, 61] }
```

The `is_causal` field exists precisely because the reference's default is the opposite. In
`Qwen3DFlashAttention.__init__`:

```python
is_causal = getattr(config, "is_causal", None)
self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
```

**A `sliding_attention` layer with no `is_causal` key is CAUSAL.** The authors had to write
`is_causal: false` into the DFlash2 checkpoint to get a non-causal block. That is the strongest
possible statement that the intended DFlash2 mask is non-causal: it is a deliberate, explicitly
published departure from the reference's default, and it is the one field that differs between
the two drafters.

### 1.2 Our tree is already non-causal

**[read from source]** — `src/ops/softmax_attention/common/context_query.cuh:275-283`, the tile
metadata lambda that decides what each query row may attend to:

```cpp
// The capture clause is written with a space so that this snippet is not mistaken for a markdown
// link by tools/release/check_doc_links.py, which does not skip fenced code blocks.
auto tile_metadata = [&] (int iteration, bool& is_query, int& key0, int& valid_keys) {
    is_query = iteration >= context_tile_count;
    if (is_query) {
        key0       = 0;
        valid_keys = valid;      // <-- every live column, for every query row
    } else {
        key0       = context_start + (tile_begin + iteration) * KeyBlock;
        valid_keys = min(KeyBlock, length - key0);
    }
};
```

There is no causal predicate on the query-block tile. Every live column of the block is visible to
every query row. The context side is the sliding window only:
`src/ops/softmax_attention/sliding_window/kernel.cuh:14,18-20` — `context_count(value) = min(value,
window_mask)` and `allow_context(query, key) = key >= query - window_mask`, which composes to the
"at most `S-1-i` context positions plus all W query rows" rule above. The single forward in
`propose_dflash2_batch` (`src/models/qwen3_5/execution/draft.cpp:262-372`) embeds, RoPEs and
attention-processes all `width = k+1` columns as one tensor, so all columns attend to each other.

Upstream NInfer's own `docs/maintainer/dflash.md` (§"Query block and attention masks") says the
same thing and adds the consequence: *"Changing W can change every proposal because attention is
non-causal. A shorter block is not required to match the prefix of a longer block."*

**Verdict: our mask is correct. The measured decay is not a causal-mask bug.**

### 1.3 One real, reachable divergence — in DFlash v1, not in our lane

Worth recording because it is a findable defect even though it does not explain our numbers.

`huggingface.co/z-lab/Qwen3.5-27B-DFlash/raw/main/config.json` **[read from source]** has **no
`is_causal` key at all**, and its six layers are
`["sliding_attention" × 5, "full_attention"]` with `sliding_window: 4096`. Under the reference's
default that means **five causal layers and one non-causal layer** for the DFlash v1 checkpoint —
a mixed mask. Our port does not model that:

- `tools/convert/qwen3_5.py:227` calls `_fixed(raw, "is_causal", False, backend)`, and
  `_fixed` (`tools/convert/qwen3_5.py:46-52`) only raises **when the key is present and
  differs**. A v1 checkpoint with the key absent converts cleanly.
- The converter's `result` dict (`tools/convert/qwen3_5.py:234-243`) does not carry `is_causal`
  forward, and the runtime parser `DraftConfig draft(...)`
  (`src/models/qwen3_5/config.cpp:211-217`) neither requires nor reads it. (It cannot: the shared
  `require_members` at `src/artifact/schema.cpp:38-42` **throws on any unknown member**.)
- So a converted DFlash v1 drafter silently gets a uniformly non-causal block, where the reference
  would give it five causal layers.

**[no evidence found]** for any published DFlash v1 acceptance number broken down by layer
`is_causal`, so I cannot say how much this costs. It is a divergence to close, not a measured
loss. It does not touch the DFlash2 27B lane, whose config does carry `is_causal: false`.

---

## 2. Q2 — Is a steep per-position decay normal for any class of drafter?

**No, and the comparison runs the opposite way from the hypothesis.** I looked for a per-position
curve in the causal-drafter class specifically, because that is what would make decay
diagnostic. It is not there.

### 2.1 The curves

| drafter | mask over the block | position 0 | position 1 | position 6 | position 14 | setting | source |
|---|---|---:|---:|---:|---:|---|---|
| **MTP (Qwen3.5-4B)** | **causal / autoregressive** | 84.57% | 80.23% | 77.36% | 77.85% | T=1.0 sampled | **[author blog]** Fig. 5 |
| DFlash 5L (Qwen3-4B) | non-causal | 85.39% | 80.31% | 72.86% | — | T=0 greedy | **[author blog]** Fig. 2 |
| DFlash 3L (Qwen3-4B) | non-causal | 85.21% | 79.26% | **64.97%** | — | T=0 greedy | **[author blog]** Fig. 2 |
| DFlash 15L (Qwen3-4B) | non-causal | 86.42% | 81.61% | 78.73% | — | T=0 greedy | **[author blog]** Fig. 2 |
| DFlash 5L + conv | non-causal | 85.83% | 80.94% | 77.61% | — | T=0 greedy | **[author blog]** Fig. 2 |
| DSpark | — | 87.24% | 84.59% | 82.97% | 79.86% | T=1.0 sampled | **[author blog]** Fig. 5 |
| **DFlash 2** | non-causal | **88.30%** | 85.30% | 85.36% | **86.48%** | T=1.0 sampled | **[author blog]** Fig. 5 |
| DFlash 2 (oracle @16) | non-causal | 99.5% | 97.3% | 87.8% | — | T=0 greedy | **[author blog]** Tab. 1 |
| **ours, window 5** | non-causal | **44.4%** | **22.2%** | 2.8% | 0.000% | T=0 greedy | **[measured here]** |

The DFlash v1 greedy rows (Fig. 2) are *per-position Recall@1* conditioned on every earlier
position being right — the same conditional statistic our `accepted_per_position` measures. The
DFlash2 and MTP rows (Fig. 5) are the same statistic for DFlash2, DSpark and native MTP.

### 2.2 What that rules out

1. **A causal drafter does not decay steeply.** MTP is the canonical causal drafter — one layer
   deep, autoregressive within the block, no target-feature conditioning — and it runs
   **84.57% → 77.85% over fifteen positions**, a 6.7-point total drop, *with no decay at all after
   position 1*. So "steep per-position decay is characteristic of a causal drafter" is not
   supported by the best available measurement of one. It is the closest published test of the
   hypothesis and it comes out flat.
2. **The steepest published curve is nowhere near ours.** DFlash 3L — the authors' deliberately
   under-capacity ablation, the configuration they argue is *too small a backbone* — falls
   85.21% → 64.97% over seven positions. That is a 20-point drop and it is the worst in the
   literature. Ours is 44.4% → 0.0% over four: a 44-point drop to **exactly zero**, and it is
   worse than the worst published case at every position after the first.
3. **A decay to exactly 0.000 within four positions is not a phenomenon any paper reports.** The
   authors' named term for the effect they *do* see is "suffix decay", and they characterise it as
   mild and gradual: *"Even the oracle decays: with perfect selection, accuracy still falls from
   99.5% at the first position to 87.8% by the last."* A curve that reaches zero at position 4 is
   outside the range in which "suffix decay" is the explanation.
4. **Position 1 is already wrong in our lane.** Ours is 44.4% against a reference range of
   85.4% (greedy, DFlash v1 5L) to 88.3% (sampled, DFlash2). That is a ~2x shortfall **before any
   decay comparison**, and it is the single fact the diagnosis most needs to explain. A mask
   question cannot explain it, because the mask is right.

Medusa, the other parallel-multi-position drafter, reinforces point 1 from the other side. Every
Medusa head predicts its position from the *same* hidden state `h_t`
(`p_t^{(k)} = softmax(W_2^{(k)}·(SiLU(W_1^{(k)}·h_t)+h_t))`, arXiv:2401.10774v2 §2.1.1), so
positions are independent by construction — DFlash's structure — and Medusa still reports
acceptance rates of **3.01-3.51** across Vicuna-7B/13B/33B and Zephyr-7B (Table 1). Medusa's paper
does note the underlying asymmetry ("we observe that `L_k` is larger when `k` is larger") and
corrects it in training with `λ_k = 0.8^k` (§2.2.1) — a *training-weighting* response, not an
acceptance collapse.

**[no evidence found]** I did not find a published per-position acceptance curve for EAGLE-2 or
EAGLE-3 in a paper or in a primary repository. The EAGLE-3 arXiv HTML renders its scaling figure
without per-position numbers, and the one third-party table I found
(fixstars, gpt-oss-120B + EAGLE-3) reports only "Acceptance rate @ Position 1 (%) – 26.75"
**[vendor, incomplete]** — a single point, unusable for a curve. DFlash's own Table 1 gives
EAGLE-3's aggregate τ at both temperatures but not per position. The MTP row above is the closest
substitute and it is measured by the DFlash authors under matched conditions.

---

## 3. Q3 — Greedy versus sampled acceptance

**The prior runs backwards: greedy acceptance is *higher* than sampled, by 10-15%, for every
drafter in the DFlash paper. Our measurement is already the favourable setting.**

**Mechanism, stated by Medusa's authors** (arXiv:2401.10774v2 §2.3.1) **[paper]**:

> this sampling strategy results in diminished efficiency as the sampling temperature increases.
> Intuitively, this can be comprehended in the extreme instance where the draft model is the same
> as the original one: Using greedy decoding, all output of the draft model will be accepted,
> therefore maximizing the efficiency. Conversely, rejection sampling introduces extra overhead,
> as the draft model and the original model are sampled independently. **Even if their
> distributions align perfectly, the output of the draft model may still be rejected.**

That is the lossless rejection-sampling rule, and it is exactly what the DFlash2 model card's
numbers were measured under. Even a *perfect* drafter is rejected sometimes purely by sampling
noise.

**Measurement, by the DFlash authors.** DFlash2 blog Table 2 (Qwen3-4B, GSM8K, five layers) gives
both temperatures for the same drafter **[author blog]**:

| variant | T=0 (greedy) | T=1 (sampled) |
|---|---:|---:|
| DFlash | 4.27 | 3.78 |
| + DSpark correction | 4.49 | 4.08 |
| + path selection (DFlash 2's selector) | **4.61** | 4.25 |

**Measurement, by the DFlash authors, every row of Table 1** (arXiv:2602.06036v2) **[paper]**. All
eight method/model combinations are higher at temperature 0 than at temperature 1:

| | T=0 avg τ | T=1 avg τ | ratio |
|---|---:|---:|---:|
| Q3-4B DFlash (16) | 6.54 | 5.69 | 0.87 |
| Q3-8B DFlash (16) | 6.49 | 5.48 | 0.84 |
| Q3-4B EAGLE-3 (16) | 3.05 | 2.95 | 0.97 |
| Q3-4B EAGLE-3 (60) | 3.48 | 3.36 | 0.97 |
| Q3-8B EAGLE-3 (16) | 2.96 | 2.83 | 0.96 |
| Q3-8B EAGLE-3 (60) | 3.40 | 3.26 | 0.96 |

**Consequences for our comparison, both of which cut against us:**

1. Our greedy measurement is *not* handicapped relative to the model card. The model card's
   4.10-5.46 (mean 4.80) were measured at T=1.0 / top-p 0.95 / top-k 20 **[model card]**, and on
   the authors' own DFlash2 selector that setting is **8% worse** than greedy. If we re-measured
   at T=1.0 to match, the gap would *widen*.
2. The like-for-like greedy comparison is not the DFlash2 blog's flat 88%→86% curve — that is a
   sampled curve (§5 below). It is the DFlash v1 greedy Recall@1 row, **85.39% → 72.86%** over
   seven positions. Against that, our 44.4% at position 1 is a 2x shortfall and our zero at
   position 4 is not on the same curve at all.

**[no evidence found]** for a published quantification of greedy-vs-sampled acceptance for DFlash2
specifically on Qwen3.8-27B. The 4.61/4.25 pair is Qwen3-4B.

---

## 4. Q4 — Target residual conditioning

**Our tree matches the reference on all three points, and one of them is a trap the paper's
notation hides.** So the conditioning is not where the defect is, but the *scale* of the
conditioning is the one thing here that would produce "mediocre rather than broken" — and ours
is not even mediocre at position 1.

**Capture point.** Reference, `extract_context_feature` **[read from source]**:

```python
def extract_context_feature(hidden_states, layer_ids):
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)
```

`hidden_states` from Hugging Face is a `num_layers+1` tuple whose entry 0 is the embedding output
and whose entry `i` is the output of block `i`. So `hidden_states[layer_id + 1]` is the residual
stream **after block `layer_id` has completed both its mixer and FFN residual additions**, and
before the next block's input norm. `target_layer_ids` `[5,19,33,47,61]` are zero-based block IDs.
This is what the paper describes (`arXiv:2602.06036v2` §4.1: features "extracted from a fixed set
of layers uniformly sampled from shallow to deep"; §5: "extracted from 5 layers uniformly selected
between the second layer and the third-to-last layer") and what upstream NInfer documents:
*"the complete residual output of zero-based target block l, after both mixer and FFN residual
additions … Features are captured before the next block's input norm or final Text norm."*
**[paper]** **[read from source]**

**Projection and normalisation order.** Reference, `DFlashDraftModel.forward` **[read from
source]**:

```python
target_hidden = self.hidden_norm(self.fc(target_hidden))
```

with `self.fc = nn.Linear(len(target_layer_ids) * hidden_size, hidden_size, bias=False)` and
`self.hidden_norm = Qwen3RMSNorm(hidden_size, eps)`. Two things matter. **Order: project first,
then norm.** **The norm is *parametric* — `Qwen3RMSNorm` carries a learned weight.**

The paper's notation is misleading here. Appendix A.3 writes
`H_t = RMSNorm(W_c[H^(l_1);…;H^(l_5)])` with no weight vector, which reads as a *plain*,
non-parametric RMSNorm. The reference has a learned one. Our tree has the learned one, and in the
right order: `src/models/qwen3_5/execution/draft.cpp:152` projects with `feature_projection` and
`:154` immediately applies `context_norm`. The converter maps them to the reference's parameter
names — `tools/convert/qwen3_5.py:819-826`, `fc.weight` → `feature_projection` and
`hidden_norm.weight` → `context_norm` **[read from source]**. **Correct, and worth knowing that the
paper's rendering would have led you to drop the weight.**

**Whether a scale error here could produce a mediocre drafter.** Yes, and this is the one place
in Q4 where a defect is plausible rather than ruled out — but our position-1 number already
argues against it. The published oracle row is the tell: with perfect selection over the top-16,
DFlash v1 recall is **99.5% at position 0** **[author blog]** Tab. 1. If the injected target
features were mis-scaled or mis-normalised, position 1 would be the first casualty, because
position 1 is the column whose entire non-context input is the anchor token plus `c_t`. Ours is
44.4% where a *mediocre* conditioning would still be 70-85%. A conditioning scale error large
enough to cause our position-1 number would have to be gross (a missing norm, a transposed
projection, a wrong `target_layer_ids` order), not subtle — and those are all cheap to check by
inspection because they are single-expression code.

**Not checked here, and I am flagging it rather than implying it:** whether the *values* our tree
materialises into the context K/V match the reference numerically. The upstream contract for that
is `include/ninfer/ops/context_kv_materialize.h` — BF16 K with no observable cast between
projection and head normalisation, and `FP16_RNE(BF16(v_raw))` V. I read the mask and the
normalisation order; I did not audit the materialiser's arithmetic.

---

## 5. Q5 — Upstream on DFlash2 draft quality

**The single most useful thing in our own tracker is issue #298, and it is an independent
measurement of a broken DFlash2 lane landing at our own numbers.**

`Neroued/ninfer` issue **#298** (*Report: WSL2 builds and serves at published speeds…*, `koldfrontier`,
2026-09-21, OPEN), §3 **[community, but a first-hand measurement]**:

> Same finetune artifact converted with `--components text,vision,mtp,dflash2` using
> `z-lab/Qwen3.8-27B-DFlash2` (rev `50307d4c`): `--spec dflash2 --draft-tokens 7` **accepted
> 3.3-5.1% of drafts and decode fell to 67-75 tok/s**, versus `--spec mtp --draft-tokens 3` at
> 175-206 tok/s with 67-85% acceptance.

Their stated reason is legitimate and does **not** apply to us — "the draft was trained on the
stock model" and the target was a finetune. But note what the number is: **3.3-5.1% at draft
window 7.** Ours is 7.27% at window 7 and 7.45% at window 6 **[measured here]**. Two independent
implementations, one of them upstream NInfer on Linux, both land in single-digit percent on a
DFlash2 lane, while the same artifact's MTP head accepts 67-85%. **The reported acceptance profile
is a known failure signature in this project, and the target is demonstrably healthy.**

The rest of #298 is worth having on record too: on the *stock* artifact under WSL2, MTP acceptance
is 60% (code), 37% (prose), 54% (thinking) at draft width 3 **[community]**. Same target
family, same engine. A healthy target.

**Issue #188** ("DFlash2 support is now available for Qwen3.8-27B", OPEN) is the primary DFlash2
channel. The unresolved `Skylux70` report — a large output-quality regression on the v2 NVFP4
artifact that **persists with DFlash disabled and MTP enabled** — remains on the table and is
about the target, not the drafter. Since MTP accepts 67-85% on that same artifact family (#298,
#298 §1), I read the quality complaint as orthogonal to acceptance: a target can be sharp at
position 1 and still be reported as "Qwen 3.5 levels" on quality benchmarks. It does not excuse a
44% position-1 draft acceptance.

**Issue #279** (chunked path re-sweeping KV for DFlash K≥8) is throughput only and raises no
accuracy concern; consistent with the prior note.

**Model-card caveats actually stated:** block size 8 (7 draft tokens) is the only published
configuration; decoding is lossless (greedy output matches the target exactly); and the
`incoai` card gives no per-position table of its own — the flat curve lives in the blog, and it is
a *sampled* curve (see below).

**One correction to our own prior note.** `docs/research/dflash2-acceptance-baseline.md` §1 records
the DFlash2 per-position curve as "**T=0**, block 16". It is not. The blog's Figure 5 caption
reads "same sampling as above", and the "above" is Table 3's stated condition: *"Sampling: thinking
enabled, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, with lossless rejection
sampling."* **[author blog]** The flat 88.3% → 86.48% curve is therefore a **temperature-1.0
rejection-sampling** measurement, and the block size is 16 (15 draft positions, 0-14), which that
note has right. The T=0 figures in the same note — DFlash v1 Recall@1 85.4% → 72.9% on GSM8K, and
the conv's 72.86% → 77.61% — are correct, and are the ones to compare our greedy numbers against.

---

## 6. What would discriminate the remaining hypotheses, cheaply

Ranked by information gained per unit of work. All of these are measurements in a tree that
already emits the necessary quantities.

**1. Split the published decomposition out of our own numbers. This is the decisive one.**
The blog reports two different rows per position for the same drafter, and they localise the fault
to one of two halves of the system:

| quantity | what it tests | our lane |
|---|---|---|
| **Recall@1** — is `candidates[i, 0]` (the drafter's own unary top pick) the target's token? | backbone + proposal head, insensitive to the selector | not currently reported |
| **Recall@16** — is the target's token *anywhere* in the 16? | backbone + proposal head, still insensitive to the selector | not currently reported |
| **path acceptance** — is the *selected* token correct? | everything, including the selector | what `accepted_per_position` reports |

The gap between them is what the authors call "pure selection headroom" — 4.27 → 6.79 τ, a 59%
gain **[author blog]** Tab. 1. The three outcomes are mutually exclusive:

- **Recall@16 healthy (≈ the published 99.5% → 87.8% shape) while path acceptance collapses** →
  the defect is in `candidate_selector_path`. Position 1 uses the anchor as its predecessor and
  would be untouched by a broken pairwise term, which is exactly the shape we measure. The
  documented contract is right — `include/ninfer/ops/candidate_selector.h:32-47`: *"Starting with
  predecessor=anchors[b], each position i in [0,K) computes … written to drafts[i,b] and becomes
  the next predecessor"* — but a header is a claim, not evidence, and the kernel has not been
  measured against it. A left-to-right walk that used the *unary argmax* predecessor instead of
  the *selected* one would produce position 1 correct and positions 2+ degrading sharply.
- **Recall@16 itself collapsing after position 1** → the backbone or the injected conditioning, not
  the selector. Go to §4's scale/normalisation audit and to the conv's `i-1` tap.
- **Recall@1 already ≈0.44 at position 1** → the head or the conditioning is weak from the very
  first column, and the whole decay conversation is downstream of that.

The data is already materialised: `frame.candidate_ids` and `scores` are produced at
`src/models/qwen3_5/execution/draft.cpp:351-355`, in the right `[16, K, B]` layout, and the target's
argmax over the same columns is what the verify pass already computes. This is a comparison, not
a new kernel.

**2. Re-run the same profile with the drafter's width and the target's verify width decoupled.**
The position-1 *count* is width-invariant at ~15 while positions 2+ are not. From our own
acceptance rates and the round counts implied by tokens/round on 64 generated tokens — window 5:
~36 rounds, so counts ≈ **16, 8, 3, 1, 0**; window 6: ~44 rounds, so ≈ **15, 2, 1, 1, 0, 0**
(arithmetic on the measured rates, not a measurement) — position 1 is unchanged by one added
column and position 2 loses 4x of its accepts. Under a non-causal mask a wider block perturbs
every column's input similarly, so a position that is *more* width-sensitive than position 1 is
the anomaly. Holding the drafter width fixed and varying only the verify width (or the reverse)
separates "the drafter's block changed" from "the target's verify path changed". The prior note's
§8 already noted that the route predicate the width sweep blames does not switch at that column.

**3. Check the two things §4 flags as single-expression, and the one thing that is cheap and
unmeasured.** In order: (a) the `target_layer_ids` capture offset — the reference indexes
`hidden_states[layer_id + 1]` and our block IDs are zero-based, so an off-by-one here silently
shifts all five taps; (b) `context_norm` is applied *after* `fc`, not before, and carries a learned
weight; (c) whether our context K/V materialiser matches
`include/ninfer/ops/context_kv_materialize.h` numerically. I verified (a) and (b) by reading and
they are right; I did not verify (c).

**4. Ruled out already, cheaply, so do not spend time on them.** `proposal_valid_columns` is set to
the full block width at both writers (`src/models/qwen3_5/program/graphs.cpp:229` and
`src/models/qwen3_5/program/decode.cpp:667`), so no column is being zeroed by the
`token >= valid_columns` early-out at `context_query.cuh:530`. A trailing-position hard zero from a
validity-count bug is excluded.

**5. Do not re-measure at T=1.0 to match the model card.** Per §3 that makes the gap larger.

---

## 7. Answers, in the order asked

1. **The mask is non-causal** — bidirectional within the block, sliding-window over context, causal
   only *between* blocks. Paper §4.2 **[paper]**; reference `_attention_mask` **[read from source]**;
   and decisively the published checkpoint's explicit `"is_causal": false`, which exists only
   because the reference defaults a `sliding_attention` layer to causal **[read from source]**.
   **Our tree already implements this** (`context_query.cuh:275-283`, `kernel.cuh:14,18-20`), so
   this is not our bug. Separately, a converted DFlash **v1** drafter does diverge from the
   reference, which would give that checkpoint five causal layers; that is a real defect but it is
   not in our lane.
2. **No drafter class shows anything like our decay.** MTP, the canonical *causal* drafter, runs
   84.57% → 77.85% over fifteen positions. The steepest published curve is DFlash 3L's
   85.21% → 64.97% over seven. Ours is 44.4% → 0.000% over four. The literature therefore does
   **not** support the causal reading, and our position 1 is already ~2x low before any decay
   comparison.
3. **Greedy is the favourable setting, by 8-16%.** Medusa's authors state the mechanism (lossless
   rejection sampling rejects even a perfect drafter, by sampling noise); the DFlash authors
   measure it twice (DFlash2's selector 4.61 at T=0 vs 4.25 at T=1; and all eight rows of the
   paper's Table 1 higher at T=0). Matching the model card's sampling would widen our gap.
4. **The conditioning matches the reference** — captured after block `l` completes, projected
   first then RMSNorm'd with a *learned* weight (the paper's `RMSNorm(W_c[…])` notation omits the
   weight and would mislead). A scale error there is plausible in principle, but it would show as a
   weak position 1, and position 1 is where our 2x shortfall lives.
5. **Issue #298 §3 is the key upstream datum**: DFlash2 at draft width 7 accepting **3.3-5.1%** with
   MTP at 67-85% on the same artifact, reported as a known-draft-mismatch case. Our 7.27% is the
   same signature; the "wrong drafter for this target" excuse does not apply to us because we run
   the stock pair.

**The short version of the diagnosis:** the mask is right, the setting is favourable, the target
is healthy, and the drafter's own top-1 at position 1 is half of what the reference reports. The
measurement that splits the remaining space is the one the authors already publish — Recall@1 and
Recall@16 alongside path acceptance — and our tree already has the candidate tensor to compute two
of the three.

---

## 8. Sources

| source | what was taken from it |
|---|---|
| `arxiv.org/html/2602.06036v2` (DFlash, ICML 2026) | §4.2 the bidirectional-within-block training mask; Fig. 4 caption; §4.1 feature extraction; §5 "5 layers uniformly selected between the second layer and the third-to-last layer"; Table 1 T=0 vs T=1 for DFlash and EAGLE-3; §5.5.4 block-size generalisation; App. A.3 `H_t = RMSNorm(W_c[…])`; App. A.1 γ=4 for block size 8 |
| `github.com/z-lab/dflash` `dflash/model.py` | `_attention_mask`; `is_causal` default `layer_type == "sliding_attention"`; `extract_context_feature` with `offset = 1`; `hidden_norm(self.fc(...))` and `fc(..., bias=False)`; `CandidateSelector.select` left-to-right walk with `predecessor = candidates.gather(index)`; `GroupedDynamicCausalConv`; `_rejection_sample` |
| `huggingface.co/z-lab/Qwen3.8-27B-DFlash2/raw/main/config.json` | **`"is_causal": false`**; `block_size: 8`; `target_layer_ids: [5,19,33,47,61]`; `sliding_window: 2048`; 5 × `sliding_attention`; `selector_rank: 256`, `selector_top_k: 16`, `mask_token_id: 248070` |
| `huggingface.co/z-lab/Qwen3.5-27B-DFlash/raw/main/config.json` | **no `is_causal` key**; 5 × `sliding_attention` + 1 × `full_attention`; `sliding_window: 4096` |
| `huggingface.co/incoai/Qwen3.8-27B-DFlash2` (model card) | acceptance length 4.10-5.46 (mean 4.80) at block 8; MTP 3.74-5.02; DSpark 3.01-4.36; conditions (T=1.0, top-p 0.95, top-k 20, `xhigh`, H200, FA3, 4096 new tokens) |
| `inco.ai/blog/dflash2/` (author blog, 18 Aug 2026) | Table 1 Recall@1 85.4→72.9 and Recall@16 99.5→87.8 (T=0, GSM8K, Qwen3-4B); Table 2 T=0 vs T=1 (DFlash 4.27/3.78, DSpark-corr 4.49/4.08, path-selection 4.61/4.25); Fig. 2 the 3L/5L/15L/conv per-position table; Fig. 5 the MTP/DFlash/DSpark/DFlash2 per-position table **and its T=1.0 sampling condition**; "suffix decay" naming; oracle 4.27→6.79 |
| `arxiv.org/html/2401.10774v2` (Medusa, ICML 2024) | §2.1.1 all heads predict from the same `h_t`; §2.2.1 `L_k` grows with `k`, `λ_k = 0.8^k`; §2.3.1 the rejection-sampling-vs-greedy mechanism; Table 1 acceptance rate 3.01-3.51 |
| `Neroued/ninfer` #298 §1, §3 (`koldfrontier`, 2026-09-21) | DFlash2 at draft width 7 accepting **3.3-5.1%** with decode 67-75 tok/s; MTP 67-85% on the same artifact; stock-artifact MTP 60%/37%/54% at width 3 |
| `Neroued/ninfer` #188 (`Skylux70`), #279 | the unresolved v2 NVFP4 target-quality report; the chunked-path throughput issue (no accuracy concern raised) |
| our tree, `upstream/master:docs/maintainer/dflash.md` | the non-causal mask rule and the sliding-window interval; target conditioning formula; capture point; the "changing W can change every proposal" note; the DFlash2 selector's left-to-right walk |
| our tree | `src/ops/softmax_attention/common/context_query.cuh:275-283,530`; `src/ops/softmax_attention/sliding_window/kernel.cuh:14,18-20`; `src/models/qwen3_5/execution/draft.cpp:152-154,262-372`; `src/models/qwen3_5/program/graphs.cpp:229`; `src/models/qwen3_5/program/decode.cpp:667`; `src/artifact/schema.cpp:30-42`; `src/models/qwen3_5/config.cpp:211-217`; `tools/convert/qwen3_5.py:46-52,221-243,819-826`; `include/ninfer/ops/candidate_selector.h:32-47` |
