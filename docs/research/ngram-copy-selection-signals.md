# Choosing between a copy proposal and a neural draft: the signals, the rules, and the numbers

Research note. Written 2026-09-27. Third in the `ngram` series; the other two are
[`ngram-drafting-designs.md`](ngram-drafting-designs.md) (GitHub-tracker survey) and
[`ngram-outside-github.md`](ngram-outside-github.md) (everything else). Neither is repeated
here except where this note corrects or sharpens it; those places say so.

Evidence classes, as in the sibling note:

| label | meaning |
|---|---|
| **[MAINT]** | written by a maintainer/author of the system described, about their own system |
| **[PAPER]** | a paper's own reported measurement, by its authors |
| **[VENDOR]** | first-party for a product being sold; not independently reviewed |
| **[INFERENCE]** | arithmetic I performed on measured inputs. Every input is named. This is not a measurement. |

Method note, and it differs from the sibling notes: I could not obtain stable `file:line`
offsets for most of the sources below, because they were read as whole files rather than as
a checked-out tree. So citations here name the **file plus the symbol, field, section or
table**, which is what a reviewer needs to go and read. Where a line number is given it is
because the source itself printed it. I did not invent any.

---

## 0. Corrections and sharpenings to the two existing notes

| existing claim | status after this round |
|---|---|
| "TensorRT-LLM has `use_sa_spec` / `sa_spec_threshold`" | **Confirmed, and it is the only shipped copy-vs-neural selector in any of the four engines.** I read the implementation (§1.1). It is not a flag on a feature list; it is a 3-line `torch.where` over a match-length mask. |
| "vLLM ships ngram-with-neural selection" (as written in the brief's summary of §7.4) | **This is wrong and needs withdrawing.** vLLM ships `method: "ngram"` and `method: "suffix"` as *alternatives*, never combined. The combination (PR #24344) is still unmerged: the `SpeculativeMethod` literal in `vllm/config/speculative.py` contains no hybrid value, and `vllm/v1/spec_decode/` contains no router (§1.4). The sibling note's §7.4 wording — "Selection has shipped in production … TensorRT-LLM added `use_sa_spec` … and vLLM ships `method: \"ngram\"`" — reads as though both engines ship selection. Only TensorRT-LLM does. |
| "n-gram copy drafting used alone rejects 91-98 % of rounds" | **Confirmed and independently corroborated from a different direction** (§3.4): adding a match-length-gated copy selector to a strong neural drafter is worth +3.4 % to +8.4 % end-to-end, but +23 % to +46 % when the drafter it gates is weak. The selector's value is inversely proportional to the drafter's strength. |
| "FastDeploy ships concatenation with no published numbers" | Not re-checked; it is a concatenation, so it has no selection rule to report. Out of scope for this note. |
| The brief's premise that selection ships in several engines | **Narrowed to one.** I searched for a shipped copy-vs-neural selector in llama.cpp, vLLM, SGLang and TensorRT-LLM. Exactly one exists, plus one academic version of the same rule and one shipped *proportional-width* retrieval drafter. Details in §1. |

---

## 1. What shipped engines actually read to decide

### 1.1 TensorRT-LLM `SADraftEnhancer` — the signal is the match length, and nothing else

`tensorrt_llm/_torch/speculative/sa_enhancer.py`, class `SADraftEnhancer`. This is the whole
selection rule, in `maybe_override_all_draft_tokens`:

```python
n = self._num_gens
K = draft_tokens.shape[1]
mask = (self.sa_match_len[:n] >= self.threshold).unsqueeze(1).expand_as(draft_tokens)
draft_tokens = torch.where(mask, self.sa_draft_tokens[:n, :K], draft_tokens)
```

Properties of the rule, each readable from the code:

- **Signal:** `self.sa_match_len` — the per-request suffix-automaton match length in tokens,
  produced by `extend_and_prepare` from the accepted tokens of the *previous* round. One
  integer per request per round. Nothing else is consulted.
- **Threshold:** `self.threshold`, an `int`, taken from `spec_config.sa_config.threshold`.
  `MTPDecodingConfig` exposes it as `sa_spec_threshold`, **default 4**; `use_sa_spec` defaults
  `False`. A reviewer flagged on the PR that the field has no validator, so 0 or negative is
  accepted. Semantics per the docs: "the minimum suffix match length required to override the
  neural draft".
- **Granularity:** per request, per round, and **binary over the whole block** — the mask is
  broadcast across all `K` positions, so a request's draft is either all SA's or all the
  neural drafter's. There is no per-position choice anywhere in this implementation.
- **Timing:** *after* the drafter. The docstring is explicit: "Call
  `maybe_override_all_draft_tokens` once after all draft layers have finished, so that neural
  draft layers never see SA tokens." The two call sites in `mtp.py` are commented "Override
  with SA draft tokens after all MTP layers have run, so that MTP layers never see SA tokens
  in their inputs."
- **Adapters:** `use_sa_spec` is wired into `MTPDecodingConfig`, `Eagle3DecodingConfig` and
  `PARDDecodingConfig` (`docs/source/features/speculative-decoding.md`, "Suffix Automaton
  (SA) Enhancement").
- **Width:** `K` is the worker's `max_draft_len`. It is the *same* `K` for the SA row and the
  neural row. See §5.4 — this matters.

Two shipped variants of the same file are worth knowing about. `sa_worker.py`
(`SADecodingConfig`, the standalone SA mode) gates differently, at `match_len > 0`, and zeroes
the row rather than masking it, so no neural draft is involved at all. And the SA
`max_matching_ngram_size` selects fixed-size versus longest-match lookup: `-1` is longest
match, a positive value is fixed size, and `SuffixAutomaton::lookupFixed(targetLen)` in
`cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomaton.h` is the
fixed-size path.

### 1.2 SAM Decoding — the same rule, stated as a "virtual match length"

Hu et al., arXiv:2411.10666v3, §3.4 "Adaptive Draft Selection". This is the paper the
TensorRT-LLM implementation is an engineering of, and its statement of the rule is more
precise than the code's:

> a straightforward idea is that the length of the suffix match can indicate the confidence of
> the draft produced by the automaton, where long matches imply that more tokens are likely to
> be acceptable.

> During each generation step, we adaptively select the drafts offered by the automaton or the
> auxiliary SD method based on the match length of the generated text within the automaton. For
> the auxiliary SD method, we set a **fixed virtual match length `l_threshold`**.

So the rule is a comparison of two numbers: measured match length against a constant that the
neural drafter is *assigned*. `l_bias = l_threshold = 5` in their setup. Ablation, Figure 6,
described in §4 of the paper: both mean accepted tokens and speedup **increase with
`l_threshold` up to 5 and decline above it**. The per-point values are in a figure, not in
text, so the shape is the finding and the magnitude is not available.

### 1.3 The three signals the brief asked about, each checked

| candidate signal | does anything shipped read it? | what the source says |
|---|---|---|
| **Previous round's outcome / running hit rate** | **Not to choose a drafter. Yes, to kill speculation outright, in two independent systems.** | TRT-LLM `speculation_gate.py`, class `SpeculationGate`: "Permanently disables speculation when the rolling average falls below the configured threshold" — an O(1) rolling mean over a `window` of per-round acceptance rates, and once `disabled` is set it never re-enables. TriForce (arXiv:2404.11912v3, §4.3) triggers a retrieval-cache rebuild "either when the rolling average acceptance rate drops below a threshold or at a designed stride." Same shape, different consequence. |
| **A confidence / probability threshold on the last drafted token** | **No, not for choosing a copy.** The only probability-thresholded copy signal in any shipped engine is a *frequency* over a suffix tree, not a model confidence — see §2.3. | vLLM's `enable_adaptive_verification` is confidence-driven but trims the *verify budget* and is gated to `method="dspark"`; TRT-LLM's `use_relaxed_acceptance_for_thinking` is a confidence-based *acceptance* rule (`relaxed_topk`, `relaxed_delta`) gated to the thinking phase, not a drafter choice. |
| **Prompt length** | **No.** | No shipped or published implementation reads context length, generated length, or position to decide between a copy and a neural draft. |
| **A static per-request flag** | **Yes, one exists, and it is the only *admission-time* selector.** | llama.cpp resolves drafters by a fixed priority list with first-successor-wins (`common_speculative.cpp`, `common_speculative_init` and `common_speculative_draft`; the sibling note's §1.6). The choice is made by *configuration and position in a list*, not by any state. It is decided once, at request admission, and never revisited. |

**The finding: the only signal any shipped copy-vs-neural selector reads is the length of the
match it just found.** Not a running rate, not the previous outcome, not a confidence, not the
prompt.

### 1.4 Negative results, stated as searches

- **vLLM: no shipped selector.** `SpeculativeMethod` in `vllm/config/speculative.py` is a
  `Literal` of `"ngram", "medusa", "mlp_speculator", "draft_model", "suffix", "custom_class"`,
  the MTP literal, `"dflash"`, `"dspark"`, `"ngram_gpu"`. No hybrid. `vllm/v1/spec_decode/`
  (fetched 2026-09-27) contains `ngram_proposer.py`, `ngram_proposer_gpu.py`,
  `suffix_decoding.py`, `dflash.py`, `llm_base_proposer.py`, `medusa.py`, `eagle.py`,
  `gemma4.py`, `step3p5.py`, `draft_model.py` and a `dynamic/` package — **no router, no
  selector**. PR #24344 remains unmerged.
- **vLLM: no shipped match-length-driven width either.** `suffix_decoding.py` is the one file
  that varies width per request per round, and it has no neural path (§5.2).
- **llama.cpp: no selector, only a fixed priority chain** (§1.3). Sibling note §1.6 stands.
- **SGLang: no shipped selector.** One algorithm at a time (`speculative_algorithm` enum,
  `arg_groups/fields/spec.py`); the hybrid is PR #37237, open, CI red, and it routes on a
  *score* rather than a match length (sibling note §4.4). Not re-measured here.

---

## 2. The decision rules that exist, in implementable form

Three distinct rules are in shipped code. They are not variants of one thing; they answer
different questions.

### 2.1 All-or-nothing on a match-length threshold

```
copy_this_round(request)  :=  match_len(request) >= T
```

Used by: TRT-LLM `SADraftEnhancer` (T = `sa_spec_threshold`, default 4) and SAM Decoding
(T = `l_threshold`, default 5, with the neural drafter assigned a fixed virtual match length).

- Stateless. Nothing carries between rounds. No warmup, no seeding, no hysteresis.
- Fixed at configuration time. Does **not** adapt during a request.
- Applies to a whole round, not a position.
- Evidence that the threshold has an interior optimum: SAM Decoding Fig. 6, decline above 5.

### 2.2 Proportional to match length

```
max_copy_tokens(request)  :=  min( K_max , max_spec_factor * prefix_match_length(request) )
```

Used by: vLLM `method: "suffix"`, wrapping Arctic Inference's `SuffixDecodingCache.speculate`.
`vllm/config/speculative.py`, verbatim:

```python
suffix_decoding_max_spec_factor: float = 1.0
"""The maximum spec factor for suffix decoding. The spec factor controls
speculation lengths based on the prefix match length: max_spec_tokens =
max_spec_factor * prefix_match_length."""

suffix_decoding_max_tree_depth: int = 24
suffix_decoding_max_cached_requests: int = 10000
```

`vllm/v1/spec_decode/suffix_decoding.py`, `SuffixDecodingProposer.propose`, states the contract
in its own docstring: "Suffix Decoding will speculate a dynamic number of tokens for each
request every decoding step, so each entry in the returned list may have different lengths."
The defaults that matter: `max_spec_factor = 1.0`, `max_tree_depth = 24`,
`max_cached_requests = 10000`, and the docs recommend setting `num_speculative_tokens` to
"16 or 32 (default)" as a *ceiling*.

This is the only shipped rule where a *longer* match buys a *longer* proposal. It is worth
noticing that with the default factor of 1.0, a 4-token match — the threshold TensorRT-LLM
defaults to — proposes 4 tokens, not 15.

### 2.3 A per-token frequency floor

```python
suffix_decoding_min_token_prob: float = 0.1
"""The minimum token probability for suffix decoding. Will only speculate
tokens with estimated probability (based on frequency counts) greater than
or equal to this value."""
```

Same drafter, same call (`SuffixDecodingProposer.propose` passes `min_token_prob` into
`speculate`). This is a **per-position** rule, applied to each candidate token, and it is the
only probability threshold in the whole set — but note what the probability is: a frequency
count over the suffix tree, i.e. how often that continuation was observed. It is a prior over
the copy source, not a model confidence, and it is available for free from the index.

### 2.4 What the rule is not

Not fixed alternation. Not a confidence threshold on the drafter's last token. Not a per-round
*mixture* of the two proposals in one verify block — every implementation surveyed either
replaces the whole block or varies its length, and the one engine that concatenates
(FastDeploy `mtp_strategy: with_ngram`) has no selector because it does not choose.

---

## 3. Measured effect of selection vs. copy-only vs. neural-only

This is the most valuable section in the note. Two independent groups measured selection against
a neural drafter with a control arm, and their results sit at **+38.9 % TPOT** (a vendor PR, on
repetition-heavy code edits) and **+3.6 % to +16.9 % throughput** (a paper, on a mixed benchmark)
— a spread of more than a factor of ten between the two ends. Both are single unreplicated runs.
The condition attached to each number matters more than the number.

### 3.1 Selection vs. neural-only, on a neural drafter — TensorRT-LLM PR #11434

NVIDIA/TensorRT-LLM PR #11434, merged 2026-03-03, author `cascade812` with `@mahmoudhas`
(Baseten). The numbers are in the PR body, read from the API, not from a summary.

Configuration: `nvidia/DeepSeek-V3.1-NVFP4`, 8× B200, TP 8, 8 MTP layers,
`SA max_ngram_size = -1` (longest match), `num_nextn_predict_layers: 8`,
`use_sa_spec: true` (threshold default 4), dataset `glaiveai/code_edits_sample`,
100 requests, average input 413.49 tokens, average output 256.0 tokens.

| configuration | output tok/s | TTFT (ms) | TPOT (ms) | accept rate | accept length |
|---|---:|---:|---:|---:|---:|
| no speculative decoding | 98.34 | 83.05 | 9.88 | — | — |
| MTP (neural only) | 192.37 | 116.55 | 4.76 | 0.2704 | 3.16 |
| **MTP + SA (selection)** | **299.31** | 113.89 | **2.91** | **0.5513** | **5.41** |

The PR's own deltas: TPOT −38.9 %; accept rate 0.5513 vs 0.2704; accept length 5.41 vs 3.16
(+71.1 %); speedup over no-spec 1.96× → 3.04×.

**Read this with its conditions attached.** It is one `trtllm-bench` run per arm on a
repetition-heavy code-edit dataset at batch size implied by 100 requests, on datacenter
hardware, with 8 MTP layers. No repetition, no error bars, no interleaving. It is the
strongest available number and it is still a single unreplicated measurement by the vendor.

A caution on units, because the brief's "58 % neural acceptance" is a different unit from the
accept lengths in these tables. TRT-LLM's `draft_acceptance_rate` is a *per-draft precision*
(`num_accepted_draft_tokens / num_draft_tokens`) and its `acceptance_length` is
`total_new_tokens / total_iterations`, i.e. it already folds in the draft width and is
comparable across arms. I verified the relationship against the raw benchmark JSON in
`basetenlabs/sa_spec` (`bench/bench_bs_1.json`, MTP+SA, 8 layers, BS 1): 3877.6 draft tokens
and 1563.3 accepted over 104 requests gives 484.7 iterations and 2048 generated tokens
(3877.6/8 + 1563.3 = 2048, exactly the dataset's output cap), so accept length = 2048/484.7 =
4.23, matching the file's reported mean of 4.41. So 58 % precision and 5.06 accept length are
consistent descriptions of the same round at k=7, and quoting one as the other would be an
error.

### 3.2 Copy-only arms, same PR — suffix automaton vs. plain n-gram

Same PR, second table. `meta-llama/Meta-Llama-3.1-8B`, single B200,
`glaiveai/code_edits_sample`, 100 requests, average input 389.9 tokens, average output 256.0.
NGram at `is_keep_all=false, is_public_pool=false`.

| configuration | output tok/s | TTFT (ms) | TPOT (ms) | accept rate | accept length |
|---|---:|---:|---:|---:|---:|
| no speculative decoding | 278.66 | 15.23 | 3.54 | — | — |
| NGram, `ngram_size=3` (parenthesised) | (740.25) | (14.28) | (1.30) | (0.66) | (3.65) |
| SA, `ngram_size=3` | 928.21 | 15.69 | 1.02 | 0.66 | 3.65 |
| SA, `ngram_size=-1` (longest match) | 958.28 | 15.33 | 0.99 | 0.70 | 3.81 |

Two things here. First, **the copy arms reach 0.66–0.70 accept rate on this workload** —
which is the opposite end from the sibling note's "copy alone rejects 91-98 % of rounds". Both
are true; they are different workloads. Second, **the two retrieval implementations have
identical acceptance (0.66 / 3.65) and differ 25 % in throughput** — the entire gap is
implementation cost, i.e. the device-resident automaton plus overlap and cudagraph versus the
host-side pool. That is a clean measurement of what a better index buys, and it has nothing
to do with the selection rule.

### 3.3 The three-arm academic measurement — SAM Decoding Table 1

Hu et al., arXiv:2411.10666v3, Table 1. Vicuna-7B-v1.3, single RTX A6000 48 GB, PyTorch 2.3.0,
CUDA 12.1, fp16, **greedy, batch size 1**, draft size 40 (16 on code datasets),
`l_bias = l_threshold = 5`. Retrieval corpus pre-built from Vicuna-7B outputs on
Stanford-alpaca, python-code-instruction-18k and GSM8k.

| method | Spec-Bench MAT / tok/s / speedup | HumanEval | HAGRID |
|---|---|---|---|
| PLD | 1.75 / 59.02 / 1.56× | 1.65 / 59.04 / 1.52× | 2.03 / 44.11 / 1.29× |
| SAM-Decoding (copy only) | 2.30 / 69.37 / 1.84× | 2.64 / 88.91 / 2.29× | 2.44 / 76.72 / 2.24× |
| Token Recycling (auxiliary) | 2.83 / 69.65 / 1.84× | 2.78 / 75.44 / 1.94× | 2.88 / 66.17 / 1.93× |
| **SAM-Decoding[T]** | 3.03 / 85.73 / 2.27× | 2.94 / 95.08 / 2.45× | 3.23 / 87.93 / 2.57× |
| EAGLE-2 (neural only) | 4.36 / 90.14 / 2.38× | 5.13 / 125.77 / 3.24× | 4.15 / 82.61 / 2.41× |
| **SAM-Decoding[E2]** | 4.62 / 97.56 / 2.58× | 4.95 / 130.28 / 3.35× | 4.75 / 96.60 / 2.81× |

**This is the only measurement in the whole series with all three arms on one rig.** Selection
versus each single-drafter arm, recomputed from the table **[INFERENCE]**:

Each cell is `selection arm / the arm it is compared against`, so the division can be redone.

| | vs. copy only (SAM-Decoding alone) | vs. neural only (the gated drafter alone) |
|---|---:|---:|
| Spec-Bench, gated on a weak model-free drafter (Token Recycling) | +23.6 % (85.73 / 69.37) | +23.1 % (85.73 / 69.65) |
| Spec-Bench, gated on EAGLE-2 | +40.6 % (97.56 / 69.37) | **+8.2 %** (97.56 / 90.14) |
| HumanEval, gated on Token Recycling | +6.9 % (95.08 / 88.91) | +26.0 % (95.08 / 75.44) |
| HumanEval, gated on EAGLE-2 | +46.5 % (130.28 / 88.91) | **+3.6 %** (130.28 / 125.77) |
| HAGRID, gated on Token Recycling | +14.6 % (87.93 / 76.72) | +32.9 % (87.93 / 66.17) |
| HAGRID, gated on EAGLE-2 | +25.9 % (96.60 / 76.72) | **+16.9 %** (96.60 / 82.61) |

Note the third and fifth rows: on HumanEval and HAGRID, **selection beats the copy arm by a
smaller margin than it beats the copy arm on Spec-Bench** (+6.9 % and +14.6 % against
Token Recycling), because those two datasets reward copy drafting enough that the copy arm
alone is already strong. The column that tracks the design's question is the right-hand one —
selection against the drafter it is gating — and there the ordering is unambiguous.

Per Spec-Bench task, from the §4 text: EAGLE-2 → SAM-Decoding[E2] gives 2.87× → 3.02× on
Multi-turn Conversation, 2.33× → 2.76× on Summarization, 2.03× → 2.23× on RAG
**[INFERENCE from the quoted ratios]**. Token Recycling → SAM-Decoding[T] gives
1.92× → 2.48×, 1.96× → 2.86×, 1.68× → 2.14× on the same three.

Note the abstract's headline "3.28 %–11.13 %" is a *different* measurement from the table's
per-task ratios: it is MT-Bench throughput across Vicuna-7B/13B/33B, not Spec-Bench speedup
ratios. Both are in the paper; they should not be quoted interchangeably.

### 3.4 The one pattern in this table that transfers to a stronger drafter

**The selector's value is inversely proportional to the strength of the drafter it gates.**

Gating a weak drafter (Token Recycling, MAT 2.83) is worth **+23 % to +33 %**. Gating a strong
one (EAGLE-2, MAT 4.36) is worth **+3.6 % to +16.9 %**. On HumanEval, where the gated drafter
is at its strongest (MAT 5.13) and the text is least copyable, the selector is worth +3.6 % of
throughput and the *acceptance length actually falls*, 5.13 → 4.95 **[PAPER]**.

Baseten states the same mechanism from the other end **[VENDOR]**: "SA Decoding shines at code
generation, where the accept length is 10+ with long context, but performs poorly on reasoning
and other writing tasks, with accept rates near 0. Meanwhile, MTP produces consistent speed-ups
across all domains, though the accept rate is usually only 2-4 tokens per iteration."

The design decision this bears on: a lane whose neural drafter already accepts ~5 tokens per
round is at the strong-drafter end of that range, where the published value of adding a copy
selector is **low single digits to mid teens of percent**, not the 30-40 % the repetition-heavy
code-edit numbers show. That is not an argument against building it; it is the range the
acceptance criterion should be written against.

---

## 4. Cost of a wrong guess

### 4.1 What the engine pays, read from the code

In the shipped design — and therefore in this project's design, which is the same shape — a
copy round is a *wasted drafter forward*. `SADraftEnhancer`'s contract is
"call `maybe_override_all_draft_tokens` once after all draft layers have finished", so on a
copy round:

1. the suffix automaton extends and searches (on a side stream, overlapping the drafter),
2. the **full neural drafter forward runs anyway**, for all K positions,
3. its output is discarded by `torch.where`,
4. the verify pass runs, and the SA tokens are accepted or rejected normally.

So a wrong guess costs **one discarded drafter output plus one normal verify pass**. It does
*not* cost an extra forward beyond what the round already pays, and it does not cost a
synchronisation: `extend_and_prepare` records an event and `maybe_override_all_draft_tokens`
is the lazy `wait_event`, and the buffers are pre-allocated in `_ensure_buffers` precisely so
the gate can be CUDA-graph-safe.

Two shipped designs deliberately do *not* pay that cost, and are the relevant contrast:
llama.cpp stops at the first implementation that returns a proposal (`dp.drafting = false`),
and the vLLM PR #24344 author keeps the drafter's prefill live and gates only its decode.

### 4.2 What is actually measured, and what is not

**Measured:** nothing, on the cost of a wrong guess.

The nearest measurement is a vendor claim about a different quantity. Baseten **[VENDOR]**: "The
absence of latency overhead was verified by setting the SA threshold to infinity, effectively
disabling SA predictions while still executing the computation, and confirming that end-to-end
latency matches that of baseline MTP." The `basetenlabs/sa_spec` README says the same: "we
tested the system with the threshold set to infinity and confirmed that performance remains on
par with vanilla MTP."

That establishes that **the suffix-automaton machinery is free** when the gate is permanently
closed. It says nothing about what a *wrong guess* costs, which is a different quantity: a
discarded drafter output, plus whatever the wider verify width costs. No number is published
for either the false-positive rate of a match-length threshold or the cost of a rejection on a
copy round. I searched the PR thread, the repo, the blog and the paper for both; they are
absent.

### 4.3 The break-even, which follows from the design and does not need a new measurement

Given the design's own numbers — neural acceptance 58 % at k=7, a copy round at k=15, and a
measured +5.7 % round time for the wider round — the break-even copy-acceptance rate can be
derived. All inputs are as stated in the task brief; the arithmetic is mine. **[INFERENCE]**

Set `t(7) = 1` and `t(15) = 1.057`. Neural tokens per round `N = 1 + 7 × 0.58 = 5.06`. Copy
tokens per round at width 15 with copy acceptance `c`: `C = 1 + 15c`. Let `p` be the fraction of
rounds spent on copy. Throughput gain over neural-only is
`[(1-p)N + pC] / [(1-p)·1 + p·1.057] / N`. Setting that to 1 and cancelling `p` on both sides:

> **`C = 1.057 · N`**, i.e. **`1 + 15c = 5.348`**, i.e. **`c = 0.290`**.

#### 4.3.1 `c` is not a free parameter — it is `p` of the copied token

Added 2026-09-28, from work on the accept path (`docs/active-work.md` item 11). The break-even above
treats `c` as something to be measured or assumed. It is neither: it is pinned by the rejection
sampling identity.

A deterministic copy proposal is a **point mass**. The drafter proposes one token, so `q = 1` on that
token and 0 elsewhere — which is a normalised distribution, so the identity applies. The accept test is
`min(1, p/q)` with the correction `normalize(max(0, p - q))`, and `P(accept) = 1 - TV(p, q)`
(Leviathan et al. 2302.01318 Thm 1). For a point mass that reduces to

> **`c = P(accept) = 1 - TV(p, q) = p(copied token)`**

— the target's own probability of the token being copied, and nothing else. Consequences:

- **The break-even becomes measurable.** `c = 0.290` is then a threshold on the target's probability of
  the copied token rather than on an unmeasurable gate statistic. Log `p` of the proposed copy token per
  round and the break-even test becomes a direct comparison, with no dependence on the match-length
  gate's unknown false-positive rate.
- **It explains the shipped signal.** §1.1 gates on suffix match length and nothing else. A longer match
  is a proxy for a higher `p`, which is why a 3-line mask over match length is a reasonable stand-in.
  It also predicts where the proxy fails: it is weakest on exactly the borderline matches, which are the
  ones that decide the break-even.
- **It bounds what a copy drafter can do alone.** A point mass cannot exceed `p(token)`, so a
  deterministic lookup is capped by how predictable the copy is. Beating that needs several candidates
  carrying real probabilities rather than one token — STAND (`arXiv 2506.04708`) stores top-k indices
  *and* their probabilities per n-gram for exactly this reason, and that is the difference between a
  selector that can be grafted onto a neural drafter and one that cannot.
- **Caveat on the framing.** This holds for the stochastic accept path. Under a greedy target
  (`temperature = 0`) the accept test is an argmax match and the arithmetic does not apply, which is
  also why the published 2-4x prompt-lookup gains are greedy-decode measurements.

Two consequences, and the second is the useful one:

- **The copy proposal has to be accepted at about 29 % — roughly half the neural drafter's
  58 % — to break even.** Equivalently, about 4.4 of the 15 copied tokens.
- **The mixing rate cancels out entirely.** Because both arms consume the same number of
  rounds and the cost difference is a fixed per-round factor, *how often* the selector chooses
  copy does not affect the break-even at all; it only affects variance. There is no separate
  "how aggressive should selection be" tuning problem. The entire problem is the **precision
  of the gate** — the false-positive rate — because every false positive is a copy round that
  returns 1 token instead of 5.06. This reframes what the decision core should be tuned for,
  and it is the one quantity for which no published number exists.

The identity's assumptions, and which way each cuts if violated: it holds only if round time is
the whole cost, if the drafter forward is paid on both arms (it is, by design), and if the
+5.7 % is the complete cost delta of the wider round including its drafter forward. A design
that *skipped* the drafter on copy rounds would change the identity and lower the break-even.

---

## 5. Adaptive width

### 5.1 Every shipped width schedule is keyed on batch size, not on anything per-round

- **vLLM**, `num_speculative_tokens_per_batch_size`, documented in
  `docs/features/speculative_decoding/dynamic_speculative_decoding.md`: entries are
  `(start_bs, end_bs, optimal_K)`. "K=3 will be used when the concurrency is in range [1, 64]".
  The key is concurrency. The doc's own use cases are "Variable concurrency workload using same
  deployment" and RL rollout tail behaviour. It also states it is tested with Eagle, Eagle-3
  and **DFlash** — this project's drafter family — and that with data parallelism enabled
  vLLM *disables it entirely* because ranks would pick different K and deadlock.
- **TensorRT-LLM**, `draft_len_schedule` (PR #10860, expanded in #12262): `{4: 4, 8: 2, 32: 1}`
  means batch 1-4 → 4, 5-8 → 2, 9-32 → 1, 33+ → 0. Again keyed on batch size, with 0 meaning
  "speculation disabled". PR #12262 explicitly extends this to "DFlash, MTP, MTP-Eagle, PARD,
  draft target, suffix automaton, and hybrid spec dec algos with suffix automaton."
- **Neither engine varies width on measured acceptance.** Neither engine's docs or PR bodies
  carry a number for the gain from a width schedule.

### 5.2 The one shipped confidence-driven width, and it has no neural path

Arctic/Snowflake Suffix Decoding, shipped in vLLM as `method: "suffix"`, is the only retrieval
drafter in any engine that varies its own width per request per round, and it does so with
`max_spec_tokens = max_spec_factor × prefix_match_length` (§2.2) plus a per-token frequency
floor (§2.3). The Snowflake engineering write-up **[VENDOR]** states the outcome: "1.96x-3.12x
on BlazeEdit and 1.0x-1.28x on SpecBench, along with 1.11x-1.17x speedups over the best N-gram
baseline on SpecBench and 1.02x-1.31x on BlazeEdit", and describes the mechanism as speculating
"a dynamic number of tokens for each request at each decoding step, so the
`num_speculative_tokens` configuration specifies the maximum". It also notes the graph-shape
consequence: fixed tree depth, default 64 tokens, and single-sequence speculation only.

There is **no arm of this that widens a copy round relative to a neural one**, because there is
no neural drafter in the same request.

### 5.3 The measurement that bears directly on k=15

SAM Decoding Figure 7, ablations §4, described in text: "As the draft size increases, there is
a positive trend in throughput until the draft size equals 40. When the draft size exceeds 40,
there is an observable decline in performance metrics, which becomes more significant as the
draft size reaches 70."

Two cautions on that sentence, both from the paper's own Table 1. Their stated mechanism is
"for draft sizes below the average acceptance length, increasing the draft size … enhances
efficiency" — but the same paper reports mean accepted tokens of 3.03 for that configuration,
against an optimal draft size of 40. The mechanism and the numbers do not line up, so the
figure's *shape* is a finding and the *explanation* is not established. And the per-point
values are in a figure, not in text.

The project's own measurement is sharper and should be used instead. Given in the brief:
widening the neural lane from k=7 to k=15 buys **+28.7 % tokens per round**. **[INFERENCE]** from
that and from 58 % acceptance: neural accept length at k=7 is 5.06, at k=15 is 5.06 × 1.287 =
6.51, so the marginal per-position acceptance on positions 8-15 is
`(6.51 − 5.06) / 8 = 0.18`, against 0.58 on positions 1-7. **Per-position acceptance decays by
about 3× across that boundary.** Two implications: a k=15 copy round cannot be assumed to hold
the k=7 neural arm's precision, and widening much past 15 is very unlikely to pay, because the
marginal positions are already the weak ones.

### 5.4 No shipped engine widens a copy round — and the reference implementation calls it future work

This is the sharpest finding in the note for the design under review.

TensorRT-LLM's SA override uses **one** `max_draft_len` for both arms: `K = draft_tokens.shape[1]`
is the neural drafter's width, and the SA row is written into the same `K` positions. There is
no width parameter anywhere in `SADraftEnhancer`. Baseten says so in prose **[VENDOR]**: the
gains come "in many cases … without requiring any changes to configuration parameters such as
draft length."

And in the same post, under the heading **"Areas for further work"**:

> Dynamic-length speculation, where the draft length is adjusted based on speculation confidence
> on a per-request, per-micro-batch basis.

So the reference implementation of this exact feature, by the authors of the feature, does not
vary the width, states the fixed width as a deployment advantage, and lists per-round dynamic
width as work not yet done. The design's variable-width choice is therefore **unvalidated by any
external source**, and it is the one part of the design where the strongest available
implementation deliberately did something different.

### 5.5 One design detail worth copying

Baseten's production engine decides **how many** tokens come from the automaton, not whether all
of them do **[VENDOR]**: "These match lengths are compared against a threshold to decide how many
draft tokens come from suffix-automaton continuations versus multi-token prediction sampling."
The open-sourced `SADraftEnhancer` is the coarser all-or-nothing version. If this project wants
a graded response to match length, the proportional rule of §2.2 is the published form of what
Baseten describes, and it is strictly between all-or-nothing and a fixed wider window.

---

## 6. What predicts copy acceptance

### 6.1 The signal is close to binary, which is why a threshold works at all

Baseten **[VENDOR]**, and it is the load-bearing fact behind every threshold rule in this note:
where SA matches on code generation "the accept length is 10+ with long context", and where it
does not apply — reasoning and writing — "accept rates near 0". SAM Decoding **[PAPER]** puts the
same assumption in one sentence: "the length of the suffix match can indicate the confidence of
the draft produced by the automaton, where long matches imply that more tokens are likely to be
acceptable."

If that bimodality is real, then a threshold is close to optimal and a better predictor cannot
help much — which is consistent with the fact that nobody has built one.

### 6.2 The authors of the rule call it heuristic, and name the fix they did not do

SAM Decoding §7, Limitation, verbatim:

> when combining SAM-Decoding with other types of methods, we use a very heuristic approach,
> i.e., we choose different methods depending on the match length. This does not fully utilize
> the exact match lengths provided by the suffix automaton, so subsequently we will try to train
> classifier to select different decoding methods at each generate round.

**So: no published, measured predictor of copy-proposal acceptance used to *choose*.** The one
group that shipped this rule names a learned classifier as future work and does not have one.
The gap in the literature is the same shape as the gap in production: match length as a scalar
threshold, nothing more.

### 6.3 The two precedents for a running acceptance rate, both for a different decision

- **TRT-LLM `speculation_gate.py`, `SpeculationGate`**: an O(1) rolling mean of per-round
  acceptance over a configurable `window`, compared to a `threshold`, which **permanently**
  disables speculation once the window is full and the mean is below. It never re-enables
  (except via `reset()`). This is a kill switch, not a router.
- **TriForce**, arXiv:2404.11912v3 §4.3: "A reconstruction of `Cr` is triggered either when the
  rolling average acceptance rate drops below a threshold or at a designed stride."

Both establish that a rolling acceptance average is an accepted control signal in this
literature. Neither uses it to choose between a copy and a neural draft, and neither publishes
a window or a threshold.

### 6.4 llama.cpp's streak counter, for completeness

`common/speculative.cpp` keeps a per-sequence `n_low` low-acceptance streak and resets the
**whole** shared hash pool when five consecutive rounds from any sequence fall below 0.25
acceptance (sibling note §6.3). It is the third shipped instance of "a counter of bad outcomes
drives a decision", and it is the only one whose consequence is destructive enough to have
caused a reported regression.

---

## 7. Findings that bear on the chosen design

Ordered by how much they should change a decision.

1. **The signal is settled and simple: the match length, against a fixed integer threshold,
   per request, per round, deciding the whole block.** This is what one engine ships and what
   the paper that produced it specifies. It requires no running state, no warmup and no
   hysteresis, which is a large simplification for a decision core that is already
   exhaustively unit-tested over 1,080 boundary cases — the boundary that matters is the
   threshold comparison, and it is one comparison.
2. **The default threshold in the one shipped engine is 4, and the paper's tuned optimum is 5,
   with degradation above it** (SAM Fig. 6). Two independent sources landing on the same
   single-digit value for a token match is the strongest external evidence available for where
   to start. Note the consequence of a too-high threshold: it does not merely lose the copy
   round, it *replaces* a good copy with a worse neural draft.
3. **The variable-width choice is unvalidated and contrary to the reference implementation's
   own practice** (§5.4). It is not wrong on the arithmetic — §4.3 puts the break-even at 29 %
   copy acceptance, and §5.3 puts a k=15 neural round at 6.51 tokens, so the margin looks real
   — but no external measurement supports widening a copy round specifically, and the authors of
   the feature list dynamic width as future work. This is the part of the design most worth a
   local A/B before it is treated as settled.
4. **The acceptance criterion should be written against low single digits to mid teens of
   percent, not 30-40 %.** §3.3's gated-on-a-strong-drafter column and the HumanEval
   acceptance-length *regression* are the honest range for a lane already at ~5 tokens per
   round. The 30-40 % figures come from repetition-heavy code-edit workloads with an 8-layer
   drafter on datacenter hardware.
5. **The tuning target is the gate's false-positive rate, not the mixing rate** (§4.3). The
   mixing rate cancels out of the break-even entirely. If the decision core is going to be tuned
   against something, tune the precision of the threshold against the 5.06-token neural
   baseline.
6. **A graded rule is available and unpublished as a comparison point**: proportional width,
   `max_spec_tokens = factor × match_length` (§2.2, vLLM `suffix_decoding_max_spec_factor`,
   default 1.0), plus a per-token frequency floor (§2.3, default 0.1). This sits between
   all-or-nothing and a fixed wider window, and it is what Baseten describes their own
   production engine doing.
7. **A host-side copy author is the low-risk choice and the sources support it.** The one
   shipped selector computes its match on a device-resident automaton on a side stream, but its
   *decision* is one integer comparison, and llama.cpp's host-side `ngram-mod` chain is the
   longest-shipping precedent. Nothing found here requires the index to be on the device for the
   *selection* to work — only for the SA implementation's specific throughput numbers in §3.2.

---

## 8. What I could not establish

Stated plainly, because these are the gaps a reviewer would otherwise assume were covered.

1. **No published false-positive or true-positive rate for any match-length threshold.** Not
   for `sa_spec_threshold`, not for `l_threshold`. I read the TensorRT-LLM PR body, its 104
   comments, the `basetenlabs/sa_spec` repository, the Baseten post and the SAM paper. SAM's
   Figure 6 is the only threshold sweep and its per-point values are not in text. This is the
   single most important missing number for this design.
2. **No measured cost of a wrong guess**, in any engine. The "zero overhead" claim is about the
   gate being permanently closed, not about a rejected copy proposal. I searched for a number
   and did not find one; the absence is the finding.
3. **No measurement of a variable-width copy round**, on any hardware, against any baseline.
   §5.4 is the evidence that it is unexplored rather than evidence that it is bad.
4. **No measurement of the combination on a masked-draft drafter (DFlash/DFlash2), and none on
   a consumer discrete GPU at RTX 5090 class.** This is unchanged from the sibling note's §7.4
   item 1 and it is still the gap that matters most for this product. The vLLM dynamic-SD docs
   listing DFlash among the tested methods is the nearest thing and it carries no numbers.
5. **No adaptive width driven by measured acceptance** in any engine, as distinct from
   batch-size-driven width. Both shipped schedules are concurrency-keyed, and neither publishes
   its gain.
6. **The role of the host/device split in the selection rule is unquantified.** TRT-LLM's SA
   beats NGram by 25 % throughput at identical acceptance (§3.2), but that is the index, not
   the gate, and the split is entangled with the overlap scheduler in that engine.
7. **Not re-checked:** FastDeploy's `with_ngram` (concatenation, so no selector), SGLang's
   `ngram_worker.py` internals, and the MLC-LLM path. The sibling notes cover the first two.
8. **Two sources are behind a figure rather than a table** and I have reported their shape
   only: SAM Decoding Figures 6 and 7. I did not read the PNGs, so I have not attributed
   magnitudes to them.

---

## 9. Sources

Read 2026-09-27. Cited by file + symbol/section, for the reason given in the method note.

**TensorRT-LLM** (`main`) — `tensorrt_llm/_torch/speculative/sa_enhancer.py`
(`SADraftEnhancer`, `__init__`, `_ensure_stream`, `_ensure_buffers`, `extend_and_prepare`,
`maybe_override_all_draft_tokens`); `speculation_gate.py` (`SpeculationGate`,
`record_acceptance_rate`); `sa_worker.py` (`SAWorker._generate_draft_tokens`);
`suffix_automaton.py`; `mtp.py` (`MTPWorker.__init__`, `sample_and_accept_draft_tokens`,
and the two override call sites); `interface.py` (`SpecMetadata.runtime_draft_len`,
`runtime_tokens_per_gen_step`); `llmapi/llm_args.py` (`MTPDecodingConfig.use_sa_spec`,
`sa_spec_threshold`); `docs/source/features/speculative-decoding.md` (SA Enhancement; MTP;
dynamic tree); `cpp/tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomaton.h`
(`lookup`, `lookupFixed`, `getRequiredMemorySize`). PRs
[#11434](https://github.com/NVIDIA/TensorRT-LLM/pull/11434) (merged 2026-03-03 — both benchmark
tables and the `sa_spec_threshold` review comment),
[#10860](https://github.com/NVIDIA/TensorRT-LLM/pull/10860) (dynamic draft length,
`draft_len_schedule`),
[#12262](https://github.com/NVIDIA/TensorRT-LLM/pull/12262) (expansion to DFlash / MTP / PARD /
SA / hybrid-SA),
[#15775](https://github.com/NVIDIA/TensorRT-LLM/pull/15775) (rejection sampling scope).

**Baseten / `sa_spec`** **[VENDOR]** — `basetenlabs/sa_spec` (`dev` branch): `README.md`,
`bench/config.yaml` (`num_nextn_predict_layers: 8`, `use_sa_spec: true`),
`bench/bench_bs_1.json`, `src/sa_spec/suffix_automaton.hpp` (`SuffixAutomaton::lookup`,
`get_draft_tokens`).
[Open-sourcing Baseten's suffix automaton MTP accelerator](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/),
Mahmoud Hassan, Baseten Model Performance Team, 2026-05-05. First-party for their own product;
used for the rule description, the per-domain acceptance claims, the "Areas for further work"
statement and the threshold=∞ check. Not independently reviewed.

**SAM Decoding** **[PAPER]** — Yuxuan Hu, Ke Wang, Xiaokang Zhang, Fanjin Zhang, Cuiping Li,
Hong Chen, Jing Zhang, "SAM Decoding: Speculative Decoding via Suffix Automaton",
[arXiv:2411.10666v3](https://arxiv.org/abs/2411.10666) (16 Dec 2024). §3.2 (static vs dynamic
automaton, `l_bias`), §3.4 (Adaptive Draft Selection, `l_threshold`), §4 Table 1 and the
Figures 5-7 discussion, §7 Limitation. HTML rendering read.

**vLLM** (`main`) — `vllm/config/speculative.py` (`SpeculativeMethod` literal,
`num_speculative_tokens_per_batch_size`, `suffix_decoding_max_tree_depth`,
`suffix_decoding_max_cached_requests`, `suffix_decoding_max_spec_factor`,
`suffix_decoding_min_token_prob`, `enable_adaptive_verification`);
`vllm/v1/spec_decode/suffix_decoding.py` (`SuffixDecodingProposer.__init__`, `.propose`);
`vllm/v1/spec_decode/` directory listing (absence of a router);
`docs/features/speculative_decoding/suffix.md`;
`docs/features/speculative_decoding/dynamic_speculative_decoding.md`;
`docs/features/speculative_decoding/n_gram.md`. Also read: vLLM-Ascend's
[Speculative Decoding Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/speculative_decoding.html)
for its statement that outer batch-size-based dynamic SD is disabled where the method is already
per-request dynamic **[community: a downstream port, first-party for its own fork]**.

**TriForce** **[PAPER]** — Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter, "TriForce:
Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding",
[arXiv:2404.11912v3](https://arxiv.org/abs/2404.11912) / OpenReview `HVK6nl3i97`, §4.3 (the
rolling-average-acceptance-rate trigger). Read for the control-signal precedent only; its
retrieval drafter is a KV cache, not a text copy.

**Arctic Inference / Snowflake** **[VENDOR]** — [SuffixDecoding at Production Scale with Arctic
Inference and vLLM](https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/),
2025-12-02. Used for the published speedup ranges and the fixed-tree-depth / per-request-width
description. The underlying technical report is arXiv:2411.04975; **I did not read the report
itself**, so the mechanism claims in this note come from the vLLM source, not from the paper.

**llama.cpp** — `common/speculative.cpp` (`n_low` streak, the reset at 0.25/`n_low >= 5`, the
priority list and `dp.drafting`), `docs/speculative.md`. Re-read only for the selection
question; the rest is the sibling note's §1.

**Not consulted:** the two prior notes' own fetch dates are unchanged and their contents are
not restated here. I did not re-verify their line numbers.
