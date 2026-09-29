# The per-domain KL instrument: design, decided before it is built

**DECIDED 2026-09-29 from four cited sources. No number in this file is measured on this project;
every figure is labelled with where it came from. The first thing to measure is whether the top-60
choice behaves here as it does there.**

The perplexity authority names the missing instrument precisely: per-domain KL against a BF16
reference, because the four lanes span **6.4 %** on `chinese_reference` against 1.8 % overall and no
lane wins everywhere ([perplexity-baseline.md](../perplexity-baseline.md)). This records the design
before it is built, because the design is the part worth keeping and two of its constraints are not
obvious until they are looked up.

## Sources, and what each contributed

**[vLLM score-mode KLD](https://github.com/phaelon74/vllm/blob/feature/score-mode-ppl-kld/docs/features/score_mode.md)** —
the two-pass protocol. Phase 1 runs the reference with `return_prompt_logits=True` and writes the
reference distribution to safetensors; phase 2 runs the candidate with `kld_mode=True` and computes
the divergence on GPU. **One model resident at a time**, which is the whole point.

**[arXiv 2606.19558, "Displacement Is Not Direction"](https://arxiv.org/pdf/2606.19558v1)** — the
top-k divergence and its floor. When a reference-important token falls outside the candidate's own
top-k, the candidate has abandoned it, the renormalized log-probability is −∞, and the divergence is
unbounded. A fixed log-probability floor keeps it finite; the paper's Eq. 5 is the definition.

**Erlidev's Qwen3.6-27B quant comparison** — the storage arithmetic and the tail measurement. Top-60
is 480 bytes per position against about 485 KB for a full 248,320-entry distribution, and the mean
reference tail mass outside the top 200 measured **0.0025**.

**llm-compressor issue 2031** — why this is not one pass, and the measured cost of doing it naively:
extracting full-vocabulary logprobs for 10 tokens with Llama-3-8B runs at 1.1 tokens/s, which puts a
WikiText KL at roughly **64 hours**.

## The decision

**Two passes, one model resident at a time, top-k reference records on disk.**

The memory argument is arithmetic and it settles the design on this machine before any preference
does. Two 17.65 GiB artifacts are **35.3 GiB of 47.8 GiB**, leaving 12.5 GiB for the KV cache and
workspace the scorer needs. Loading both is not merely inelegant here; it leaves too little room, and
the failure would be an allocation error part way through a run rather than a refusal at the start.

The storage argument settles top-k over the full vocabulary, checked against this project's own corpus
of 1,044,876 scored tokens:

| record | per position | over this corpus |
|---|---:|---:|
| full 248,320-entry distribution | 485 KB | **483 GiB** |
| top-60 (I32 index + F32 log-prob) | 480 B | **478 MiB** |

A factor of **1035**. That is the difference between impossible and routine.

Top-60 rather than top-200 because the tail argument cuts the other way at this vocabulary: the mean
reference mass outside the top 200 measured 0.0025, so 60 costs almost none of the mass and a third
of the I/O of 200.

## What this port already has

* `plan_windows(tokens, context, stride)` in [evaluation.h](../../apps/perplexity/evaluation.h) — the
  sliding window, in the same shape EXL3 uses (2048 context, 512 stride, every position in every
  window), so the KLD positions are the perplexity positions and the two tables are comparable.
* `ScoreAggregate` accumulating NLL per domain, so the per-domain reduction is a second accumulator
  over the same windows rather than a second scoring pass.
* `ProgramImpl::causal_score` already materialises the full `[vocab, columns]` BF16 logits per tile
  before reducing them, so the candidate's top-k reduces a tensor that already exists.

## What has to be built

1. A `topk_logprobs` Op beside the existing `target_logprobs`, reducing a tile to k per column.
2. A scoring mode on `causal_score` returning k indices and k log-probabilities per position instead
   of the target token's single log-probability.
3. The engine surface for it, and a reference-record format on disk.
4. The per-domain aggregation, with the floor of arXiv 2606.19558 Eq. 5.

## Why this instrument and not more perplexity

The GDN fused-scale audit already supplies the argument. Its defect makes long-context perplexity
*better than BF16* while reasoning degrades, because a broken forget gate makes the state hold
everything. Perplexity cannot see that. KL against a reference can, because it compares
distributions rather than the likelihood of the tokens that happened to occur.

The paper's measurement protocol is a warning rather than a method, and it is one this project already
satisfies. It reports that BF16 Qwen3.8-27B scores the same tokens *worse* inside a 32K request than in
isolated 4K windows — perplexity 6.95 to 10.35, deterministic, reproduced to three decimals — so PPL@32K
is only meaningful within one serving path and window protocol. This project scores at 4096 context
with 2048 stride, entirely inside the regime where that inversion does not apply, and all four lanes
share one protocol. A KL table produced on the same windows would be comparable to the existing table.

## What is not established

No number here is measured on this project. The design is chosen from the cited figures, and the first
thing to measure is whether the top-60 floor behaves here as it does there, **on this vocabulary**. If
the tail mass outside the top 60 is materially larger than 0.0025 on this model, k has to rise and the
storage argument has to be re-run rather than assumed.

Two further gaps the design does not close. The response-only protocol from the same paper — scoring
divergence only where the next token is a response token — needs a corpus of prompt/response pairs,
and this project's corpus is plain text, so the first version would be whole-corpus divergence only.
And KL is asymmetric; the direction matters, and the headline should be `KL(p_BF16 ‖ p_quant)`, the
expectation under the reference, because that penalizes the quant precisely where the reference places
mass.
