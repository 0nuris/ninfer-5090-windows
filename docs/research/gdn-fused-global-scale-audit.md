# The GDN fused-global-scale defect: audited, not present in the four shipped lanes

**READ-IN-SOURCE 2026-09-29. Found by reading a paper about this exact model, then checked against
this tree's own code. Nothing changed and no number was re-measured. The correctness claim is
structural; the optimality claim is not made.**

## Why this was worth checking

[arXiv 2609.04098](https://arxiv.org/abs/2609.04098), "Why Gated DeltaNet Survives 4-Bit
Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM" (2026-09-03), is about
**Qwen3.8-27B** — the model these four lanes serve. Its section 6 documents a defect that silently
corrupts a hybrid NVFP4 deployment, and its worst property is that the corruption *improves* the
metric this project measures:

> The corrupted model is deceptively plausible: reasoning degrades moderately (AIME 80.8 vs. 86.7
> repaired) while long-context perplexity gets *better than BF16* (a flat 6.86 at 32K vs. the true
> 10.84) — a broken forget gate makes the state hold everything, which happens to help next-token
> prediction on WikiText.

[perplexity-baseline.md](../perplexity-baseline.md) is the authority for four lanes, and several
closed items in [active-work.md](../active-work.md) rest on it. A model whose long-context perplexity
is *too good* would not have been questioned by anything this tree runs.

## The mechanism

A checkpoint calibrates one FP32 global scale per linear **module**. A serving kernel fuses those
modules into **one GEMM** and takes the maximum of the constituent global scales without rescaling
the local ones. The fused group is then computed with mis-scaled weights. In the paper's checkpoint the
paired scales differ by **1.82x** (`in_proj_qkv` + `in_proj_z`) and **2.75x** (`in_proj_b` + `in_proj_a`)
in **all 48 GDN layers**, so the decay and write gates are computed wrong.

The checkpoint-side repair is to rewrite each fused group to the shared global scale and fold the ratio
into the per-block E4M3 scales (94 scale sets, worst ratio 2.81x, re-rounding error under 6.2%).

The paper notes this is invisible on checkpoints that keep fused-adjacent modules at equal scales — it
audited Unsloth and RadixArk and found their fused groups uniform. So this is a check worth making
rather than a hazard to assume.

## Check one: the GDN control projections are BF16 by construction

The `b`/`a` pair is the one carrying the paper's larger 2.75x divergence, and this port cannot hold it
at NVFP4 in the first place.

`src/ops/weight_input.cpp:191-207` — `prepare_gdn_gating_proj_weights` takes `a` and `b` as a pair,
which *is* the fused GEMM, and requires BF16 on both paths:

```cpp
ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a, const WeightInput& b) {
    ...
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    require(shape[0] == 48, "GDN control: this geometry requires a combined parent");
    ...
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
```

The call site is `src/models/qwen3_5/execution/parameters.cpp:103-104`, which passes
`a_projection` and `b_projection` into that function. All three guard strings
(`GDN control requires BF16 weights`, `GDN control: unsupported A/B geometry`, `GDN control: this
geometry requires a combined parent`) are present in the shipped `build/apps/ninfer-serve.exe`, so the
guard is compiled in rather than only described.

**The reason is geometric, and it is in our own converter.**
`tools/convert/official_recipes.py:220-224` records it:

> QUASAR's export carries 496 fused NVFP4 sites covering every text linear, and it quantizes
> `gdn/a_projection` and `gdn/b_projection` too, where the other two sources leave them BF16. Those
> two are BF16 from the base checkpoint regardless: they are (96, 5120) and
> `block_scale_k16_m128x4_v1` requires N divisible by 128, so the layout cannot hold them at all.

Concatenated, `a` and `b` are `(96, 5120)`, and `96 % 128 == 96`. The block layout cannot represent
them. So every recipe keeps them BF16, the fused `b`/`a` scale pair does not exist in any of the four
artifacts, and the paper's 2.75x divergence has nowhere to live.

This is worth stating plainly because it is a coincidence of geometry, not a designed defence. It
happens to produce the right answer for a reason unrelated to the paper's finding — which is also why
the second check matters.

## Check two: fused NVFP4 parents must agree on the activation divisor

`src/ops/weight_input.cpp:97-106` compares divisors **bit-for-bit** across a fused group whenever the
A4 route is enabled, and refuses the load otherwise:

```cpp
} else if (allows_a4(policy)) {
    // Current NVFP4 consumers use A16 for AllowA8 as well. Only their A4 route
    // quantizes the shared activation and therefore requires a common divisor.
    require(std::bit_cast<std::uint32_t>(divisor) ==
                std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
            "shared NVFP4 native input requires identical activation divisors");
}
```

The recipes pass `activation_policy="AllowA4"` on the NVFP4 sites, so the check is live rather than
dead code. A fused group whose members disagree is rejected at load time with a named reason.

## Verdict

The four shipped lanes are **not** in the corrupted state the paper describes, and one of the two
scale pairs that cause it cannot exist here by layout rather than by policy. Both checks are read from
the source at the line, and the BF16 guard is confirmed in the shipped binary.

**What this does not establish.** It is a correctness audit of one specific defect class, not a claim
that the artifacts are optimal, and not a claim that no other quantization defect exists. It changes
no number in [perplexity-baseline.md](../perplexity-baseline.md).

## The two claims in the same paper that would be gains, and remain untested

Neither is a correctness issue, and neither has been measured here.

**The community's protection of the GDN gate projections is unnecessary.** Every public 4-bit build of
this model keeps `a` and `b` in BF16, on the intuition that a recurrence accumulates error. The paper
measures the opposite: fully quantizing those two moves the layer output by 2.1% and 2.6%, the *smallest*
effects of any projection, because the log-space `softplus`/`exp` parameterization compresses an ~11%
GEMM error to a ~2% output error. Its Minima build quantizes all 496 linears including the GDN block,
and the 48 GDN layers are 5.5B parameters, about 23% of decode weight bytes. If that transfers, the
saving is real — **but this port cannot take it anyway**, because of the `(96, 5120)` geometry above.
The 2.3 GiB that would save is the same 2.3 GiB the community is already not saving. That is a
notable instance of a real result that does not apply to this design.

**Calibrated FP8 KV scales recover 83% of the long-context KV penalty.** The paper's numbers: FP8 KV
with scale 1.0 costs +0.13 perplexity at 32K for BF16 and +0.41 for the quantized model, 3x larger
because W4A4 K/V projections leave less headroom before the cache rounds again. Shipping calibrated
per-layer static scales drops 10.84 to 10.50, recovering 83%, with throughput unchanged within 0.4%.
This port already measures all four lanes on `--kv-dtype fp8`, so this is a live question about a
number the project reports — see item 10 in [active-work.md](../active-work.md), which concerns the A4
activation divisor and is adjacent but not identical.

## One methodological finding that outranks both

The paper's own section 6 lists three ways a Qwen3.8-27B NVFP4 measurement can be wrong for reasons
unrelated to quantization. The most important for this project:

> BF16 Qwen3.8-27B scores the *same* tokens worse inside a 32K request than in isolated 4K windows
> (PPL 6.95 to 10.35; deterministic; reproduced identically in vLLM and in the reference
> implementation to three decimals; retrieval at 64K remains 100%). This is a property of the model,
> not of quantization — but it means "PPL@32K" comparisons are only meaningful within one serving path
> and window protocol.

This port's corpus is scored at **4096 context with 2048 stride**, so it sits entirely in the window
regime where that inversion does not apply, and all four lanes are scored under one fixed protocol —
which is the condition the paper says the comparison needs. The per-domain spread measured on
2026-09-29 (see [nvfp4-block-scale-4-vs-6.md](nvfp4-block-scale-4-vs-6.md), where a 3.56% swing on
`chinese_reference` cancelled a 2.82% swing on `english_reference` into a 0.36% overall move) is a
different effect at the same scale, and the paper's mechanism does not explain it.

Two further paper warnings, recorded for whoever measures next: serving a **multimodal composite**
makes vLLM take a multimodal position-encoding path even for pure text, scoring long context
differently (PPL@32K 10.04 composite vs 10.22 text-only for the *same* BF16 model) — this port's
artifacts carry a vision component, so a text-only extraction is the clean comparison; and
**raw-completion harnesses are invalid for this model**, because "thinking disabled" never reaches it
without a chat template, producing swings of +/-40-60 points in both directions.
