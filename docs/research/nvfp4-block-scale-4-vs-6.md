# NVFP4 block scale: 4 or 6

**MEASURED-HERE 2026-09-29. The answer for this model on this corpus is 6, which is what the code
already did. The change was reverted; this file is the record of why it was tried and what it cost.**

## The hypothesis

arXiv 2512.02010, "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling", makes
a specific claim about LLM post-training quantization:

> We find that for many blocks in weight, gradient, and activation tensors during both pre-training and
> post-training quantization (PTQ), scaling to 4 rather than 6 introduces less error, leading to more
> accurate models.

> We find that error from scale factors has a minimal effect on model performance, and that NVFP4
> performance degradation can be entirely attributed to error introduced by casting values to FP4.

E2M1's magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6. Scaling a block so its largest value reaches 6
puts the block's values in the coarse upper half of that grid, where the step between 4 and 6 is 2.
Scaling to 4 instead puts them in the dense lower half, at the cost of the single largest value
saturating. The paper's claim is that the trade is usually worth it.

## Why it was worth trying here, and why the prior was against it

`tools/convert/quantization/nvfp4.py` did not choose 6 arbitrarily. Its own docstring records that the
two scales follow **Transformer Engine's NVFP4 recipe**, and that this was "confirmed against a
ModelOpt checkpoint whose scaled codes saturate the format maximum exactly". So 6 is NVIDIA's own
choice, verified against a real NVIDIA artifact, and the experiment is a deliberate deviation from it
rather than a fix to a defect. That is the correct thing to try once, and then stop.

## Method

Two constants move together, or the change is not the one the paper describes. The block scale is
`block_amax / D` and the stored weight divisor is `FULL_RANGE / global_amax` with `FULL_RANGE = D * 448`,
so that `s_block` still maxes at E4M3's 448. Leaving the global scale at 6 while the block scale moved
to 4 would put `s_block` at `448 * 6/4 = 672`, past the E4M3 maximum, and every large block would clamp
-- which would have measured the clamp rather than the block scale. The activation side
(`tools/convert/calibration.py`, still `FULL_RANGE = 2688`) was deliberately left at 6 so the
comparison isolates the weight block scale.

Both arms were freshly converted from `Qwen3.8-27B-NVFP4-unsloth` with recipe
`qwen3_8_27b_nvfp4_unsloth`, then measured on the full corpus: 1,044,876 tokens, fp8 KV, 4096/2048.

The control is what makes this trustworthy. A fresh conversion with the **unmodified** recipe measured
**4.998419 overall with all four per-domain values identical to the shipped `nvfp4full` artifact**,
which had been measured the same morning. Acceptance is bit-reproducible and perplexity is not, so
without that agreement a difference in the variant could have been run-to-run variation. Conversion
took 124 s and produced 1150 objects; the variant produced the same 1150 objects and the same file size,
as expected when the format is unchanged and only the codes and scales differ.

## Result

| domain | scale 6 (control) | scale 4 (variant) | change |
|---|---:|---:|---:|
| `chinese_reference` | 6.509076 | 6.740745 | **+3.56 %** |
| `english_long_form` | 8.304946 | 8.349742 | +0.54 % |
| `english_reference` | 6.779242 | 6.588411 | **-2.82 %** |
| `ninfer_code` | 1.691167 | 1.695662 | +0.27 % |
| **overall** | **4.998419** | **5.016626** | **+0.36 %** |

**Overall is worse by 0.36 %, so the paper's recommendation does not hold for this model on this
corpus, and NVIDIA's 6 stands.** The change is reverted.

## The part worth keeping

The overall figure is the least interesting number in the table. The per-domain swings are three to
thirteen times larger than it, and they do not agree in sign: `english_reference` improves by 2.82 %
while `chinese_reference` degrades by 3.56 %, and the two largely cancel, leaving only +0.36 % visible.

That is the concrete case for a per-domain instrument on this project, and it is the third independent
observation pointing at the same thing. An overall perplexity number on this corpus can hide a 3.5 %
domain swing, which means it can hide a regression in exactly the domain a product is used for. It also
means the reverse: a change that helps one domain and hurts another is indistinguishable from no change
at all, so it gets discarded.

`ninfer_code` moved least, at +0.27 %. Whatever the block scale is doing, it is doing least of it to
code.

## What this does not establish

One model, one lane, one corpus, one seed of the conversion. It does not show that 4 is worse for
other models, that the paper is wrong, or that a hybrid -- 6 for some sites and 4 for others -- would
not win. It does show that the sweep is a real lever with per-domain effects an aggregate cannot see,
which is worth having measured before spending more time on it.

The other candidate from the same research, per-channel smoothing, was not tried. Its headline numbers
are from diffusion models; the LLM evidence is NVIDIA's ModelOpt integration and a QAT-comparable
quality claim, not a published perplexity delta on a model of this size. It remains untested here and
should not be assumed to behave like this experiment did.
