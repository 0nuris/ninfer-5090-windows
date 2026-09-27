# N-gram / prompt-lookup speculative drafting: evidence outside the GitHub trackers

Research note. Written 2026-09-27. Companion to
[`ngram-drafting-designs.md`](ngram-drafting-designs.md) (the GitHub-tracker survey), which this note
does **not** repeat. Where the two disagree, the disagreement is called out in §7.

Every claim below carries its number, its configuration, and an evidence class:

| label | meaning |
|---|---|
| **[MAINT]** | written by a maintainer/author of the system described, about their own system |
| **[VENDOR]** | first-party for a product being sold; not independently reviewed |
| **[USER]** | independent user report, not a maintainer of the engine |
| **[UNVERIF]** | a number I could not trace to a measurement, a config, or even a consistent source |

**Sources actually searched**, and what came back:

| source type | searched | result |
|---|---|---|
| HuggingFace discussions | yes | 2 substantive (both Unsloth GGUF repos, both from `solarkyle` and `thc1006`) — §3, §4 |
| Reddit r/LocalLLaMA | yes | 8+ threads. **Caveat:** direct fetch of `reddit.com` returned HTTP 403 and `old.reddit.com` returned an empty shell for this session, so the Reddit threads below are extracted from the search index, not read from source. Treat Reddit numbers as **[UNVERIF]** on that basis, except where the same figure appears in a fetchable primary. |
| Hacker News (Algolia API) | yes | 6 relevant hits — §6.1, §6.2 |
| arXiv beyond the two known papers | yes | 8 further papers — §5 |
| Engine-author blog posts / engineering write-ups | yes | NVIDIA, Baseten, `0xBakeer`, `thefrontierlab.ai`, `ovidiudan`, DEV.to — §3, §4, §6 |
| YouTube | yes | 1 video, description/transcript only — §6.4 |
| Chinese-language sources | yes | FastDeploy (Baidu), vLLM-Ascend (Huawei), PaddleNLP, Huawei ModelArts, AtomGit/GitCode, WeChat — §5.3, §6.5 |
| Engine changelogs / release notes | searched only via the above | **[no evidence found]** — no llama.cpp/vLLM/SGLang/MLC/Ollama changelog carries a measured ngram figure that is not already in the sibling note |
| "MSpec" | yes | **[no evidence found]** — see §8.3 |

---

## 1. The direct answer to the question asked

> Does n-gram / prompt-lookup drafting pay off when added **on top of** a neural drafter, on a
> high-end **consumer** GPU with limited VRAM, on a **Qwen-family** model?

**There is exactly one measurement that hits all three axes, and it is negative.** There are two
that answer the same question on datacenter hardware, and they disagree with each other by an order
of magnitude. Nothing at all measures the combination on Windows, and nothing measures the VRAM
cost of turning ngram on in any engine.

### 1.1 The one measurement on consumer-class hardware, Qwen3.6, with a control arm

**[USER]** Erik, [thefrontierlab.ai](https://thefrontierlab.ai/mtp-defaults-are-a-trap/),
2026-05-28. llama.cpp build `b9295`. Hardware: **Bosgame M5, Ryzen AI MAX+ 395, 96 GB unified
memory, gfx1151, Vulkan/RADV, one queue** — an integrated consumer SoC, not a discrete 4090/5090,
and ROCm was unavailable. Model quant: **UD-Q5_K_XL**. Context: 461-token ("medium") and
3799-token ("long") prompts. Sampler: presence-penalty 0.0. Method: **five runs per cell, first run
discarded as warmup, median of the remaining four**, 260 runs total.

Chaining test, at each model's own MTP sweet spot:

| Model | MTP only | + ngram-mod | + ngram-mod + ngram-map-k4v |
|---|---:|---:|---:|
| Qwen3.6 27B dense (`--spec-draft-n-max 3`) | 15.60 | **15.95 (+2.2 %)** | 15.81 (+1.3 %) |
| Qwen3.6 35B-A3B MoE (`--spec-draft-n-max 2`) | 52.73 | **50.11 (−5.0 %)** | 51.63 (−2.1 %) |

**Control arm: real.** All three arms are the same binary, same model, same quant, same workload,
same sweet-spot depth, median-of-four. This is the strongest-controlled consumer-hardware answer I
found, and it is the shape of result the question was asking about.

The author's own reading, and the mechanism: "The dense model's 1-2% gain is within noise; the MoE
model measurably loses. On a single integrated GPU with one Vulkan queue, the extra drafters
compete for the same compute the verification pass needs, and that competition outweighs the
marginal acceptance gain."

The author also states the boundary of the claim honestly: PR #23269 (which enabled the chaining)
cites a community user reporting 75 t/s on a **dual 5070 Ti** with chained configs, which the
author calls "suggestive but not isolated (we don't know how much of that came from chaining versus
MTP alone)". So the only discrete-multi-consumer-GPU datapoint in existence is, by the admission of
the person citing it, uninterpretable.

The same post supplies the surrounding control that makes the 1-2 % uninterpretable anyway. His
own MTP depth sweep on the 35B-A3B: 52.98 t/s at `n_max` 0, 58.20 at 2, **52.34 at 3**, 45.58 at
4, 34.22 at 6, 21.92 at 8, 13.35 at the old default of 16 (−75 %). A ±5 % effect is inside the
width of a one-step draft-depth change.

### 1.2 Two datacenter measurements that disagree, both well-controlled

**Positive, for MTP — [VENDOR]/[MAINT]**, NVIDIA TensorRT-LLM PR #11434 (merged) plus the
[Baseten blog post](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/)
(2026-05-05) and [basetenlabs/sa_spec](https://github.com/basetenlabs/sa_spec). DeepSeek-V3.1-NVFP4,
**8×B200, TP 8, 8 MTP layers**, `max_ngram_size = -1` (longest match), dataset
`glaiveai/code_edits_sample`, 100 requests, avg input **413 tokens**, avg output 256:

| Configuration | Output tok/s | TTFT ms | TPOT ms | Accept rate | Accept length |
|---|---:|---:|---:|---:|---:|
| Baseline (no SD) | 98.34 | 83.05 | 9.88 | — | — |
| MTP | 192.37 | 116.55 | 4.76 | 0.2704 | 3.16 |
| **MTP + SA (ngram)** | **299.31** | 113.89 | **2.91** | 0.5513 | 5.41 |

**Control arm: real.** Same model, same hardware, same dataset, three arms. And there is a second,
cleverer control that exists only here: Baseten verified the *routing machinery itself* costs
nothing by setting the SA threshold to infinity — which disables SA drafts while still executing
every SA computation — "and confirming that end-to-end latency matches that of baseline MTP". That
isolates the +38.9 % TPOT to the accepted drafts themselves, not to the extra compute. **No other
source in this note or the sibling note has such a control.**

**Negative, for EAGLE3 — [MAINT]**, TensorRT-LLM PR #12130 (2026-03-12). gpt-oss-120b +
`gpt-oss-120b-Eagle3`, **8×H200, TP 8**, MT-Bench, 80 samples, 2048-token outputs, speedup vs
baseline:

| Batch size | Baseline tok/s | EAGLE3 | EAGLE3 + SA | EAGLE3 + SA + Global |
|---:|---:|---:|---:|---:|
| 1 | 370.45 | 731.33 (**1.97×**) | 617.05 (1.67×) | 979.85 (2.64×) |
| 4 | 1244.39 | 2195.98 (**1.76×**) | 2059.79 (1.66×) | 2736.63 (2.20×) |
| 8 | 2175.31 | 3601.65 (**1.66×**) | 3266.49 (1.50×) | 4295.22 (1.97×) |
| 16 | 3761.68 | 5523.94 (**1.47×**) | 5054.51 (1.34×) | 6315.13 (1.68×) |
| 32 | 5594.57 | 7303.50 (**1.31×**) | 6986.78 (1.25×) | 9873.33 (1.76×) |
| 64 | 7585.72 | 11305.00 (**1.49×**) | 10092.35 (1.33×) | 12677.57 (1.67×) |
| 128 | 11806.55 | 11820.91 (1.00×) | 12760.32 (1.08×) | 13443.63 (1.14×) |

**Control arm: real.** Adding the suffix automaton to EAGLE3 costs **−15 % to −19 % at every
batch size from 1 to 64**. The same PR shows the identical ngram machinery is *positive* on MTP
(+4.6 % to +8.6 % throughput at BS 16 on the two long-context datasets) — and, in the same table,
that on those same two long-context datasets **MTP+SA is 2.3 % and 3.0 % *slower* than MTP alone at
BS 16**. So within one vendor's own PR, on one model family: ngram-on-top-of-MTP is +55 % on
413-token code edits and −3 % on 1.4K–2.1K-token contexts.

**Control arm: real, and the hardware is the whole answer.** 8×B200 and 8×H200, i.e. ~1.4–1.8 TB
of HBM and no memory-bandwidth ceiling at batch 1. Neither §1.1's iGPU nor any consumer GPU in
§3–§4 has that. The mechanism the vendor relies on — device-side suffix automaton updated in CUDA
with no host round-trip, amortised O(1) per update — is exactly what a bandwidth- or
compute-starved consumer card cannot absorb.

### 1.3 What the sources support, stated plainly

1. On consumer / bandwidth-limited hardware, adding ngram drafting to a neural drafter is worth
   between **−5 % and +2 %**, and the +2 % is inside the noise of a one-step draft-depth change
   (§1.1). One measurement, real control arm, Qwen-family. Not a 4090/5090, but a single-queue
   consumer SoC with the same "no spare compute during verification" property.
2. On datacenter hardware, the answer **depends on which neural drafter**: +38.9 % TPOT for MTP on
   short-context code edits, −15 % to −19 % for EAGLE3 at every concurrency 1–64 (§1.2).
3. **No source anywhere measures the combination on an RTX 4090 or 5090**, and none measures it on
   Windows. I searched for both and found neither. (§6.6, §8.2.)
4. No source measures the VRAM cost of enabling ngram in any engine. (§4.3.)

---

## 2. Real deployments with ngram enabled

Asked for explicitly. Three exist; only one is a shipping product with a public number, and that
number is not a controlled ngram measurement.

### 2.1 Sweep — the JetBrains autocomplete plugin **[VENDOR]**

[Show HN thread](https://news.ycombinator.com/item?id=45505487), 2025-10-07, by `williamzeng0`
(Sweep). Quoted: "Out of the box with vLLM, each request had a median latency of 1500ms (too long).
To optimize this, we rewrote TensorRT-LLM to support N-gram speculative decoding, which lets us
serve completions at a median latency of 94ms."

**Control arm: none, and it cannot be repaired from the post.** The 1500 ms → 94 ms (16×) spans
*four* changes at once: a Python engine replaced by a C++ one, a rewritten inference stack, the
N-gram speculative decoding, and a fine-tuned next-edit model. The author says as much in the
[LinkedIn thread](https://www.linkedin.com/posts/william-zeng_we-built-next-edit-autocomplete-for-jetbrains-activity-7377049648684965888-YUEC)
— the stated reason for leaving vLLM was Python overhead. No GPU, no model size, no quant, no draft
width, no batch/concurrency figure is published.

**[UNVERIF]** as an ngram measurement; **[VENDOR]** as a deployment. It does establish that
somebody shipped ngram drafting in a product people pay for. A response on the HN thread from
`disguiseddumpling` is worth quoting because it is the same mechanism seen from the other side:
"I think what your experiments show is that in a lot cases autocomplete can be solved with simpler
methods like n-gram without LLMs. That's why use such a huge boost in performance coming from
speculative decoding. **Usually on any non-trivial tasks spec-decoding can introduce additional
overhead and be slower than just running your original LLM** (from my experience)."

### 2.2 Baseten Speculation Engine **[VENDOR]**

[baseten.co blog](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/):
"On production agentic coding workloads, we see **up to 40 % higher throughput** at equal latency
and **up to 40 % lower latency** at equal throughput, compared to MTP alone." Workload named as
"production agentic coding". The same post states the two regimes it exploits and, usefully, the
regime where it does not:

> "SA Decoding shines at code generation, where the accept length is 10+ with long context, but
> **performs poorly on reasoning and other writing tasks, with accept rates near 0**. Meanwhile,
> MTP produces consistent speed-ups across all domains, though the accept rate is usually only 2-4
> tokens per iteration."

Its own caveat: "The level of speedup depends heavily on the task, with a more pronounced
increase on agentic coding and math tasks." The 40 % is a ceiling on one workload class, not a
general factor. The reproduction numbers are §1.2.

### 2.3 FastDeploy `mtp_strategy: with_ngram` — shipped, documented, **zero published numbers**

Baidu's FastDeploy is the only engine I found that **concatenates** rather than routes: the
[Chinese doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/features/speculative_decoding.md)
and the [English doc](https://paddlepaddle.github.io/FastDeploy/features/speculative_decoding/)
both describe it as "First, MTP generates N draft tokens, then Ngram matching is used to
**supplement** additional draft tokens" — i.e. 2 MTP + 3 ngram into one 5-token verify:

```
--speculative-config '{"method": "mtp", "num_model_steps": 2, "mtp_strategy": "with_ngram",
                      "num_speculative_tokens": 5, "model": "<model_path>/mtp"}'
```

It is not a prototype: PR #3610 added it (2025-08-26), PR #4047 revised it (merged 2025-09-15),
and PR #7103 (2026-03-31) ported `hybrid_mtp_ngram` from a host/serial CPU implementation to a
three-phase CUDA kernel to cut latency and host↔device copies.

**[no evidence found]** — I read the feature docs, both PRs and the PR body for the CUDA port, and
searched for a benchmark. Every published FastDeploy ngram and MTP figure is on **4×H100 or
8×H100, WINT4** (`eb45t-32k-wint4-mtp-h100-tp4.yaml`), and each is measured as a *standalone*
method — the doc gives a command for `method: ngram` and a separate one for `method: mtp` with
`num_speculative_tokens: 1`, and never a hybrid arm. So the concatenation that everyone else has
refused on architectural grounds has shipped, in a production engine, with **no published
measurement of it at all**. Its config knob name, `mtp_strategy`, is itself the design in one
identifier.

### 2.4 A maintainer using ngram, without a number **[MAINT]**

`ggerganov` on Hacker News, 2026-06-16, on
[Vicki Boykis's "Running local models is good now"](https://vickiboykis.com/2026/06/15/running-local-models-is-good-now/):

> "I use the agent for very targeted sessions — basically things that are clear to me how to do,
> just want to automate them. My workflow is usually: new session → read this, this and this →
> do that. I.e. I don't let it wander at all in the codebase, so I rarely exceed the context
> window. Also, I get a lot of mileage from the ngram-based speculative decoding functionality as
> it allows me to iterate on the implementation much faster."

This is the closest thing to a maintainer endorsement, and it carries two conditions worth noting
because they are the same two conditions every measured result in §3 depends on: the workload is
**highly repetitive against the context**, and the context is **short**. His named RTX 5090
figures in the same comment are prefill-only (3714 ± 10.85 t/s at pp2048/d512 on
qwen35-27B Q4_K_M); he publishes no decode figure for the ngram configuration.

**[no evidence found]** — I found no Reddit or HN report of `--spec-type draft-mtp,ngram-mod`
chaining that includes a ngram-off arm. The closest, r/LocalLLaMA
["Need help understanding how spec decode affects token throughput"](https://www.reddit.com/r/LocalLLaMA/comments/1u7llqs/need_help_understanding_how_spec_decode_affects/)
(2026-06-16), runs exactly the combination asked about — `--spec-type draft-mtp,ngram-mod`,
`--spec-draft-n-max 3`, `--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32`,
gemma-4-12B-it QAT UD-Q4_K_XL + `-MTP` GGUF, single slot, `q8_0` KV — and reports 40–70 t/s with
"highest acceptance rate … on MTP is 92 %", but gives no ngram-off number on the same workload. It
is also **gemma, not Qwen**. **[UNVERIF]**.

---

## 3. N-gram alone on consumer GPUs: the negative and positive sides, with controls

Not the combination question, but the baselines any combination claim is measured against, and the
only place where consumer-GPU ngram numbers with real control arms exist.

### 3.1 RTX 3090, Qwen3.6-35B-A3B, ngram **loses** at 100 % acceptance **[USER]**

`thc1006`, published as a
[HuggingFace discussion](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14)
(2026-04-21) and a repo and a
[HackMD report](https://hackmd.io/@thc1006/SJly6IE6Wx). Single RTX 3090 24 GB, SM 8.6, Q4_K_XL,
llama.cpp `9789512`, greedy, **batch 1, Q8_0 KV**, 10 prompts spanning chat / reasoning / code /
multi-turn / 繁體中文:

| config | mean t/s | min | std | draft accept |
|---|---:|---:|---:|---:|
| **baseline (no spec)** | **135.7** | 135.3 | 0.3 | — |
| `ngmod-n32` | 133.7 | 133.5 | 0.1 | 0 % (never hits) |
| `ngram-mod-n24` (srogmann params) | 131.1 | 129.6 | 2.6 | **100 % (35/35)** |
| `ngmod-n{8,12,16,20}` | 129.6–130.0 | 119.8–128.8 | 2–5 | 100 % |
| `ngram-cache` | 119.1 | 65.3 | 27.8 | 100 % (96/96) |
| `ngram-cache` rerun | 118.8 | 65.6 | 27.5 | reproduction |
| `ngram-cache`, 1000-tok output | 115.9 | 60.0 | 28.7 | 100 % (317/317) |
| classic draft Qwen3.5-0.8B `--draft-max 8` | 121.1 | 59.2 | 30.9 | 100 % (270/270) |

**Control arm: real, and unusually thorough.** `baseline-rerun 135.5` and `ngcache-rerun 118.8`
are explicit reproductions; within-config std ≤ 0.4. Ruled out as causes: KV quantisation
(fp16 KV leaves `ngram-cache` at 121.3), output length (300 → 1000 tokens keeps every ratio), and
draft-model vocab mismatch (an earlier `qwen3:0.6b` run silently failed to attach and was
correctly re-run with the vocab-matched Qwen3.5-0.8B).

The result that matters for the combination question: **100 % acceptance and a net loss.** The
author's mechanism, consistent with §1.1: A3B routes 8-of-256 experts per token, sparsity ρ ≈
0.031, so the batch needed to saturate the expert set is `T_thres = log_{1-ρ}(0.05) ≈ 94`. At
draft width K = 5–64 ≪ 94, the verification pass loads the *union* of the per-token expert sets,
paying full memory traffic for every weight. "There is nothing to amortize."

Their later **code/JSON-only** workload variant (5 structured prompts × 3 trials × 3 configs):
baseline 139.22 ± 0.46, `--draft-min 2 --draft-max 32` 66.57 ± 7.57 (**−52 %**), `--draft-min 48
--draft-max 64` 83.84 ± 1.80 (**−40 %**). So the workload-shape hypothesis ("structured prompts
might let ngram win") was tested and refuted on this stack.

**Honest boundary the author states himself:** a v3 retest in a sibling repo on the same 3090s with
**vLLM 0.19.1**, matched flags and `--no-enable-prefix-caching`, flips MTP k=1 to **−21.6 % decode
TPOT (≡ +27.5 % faster decode rate)**, N=5 trials × 5 prompts. So the negative finding is specific
to llama.cpp's draft-then-verify path at K = 5–64, **not** to MTP and **not** to the hardware
class. He also retracts an earlier over-general framing of his own result. I report the
retraction because it is the same discipline §4.1's author applies.

### 3.2 RTX 5060 Ti 16 GB, Qwen3.8-Flash-Next: the ngram **variant** matters more than anything else **[USER]**

`solarkyle`,
[HF discussion #34 on unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/34)
(2026-08-27). RTX 5060 Ti 16 GB / Ryzen 9 9950X / 128 GB DDR5-5600, **UD-IQ4_XS**, llama.cpp PR
#27742 @ `af1ffaf37` (build 10706), one variable changed, `-np 1`, one copy-heavy code edit and one
freeform prose prompt:

| `--spec-type` | code t/s | prose t/s | accept |
|---|---:|---:|---:|
| `none` | 20.75 | 21.12 | — |
| `ngram-mod` | 27.51 (+32.6 %) | 20.96 (**−0.8 %**) | 79.7 % |
| `ngram-simple` | 29.97 (+44.4 %) | — | 90.7 % |
| **`ngram-map-k`** | **30.02 (+44.7 %)** | **21.24 (+0.6 %)** | 92.0 % |
| `ngram-map-k4v` | 31.08 (+49.8 %) | — | 95.5 % |
| `ngram-cache` | **32.46 (+56.4 %)** | **16.06 (−24.0 %)** | 87.9 % |

**Control arm: real** (`none` measured in the same server, same session). Two things the author
draws from it that matter more than the table:

- "`ngram-cache` looks best on code and costs −24 % on prose. **If you only benchmark the workload
  speculation helps, you'll pick it and ship a silent regression on everything else.**"
- `--spec-draft-n-max` has **no effect** on any ngram variant — tested 2, 4, 6, 8, 12, identical
  speed and identical acceptance every time. (`vasya100` in the same thread: "I don't think those
  ngrams work with `--spec-draft-n-max`".)

**No MTP arm**, and the reason is structural: "`--spec-type draft-mtp` fails with *model doesn't
contain MTP layers* — the MTP head isn't in the GGUF." So this is a Qwen-family consumer-GPU
measurement of ngram-vs-nothing, not of the combination.

Author's own noise floor, stated up front: "Single machine, one run per config unless noted —
**differences under ~3 % are noise**." On that floor, `ngram-mod`'s +2.2 % in §1.1 is
indistinguishable from nothing, and `ngram-mod`'s prose number here (−0.8 %) is a real
(no-benefit) reading.

### 3.3 The same author's VRAM table — the only ngram-relevant VRAM numbers I found

Same discussion, same box, `--n-cpu-moe` sweep (expert residency, not ngram — included because it
is the only published VRAM-headroom budget for a Qwen MoE on a 16 GB consumer card):

| `--n-cpu-moe` | layers on GPU | VRAM | tok/s |
|---:|---:|---:|---:|
| 48 | 0 | 8,130 MB | 19.12 |
| 46 | 2 | 11,216 MB | 19.67 |
| 44 | 4 | 13,490 MB | 20.10 |
| 43 | 5 | 14,628 MB | 20.43 |

and: "guides suggesting `--n-cpu-moe 38` or `36` won't fit on a 16GB card — expert weights are
1.34 GB/layer with an 8.1 GB zero-expert baseline"; "`262144` hits the VRAM ceiling — 15.6/16 GB
and drops to 22.54." His landed config uses **~14.3 GB VRAM** at `-c 131072` with ngram enabled
(`--spec-type ngram-map-k`). **[no evidence found]** — he does not report a VRAM delta for turning
ngram on; the ngram variants share one VRAM reading.

### 3.4 DGX Spark, Qwen3.5-122B-A10B: ngram loses, MTP wins, on the same box **[USER]**

[DEV.to, 2026-07-18](https://dev.to/kstoyanovai/speculative-decoding-on-local-hardware-benchmarking-n-gram-mtp-eagle3-and-dflash-on-the-nvidia-2kec).
Single NVIDIA DGX Spark (GB10, SM121), `Qwen3.5-122B-A10B-hybrid-int4-fp8`:

| Method | Avg tok/s | Peak tok/s | Notes |
|---|---:|---:|---|
| **None (baseline)** | **~36–37** | ~38 | reference |
| **N-Gram** | **~24–30** | ~31 | **worse than baseline**; "The overhead of proposing tokens that get rejected outweighs any gain" |
| MTP-1 | ~44–45 | ~46 | accept rate mid-to-high 80s % |
| MTP-2 | ~48–49 | ~51 | accept ~80 %+ |
| MTP-4 | ~34–35 | ~38 | **worse than baseline** |

**Control arm: real** for ngram-vs-nothing. **[UNVERIF]** on configuration: the post does not name
the serving engine in the part I could read, and gives no draft width, match length, prompt
template, or per-run count. I could not fetch it directly to check (the extract is search-index
only), so treat the engine attribution as unknown. Its MTP column is worth keeping only as
directional: it reproduces §1.1's shape — MTP peaks at a shallow depth and falls below baseline by
depth 4 — on independent hardware.

### 3.5 The most methodologically careful ngram measurement available **[USER]**

`0xBakeer`,
[docs/speculative-decoding.md](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/main/docs/speculative-decoding.md).
DGX Spark (GB10, 121 GiB unified memory), `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL, llama.cpp PR
#27742. `run.sh` enables `--spec-type ngram-mod` **by default**. Throughput by novelty, not by task
type:

| Task | Decode tok/s |
|---|---:|
| Return a given file with one line changed | ~70 |
| Fix a bug in a file you were given | ~60 |
| Add a function to a file you were given | ~35 |
| Write prose | ~22 |

Four measured traps, each of which has produced a wrong number in the sources in §3 and §6:

1. **Repeating an identical prompt.** Temperature 0, four runs: identical prompt 59.8, 59.8,
   **169.0, 166.8**; varied prompt 61.7, 70.3, 26.8, 64.1. "It needs two observations before the
   n-grams are usable, which is why nothing happens until the third run, and then it is about
   **2.8×**. More repetitions make it worse, not better." From a chat UI, five identical requests
   (md5-verified identical output) gave 43.2, 95.4, 100.3, 100.3, 100.2 tok/s — "The honest number
   for that task is **43, not 100**."
2. **Not discarding the first run.** run1 132.8, then 172.9 / 172.7 / 171.4 / 171.4 / 171.3 /
   170.7 / 170.4. "Run 1 is 23 % low." — **and then he retracts the inference**: a proper n=3 A/B
   with warmup discarded finds 88.5 cold vs 86.2 warm, ranges 78.6–109.1 and 78.1–114.2, prose
   identical at 27.8, i.e. "no measurable difference either way."
3. **`max_tokens` running past the copied region.** "The same task measured **171 tok/s at one
   `max_tokens` and 52 at another**."
4. **Calling a difference from single runs.** "Identical single runs on this box spread by
   **6.5 %** with nothing changed between them; a matching **6.9 %** spread has been reported under
   vLLM on another DGX Spark with a different quantization and drafter. So two single runs that
   differ by less than roughly **10 % support no conclusion**."

Also: reasoning tokens are generated, never drafted — 33 tok/s with `reasoning_effort: xhigh`
default vs 43 with it off, on the same task. And the MTP comparison, which **is** the
ngram-vs-neural comparison on a Qwen-family MoE: the GGUF converter drops the MTP head, so llama.cpp
cannot run it; via vLLM on the NVFP4 checkpoint, "Measured gain on prose: about **1.16×** across
engines (27.8 → 32.2), flat across task shapes rather than the 3× swing `ngram-mod` shows.
Modest, because on a top-10-of-512 MoE verifying *k* draft tokens activates the union of experts
across those positions." A third-party DGX Spark in-engine MTP-off A/B is cited at **+35 % at one
caller, not measurable at 16 concurrent**.

**A vocabulary-compatible draft model was the thing to reach for and it did not work**: Qwen3.5-0.8B
(vocab 248320, same as target) "gave **no speedup at all**: mean accepted length 2.88, decode
unchanged at ~23 tok/s" — same expert-union argument as §3.1, arriving at the same place from a
different engine.

---

## 4. Reports of ngram *hurting* throughput, and the mechanisms given

Every one of these has a control arm. That is the main reason they are in this note.

| # | finding | configuration | control arm | class |
|---|---|---|---|---|
| a | ngram **−5.0 %** added to MTP on MoE; **+2.2 %** on dense | Qwen3.6 27B + 35B-A3B, UD-Q5_K_XL, llama.cpp b9295, Strix Halo Vulkan, ≤3800 ctx | median-of-4, 5 runs/cell, 260 runs | [USER] §1.1 |
| b | ngram **−15 % to −19 %** added to EAGLE3 at **every** batch size 1–64 | gpt-oss-120b + Eagle3, 8×H200, MT-Bench 2048 out | full 7-point batch sweep, both arms | [MAINT] §1.2 |
| c | MTP+SA **−2.3 % / −3.0 %** throughput vs MTP alone at BS 16 | DeepSeek-V3.1-NVFP4, 8×B200, CrossCodeEval 2118-tok in / SWE-Bench 1359-tok in | same PR, same batch size | [MAINT] §1.2 |
| d | ngram **−24 %** on prose while **+56 %** on code | Qwen3.8-Flash-Next UD-IQ4_XS, RTX 5060 Ti 16 GB | `none` in same session | [USER] §3.2 |
| e | ngram-mod **−3 to −4 %**, ngram-cache **−12 %**, at **100 % acceptance** | Qwen3.6-35B-A3B UD-Q4_K_XL, RTX 3090, batch 1 | baseline + rerun reproductions | [USER] §3.1 |
| f | N-Gram **~24–30 vs ~36–37 baseline** | Qwen3.5-122B-A10B int4-fp8, DGX Spark | baseline in same sweep | [USER] §3.4 |
| g | ngram-cache **−24 %**, and the same variant is the **best** on code | as (d) | as (d) | [USER] §3.2 |
| h | "net slowdowns with N-Gram speculation at this concurrency, **as ARs fail to justify validation costs**" | SPEED-Bench, B200, **BS = 32**, 9 domains | per-domain speedup table | [MAINT-academic] §5.2 |
| i | "ngram-mod, **which I found slows it down**" (R9700, Qwen3.8-27B) while MTP `n-max 7` took 30 → 80 t/s, MTP peak ~65 t/s | `naasking`, HN, 2026-09-18 | none | [UNVERIF] §6.1 |
| j | baseline arm **not beaten at all**; a spec run at 19.4 t/s is 25 % slower than no spec at all | llama.cpp #27852, RTX A4500, Qwen3.8-Flash-Next UD-Q4_K_XL, one slot | yes (three-arm) | in sibling note |
| k | `--spec-type ngram-mod` = **−52 % / −40 %** on code/JSON-only prompts | Qwen3.6-35B-A3B, RTX 3090, `--draft-min 2 --draft-max 32` / `48/64` | baseline 139.22 ± 0.46 | [USER] §3.1 |

### 4.1 The three distinct mechanisms, as stated by their sources

- **Memory-bandwidth × MoE expert-union.** §3.1, and independently §3.5 and §1.1's author reach
  the same conclusion by three routes. At draft width K far below the expert-saturation threshold
  `T_thres`, the verify pass pulls the *union* of K positions' expert sets through the memory
  hierarchy instead of one token's worth, so weight traffic scales with K and there is nothing to
  amortise. §3.1 puts `T_thres ≈ 94` for A3B; §3.5 says the same for top-10-of-512.
- **Compute contention with verification on a single queue.** §1.1, stated in the source's own
  words: "On a single integrated GPU with one Vulkan queue, the extra drafters compete for the same
  compute the verification pass needs, and that competition outweighs the marginal acceptance gain."
  This is the mechanism that specifically does *not* apply on 8×B200, which is where §1.2's +38.9 %
  comes from.
- **Validation cost not amortised at concurrency.** SPEED-Bench's prose (§5.2) and the TRT-LLM
  EAGLE3 column (b) are the same effect: raising accepted length does not raise throughput when
  `BS × K` pushes verification past the critical batch size.

### 4.2 A mechanism nobody costs: the *selection* overhead is measurable, and can be measured away

Baseten's threshold-to-infinity control (§1.2) is the only measurement in either note of the cost
of the *router* itself rather than of the drafts. It came out at "performance remains on par with
vanilla MTP", achieved by keeping the suffix automaton device-side and updating it in CUDA
alongside MTP draft sampling so no new synchronisation point is introduced. The honest reading is
that the overhead is small **only because it was architected away**; a host-side or
synchronisation-crossing router was never measured here.

### 4.3 How much VRAM does the ngram index cost? **[no evidence found]**, with two exceptions

Asked for explicitly. Outside the sibling note's llama.cpp figure of a fixed 16 MiB `int32` pool:

- **llama.cpp: no published VRAM delta for enabling ngram.** §3.2's VRAM table is an
  `--n-cpu-moe` sweep, and his ngram-enabled config's ~14.3 GB is a single reading, not a delta.
  The one measured observation is the sibling note's #23154, where a reporter's VRAM climbed 79 %
  → 100 % with `ngram-mod` on and stayed at 76 % with it off on an RTX 4090 — diagnosed by nobody.
- **TensorRT-LLM / Baseten: no published size.** The device-side suffix automaton states live in "a
  pre-allocated global buffer with a pre-defined constants for the maximum batch size and maximum
  sequence length" (basetenlabs/sa_spec README), with a standing `_todo: make the shapes dynamic`.
  So its footprint is a compile-time function of max-batch × max-seq-len, and no one has published
  the function.
- **SGLang: no published VRAM figure** (sibling note §7.4 item 3 already recorded this).
- **The one large number in the wild is a different design.** `cmrdporcupine`, HN 2026-08-26, on
  [Qwen3.8-Flash-Next](https://news.ycombinator.com/item?id=49448210) and a DGX Spark: "I have nvfp4
  quant fitting fine in 128GB on DGX Spark, but **with paging (from nVME) of the n-gram table**.
  Resident ~80GiB for weights & context… 12 tok/sec decode without speculative decoding (will come
  later)." A questioner in the same thread: "**32GB dedicated to an N-gram table** instead of a
  draft model is an unusual choice for speculative decoding — what made it win over the more
  common draft-model approach here?" — **which the thread does not answer.** Treat 32 GB as one
  user's storage design, not as an implementation cost.
- **llama.cpp's disk-backed corpora** (the `ngram-cache` / `examples/lookup` path) are described in
  detail in a [Chinese write-up](https://blog.gitcode.com/280b2cfc07a983a16650fa231edf5933.html)
  (AtomGit, 2026-09-04) — three tiers (`nc_context` in-memory per prompt, `nc_dynamic` accumulated
  and merged back to disk at exit, `nc_static` a read-only offline corpus used both as a drafting
  source and as a *validator*, since the candidate score is `context_count × static_count`) — with
  no size figure and no throughput number.

---

## 5. arXiv beyond the two known papers

Eight further papers read. Two are directly load-bearing for the combination question.

### 5.1 CopySpec — copy drafting integrated **with** EAGLE, and with an appendix comparison **[MAINT-academic]**

[arXiv 2502.08923](https://arxiv.org/pdf/2502.08923). Rolling-hash lookup of a γ-token suffix
anywhere in context; verified directly. "These candidate tokens are verified directly, enabling
**integration with any speculative decoding setup** and yielding additional speedups without
compromising output quality." Reports "an average **49 % additional speedup** over speculative
decoding for the second [stage]". §3.3: "we mainly focus on the integration with vanilla Speculative
Decoding … for generalizability **but we also integrate it as part of the EAGLE framework and
compare it against baselines in Appendix C** to showcase the technique's potential."

**[no evidence found]** — I did not reach Appendix C's numbers, and the paper's own framing
(select-with-fallback, the same design as vLLM #24344 and TRT-LLM's threshold) means the 49 % is
*not* a concatenation result. No GPU class or model is stated in the abstract. Treat "49 %" as an
unqualified figure until the appendix is read.

### 5.2 SPEED-Bench — the largest cross-domain SD benchmark, and it also has no combination arm **[MAINT-academic]**

[arXiv 2604.09557](https://arxiv.org/html/2604.09557v2) (2026-05-28). Five models, single **B200**
(8× for DeepSeek/Qwen inference and GPT-OSS EAGLE3 training), TensorRT-LLM and SGLang for Qwen3,
**BS 32**, 9 domains, draft chains only. N-Gram speedup at T=0:

| Domain | Llama 3.3 70B, N-Gram | GPT-OSS 120B, N-Gram | (for scale) DeepSeek R1, MTP | Qwen3-Next, MTP |
|---|---:|---:|---:|---:|
| Coding | 1.54 | 1.31 | 2.76 | 3.34 |
| Humanities | 1.39 | 1.35 | 2.53 | 2.68 |
| Math | 1.43 | 1.30 | 2.77 | 3.13 |
| Multilingual | **1.91** | **1.58** | 2.68 | 3.19 |
| QA | **1.21** | **1.27** | 2.52 | 2.71 |
| RAG | 1.51 | 1.31 | 2.61 | 2.94 |
| Reasoning | 1.34 | 1.31 | 2.62 | 2.89 |
| Roleplay | **1.15** | **1.25** | 2.14 | 2.09 |
| STEM | 1.38 | 1.30 | 2.62 | 2.85 |
| Summarization | 1.36 | 1.25 | 2.47 | 2.66 |

**Control arm: real** (speedup vs no-SD, same server). Two caveats I have to state: the table is
labelled `Temperature=0` and appears to be an accepted-length table, so it is not directly
comparable to a throughput figure; and the prose separately claims "we observe **net slowdowns**
with N-Gram speculation at this concurrency, as ARs fail to justify validation costs", which
those >1 numbers do not obviously support. Both are reported as found.

**Critically: SPEED-Bench does not test ngram + a neural drafter.** N-Gram, Vanilla SD, EAGLE3 and
MTP are separate columns. Its MTP models (DeepSeek R1, Qwen3 235B, Qwen3-Next) have **no N-Gram
column at all**, so the two Qwen-family MTP rows above have no ngram counterpart. This is the same
gap as the sibling note's §7.4 item 1, now confirmed at benchmark scale.

### 5.3 The rest

| paper | what it says | usable? |
|---|---|---|
| **PLD+** [2412.01447](https://arxiv.org/html/2412.01447v1) | "In the greedy setting, it even **outperforms EAGLE on four of the tasks** (by a margin of up to 2.31 in terms of avg. speedup)." Ranks candidate spans by attention/hidden-state similarity instead of heuristically. | **Replaces** the neural drafter, does not add to it. Usable as a bound on how much ngram-family drafting can win *instead*, not on the combination. |
| **DReSD** [ACL Findings 2025, 1017](https://aclanthology.org/2025.findings-acl.1017.pdf) | Dense retrieval for drafting: "+87 % acceptance rates, +65 % longer accepted tokens, +19 % faster generation than sparse retrieval (REST)"; best config 4.64× over AR. | **Different mechanism**: approximate-NN over a *prebuilt external datastore*, not in-context n-gram. Its own §2.2.1 separates static from dynamic datastores and says dynamic ones (i.e. PLD) "are not appropriate for a direct comparison". Do not conflate. Also: "PLD is most effective for very repetitive outputs… the effectiveness of PLD **drops sharply for instruction-tuned models**." |
| **SAM decoding** [2411.10666](https://arxiv.org/pdf/2411.10666) | The academic treatment of suffix-automaton drafting that Baseten's shipped work builds on. | Background; no Qwen/consumer measurement extracted. |
| **ANPD** [2404.08698](https://arxiv.org/abs/2404.08698) | Adaptive N-Gram Parallel Decoding; the origin of llama.cpp PR #2926. An HN commenter (`jncraton`) summarises the two mechanisms as **exclusive**: "These two enhancements can't be meaningfully stacked." | **[UNVERIF]** — an anonymous comment, not the paper. Note the paper's own abstract claims ngram is "exactly the difference between ANPD and the previous speculative decoding methods", which is a *substitution* claim. |
| **PROMTEC** [ACL Findings 2025, 355](https://aclanthology.org/2025.findings-acl.355.pdf) | Multi-candidate prompt lookup (reversed Z-function → several candidate spans → token **tree**) + a normalised-pattern template trie. 3.87–3.91× over AR for Llama2 on miniF2F. | Tree-of-candidates is the mechanism §4.6 of the sibling note names as the prerequisite for concatenating. No combination with a neural drafter; math/code only. |
| **LogitSpec** [2507.01449](https://arxiv.org/html/2507.01449) | Retrieval SD guided by the last logit; up to 2.61×, 3.28 MAT. Explicitly self-limiting: "While our LogitSpec is a fully plug-and-play SD framework, **its real-world inference acceleration is less competitive**. Our future works involve integrating LogitSpec into existing draft-model-based SD methods" — i.e. the combination is *not yet done* by the authors either. | Confirms the combination is open, from a 2025 paper. A100; ~1.4× on translation, 2.01× on LongBench. |
| **MMSpec** [2603.14989](https://arxiv.org/pdf/2603.14989v1) | 10 SD methods on VLMs, 4×A100. "Model-free methods, including Lookahead, Recycling, **PLD**, and SAM, consistently show only marginal improvements and, **in some cases, even lead to slowdowns**… In several subtasks, the speedup even **drops below 1×**." PLD config `ngram=4, n_pred=10`. | The one peer-reviewed-style negative on PLD **for a Qwen** (Qwen2.5-VL-7B). Vision-language, so the token distribution is unlike the text case — but it is a measured, controlled negative on a Qwen model. |
| **AngelSpec** [2607.25852](https://arxiv.org/pdf/2607.25852.pdf) | MTP vs DFlash vs DSpark on Qwen3-8B and Hy3-A21B; mean acceptance length MTP 3.24, DFlash 4.57, DSpark 5.32, DFly 5.41. | **No ngram arm.** Included only because it is the best current statement of the MTP acceptance-length ceiling any ngram method would be stacked onto. |

**[no evidence found]** — no arXiv paper in the searches above reports ngram drafting stacked on
MTP or EAGLE with a control arm on consumer hardware, or on any Qwen3.x model with a control arm.
The one paper claiming an EAGLE integration (CopySpec) does not surface its comparison.

---

## 6. Blogs, forums, video: the un-controlled numbers

### 6.1 Hacker News (Algolia, `query=ngram speculative decoding`, 13 hits total)

| comment | content | class |
|---|---|---|
| `ggerganov`, 2026-06-16, on vickiboykis.com | "I get a lot of mileage from the ngram-based speculative decoding functionality as it allows me to iterate on the implementation much faster." Workload: targeted agent sessions, rarely exceeds context window. | **[MAINT]**, no number — see §2.4 |
| `naasking`, 2026-09-18, on byteshape.com | R9700, Qwen3.8-27B fine-tune: "no other speculative decoding like **ngram-mod, which I found slows it down**"; MTP `n-max 7` with acceptance ~0.55 took 30 → 80 t/s peak, "**MTP peaked at ~65 t/s**". | **[UNVERIF]** — no ngram-off or MTP-off number, no workload |
| `flutetornado`, 2026-09-17, on prismml.com | DGX Spark, Bonsai 2 27B, 34.38 t/s gen: "**ngram speculative decoding did not help too much either — not enough accepted tokens.**" | **[UNVERIF]** — a one-line remark, no numbers |
| `kgeist`, 2026-09-02, on baseten.co | Building a hybrid llama.cpp+vLLM/SGLang engine for cheap multi-GPU hardware. "speculative decoding (including domain-specific ngrams, they already can speed up code generation considerably **without the overhead of a draft model**)." | **[UNVERIF]** — an opinion, no numbers. Worth noting as a *mechanism* claim: the "no draft model" framing is exactly the argument that §1.1 and §3.1's evidence does not support on bandwidth-limited hardware |
| `williamzeng0`, 2025-10-07 | Show HN: Sweep. 1500 ms → 94 ms. | **[VENDOR]** — see §2.1 |
| `ydj`, 2026-06-13, on imil.net | 5080+3090, Qwen3.6-27B Q8, 80 t/s: "**Would like to see the perf of their setup with and without mtp and ngram speculative decoding** though." | A request for the measurement, not the measurement |
| `siris9476`, 2026-09-01, on carloslfu/slotstream | The 32 GB n-gram-table question. | **[UNVERIF]** — see §4.3 |
| `cmrdporcupine`, 2026-08-26, on qwen.ai | n-gram table paged from NVMe, 12 t/s decode, no SD yet. | **[UNVERIF]** |

### 6.2 A Reddit thread that is really a control-arm lesson **[USER]**

r/LocalLLaMA
["spec : add ngram-mod by ggerganov · PR #19164"](https://www.reddit.com/r/LocalLLaMA/comments/1qrbfez/spec_add_ngrammod_by_ggerganov_pull_request_19164/)
(2026-01-30). Top comment: "this is HUGE im already seeing **almost 2x speed up** on my opencode with 4.7
flash." Another user pastes full llama-server timings: `prompt eval 4520.06 ms / 4476 tokens
(990.25 t/s)`, `eval 6675.55 ms / 378 tokens (56.62 t/s)`. **[UNVERIF]** — no baseline arm, GLM
4.7 Flash, and the "2x" is against an unstated comparison. Included only because it is the
originating instance of the "ngram-mod is a huge win" claim that recurs throughout §6.

### 6.3 The two "huge win" Reddit posts that are not measurements **[UNVERIF]**

- ["600tk/s+ speed … (rtx 3090)"](https://www.reddit.com/r/LocalLLaMA/comments/1rjpvdd/600tks_speed_on_local_hardware_with_self/):
  "For couple of new, simple lines on 4k tokens of code and text, I get 600+ tk/s gen speed, 300tk/s
  with major changes." Devstral-Small-2-24B IQ4_NL, `--spec-type ngram-mod --spec-ngram-size-n 24
  --draft-min 48 --draft-max 64`. A reply in the same thread supplies the missing control: "Your
  numbers make sense if you are, say, fixing a syntax error bug in … code file and outputting the
  entire fixed … . In that case 99. … of the output predicted will be copying the original file."
  No ngram-off number.
- ["llama.cpp speculative checkpointing was merged"](https://www.reddit.com/r/LocalLLaMA/comments/1sprdm8/llamacpp_speculative_checkpointing_was_merged/):
  "For coding, I got some 10%~15% speedup with these params" — same flags. No baseline. A reply
  diagnoses the variance as the mechanism §3.5 documents: "code heavy on boilerplate/repeated
  variable names … should see the high end of 0-50 % [acceptance]. one-off logic or reasoning chains
  will be near zero."
- ["Brief Ngram-Mod Test Results — R9700/Qwen3.6 27B"](https://www.reddit.com/r/LocalLLaMA/comments/1swt19w/brief_ngrammod_test_results_r9700qwen36_27b/):
  mean 28.80 t/s, median 28.20, P95 45.34, range 16.49–53.63, `-np 1`, Vulkan, Qwen3.6-27B
  UD-Q4_K_XL, `--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 12 --draft-max 48`. **No
  ngram-off arm at all** — this is exactly the defect the sibling note flags in upstream NInfer
  issue #234, reproduced in the wild. The 28.8 t/s mean is not interpretable as a result; only
  the 3.3× spread (16.49 → 53.63) is informative, and it matches §3.5's "throughput is a function of
  novelty".

### 6.4 YouTube — the one combination video **[UNVERIF]**

["MTP + Ngram Stacked in llama.cpp — Qwen3.6 27B at 56 tok/s Locally"](https://www.youtube.com/watch?v=71T4aQiwZ_I)
(Fahad Mirza, 2026-05-27). The described progression: **22 t/s** (no SD) → **42 t/s** (MTP alone,
`--spec-draft-n-max 2`) → **56.6 t/s** (adding `--spec-type ngram-mod`, `--spec-ngram-mod-n-match
24`). Mainline llama.cpp, Qwen3.6-27B, no forks, no second model.

**Why this is not a result, stated carefully.** The GPU is not named anywhere in the description or
transcript I could read. The baseline is explicitly cross-video: "it started at 22 tokens per
second when we used it **earlier**" — i.e. the 22 came from a prior video, not from a same-session
ngram-off arm. The video's own claim of a same-workload comparison is only "Same model, same GPU,
same build, just four extra flags" — which holds for 42 → 56.6, not for 22 → 42. And the presenter
quotes a *second-hand* community number — "The main PR author himself, somewhere showed 68 tokens
per second on a coding task on a single RTX 3090 with both stacked and **254 token per second** on a
follow-up code edit" — the 254 figure being the §3.5 trap-1 pattern (a follow-up edit on an
already-copied file). Taken as a community report the interesting part is that +35 % for the ngram
increment (42 → 56.6) is larger than anything in §1.1; taken as a measurement it is unusable.

### 6.5 Chinese-language sources — one shipped hybrid, and a table with no source

- **FastDeploy `Hybrid-MTP-with-Ngram`** — §2.3. Shipped, concatenating, no published numbers.
  The Chinese doc is explicit about the intent: "适合在需要更多草稿Token时使用，兼顾MTP生成能力与
  Ngram匹配的高效性" (suited when more draft tokens are needed, balancing MTP's generative ability
  against ngram matching's efficiency). Max `num_speculative_tokens` is **5** and "当前 MTP 仅支持
  1", so the hybrid path is 2 + 3 at most on this engine.
- **vLLM Ascend (Huawei)** — [中文 docs](https://docs.vllm.ai/projects/ascend/zh-cn/main/user_guide/feature_guide/speculative_decoding.html).
  Same single-choice method enum as upstream (`ngram`, `suffix`, `medusa`, `eagle`, `eagle3`,
  `mtp`, `dflash`, `dspark`, `draft_model`) — no combination. What it adds is a dynamic-K-by-batch-size
  feature that explicitly supports `n-gram` (`num_speculative_tokens_per_batch_size`), and it
  restates the BS×K mechanism: "当 `BS * K`超过临界 batch size 时，投机解码反而可能损害解码速度
  （TPOT）" — and that Suffix Decoding is already per-request dynamic, so an outer dynamic K is
  "冗余或冲突" (redundant or conflicting) for it. No numbers.
- **Huawei ModelArts best-practice doc** — the only Chinese source that gives *tuning* values for
  ngram, and it is a vendor recommendation with no measurement attached: `prompt_lookup_min`
  经验值 1 或 2 (rule of thumb 1 or 2), `prompt_lookup_max` 经验值 4~16 (4–16), and
  `num_speculative_tokens`: "若设置过大会导致性能劣化，推荐根据接受率设置1/2/3，其中推荐先设置
  为1，若接受率高于70%，可尝试设置为2" (too large degrades performance; start at 1, try 2 if
  acceptance > 70 %). **[VENDOR]**, unmeasured — and the advice directly contradicts both
  llama.cpp's shipped `n_min 48 / n_max 64` and TRT-LLM's `max_draft_len 5` defaults.
- **PaddleNLP** — `speculate_method: inference_with_reference` is its ngram path,
  `speculate_max_ngram_size` default 1, `speculate_max_draft_token_num` max 5, max `batch_size` 128.
  No ngram/MTP combination and no ngram numbers; its MTP table (45.2 → 128.7 → 152.3 t/s) is
  unlabelled as to model, GPU and quant, so **[UNVERIF]**.
- **AtomGit/GitCode, the `examples/lookup` deep-dive** — §4.3. Useful design detail, no numbers.
- **A table to distrust** — [inference.lei6393.com](https://inference.lei6393.com/sglang/05-performance/20-speculative-decoding/)
  publishes a per-domain EAGLE-vs-NGRAM speedup grid: code 2–3× vs **1.5–2×**; translation 2× vs
  1.2×; summarisation 1.8× vs 1.1×; creative writing 1.5× vs **1.0×**. It cites no paper, no
  model, no GPU, no dataset, and the numbers are suspiciously clean. **[UNVERIF]** — flagged
  because it is the single most quotable ngram-vs-EAGLE table I found outside a paper, and it
  cannot be traced.

### 6.6 Windows

- **A Windows user, independently reproducing the CRLF mechanism.**
  [r/LocalLLaMA, "PSA on llama.cpp —spec-type ngram-mod (use LF not CRLF)"](https://www.reddit.com/r/LocalLLaMA/comments/1r1k5gn/psa_on_llamacpp_spectype_ngrammod_use_lf_not_crlf/)
  (2026-02-27): "My files (**I am using a Windows computer**) used CRLF… So most of the ngrams
  created from my pasted file were useless because of the `\r\n`." Devstral-2-123B-Instruct-2512
  UD-Q5_K_XL. The sibling note has the same defect measured with numbers (EndeavoringOrb, aarch64:
  2.32 t/s CRLF vs 29.79 t/s LF, 12.8×). This is an independent report of the same mechanism from a
  **Windows** user, with no numbers — it corroborates the mechanism and the platform, and adds
  nothing quantitative.
- **vLLM on Windows, 2× RTX 3090, Qwen3.6-27B INT4** —
  [devnen/qwen3.6-windows-server, docs/SPEC_DECODE_MATRIX.md](https://github.com/devnen/qwen3.6-windows-server/blob/main/docs/SPEC_DECODE_MATRIX.md).
  "TP=1 + MTP (n=3..6) | **Works.** 53–72 tok/s on Qwen3.6-27B INT4 depending on N and prompt
  class." "PP=2 + ngram | `RuntimeError: 'GPUModelRunner' object has no attribute 'drafter'` at
  worker rank during `determine_available_memory`. vLLM 0.19.0 bug." And the bottom line: "pick
  **either** speed (MTP on a single GPU) **or** context (PP=2 across both GPUs with no
  spec-decode). You cannot have both on this wheel." **[USER]**, a GitHub source. No
  ngram-with-MTP measurement, and the author reports ngram as blocked.
- **[no evidence found]** — no measured ngram result, alone or combined with a neural drafter, on
  Windows, from any source. No 4090 or 5090 measurement of the combination from any source.

### 6.7 A blog whose own summary contradicts its own table **[UNVERIF]**

The [BitTide mirror](https://bittide.aicompass.dev/article/4bc43450-93ab-451f-939a-12ac6445befd)
of an r/LocalLLaMA post ([original thread](https://www.reddit.com/r/LocalLLaMA/comments/1ujo46r/), 2026-06-30;
BitTide was the fetchable copy — reddit.com returned 403 for this session). Xeon E5-2666v3, 64 GB,
single **RTX 3090 24 GB**, Qwen3.6-27B, 5 engines, `club-3090` bench script, code + narrative
decode t/s:

| fork / engine | spec type | quant | code DP | narrative DP | VRAM MiB |
|---|---|---|---:|---:|---:|
| ik_llama (ubergarm config) | MTP `n_max=4` | IQ4_KS | **89.2** | **63.9** | 22304 |
| **ik_llama + ngram** | **ngram + MTP (`n_max=3`)** | IQ4_KS | 87.8 | 58.6 | 20508 |
| ik_llama (standard) | MTP `n_max=2` | IQ4_KS | 73.1 | 61.7 | 20208 |
| mainline llama.cpp | MTP `n_max=1` | Q4_K_M | 64.7 | 52.5 | 21354 |
| beellama | DFlash | Q4_K_M | 96.8 | 45.6 | 20814 |

**There is no clean control here, and the reason is visible in the table:** the ngram arm changes
the MTP depth (3 vs 4) *and* the fork config *and* the quant is the same but the extra flags differ.
87.8 sits between MTP-2's 73.1 and MTP-4's 89.2, so the ngram contribution cannot be separated
from the depth change. On the **narrative** column, which is the less copy-heavy workload, the
ngram arm is the **worst** of the three ik_llama configs (58.6 vs 61.7 and 63.9) — the same sign as
§3.2's prose regression and §1.1's MoE regression, but not separable here.

Two things I can extract. **(a)** The only VRAM delta attributable to adding ngram in any source:
20508 − 20208 = **+300 MiB**, on the same fork and quant, though confounded by the MTP-depth change
and by the standard arm lacking ubergarm's `--slot-save-path --ctx-checkpoints`. **(b)** The post's
own recommendation table reports "Code speed — ik_llama MTP+ngram — **98.5 DP**, double the
baseline", while its measurement table above reports 87.8. The 98.5 appears nowhere else in the
post. **The recommendation figure is not reproducible from the author's own data**, and it is the
one that would be quoted.

---

## 7. Where this note contradicts the sibling note

Two places, both material.

**7.1 TRT-LLM's `SA` is not "a separate option from `NGram`, not a combination with it."**
The sibling note (§3.2, from `decoding_type`'s enum and `sa_worker.py`) concluded that SA and a
neural drafter are mutually exclusive. That was true of the code it read. NVIDIA merged
[PR #11434](https://github.com/NVIDIA/TensorRT-LLM/pull/11434) (2026-02-10) which "Added optional
suffix automaton integration to **MTP** speculative decoding mode" via
`MTPDecodingConfig.use_sa_spec` and `sa_spec_threshold`, and [PR #12130](https://github.com/NVIDIA/TensorRT-LLM/pull/12130)
(2026-03-12) which added a global pool on top and removed the deprecated `is_keep_all` /
`is_use_oldest` / `is_public_pool` flags in favour of SA. The `decoding_type` enum the sibling note
read is not the shipped configuration surface any more. §1.2's numbers are the consequence.

**7.2 "Three attempts exist; none has shipped"** (sibling §4, §7.3 item 1). At least two more have
shipped: TRT-LLM's MTP+SA (NVIDIA, merged, with an `AUTO`-adjacent heuristic) and FastDeploy's
`mtp_strategy: with_ngram` (merged, CUDA kernels). Baseten's `sa_spec` is the third, and the
selection design the vLLM RFC (#18633) proposed is what shipped — twice — rather than the
concatenation. So the accurate statement is the inverse of the sibling note's: **selection has
shipped in two production engines and has published numbers; concatenation has shipped in one
(FastDeploy) and has none.**

---

## 8. What I could not find

Stated plainly rather than inferred.

1. **No measurement of the combination on a discrete consumer GPU (4090/5090/5080/5070 Ti).** The
   only consumer-hardware datapoint (§1.1) is an integrated SoC; the only discrete-GPU combination
   datapoint (§6.7) has no clean control. The 75 t/s dual-5070 Ti figure that §1.1 cites is
   explicitly acknowledged by its citer as uninterpretable.
2. **No measurement of the combination on Windows.** §6.6 gives a Windows CRLF corroboration and a
   Windows vLLM matrix in which ngram is blocked by an upstream bug. Nothing else.
3. **No "MSpec" system exists in this area.** I searched for it. The only `MSpec` in the literature
   is a 2012 concurrency design pattern (Scott et al., *IEEE Trans. Parallel Distrib. Syst.*,
   [PDF](https://cs.rochester.edu/u/scott/papers/2012_TRANSACT_mspec.pdf)) with no relationship to
   LLM inference, plus a 2020 speech-conversion network. The plausible intended referents are
   **MMSpec** (arXiv 2603.14989, a VLM SD benchmark — §5.3) or **SAM decoding** (arXiv 2411.10666).
   If "MSpec / ngram-mod on Qwen3.x" was meant as one system, I found nothing under any of the three
   names that benchmarks ngram-mod on a Qwen3.x model with a control arm.
4. **No published VRAM figure for a ngram index**, in any of llama.cpp, vLLM, SGLang or
   TensorRT-LLM/Baseten, except the sibling note's 16 MiB flat pool (§4.3).
5. **No acceptance-rate figure for ngram drafting under a JSON/XML grammar** from any non-GitHub
   source (the sibling note already records the GitHub-side absence).
6. **No ngram-off control arm in the following widely-circulated claims**, each of which is
   therefore not a result: Reddit "almost 2x on opencode with 4.7 flash" (§6.2); Reddit 600+ t/s on
   a 3090 (§6.3); Reddit 10–15 % coding speedup (§6.3); Reddit R9700 28.80 t/s mean (§6.3); the
   BitTide/Reddit 98.5 DP recommendation figure (§6.7); the YouTube 22 → 42 → 56.6 progression's
   22 t/s endpoint (§6.4); and the `inference.lei6393.com` EAGLE-vs-NGRAM grid (§6.5).
7. **I could not read Reddit directly.** `reddit.com` returned HTTP 403 and `old.reddit.com` an
   empty shell for every thread fetch in this session, including the `.json` API. All Reddit content
   above is search-index extraction. I did not verify any Reddit figure against the thread body, and
   a `Select-First`-style truncation of the extract is possible in principle. This is the weakest
   link in the note and it is why every Reddit row is labelled **[UNVERIF]**.

---

## 9. Source list

**HuggingFace** — [unsloth/Qwen3.8-Flash-Next-GGUF discussion #34](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/discussions/34)
(solarkyle, 2026-08-27); [unsloth/Qwen3.6-35B-A3B-GGUF discussion #14](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14)
(thc1006, 2026-04-21); [joaogante on ngram speculation in transformers](https://huggingface.co/posts/joaogante/647421262068164)
(**[MAINT]** — the HF transformers author; "the penalty of gathering and testing the candidates is
so small that you should use this technique whenever possible", plus the real limitation: "you are
limited to a batch size of one").

**Hacker News (Algolia `search?query=ngram speculative decoding`, 13 hits)** — item 45505487 (Sweep
Show HN), 48558288 (ggerganov), 49753613 (naasking), 49748346 (flutetornado), 49533089 (kgeist),
49528769 (siris9476), 49453020 (cmrdporcupine), 48518772 (ydj), 40108740 / 40108806 (ANPD thread).

**arXiv** — 2502.08923 (CopySpec), 2412.01447 (PLD+), 2411.10666 (SAM decoding), 2404.08698 (ANPD),
2507.01449 (LogitSpec), 2604.09557 (SPEED-Bench), 2603.14989 (MMSpec), 2607.25852 (AngelSpec);
ACL Findings 2025 1017 (DReSD) and 355 (PROMTEC).

**Blogs / write-ups** — [thefrontierlab.ai](https://thefrontierlab.ai/mtp-defaults-are-a-trap/)
(2026-05-28); [baseten.co](https://www.baseten.co/blog/boosting-mtp-acceptance-rates-in-baseten-speculation-engine/)
(2026-05-05); [NVIDIA TensorRT-LLM tech blog 07](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.html)
**[VENDOR]** (already in the sibling note); [0xBakeer/qwen38-flash-next-spark docs](https://github.com/0xBakeer/qwen38-flash-next-spark/blob/main/docs/speculative-decoding.md);
[DEV.to kstoyanovai](https://dev.to/kstoyanovai/speculative-decoding-on-local-hardware-benchmarking-n-gram-mtp-eagle3-and-dflash-on-the-nvidia-2kec)
(2026-07-18); [ovidiudan.com](https://www.ovidiudan.com/2025/10/26/speculative-decoding.html)
(2025-10-26, Qwen3-32B + `Qwen3-32B-speculator.eagle3`, vLLM, 1.82×, mean acceptance length 2.01 —
**[VENDOR]**, a draft-model result, no ngram arm); [BitTide mirror](https://bittide.aicompass.dev/article/4bc43450-93ab-451f-939a-12ac6445befd).

**Video** — ["MTP + Ngram Stacked in llama.cpp — Qwen3.6 27B at 56 tok/s Locally"](https://www.youtube.com/watch?v=71T4aQiwZ_I)
(Fahad Mirza, 2026-05-27). Description and transcript only; no frames inspected.

**Chinese-language** — [FastDeploy 中文 doc](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/features/speculative_decoding.md)
and [English doc](https://paddlepaddle.github.io/FastDeploy/features/speculative_decoding/);
[vLLM Ascend 推测解码](https://docs.vllm.ai/projects/ascend/zh-cn/main/user_guide/feature_guide/speculative_decoding.html);
[华为云 ModelArts N-Gram 投机最佳实践](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5906023.html);
[PaddleNLP 投机解码教程](https://paddlenlp.readthedocs.io/zh/latest/llm/docs/predict/speculative_decoding.html);
[AtomGit llama.cpp lookup 示例深度解析](https://blog.gitcode.com/280b2cfc07a983a16650fa231edf5933.html);
[sglang 投机采样 mtp 笔记](http://gogongxt.com/posts/c2e4af69.html);
[MTP 推测解码浅析](https://zhen8838.github.io/posts/mtp-speculative-decoding-deep-dive.html);
[leijiao.com 投机解码](https://inference.lei6393.com/sglang/05-performance/20-speculative-decoding/)
**[UNVERIF]**; [推测解码算法在 MTT GPU 的应用实践](https://jishuzhan.net/article/1926281104936521730)
(EAGLE on an S4000, no ngram arm).

**Incidental GitHub sources** (outside the trackers surveyed by the sibling note; included because
they carry the measurements) — NVIDIA/TensorRT-LLM
[#11434](https://github.com/NVIDIA/TensorRT-LLM/pull/11434),
[#12130](https://github.com/NVIDIA/TensorRT-LLM/pull/12130);
[basetenlabs/sa_spec](https://github.com/basetenlabs/sa_spec);
[thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)
and its [HackMD report](https://hackmd.io/@thc1006/SJly6IE6Wx);
[solarkyle/qwen38-flashnext-16gb](https://github.com/solarkyle/qwen38-flashnext-16gb);
[0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark);
[thefrontierlab/post4-bench](https://github.com/thefrontierlab/post4-bench);
[devnen/qwen3.6-windows-server](https://github.com/devnen/qwen3.6-windows-server/blob/main/docs/SPEC_DECODE_MATRIX.md);
PaddlePaddle/FastDeploy [#3610](https://github.com/PaddlePaddle/FastDeploy/pull/3610),
[#4047](https://github.com/PaddlePaddle/FastDeploy/pull/4047),
[#7103](https://github.com/PaddlePaddle/FastDeploy/pull/7103).

**Method note — fetch provenance, read this before citing anything here.** Only **eight** sources
were fetched directly with `webfetch` in this session: the HN Algolia API, the HF discussion #34,
`thefrontierlab.ai`, the `0xBakeer` raw markdown, `baseten.co`, `basetenlabs/sa_spec`, the BitTide
mirror, and the TRT-LLM blog (via the search index). Everything else — including every arXiv table,
every FastDeploy/vLLM-Ascend doc claim, the TensorRT-LLM PR benchmark tables, the YouTube
transcript, and every Reddit thread — is **quoted from the search engine's result extract**, not read
from the page. Those extracts are long and specific, but a search index can truncate a table and I
have not re-verified a single cell. The 219 relative links and 26 heading anchors do resolve
(`tools/release/check_doc_links.py`, PASS); the 142 external URLs are **not** fetched by that
checker and I did not fetch them all. Treat §1.1, §1.2, §3.2, §3.5 and §6.7 as primary-read and the
rest as second-hand until someone re-fetches the page.

**Method note — counting rules.** Two rules were applied and are worth stating because they changed
what got counted. (1) A before/after with no same-workload ngram-off measurement is recorded as
**not a result**, and the number is listed in §8.6 so it is not silently re-cited later — this is
the same defect the sibling note records in upstream NInfer issue #234, and it recurs in the wild at
the same rate. (2) Where a source retracts or revises its own earlier number (§3.1, §3.5, §6.7),
the retraction is reported rather than the superseded figure. No implementation is recommended here;
the sibling note's §7 remains the place where design shapes are compared.
