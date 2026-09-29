# ADR-0005: Spec route, draft depth and the proposal head are measured per profile

**Status:** accepted

## Context

`--lm-head-draft` is not uniformly good, and depth optima differ by artifact:

- QUASAR: the flag is worth +9% (DFlash2) and +18% (MTP d4).
- NVFP4-full MTP: worth +34% at d5.
- NVFP4-full DFlash2: worth about +2%, while costing 0.33 GiB of headroom.
- The retired NVFP4 image: the flag cost about 13% on DFlash2, plus 16,384 of context.

Depth optima, as re-measured 2026-09-28: **d5 on QUASAR and NVFP4-full, d4 on Swift and NVIDIA.** The
first statement of this ADR gave d4 on QUASAR and d5 on NVFP4-full, from the 2026-09-17 records; those
records mix runs with and without the first-request warmup transient, and re-measuring all sixteen
lane-by-depth combinations with it excluded moved three of the four lanes.

**Amendment 2026-09-29: the 2026-09-28 optima are themselves unresolved, and are not to be cited as
settled.** Two problems, both about the harness rather than the depths:

- **One domain.** That sixteen-cell sweep measured `CODE_PROMPT`, a single synthetic code prompt, and
  `measure_decode` hardcoded it — the bench could not measure any other workload. Code is the most
  favourable domain for speculation (measured 3.18 to 5.71 tokens per round against 1.18 to 1.85 on
  Chinese), so a code-prompt figure sits at the high end of the range, and the same session's domain
  sweep reversed depth winners that the code prompt had ordered.
- **The wrong sampling.** That sweep predates `d1ae5274`, which put the bench on the sampling
  configuration the model card specifies for coding (temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
  presence_penalty 0.0). It ran at 0.6 / 0.95 with no penalty, a point the card specifies for neither
  mode, and the same session's sweep at the documented settings did not reproduce the code-prompt
  ordering.

**The contradicting measurements cannot be reproduced or checked, because they were never persisted.**
They are in no repository document, and not in the two places a sweep's data would normally land:
`matrix_v3.jsonl` carries `tag/art/spec/draft/vision/max_context/ready/vram/log/spec_lines/refusal` and
no domain, no temperature, and no throughput on a refused run; everything under `profiles/` is a
perplexity report. Those sweeps ran from scratch scripts in a temp directory and their results died
with the session.

So the tree holds depths justified by evidence now known to be inadequate, while the better evidence
that would overturn them is unrecoverable. **The decision is therefore neither to revert nor to
endorse.** Reverting would restore depths whose own recorded basis is the contaminated 2026-09-17
records, on the strength of a contradicting measurement nobody can reproduce. Both states are
defensible to ship; presenting either as the measured optimum is not.

**The structural cause is fixed, and that is the durable part.** `measure_decode` takes a `--domain`
from a named set (`code`, `prose`, `chinese`, `dialogue`, `repetition`), and every result now records
the domain and the sampling beside the number, with the domain also in the log tag so two domains of
one lane cannot overwrite each other's request log. A sweep can vary the workload, and a figure carries
the workload it was taken on. `repetition` is there for a specific reason: it is copy-heavy by
construction, which is the workload the ngram copy selector's break-even has to be measured on rather
than assumed from a code prompt.

**What the first domain-aware measurement says, so the argument above is not only structural.** QUASAR
DFlash2 at its shipped depth of 7, at the documented sampling, changing nothing but the workload
(2026-09-29; one lane, and the prose figure is three decode runs whose spread is 0.2 tok/s):

| domain | decode tok/s | acceptance |
|---|---|---|
| `code` | 230.2 to 235.6 over 12 runs | 38.3 % |
| `code` | 218.0 to 218.1 over 3 runs | 35.3 % |
| `prose` | 152.0 | 19.3 % |

**A 30 % throughput drop and 16 acceptance points from the workload alone, on a shipping lane at a
shipping depth.** At 19.3 % acceptance a draft window of 7 is worth about 1.19 tokens per round, so
speculation there is close to worthless, and the profile table's figure — a code prompt — is the ceiling
of the range rather than a representative point in it. This is one lane and one non-code domain, so it
is not a domain survey; it is enough to show that a depth chosen where speculation looks best is not
evidence about a depth elsewhere.

**Done when:** the sixteen lane-by-depth combinations are re-measured on the corrected harness at the
documented sampling, across at least `code` and `chinese`, interleaved between lanes and rotated
between two rounds, with the domain and sampling recorded per cell — and the profile table cites those
records rather than this paragraph.

## Decision

Set the spec route, draft depth and proposal head per profile from a record. Never from
convention, and never uniformly across profiles. The three cache-bound flags likewise.

## Consequences

- A profile's flags cannot be derived from another profile's, even on the same artifact.
- Acceptance rate is not the selection criterion; measured decode rate is.
- Re-measuring means re-generating the launchers from the profile table, not editing them.
- **A measurement taken with a defective harness is not a weaker measurement, it is a wrong one.** The
  depth optima above were each defensible when recorded and three of them were wrong, because the
  harness averaged a startup transient into the first run of every profile. The optima that moved were
  the ones decided by a small margin -- QUASAR's 3.4% and Swift's 2.6% -- while NVIDIA's, decided by
  3.4% against a figure that was itself inflated by 33%, moved the most. A profile table is only as
  good as the harness that produced it, and nothing in the release check inspects the harness.

## Why this needs recording

The natural instinct is to apply one setting to every profile, which is measurably worse on at
least three of them. The earlier uniform d5 was wrong on QUASAR.
