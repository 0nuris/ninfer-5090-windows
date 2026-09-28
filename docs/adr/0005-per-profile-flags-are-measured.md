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
