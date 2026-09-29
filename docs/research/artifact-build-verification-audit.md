# How far the four NVFP4 lanes are actually verified

**READ-IN-SOURCE 2026-09-29, with two full-file digests measured. The summary sentence "4 correct,
well-built artifacts" was not earned; this is what the tree actually checks.**

Asked directly whether the four shipped lanes were built correctly, and what was missed while building
them. Answering that honestly means separating three different questions that "correct" conflates:
were the *encoders* right, were the *artifacts on disk* the ones the recipes say, and is there
*evidence the resulting model behaves well*.

## What is verified

**The encoder math has direct unit tests.** `tests/convert/quantization/` covers the three numerical
modules -- `test_nvfp4.py` (4 tests), `test_groupwise.py` (4), `test_fp8_row.py` (2). The NVFP4 tests
include the E2M1 round-to-nearest-even tie table and a layout round trip through the codec, and they
assert the block scale cannot exceed E4M3FN's 448.

**Imported payloads are checked against an independent decode oracle.** `test_sources.py` has
`nvfp4_source_preserves_words_and_decodes_independently` and
`modelopt_nvfp4_preserves_words_and_inverts_the_stored_multiplier` -- the second is the one that would
catch a wrong stored multiplier, which is a factor-of-six class of bug.

**The runtime decode has an independent oracle.** `tests/ops/test_candidate_selector.cpp` sweeps the
NVFP4 candidate-selector route over kSteps 1..15 against E2M1/E4M3FN built from the format
definitions. That covers the engine's decode, not the converter's encode.

**The artifacts on disk are the ones the documents describe.** Measured here, full SHA-256 over each
whole file:

| artifact | bytes on disk | recorded sha256 | verdict |
|---|---:|---|---|
| `qwen3_8_27b_nvfp4qat.v3.ninfer` | 18,946,877,188 | `814db0dbc367a82f…` | matches |
| `qwen3_8_27b_nvfp4nvidia.v3.ninfer` | 18,946,877,188 | `76131f792241ff0a…` | matches |
| `qwen3_8_27b_nvfp4full.v3.ninfer` | 19,715,597,060 | `f8dc64701daca3eb7…` | matches |

`qwen3_8_27b_nvfp4qat` and `qwen3_8_27b_nvfp4nvidia` have **byte-identical file sizes**
(18,946,877,188) and were checked for being the same file, because two different quantization sources
producing the same size is worth a moment's suspicion. They are not the same file: their digests differ
in the first eight hex characters, and both match what the document records. The size coincidence is
structural -- both have 1513 bindings over 1600 objects with the same component set -- not a copy.

## What is not verified, and the largest gap

**The fork's whole-artifact verification harness is not in this tree.** This is the biggest one.
`docs/maintainer/artifact-conventions.md:113-115` says:

> The lines the fork published ship a `verify_*` entry point for this (`verify_nvfp4qat`,
> `verify_nvfp4full`), which revalidates the complete ordered directory, both W8 endpoints against
> base rows, and the input divisors.

There is no `verify_nvfp4*` in `tools/convert/`. What this port has instead, per the same paragraph, is
"hashing every binding against the predecessor they replace" -- but that only works for an artifact
with a predecessor. `qwen3_8_27b_nvfp4nvidia` "has no published predecessor here" (section 19), so for
that lane **no payload was ever checked against anything**. Its 159 locally encoded attention and
linear-attention matrices are unverified beyond the encoder unit tests.

This is a claim about *our own* code, so it is worth being exact: a full-artifact verification pass --
walk the ordered directory, decode every NVFP4 word through an independent path, check both W8
endpoints against their base rows, check every input divisor -- does not exist here. The encoder is
unit-tested and the runtime decode is unit-tested, and those are different claims from "this 18 GiB
file was checked."

**Only 3 of 19 converter modules have a direct test.** `recipe.py`, `qwen3_5.py`, `model.py`,
`pipeline.py`, `official_recipes.py`, `resources.py`, `proposal.py`, and all four of `sources/` have
none. `official_recipes.py` is the file that decides which site gets which format, which is exactly the
decision the GDN audit had to read by hand. Its behaviour is covered only indirectly, through
`test_recipe.py` on the recipe abstraction.

**There is no KL instrument.** `perplexity-baseline.md:142-145` names the missing instrument
precisely -- per-domain **KL against the BF16 reference** -- and it is still missing. There is no
KL code in `tools/`, `src/` or `tests/`; the `kl` identifiers in `rowsplit_grouped_mma.cuh` are
swizzle offsets and unrelated. The document's own argument for it is the one that has since been
confirmed: the four artifacts span **6.4 %** on `chinese_reference` against 1.8 % overall, and no
artifact wins everywhere, so the aggregate lets one domain's ordering decide the headline. Perplexity
cannot say *which* domain a quantization damaged; KL against a fixed reference can, and it is
per-token rather than per-sequence, so it does not have the aggregation property that hides the
effect.

**The `nvfp4swift` lane has no local build record.** Three of the four lanes have a
`*.conversion.json` in `out/` naming the recipe that produced them. Swift has none, so unlike the other
three it cannot be traced to a conversion in this tree. This is also an open item in the block-scale
work: the 1.13 % Swift-vs-`nvfp4full` gap is unattributed, because no conversion report describes how
the shipped Swift artifact was built.

## One claim in an authority document that needed checking

`artifact-conventions.md:145` ends with "sections 17 to 19 record the results." That file has only six
numbered sections, so read literally the reference is wrong. It resolves correctly against a different
document: `docs/maintainer/qwen3.8-27b-artifact.md` has sections 17 (QAT), 18 (unsloth) and 19
(NVIDIA), and they are substantive -- identity, contents, sources, and per-binding counts. So the
content exists and is real; the reference is ambiguous rather than false, because two maintainer
documents use the same numbering. Worth disambiguating, since a reader looking in the named file finds
nothing.

## What this means for the question asked

The four lanes are **structurally sound and correctly built to the extent this tree can show it**:
the encoder math is unit-tested, imported words are checked against an independent decode oracle, the
runtime decode has its own oracle, and two of the artifacts were confirmed here by full-file digest.
Perplexity is measured on all four on one fixed protocol at 4096/2048, within ±0.1 % across the merge,
and the GDN fused-scale defect class is structurally absent.

They are **not** fully verified, and the specific unverified things are: the `nvidia` lane's 159
locally encoded matrices have never been checked against a source, no whole-artifact pass exists to
check them with, `official_recipes.py` has no direct test, there is no KL instrument, and `swift` has
no build record. Those are the honest gaps, and each is answerable -- the first three by writing the
verification pass the fork published, the fourth by building the instrument
`perplexity-baseline.md` already specifies.

## Ranked by what they would buy

1. **The `verify_*` pass.** Converts "the encoder is tested" into "these files are checked," and it is
   the only item that would close the `nvidia` lane's unverified 159 matrices. The fork's design is
   already described in the authority document; it was never ported.
2. **Per-domain KL against BF16.** The instrument the perplexity authority names as missing, aimed at
   the 6.4 % `chinese_reference` spread that the aggregate hides. It is also the measurement that
   would finally answer whether any lane's activation divisor is hurting it -- the hypothesis raised by
   the calibration-corpus discussion, which a corpus swap cannot answer on its own.
3. **A `tests/convert/test_official_recipes.py`.** The site-to-format decision table is currently
   readable only by hand, and the GDN audit is the proof that this is a real cost rather than a
   hypothetical one.
4. **Swift's build record.** Cheapest of the four, and it unblocks the unattributed 1.13 % gap.
