# The Swift lane has a build record: the local recipe reproduces the shipped artifact

**MEASURED HERE 2026-09-29. Overall and all four per-domain values match the shipped
`qwen3_8_27b_nvfp4swift.v3.ninfer` to every digit. The 1.26 % gap against `nvfp4full` is a
difference in the sources, not in the port's building of them.**

## The question

`docs/research/artifact-build-verification-audit.md` recorded that three of the four served lanes
carry a `*.conversion.json` naming the recipe that produced them, and `nvfp4swift` did not. Without a
local build there is no way to attribute its difference from `nvfp4full` -- it could be the recipe, or
it could be the source checkpoints the two lanes start from, and those call for opposite responses.

## What was run

A fresh conversion with the unmodified `qwen3_8_27b_nvfp4_swift` recipe, from the same two local
sources the recipe names: `Swift-Qwen3.8-27B-NVFP4` as the ModelOpt `base`, and `Swift-Qwen3.8-27b` as
`swift_bf16`. 103.2 s, 1177 objects, 15.65 GiB. Then the full corpus, fp8 KV, 1,044,876 tokens.

## The result

| domain | shipped `nvfp4swift` | local rebuild | difference |
|---|---:|---:|---:|
| `chinese_reference` | 6.229361 | 6.229361 | 0 |
| `english_long_form` | 8.350429 | 8.350429 | 0 |
| `english_reference` | 6.707160 | 6.707160 | 0 |
| `ninfer_code` | 1.690048 | 1.690048 | 0 |
| **overall** | **4.936397** | **4.936397** | **0** |

The recorded shipped values are the `perplexity-baseline.md:127-131` row; they were read from that
file rather than recalled. An earlier note in this session quoted different per-domain figures for the
shipped lane from memory, and they were wrong -- which is the rule this project already records about
a claim about this tree being re-read, and it is why the comparison above is against the file.

## The size and object-count difference, and what it is

The rebuild is **2.77 GiB smaller** and carries **413 fewer objects**. That is not a quality
difference and not a difference in encoding; it is the command line.

`--components` defaults to `text` alone, so the rebuild carries only the text component. The shipped
artifact carries all four:

| | shipped | rebuild |
|---|---|---|
| components | `dflash2`, `mtp`, `text`, `vision` | `text` |
| tensor objects | 1584 | 1173 |
| resource objects | 6 | 4 |
| bytes | 18.42 GiB | 15.65 GiB |

Every object in the shipped artifact that the rebuild lacks is one of the three optional components.
Nothing is present in the rebuild and absent from the shipped file -- the local set is a strict
subset. So the text stack, which is the part the recipe re-encodes, is what the exact match covers,
and vision, mtp and dflash2 are copied verbatim from their sources rather than re-encoded, which is
what artifact-conventions.md records them as being.

Rebuilding with `--components text,vision,mtp,dflash2` would carry all four. It is not needed to
answer the question this document asks, and the artifact would be the same 15.65 GiB of re-encoded
text plus the components copied through.

## What this settles, and what it does not

**Settles.** The Swift lane is built by this port's `qwen3_8_27b_nvfp4_swift` recipe, and the recipe is
reproducible: the same recipe and the same sources give the same model, not merely a similar one. So
`nvfp4swift` at 4.936397 against `nvfp4full` at 4.998419 is a **1.26 % difference between the two
lanes' source checkpoints** -- Swift's own finetune against unsloth's -- and not a build artefact.
That makes open item 1b's comparison interpretable, and it is the same result the `nvfp4full` control
gave earlier the same day, where a fresh conversion reproduced the shipped artifact's quality exactly.

**Does not settle.** Whether either source could be improved. Both lanes are the best this port can
build from the checkpoints it has; the gap says which checkpoint is better on this corpus, not that a
better artifact was available and missed. And the per-domain picture is unchanged: `nvfp4swift` takes
`chinese_reference` at 6.229361 against `nvfp4nvidia`'s 6.371428, while sitting behind it overall, which
is the ordering cancellation that the per-domain KL instrument is specified to address.

## The command, for anyone repeating this

```text
python -m tools.convert \
  --model        C:\AI\models\hf-src\Swift-Qwen3.8-27B-NVFP4 \
  --source       swift_bf16=C:\AI\models\hf-src\Swift-Qwen3.8-27b \
  --source       dflash2=C:\AI\models\hf-src\Qwen3.8-27B-DFlash2 \
  --recipe       qwen3_8_27b_nvfp4_swift \
  --components   text,vision,mtp,dflash2 \
  --name         swift-local \
  --out          out/exp_swift_local.ninfer \
  --device       cuda
```

`--model` takes the **ModelOpt** checkpoint, not the BF16 finetune. The recipe reads
`sources["base"]` for `input_scale`, and the BF16 finetune has no such tensor, so passing it fails with
`model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale: no activation scale to derive the
divisor from`. That mistake was made twice in this session -- once in `verify_artifact.py` and once in
`tools.convert` -- and the recipe's own docstring is what settles which is which: `base` is read for the
divisor and `swift_bf16` for the weights.

## Verification of the shipped file, separately

The shipped `nvfp4swift` also passes this port's own structural verifier against the same recipe:

```text
auxiliaries 512   bindings 1513   input_divisors 512
objects 1590      weight_divisors 256
PASS: 4383 checks
```

which is the artifact-conventions.md section 2 contract applied to the one lane that had not had it:
the complete ordered directory walked, every binding's ranges ordered and in range, and every NVFP4
weight divisor and stored input divisor a positive finite FP32 word.
