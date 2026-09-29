"""What the BF16 base recipe decides for every site, without a checkpoint on disk.

`tools/convert/official_recipes.py` chooses the format of every weight in every shipping lane, and
codegraph reports `RECIPES` has one caller and no tests within three hops. On 2026-09-29 the GDN
fused-scale audit had to read that file by hand to establish that the GDN control projections are
BF16 -- the single decision that makes a documented defect class structurally impossible here. That
is worth a test rather than a re-read.

The seam is the converter's own: `RECIPES[name](model, recipe, sources)` followed by
`recipe.prepare(device="cpu")`, whose `.weights` carry each site's `spec.format` and `method_name`.
Nothing reaches inside a recipe; the assertions are about the layout each recipe would produce.

The source is a real safetensors file from `checkpoint_fixture.py`, not a stub. Stubbing was tried and
rejected: the recipes reach through the source four different ways -- `sources["base"].config`,
`store.has(...)`, `store.read_flat(...)`, and `model.source(name, store, format)` -- so the stub grew
a method per call and each one risked answering something the real source would not.

## What this covers, and what it does not

Covered: the `qwen3_8_27b` BF16 base recipe's whole decision table, which is the port's own precision
policy -- the Q4/Q5 split by role, both W8 endpoints, and the GDN control staying BF16.

Not covered: the four NVFP4 lanes' `import_encoded` sites. Reading one needs a source that carries
both a `shape` and lazily-read packed codes, and only `qwen3_5.build_model` installs such a factory;
`EncodedRows` is a bare four-field record without a shape, so substituting it fails at
`recipe.py:160` rather than exercising the path. Standing up a second builder for it was judged more
effort than the assertion is worth, and the fact it would pin is already established on the real
artifacts instead: `verify_artifact.py --values` reports 192 `imported_words` and 363-366
`encoded_values` on the nvidia lane, which is the import-vs-encode split read off four real
18 GiB artifacts rather than a synthetic model. See docs/research/artifact-build-verification-audit.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
# Both the repository root and this directory: the fixture is a sibling module here, and the
# converter package is imported as `tools.convert.*` from the root.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.convert.official_recipes as official_recipes  # noqa: E402
from checkpoint_fixture import write_checkpoint  # noqa: E402
from tools.convert.model import Model, Parameter  # noqa: E402
from tools.convert.official_recipes import RECIPES  # noqa: E402
from tools.convert.recipe import Recipe  # noqa: E402
from tools.convert.sources.logical import array_source  # noqa: E402
from tools.convert.sources.safetensors import SafetensorsSource  # noqa: E402

HIDDEN = 5120
VOCAB = 4096
LAYER = 0

# The logical sites the recipe branches on, with their real Qwen3.8-27B shapes. The GDN control pair
# is 48x5120 each, which is 96 rows concatenated -- the geometry that cannot be NVFP4, because
# block_scale_k16_m128x4_v1 requires N divisible by 128.
SITES: tuple[tuple[str, tuple[int, int]], ...] = (
    (f"text/layers/{LAYER}/attention/query", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/attention/gate", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/attention/key", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/attention/value", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/attention/output", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/query", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/key", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/value", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/z", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/output", (HIDDEN, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/a_projection", (48, HIDDEN)),
    (f"text/layers/{LAYER}/gdn/b_projection", (48, HIDDEN)),
    (f"text/layers/{LAYER}/mlp/gate", (17408, HIDDEN)),
    (f"text/layers/{LAYER}/mlp/up", (17408, HIDDEN)),
    (f"text/layers/{LAYER}/mlp/down", (HIDDEN, 17408)),
    ("text/token_embedding", (HIDDEN, VOCAB)),
    ("text/output_head", (HIDDEN, VOCAB)),
    ("dflash2/layers/0/attention/query", (HIDDEN, HIDDEN)),
    ("dflash2/layers/0/attention/key", (HIDDEN, HIDDEN)),
    ("dflash2/layers/0/attention/value", (HIDDEN, HIDDEN)),
    # _optional shares context_key/context_value onto key/value, and a share whose target is absent
    # raises "shared parameters must exist" -- so the shared pair is a precondition of the seam.
    ("dflash2/layers/0/attention/context_key", (HIDDEN, HIDDEN)),
    ("dflash2/layers/0/attention/context_value", (HIDDEN, HIDDEN)),
    ("dflash2/layers/0/mlp/gate", (HIDDEN, HIDDEN)),
    ("dflash2/layers/0/mlp/up", (HIDDEN, HIDDEN)),
    ("dflash2/layers/0/mlp/down", (HIDDEN, 1024)),
)

GDN_CONTROL = (
    f"text/layers/{LAYER}/gdn/a_projection",
    f"text/layers/{LAYER}/gdn/b_projection",
)


def _build_model() -> Model:
    """A Model carrying the real site names, each with a source_factory the recipes can call.

    The factory returns a zero matrix rather than reading the store, because what is under test is the
    recipe's *decision* -- which format, which method -- and not the bytes. The decision is read from
    the prepared jobs, so nothing here depends on the values.
    """
    model = Model(
        {
            "text": {
                "config": {
                    "num_hidden_layers": 1,
                    "hidden_size": HIDDEN,
                    "vocab_size": VOCAB,
                }
            },
            "dflash2": {"config": {"num_hidden_layers": 1, "hidden_size": HIDDEN}},
        }
    )
    for name, shape in SITES:

        def factory(store, fmt=None, _name=name, _shape=shape):  # noqa: ANN001
            return array_source(torch.zeros(_shape, dtype=torch.bfloat16), _name)

        model.add(
            Parameter(
                name,
                shape,
                array_source(torch.zeros(shape, dtype=torch.bfloat16), name),
                source_factory=factory,
                inputs=("input",),
            )
        )
    return model


def _format_table(recipe_name: str, sources: dict[str, object]) -> dict[str, tuple[str, str]]:
    """Run a recipe and return site -> (format, method), the way the converter reads it.

    The BF16 base recipe declares no proposal head, so nothing has to be stood down for it. The four
    NVFP4 lanes do, and each ends with `add_proposal(recipe, source=...)` -- a drafter component whose
    default ranking is the repository's real 11,919,360-byte frequency table for the real
    1,489,920-entry vocabulary, which a synthetic 4096-entry model cannot divide evenly. Replacing
    it is kept here rather than left to a fixture, because an earlier version of this file declared
    the replacement in a fixture and left the call uncovered, so nothing was patched and the failure
    looked like a missing fixture instead of an unpatched name.
    """
    model = _build_model()
    recipe = Recipe(model)
    with mock.patch.object(official_recipes, "add_proposal", lambda *a, **k: None):
        RECIPES[recipe_name](model, recipe, sources)
    return {
        parameter: (job.spec.format, job.method_name)
        for job in recipe.prepare(device="cpu").weights
        for parameter in job.parameters
    }


@pytest.fixture(scope="module")
def bf16_sources(tmp_path_factory):
    """A plain BF16 checkpoint: no `weight_scale_2` anywhere, so every site is treated as FP8."""
    root = tmp_path_factory.mktemp("bf16")
    path = write_checkpoint(root / "ckpt", nvfp4=False, composite=False)
    with SafetensorsSource(path) as store:
        yield {"base": store}


def test_the_gdn_control_stays_bf16(bf16_sources) -> None:
    """a_projection and b_projection are BF16, and that is what makes the fused-GEMM defect
    structurally impossible rather than merely absent.

    The reasoning is geometric, not stylistic: the two are 48x5120 each, so concatenated they are
    96x5120, and block_scale_k16_m128x4_v1 requires N divisible by 128. 96 cannot satisfy it, so the
    layout cannot hold them at NVFP4 whatever a recipe asks for.
    """
    table = _format_table("qwen3_8_27b", bf16_sources)
    for site in GDN_CONTROL:
        assert table[site][0] == "bf16", f"{site} is {table[site][0]}, expected bf16"
    assert (48 + 48) % 128 == 96, "the concatenated geometry is the reason, asserted here"


def test_q4_and_q5_are_split_by_role(bf16_sources) -> None:
    """query, key, gdn/query, gdn/key, mlp/gate and mlp/up are Q4; the rest of a text layer is Q5.

    This is the `_dense_groupwise` table, the port's own precision policy rather than anything the
    format requires, so it is exactly the kind of decision that should fail a test when it is edited.
    """
    table = _format_table("qwen3_8_27b", bf16_sources)
    for site in (
        f"text/layers/{LAYER}/attention/query",
        f"text/layers/{LAYER}/attention/key",
        f"text/layers/{LAYER}/gdn/query",
        f"text/layers/{LAYER}/gdn/key",
        f"text/layers/{LAYER}/mlp/gate",
        f"text/layers/{LAYER}/mlp/up",
    ):
        assert table[site][0] == "q4_g64_fp16", f"{site} is {table[site][0]}, expected Q4"
    for site in (
        f"text/layers/{LAYER}/attention/value",
        f"text/layers/{LAYER}/attention/output",
        f"text/layers/{LAYER}/gdn/z",
        f"text/layers/{LAYER}/gdn/output",
        f"text/layers/{LAYER}/mlp/down",
    ):
        assert table[site][0] == "q5_g64_fp16", f"{site} is {table[site][0]}, expected Q5"


def test_both_w8_endpoints_are_q8(bf16_sources) -> None:
    """token_embedding and output_head are the two W8 endpoints, and both are Q8.

    artifact-conventions.md records them as 2.52 GiB, 14.3 % of everything bound, so their format is a
    shipping decision rather than a detail.
    """
    table = _format_table("qwen3_8_27b", bf16_sources)
    assert table["text/token_embedding"][0] == "q8_g32_fp16"
    assert table["text/output_head"][0] == "q8_g32_fp16"


def test_the_draft_projection_is_q8(bf16_sources) -> None:
    """_optional gives the dflash2 projections Q8, and skips the five listed upstream names.

    The skip list is upstream's and was measured: including the convolution kernel projections dropped
    acceptance from 58.0 % to 42.2 % on this port's own bench, so a site quietly added back to the
    list would be a measured regression rather than a tidy-up.
    """
    table = _format_table("qwen3_8_27b", bf16_sources)
    assert table["dflash2/layers/0/attention/query"][0] == "q8_g32_fp16"
    assert table["dflash2/layers/0/mlp/gate"][0] == "q8_g32_fp16"


def test_no_text_projection_is_left_unassigned(bf16_sources) -> None:
    """Every text projection appears in the table. A site a recipe forgets is a site the artifact
    would be missing, and that fails silently -- nothing downstream reports an absent weight."""
    expected = {
        site
        for site, parameter in _build_model().parameters.items()
        if site.startswith("text/") and parameter.projection
    }
    assert not expected - set(_format_table("qwen3_8_27b", bf16_sources))
