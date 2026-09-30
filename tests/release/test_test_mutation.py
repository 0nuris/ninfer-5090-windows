"""That the perturbation gate can actually fail.

An assertion that cannot fail is worse than no assertion, and this gate is the thing standing between
a green Op test and a false claim that the Op is correct. Its teeth are proved here rather than
asserted, because a gate whose failure paths have never been executed is indistinguishable from a gate
that always passes.

The gate reads a manifest path and runs ctest, so every case here builds a manifest under
``tmp_path`` and points ``load`` at it. Nothing in the repository is edited, which matters for a test
whose whole purpose is to make things go wrong.

These cases cover the *manifest* half: the shape of a declaration and every way it can lie. The half
that cannot be tested this way -- that setting ``NINFER_OP_MUTATION`` really does turn the suite red
-- needs a built tree and a GPU, so it is proved by running the gate itself, and by
``tools/release/check_test_mutation.py --list`` being able to read the real manifest at all.

The unmodified real manifest is the control. If it does not load, every other case here proves
nothing, and a gate that cannot parse the repository's own manifest is the failure this test exists
to catch first.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

# The release scripts run standalone and put their own directory on sys.path; importing by dotted
# name would resolve the same file under two module names.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "release"))

import check_test_mutation as gate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def valid() -> dict:
    """A minimal manifest that must load, used as the base for every mutation of it."""
    return {
        "schema": 1,
        "ops": {
            "synthetic": {
                "env": "NINFER_OP_MUTATION",
                "test": "ninfer_synthetic_test",
                "mutations": [
                    {"id": 1, "site": "a site", "why": "a reason"},
                ],
            }
        },
    }


def write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "mutations.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="")
    return path


def test_the_reals_manifest_loads_and_declares_the_ops_it_claims() -> None:
    """The control. The repository's own manifest must parse, or the gate cannot run at all."""
    ops = gate.load(REPO_ROOT / "tests" / "ops" / "mutations.json")
    assert "topk_logprobs" in ops, "the shipped manifest no longer names the op it was written for"
    for name, op in ops.items():
        assert op.mutations, f"{name} declares no mutations, which load() is meant to reject"
        for mutation in op.mutations:
            assert mutation.id > 0
            assert mutation.site.strip()
            assert mutation.why.strip(), "a mutation with no recorded purpose is decoration"


def test_a_valid_manifest_loads(tmp_path: Path, valid: dict) -> None:
    ops = gate.load(write(tmp_path, valid))
    assert set(ops) == {"synthetic"}
    assert ops["synthetic"].test == "ninfer_synthetic_test"
    assert [m.id for m in ops["synthetic"].mutations] == [1]


def test_a_missing_manifest_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(gate.ManifestError, match="not found"):
        gate.load(tmp_path / "absent.json")


def test_malformed_json_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "mutations.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(gate.ManifestError, match="not valid JSON"):
        gate.load(path)


def test_an_unknown_schema_is_fatal(tmp_path: Path, valid: dict) -> None:
    """A future schema must not be read as today's. Silently accepting it is how a manifest rots."""
    valid["schema"] = gate.SCHEMA + 1
    with pytest.raises(gate.ManifestError, match="expected schema"):
        gate.load(write(tmp_path, valid))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["ops"]["synthetic"].__setitem__("test", 7), "non-empty string"),
        (lambda d: d["ops"]["synthetic"].__setitem__("test", ""), "non-empty string"),
        (lambda d: d["ops"]["synthetic"].__setitem__("test", "ninfer bad name"), "bare test name"),
        (lambda d: d["ops"]["synthetic"].__setitem__("test", "^ninfer_x_test$"), "bare test name"),
        (lambda d: d["ops"]["synthetic"].__setitem__("env", "lower_case"), "UPPER_SNAKE"),
        (lambda d: d["ops"]["synthetic"].__setitem__("mutations", []), "non-empty list"),
        (lambda d: d["ops"]["synthetic"].__setitem__("mutations", {}), "non-empty list"),
        (lambda d: d["ops"]["synthetic"].__setitem__("mutations", [7]), "must be objects"),
        (lambda d: d["ops"]["synthetic"]["mutations"][0].__setitem__("id", 0), "positive int"),
        (lambda d: d["ops"]["synthetic"]["mutations"][0].__setitem__("id", "1"), "positive int"),
        (lambda d: d["ops"]["synthetic"]["mutations"][0].pop("site"), "needs a site"),
        (lambda d: d["ops"]["synthetic"]["mutations"][0].__setitem__("why", "  "), "needs a why"),
    ],
)
def test_a_lying_declaration_is_rejected(tmp_path: Path, valid: dict, mutate, message: str) -> None:
    """Each field that could make a mutation look covered while covering nothing."""
    document = copy.deepcopy(valid)
    mutate(document)
    with pytest.raises(gate.ManifestError, match=message):
        gate.load(write(tmp_path, document))


def test_a_duplicate_mutation_id_is_rejected(tmp_path: Path, valid: dict) -> None:
    """Two entries claiming the same id would report one as a pass without ever running it."""
    valid["ops"]["synthetic"]["mutations"].append(
        {"id": 1, "site": "the same id again", "why": "should be rejected"}
    )
    with pytest.raises(gate.ManifestError, match="twice"):
        gate.load(write(tmp_path, valid))


def test_an_op_entry_that_is_not_an_object_is_rejected(tmp_path: Path, valid: dict) -> None:
    valid["ops"]["synthetic"] = ["not", "an", "object"]  # type: ignore[assignment]
    with pytest.raises(gate.ManifestError, match="must be an object"):
        gate.load(write(tmp_path, valid))


@pytest.mark.parametrize(
    ("output", "pending"),
    [
        ("[1/4] Building CUDA object src/ops/foo.cu.obj", True),
        ("[1/4] a\n[2/4] b\n[3/4] c\n[4/4] d\n", True),
        ("ninja: no work to do.", False),
        ("", False),
        # The word "build" must not be mistaken for a pending step. This is why the check is
        # structural: a prose match on "build" would report this tree as dirty when it is current.
        ("Build files have been written to: C:/AI/ninfer-v3-windows/build-test", False),
        ("[FAILED] Building CUDA object", False),
    ],
)
def test_pending_build_steps_are_detected_structurally(output: str, pending: bool) -> None:
    """A dry run exits 0 whether or not there is work, so the step lines are the only signal.

    The negative cases matter more than the positive one: a guard that matched the word "build"
    would fail every current tree, and one that matched "[FAILED]" would report a failure as pending.
    """
    assert bool(gate.NINJA_STEP.findall(output)) is pending
