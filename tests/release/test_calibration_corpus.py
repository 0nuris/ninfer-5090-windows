"""That the calibration-corpus gate can actually fail.

An assertion that cannot fail is worse than no assertion, and this gate is the thing standing between
a coding-first corpus proposal and the creator's ten documents. Its teeth are proved here rather than
asserted, because a gate whose failure paths have never been executed is indistinguishable from a gate
that always passes.

The gate derives its tree root from ``__file__``, so each case builds a mirror of the four files it
reads under ``tmp_path`` and points the module's ``WT`` at that. Nothing in the repository is edited,
which matters for a test whose whole purpose is to make things go wrong. The unmodified mirror is the
control: if it does not pass, the other cases prove nothing.

Six cases. The control passes, and five must fail:

1. a document edited, so the pinned bytes no longer match
2. the encoder's ``full_range`` moved alone, which is the clamp the 2026-09-29 experiment nearly shipped
3. all three carriers moved together to ``4 * 448``, which the format's own definition must reject even
   though they agree -- the pin fires too, so this case asserts *both* checks report, not just one
4. a second calibration corpus appearing, so which file is used is no longer decided by the path
5. the calibrator's constant surviving only inside a comment, which the ``ast`` read must not accept
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

# The release scripts run standalone and put their own directory on sys.path, as the doc-link test
# records; importing by dotted name would resolve the same file under two module names.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "release"))

import check_calibration_corpus as gate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
GATE = "tools/release/check_calibration_corpus.py"
CORPUS = "tools/convert/calibration_corpus.json"
CALIBRATOR = "tools/convert/calibration.py"
ENCODER = "tools/convert/quantization/nvfp4.py"
MIRRORED = (GATE, CORPUS, CALIBRATOR, ENCODER)


def mirror(tmp_path: Path) -> Path:
    """Copy the four files the gate reads into a scratch tree and return its root."""
    for relative in MIRRORED:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    return tmp_path


def run(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], str]:
    """Point the gate at a scratch tree, run it, and hand back what it reported.

    The return is the failure list and the combined report rather than an exit code, because the
    assertions are about which check fired -- the pin and the format check can both fire on one
    perturbation, and a single int would hide the second.
    """
    monkeypatch.setattr(gate, "WT", root)
    assert gate.main() in (0, 1)
    return list(gate.failures), "\n".join(gate.failures + gate.notes)


def swap(root: Path, relative: str, old: str, new: str) -> None:
    target = root / relative
    text = target.read_text(encoding="utf-8")
    assert old in text, f"{relative}: {old!r} not found, so this case would not perturb anything"
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def set_corpus_full_range(root: Path, value: float) -> None:
    path = root / CORPUS
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["full_range"] = value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_the_unmodified_mirror_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control. Without it a passing perturbation case would mean nothing."""
    root = mirror(tmp_path / "control")
    failures, _ = run(root, monkeypatch)
    assert failures == []


def test_an_edited_document_fails_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = mirror(tmp_path / "edited")
    path = root / CORPUS
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["documents"][2] += "\n# one character of drift"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failures, _ = run(root, monkeypatch)
    assert any("not the creator's as pinned" in failure for failure in failures), failures


def test_the_encoder_moving_alone_fails_the_agreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The clamp. One carrier of three moving is a block scale no one else agrees with."""
    root = mirror(tmp_path / "encoder-only")
    swap(root, ENCODER, "FULL_RANGE = 2688.0", "FULL_RANGE = 1792.0")
    failures, _ = run(root, monkeypatch)
    assert any("full_range disagrees" in failure for failure in failures), failures


def test_a_consistent_but_non_format_value_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agreement is not the same as correctness: 4 * 448 is agreed and still wrong.

    The pin fires here too, so this asserts the format check is reported *in addition to* rather than
    instead of it. A gate that reported only the pin would pass this case while quietly losing the
    check that distinguishes a consistent edit from a drift.
    """
    root = mirror(tmp_path / "four")
    set_corpus_full_range(root, 1792.0)
    swap(root, CALIBRATOR, "FULL_RANGE = 2688.0", "FULL_RANGE = 1792.0")
    swap(root, ENCODER, "FULL_RANGE = 2688.0", "FULL_RANGE = 1792.0")
    failures, _ = run(root, monkeypatch)
    assert any("E2M1's maximum times E4M3FN's" in failure for failure in failures), failures
    assert any("not the creator's as pinned" in failure for failure in failures), failures


def test_a_second_corpus_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = mirror(tmp_path / "second-corpus")
    shutil.copy2(root / CORPUS, root / "tools" / "convert" / "calibration_corpus_coding.json")
    failures, _ = run(root, monkeypatch)
    assert any("second calibration corpus" in failure for failure in failures), failures


def test_a_constant_only_in_a_comment_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason this reads with ast. A regex over FULL_RANGE also matches a comment."""
    root = mirror(tmp_path / "commented")
    swap(root, CALIBRATOR, "FULL_RANGE = 2688.0", "# FULL_RANGE = 2688.0, now only prose")
    failures, _ = run(root, monkeypatch)
    assert any("no module-level FULL_RANGE" in failure for failure in failures), failures


def test_the_gate_excludes_itself_from_the_consumer_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first version of this gate failed on the real tree for exactly this reason.

    The gate names the corpus in its own source, so the consumer scan counted the gate. It is worth a
    test rather than a comment, because the failure looks like a corpus problem and is not one.
    """
    root = mirror(tmp_path / "self-reference")
    failures, report = run(root, monkeypatch)
    assert "check_calibration_corpus.py" not in report, report
    assert failures == []
