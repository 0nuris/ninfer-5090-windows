#!/usr/bin/env python3
"""Pin the calibration corpus to the one the artifact's creator used, and check what must agree with it.

Product decision, 2026-09-29: this port always calibrates with the original creator's corpus. A
coding-first replacement was proposed and declined, so the property is enforced here rather than left
as prose in the corpus file's own `note` field -- a note is a comment, and this has to be a
constraint. Reopening it means changing PINNED below and recording the measurement that justified it,
which is a deliberate act rather than an accident.

The provenance cannot be verified against a public tree: `calibrate_nvfp4full.py`, the script the
corpus note names as its source, is absent from both Neroued/ninfer (1,907 entries) and cometkim/ninfer
(1,900). So a byte pin is the strongest guarantee available, and it is worth being exact about what it
does and does not establish. It establishes that the corpus has not changed since 46e1309f. It does not
establish that those ten documents are what that script read. The reason to trust them anyway is
recorded in docs/maintainer/artifact-conventions.md: the same corpus and arithmetic reproduce the
published nvfp4full divisors, and that is the evidence the corpus is the creator's.

Two things make this gate more than a hash comparison.

The pin is over the newline-normalized bytes, not the raw bytes, because the two already differ in
this tree. The working copy is CRLF (11,731 bytes) and the index is LF (11,716), so a raw pin reads
`c3cd10de...` here and `1bdb7a25...` in a clean checkout. A pin that flips on a fresh clone is a gate
people learn to bypass, so both hashes are named below and only the normalized one is enforced.

`full_range` appears in three places that have no coupling: the corpus JSON (which the converter never
reads -- it is documentation), `calibration.py`, and the NVFP4 encoder. They all orient the same E2M1
block scale, and a block-scale experiment on 2026-09-29 changed two of the three at once. If only the
encoder moved, every large block's s_block would have reached 448 * 6/4 = 672 and clamped against the
E4M3FN maximum, and the measurement would have been of the clamp. This gate is that coupling made
structural. It is read with ast rather than a regex, because a regex over `FULL_RANGE` also matches the
name inside a comment or docstring, and a guard that matches prose is a guard that passes for the wrong
reason.

Exit code 1 on any failure.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]

# Held relative and joined with WT inside main(), not resolved at import. tests/release redirects WT at
# a scratch mirror of these same four files, and a path captured at import would then name the real
# tree while every file being read came from the mirror -- so the stray scan would report the mirror's
# own corpus as a stray and the consumer scan would count the mirrored gate. Comparing against
# __file__ has the same fault for a different reason. Both are derived per call instead.
CORPUS_REL = "tools/convert/calibration_corpus.json"
GATE_REL = "tools/release/check_calibration_corpus.py"

# sha256 over CRLF-normalized-to-LF bytes. Portable across git's line-ending handling; the raw
# worktree hash is c3cd10de24c4fd1444b0be7c84d3f4ff886bcf65ca95778108a12ba7e0fe9b78 and the raw index
# hash is 1bdb7a253949667b68cf1dd95a86be872da8c00e3d3b3ad6cd7682cac4a4152e, because the working copy is
# CRLF and the index is LF. Verified equal after normalization.
PINNED = "1bdb7a253949667b68cf1dd95a86be872da8c00e3d3b3ad6cd7682cac4a4152e"

DOCUMENTS = 10

# The format's own definition, not a choice of ours: E2M1's largest magnitude is 6, E4M3FN's is 448,
# and the global scale orients a block to the format maximum. Asserted separately from the
# three-way agreement so a change that moves all three consistently still has to explain itself.
E2M1_MAX = 6.0
E4M3_MAX = 448.0

# full_range must be one number, agreed by the corpus file, the calibrator, and the encoder.
CARRIERS = [
    ("tools/convert/calibration.py", "FULL_RANGE"),
    ("tools/convert/quantization/nvfp4.py", "FULL_RANGE"),
]

failures: list[str] = []
notes: list[str] = []


def module_constant(relative: str, name: str) -> float | None:
    """Return a module-level float constant, or None if it is absent or not a plain literal.

    ast rather than a regex: the point is to read the assignment, not any mention of the name.
    """
    path = WT / relative
    if not path.is_file():
        failures.append(f"{relative}: missing")
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        failures.append(f"{relative}: does not parse ({exc})")
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == name for t in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return float(value.value)
        failures.append(f"{relative}: {name} is not a plain numeric literal, so it cannot be checked")
        return None
    failures.append(f"{relative}: no module-level {name}")
    return None


def main() -> int:
    # Cleared rather than rebound, so the gate can be called more than once in one interpreter --
    # tests/release/test_calibration_corpus.py does exactly that, and accumulating a previous case's
    # failures into the next case's report would let a broken mirror read as a clean one.
    failures.clear()
    notes.clear()
    corpus = WT / CORPUS_REL
    this_gate = (WT / GATE_REL).resolve()

    # 1. the pinned bytes
    if not corpus.is_file():
        failures.append(f"calibration corpus absent: {corpus.relative_to(WT).as_posix()}")
    else:
        raw = corpus.read_bytes()
        normalized = raw.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        if digest == PINNED:
            notes.append(f"corpus pin matches ({len(raw)} bytes on disk, {len(normalized)} normalized)")
        else:
            failures.append(
                "calibration corpus is not the creator's as pinned:\n"
                f"      expected {PINNED}\n"
                f"      actual   {digest}\n"
                "    This port always calibrates with the original creator's corpus. If a change here is\n"
                "    intended, re-measure full-corpus perplexity for every affected lane first, then\n"
                "    update PINNED in this file and record the measurement alongside it."
            )

    # 2. one corpus, one production consumer
    strays = sorted(
        p.relative_to(WT).as_posix()
        for p in (WT / "tools").rglob("*calibration_corpus*.json")
        if p.is_file() and p.resolve() != corpus.resolve()
    )
    if strays:
        failures.append(
            "a second calibration corpus exists, so which one is used is no longer decided by the "
            "path alone: " + ", ".join(strays)
        )
    consumers = sorted(
        p.relative_to(WT).as_posix()
        for p in (WT / "tools").rglob("*.py")
        if p.is_file()
        and p.resolve() != this_gate  # this gate names the corpus, so it is not a consumer
        and "calibration_corpus" in p.read_text(encoding="utf-8", errors="replace")
    )
    if consumers != ["tools/convert/calibration.py"]:
        failures.append(
            "the corpus is read by something other than the calibrator alone: "
            + (", ".join(consumers) or "nothing at all")
        )
    else:
        notes.append("single consumer: tools/convert/calibration.py")

    # 3. the three-way full_range agreement, and the format's own definition
    values: dict[str, float] = {}
    if corpus.is_file():
        try:
            payload = json.loads(corpus.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"calibration_corpus.json does not parse ({exc})")
            payload = None
        if isinstance(payload, dict):
            if "full_range" not in payload:
                failures.append("calibration_corpus.json: full_range key absent")
            else:
                values[CORPUS_REL] = float(payload["full_range"])
                docs = payload.get("documents")
                if not isinstance(docs, list) or len(docs) != DOCUMENTS:
                    failures.append(
                        f"calibration_corpus.json: {DOCUMENTS} documents expected, found "
                        f"{len(docs) if isinstance(docs, list) else type(docs).__name__}"
                    )
    for relative, name in CARRIERS:
        found = module_constant(relative, name)
        if found is not None:
            values[relative] = found

    if values:
        distinct = sorted(set(values.values()))
        if len(distinct) != 1:
            detail = ", ".join(f"{k}={v:g}" for k, v in sorted(values.items()))
            failures.append(
                "full_range disagrees between the corpus, the calibrator and the encoder. These "
                f"orient one E2M1 block scale; a block scale that only one of them moves is a scale "
                f"that clamps. Measured here on 2026-09-29: {detail}"
            )
        else:
            only = distinct[0]
            if only != E2M1_MAX * E4M3_MAX:
                failures.append(
                    f"full_range is {only:g}, but E2M1's maximum times E4M3FN's is "
                    f"{E2M1_MAX * E4M3_MAX:g}. All three carriers agree, so this is deliberate "
                    "rather than a drift; the 6-over-4 question is settled in "
                    "docs/research/nvfp4-block-scale-4-vs-6.md and is not to be reopened here."
                )
            else:
                notes.append(
                    f"full_range = {only:g} = {E2M1_MAX:g} (E2M1 max) * {E4M3_MAX:g} (E4M3FN max), "
                    "agreed by the corpus, the calibrator and the encoder"
                )

    for note in notes:
        print(f"  ok   {note}")
    for failure in failures:
        print(f"  FAIL {failure}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} calibration-corpus check(s) failed.", file=sys.stderr)
        return 1
    print("  PASS: the calibration corpus is the creator's, and the three full_range carriers agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
