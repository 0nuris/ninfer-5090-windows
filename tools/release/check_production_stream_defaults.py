#!/usr/bin/env python3
"""Ratchet: the set of defaulted-stream sites in production code may not change silently.

A function that defaults its `cudaStream_t` to `nullptr` publishes on the legacy default stream.
This codebase creates its compute and load streams with `cudaStreamNonBlocking`
(`src/core/device.cu`), and a non-blocking stream has, by definition, no implicit ordering
relationship with the legacy one. So a defaulted stream in production code is a write whose
ordering against the kernels that read it is unspecified.

That is a real defect class, documented in docs/research/kv-publish-stream-ordering.md and reported
upstream five times (#208, #210, #320, #329, #333). It is not being fixed here on speculation,
because there is no cheap test that shows a fix helped -- the fault is sporadic and hours-long, and
compute-sanitizer was measured as too slow to catch it. What this gate does instead is stop the
surface from widening while we wait.

It is a ratchet, not a detector, and the limit is worth stating plainly rather than letting the name
imply more than it delivers:

  * it catches exactly one thing -- someone adding a defaulted stream, in a new file or a new
    function in a pinned one
  * it does NOT catch a correct-looking call that passes the WRONG stream
  * it does NOT catch a new publish path that lives somewhere this pattern does not match
  * it does NOT tell you whether the race fires, here or anywhere

A difference in EITHER direction is a failure. Growth is a regression. Shrinkage means the upstream
fix arrived, and updating the pin is then a deliberate act with a commit message, not a silent pass.
A gate that quietly re-baselines itself is a gate that cannot fail, which is the defect
tools/release/check_test_mutation.py exists to prevent.

Tests are excluded deliberately: a defaulted stream in a test is correct, because there is no
compute stream to race.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

PIN = Path("tools/release/production_stream_defaults.json")
SCAN_ROOTS = (Path("src"), Path("include"))
SOURCE_SUFFIXES = (".h", ".hpp", ".cu", ".cuh", ".cpp")
SCHEMA = 1

# Matched on the declaration, tolerating spacing, and anchored on cudaStream_t so a comment or an
# unrelated identifier cannot trigger it. `const cudaStream_t stream = nullptr` also matches, which
# is correct: the const is what makes it a by-value kernel-side default.
DEFAULTED = re.compile(r"cudaStream_t\s+(?:const\s+)?\w+\s*=\s*nullptr")
SKIP_DIRECTORIES = {".git", "build", "build-test", "out", "node_modules"}


def count_sites(roots: Sequence[Path]) -> dict[str, int]:
    """Per-file count of defaulted-stream declarations, keyed by repo-relative POSIX path."""
    counts: dict[str, int] = {}
    for directory in roots:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            if SKIP_DIRECTORIES & set(path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found = len(DEFAULTED.findall(text))
            if found:
                counts[path.as_posix()] = found
    return counts


def load_pin(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise SystemExit(f"pin not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: expected schema {SCHEMA}, got {raw.get('schema')!r}")
    sites = raw.get("sites")
    if not isinstance(sites, dict) or not sites:
        raise SystemExit(f"{path}: sites must be a non-empty object")
    bad = [k for k, v in sites.items() if not isinstance(v, int) or v < 1]
    if bad:
        raise SystemExit(f"{path}: counts must be positive ints; bad entries {bad}")
    return sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin", type=Path, default=PIN)
    parser.add_argument("--list", action="store_true", help="print the current sites and exit")
    args = parser.parse_args()

    try:
        pinned = load_pin(args.pin)
    except (json.JSONDecodeError, SystemExit) as error:
        print(f"PIN ERROR: {error}", file=sys.stderr)
        return 1

    current = count_sites(SCAN_ROOTS)

    if args.list:
        total = sum(current.values())
        for name in sorted(current):
            print(f"  {current[name]:>3}  {name}")
        print(f"  {total:>3}  total across {len(current)} file(s)")
        return 0

    grew = {f: n for f, n in current.items() if f not in pinned or n > pinned[f]}
    shrank = {f: pinned[f] for f, n in current.items() if f in pinned and n < pinned[f]}
    vanished = sorted(set(pinned) - set(current))

    if not grew and not shrank and not vanished:
        total = sum(current.values())
        print(f"RATCHET OK: {total} known defaulted-stream site(s) across {len(current)} file(s), "
              "unchanged from the pin.")
        return 0

    if grew:
        print("RATCHET FAILED: a production function now defaults its stream, or a pinned file grew.")
        for name in sorted(grew):
            was = pinned.get(name, 0)
            print(f"  + {name}: {was} -> {current[name]}")
        print("  A defaulted stream publishes on the legacy stream, which is unordered against the")
        print("  non-blocking compute streams. See docs/research/kv-publish-stream-ordering.md.")

    if shrank or vanished:
        print("RATCHET: the known deficit shrank, which means a fix landed.")
        for name in sorted(shrank):
            print(f"  - {name}: {pinned[name]} -> {current[name]}")
        for name in vanished:
            print(f"  - {name}: {pinned[name]} -> 0 (file no longer has one, or is gone)")
        print("  Update the pin deliberately and say in the commit message why. Do not let this")
        print("  re-baseline itself: a gate that rewrites its own expectations cannot fail.")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
