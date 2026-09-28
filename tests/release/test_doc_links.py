"""What the documentation-link checker treats as a reference, and what it must not.

A backticked repository path is a reference to resolve, except inside a fenced code block, where it
is a command literal. Resolving those reports correct documentation as broken, so the distinction is
worth a test rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The release scripts run standalone and put their own directory on sys.path, which is also why
# mypy.ini lists tools/release in mypy_path. Importing by dotted name instead would resolve the same
# file under two module names and fail the type check, so the path is derived and added as the
# scripts do it rather than as a package import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "release"))

from check_doc_links import code_paths  # noqa: E402


def _paths(text: str) -> list[str]:
    return [target for _, target in code_paths(text)]


def test_a_backticked_path_outside_a_fence_is_a_reference() -> None:
    assert _paths("see `docs/README.md` for more.\n") == ["docs/README.md"]


def test_a_path_inside_a_fenced_block_is_not_a_reference() -> None:
    """A command naming a file that exists only on the author's machine is not a broken link."""
    assert _paths("```bash\npython -m x --out `docs/does_not_exist.md`\n```\n") == []


def test_tilde_fences_fence_too() -> None:
    assert _paths("~~~\n`src/nope.h`\n~~~\n") == []


def test_a_backtick_fence_inside_a_tilde_fence_is_content() -> None:
    """The closing marker must match the opening one, or an inner fence ends the block early."""
    assert _paths("~~~\n```\n`tools/nope.py`\n```\n~~~\n") == []


def test_a_path_after_a_closed_fence_is_a_reference_again() -> None:
    """If the fence failed to close, real broken links would stop being reported."""
    assert _paths("```\n`src/a.h`\n```\n`docs/broken.md`\n") == ["docs/broken.md"]


def test_a_fence_may_be_indented_by_up_to_three_spaces() -> None:
    assert _paths("   ```\n   `docs/indent.md`\n   ```\n") == []


def test_an_unterminated_fence_swallows_the_rest_of_the_document() -> None:
    assert _paths("```\n`docs/x.md`\n") == []
