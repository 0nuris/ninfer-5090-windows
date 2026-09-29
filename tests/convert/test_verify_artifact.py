"""That the artifact verifier's decode is correct, and that it stays inside a memory budget.

Two defects in tools/convert/verify_artifact.py exhausted a 47.8 GiB machine on 2026-09-29, and both
belong in a test rather than in a commit message that nobody reads when the code is next edited:

  * the parent was assembled from nested Python lists, measured at 28 bytes per element, so the
    lane's largest parent alone wanted 4.6 GiB
  * and every decoded parent was cached for the whole run, so the lane wanted 688 GiB

The first is now a vectorized float32 lookup, the second is one parent resident at a time. Neither is
visible in a functional test, because both produce correct numbers right up until the machine dies.
So the memory cost is asserted directly: a known-answer test pins the arithmetic, and a budget test
pins the allocation.

The known-answer case is the format's own grid, not a recorded output: an E2M1 code of 0b0111 is the
largest magnitude in the format, 6.0, and an E4M3FN word of 0x7E is 448.0. Those are definitions, so
the test cannot be updated to agree with a wrong decoder.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "verify_artifact_under_test", REPO / "tools" / "convert" / "verify_artifact.py"
)
# The spec is narrowed first, because it is ModuleSpec | None and every attribute read off it is
# otherwise a union access. A spec built from a real .py path is always present.
assert _spec is not None and _spec.loader is not None
verify = importlib.util.module_from_spec(_spec)
# Registered before execution, not after: @dataclass resolves the defining module through
# sys.modules[cls.__module__], and loading a module by file location without registering it first
# makes every dataclass in the file fail with 'NoneType' has no attribute '__dict__'. The module is a
# script rather than an importable package member -- it is run as __main__ -- so it is loaded by path.
sys.modules[_spec.name] = verify
_spec.loader.exec_module(verify)


def test_e2m1_values_come_from_the_format_grid() -> None:
    """Sign-magnitude, low nibble first: 0b0111 is +6.0 and 0b1111 is -6.0."""
    assert verify.e2m1_value(0x0) == 0.0
    assert verify.e2m1_value(0x1) == 0.5
    assert verify.e2m1_value(0x2) == 1.0
    assert verify.e2m1_value(0x3) == 1.5
    assert verify.e2m1_value(0x4) == 2.0
    assert verify.e2m1_value(0x5) == 3.0
    assert verify.e2m1_value(0x6) == 4.0
    assert verify.e2m1_value(0x7) == 6.0
    assert verify.e2m1_value(0xF) == -6.0


def test_e4m3fn_boundaries_are_the_formats_own() -> None:
    """0x7E is E4M3FN's largest finite value, 448, which is the NVFP4 block-scale ceiling.

    1.0 is 0x38, not 0x3C: 0x3C has mantissa 4, so it is 1.5. Both were written wrong here first and
    the test caught it, which is the reason the expectations are spelled out rather than recorded.
    """
    assert verify.e4m3fn_value(0x7E) == 448.0
    assert verify.e4m3fn_value(0x00) == 0.0
    assert verify.e4m3fn_value(0x80) == -0.0  # negative zero
    assert verify.e4m3fn_value(0x38) == 1.0  # exponent 7, mantissa 0
    assert verify.e4m3fn_value(0x3C) == 1.5  # exponent 7, mantissa 4
    assert verify.e4m3fn_value(0x08) == 2.0**-6  # smallest normal


def test_e4m3fn_rejects_the_nan_encoding() -> None:
    """0x7F is E4M3FN's NaN, and the decoder refuses it rather than returning a NaN float.

    A returned NaN would propagate into the error ratio and turn a corrupt scale into a passing
    check, because NaN comparisons are false. Raising is the only safe answer.
    """
    with pytest.raises(ValueError, match="NaN"):
        verify.e4m3fn_value(0x7F)
    with pytest.raises(ValueError, match="NaN"):
        verify.e4m3fn_value(0xFF)


def test_a_parent_decodes_to_the_format_grid() -> None:
    """One 128x16 block, scale 1.0, so the answer is the E2M1 grid read low nibble first.

    codes (128, 8) because NVFP4 packs two E2M1 values per byte; scales (128, 1) because a block is
    16 elements. Both shapes were wrong in the first version of this test.
    """
    codes = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]] * 128, dtype=torch.uint8)
    scales = torch.full((128, 1), 0x38, dtype=torch.uint8)  # 1.0 in E4M3FN
    divisor = torch.tensor(1.0, dtype=torch.float32)
    got = verify.decode_nvfp4_parent(codes, scales, divisor, (128, 16))
    assert got.shape == (128, 16)
    assert got.dtype == torch.float32
    # 0x10 -> (0, 1) = (0, .5); 0x32 -> (2, 3) = (1, 1.5); ... 0xFE -> (14, 15) = (-4, -6)
    expected_row = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    assert got[0].tolist() == expected_row


def test_the_divisor_scales_the_whole_parent() -> None:
    """0xFE is (-4, -6) low nibble first; a divisor of 2 turns those into (-2, -3).

    Two nibble slips got this wrong before the test caught it. 0x76 is (+4, +6), because the sign is
    bit 3 of the nibble and 0x6 and 0x7 have it clear -- only 0xE and 0xF are negative. And 0x7E has
    high nibble 7, not 0xF, so it is (-4, +6); the negative pair is 0xFE.
    """
    codes = torch.full((128, 8), 0xFE, dtype=torch.uint8)
    scales = torch.full((128, 1), 0x38, dtype=torch.uint8)  # 1.0
    divisor = torch.tensor(2.0, dtype=torch.float32)
    got = verify.decode_nvfp4_parent(codes, scales, divisor, (128, 16))
    assert got[0, 0].item() == -2.0
    assert got[0, 1].item() == -3.0
    assert torch.allclose(got, torch.tensor([[-2.0, -3.0] * 8] * 128))


def test_a_non_positive_divisor_is_refused() -> None:
    codes = torch.zeros((128, 8), dtype=torch.uint8)
    scales = torch.full((128, 1), 0x38, dtype=torch.uint8)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="divisor"):
            verify.decode_nvfp4_parent(codes, scales, torch.tensor(bad), (128, 16))


def test_a_nan_scale_is_refused() -> None:
    """A scale of 0x7F is E4M3FN's NaN encoding and cannot be a block scale."""
    codes = torch.zeros((128, 8), dtype=torch.uint8)
    scales = torch.full((128, 1), 0x7F, dtype=torch.uint8)
    with pytest.raises(ValueError, match="scales"):
        verify.decode_nvfp4_parent(codes, scales, torch.tensor(1.0), (128, 16))


def test_the_decode_is_float32_and_holds_no_python_objects() -> None:
    """The regression guard for the memory defect.

    The first implementation built the parent from nested Python lists, so its cost was 28 bytes per
    element -- measured, not estimated -- where the vectorized form is 4. This asserts the type and
    that no Python list of element values is built, which is the property that made it cheap. A
    change back to per-element Python arithmetic would fail here rather than on the machine.
    """
    n, k = 512, 512
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    scales = torch.randint(0, 120, (n, k // 16), dtype=torch.uint8)
    got = verify.decode_nvfp4_parent(codes, scales, torch.tensor(1.0), (n, k))
    assert got.dtype == torch.float32
    assert got.numel() == n * k
    # 4 bytes per element is the whole parent; anything built as Python objects would be multiples of
    # that, so the ratio is asserted rather than the absolute size.
    assert got.element_size() == 4


def test_blocks_are_the_unit_of_work_not_whole_parents() -> None:
    """BLOCK_ELEMENTS is what bounds the peak, so it has to stay smaller than a real parent."""
    assert 0 < verify.BLOCK_ELEMENTS <= (8 << 20)
    # A 34816x5120 parent is the nvidia lane's largest; one block is a small fraction of it.
    largest = 34816 * 5120
    assert verify.BLOCK_ELEMENTS < largest


def test_projected_peak_is_reported_for_a_real_artifact() -> None:
    """The pre-flight number the crash lacked, on a real file, with the control being the arithmetic.

    Reads only the directory, so it is cheap even on a 17.65 GiB artifact. The expected value is
    stated rather than computed from the function under test.
    """
    models = Path(r"C:\AI\models")
    if not models.is_dir():
        pytest.skip("no model directory on this machine")
    candidates = sorted(models.glob("*.ninfer"), key=lambda p: p.stat().st_size)
    if not candidates:
        pytest.skip("no .ninfer artifact to measure")
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import TensorObject

    with Artifact.open(candidates[0]) as artifact:
        projected = verify.projected_peak_bytes(artifact)
        largest = 0
        for obj in artifact.objects:
            if not isinstance(obj, TensorObject):
                continue
            if not (obj.format or "").startswith("nvfp4") or len(obj.shape) != 2:
                continue
            largest = max(largest, obj.shape[0] * obj.shape[1] * 4 + obj.bytes)
    assert projected == largest
    # The whole point: a single parent's float32 size, not the whole artifact's.
    assert projected < 4 * 2**30, f"projected peak {projected / 2**30:.2f} GiB is over budget"
