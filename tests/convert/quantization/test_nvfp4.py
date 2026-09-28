"""The local NVFP4 encoder: its arithmetic, its divisor, and its word contract."""

from __future__ import annotations

import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.convert.quantization.nvfp4 import (
    FULL_RANGE,
    e2m1_rne_codes,
    encode_block,
    search_block_ratios,
)


def _error(values: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
           divisor: float) -> float:
    """Squared reconstruction error, decoded with the independent table above."""
    return (_decode(codes, scales, divisor) - values).pow(2).sum().item()


def _searched(values: torch.Tensor, divisor: float):
    rows, columns = values.shape
    blocks = (values.float() * divisor).reshape(rows, columns // 16, 16)
    return encode_block(values, divisor, search_block_ratios(blocks))


def _outlier_heavy(rows: int, columns: int, seed: int) -> torch.Tensor:
    """Weights whose per-block maximum is a rare spike, which is what the search exists for.

    Real weight matrices are not Gaussian within a 16-wide block: most entries are small and a few
    are much larger, so a scale that maps the spike onto 6.0 leaves the other fifteen values on the
    coarse end of the E2M1 grid.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator) * 0.25
    spikes = torch.rand(rows, columns // 16, generator=generator) < 0.06
    values = values + spikes.repeat_interleave(16, dim=1) * torch.tensor(9.0)
    return values


def test_the_searched_scale_never_increases_reconstruction_error() -> None:
    """The whole point: the max-abs scale is in the candidate set, so this can only match or win."""
    for seed in range(6):
        values = _outlier_heavy(128, 256, seed)
        divisor = FULL_RANGE / values.abs().max().item()
        maxabs_codes, maxabs_scales = encode_block(values, divisor)
        codes, scales = _searched(values, divisor)
        baseline = _error(values, maxabs_codes, maxabs_scales, divisor)
        assert _error(values, codes, scales, divisor) <= baseline + 1e-9, (
            f"seed {seed}: searched {_error(values, codes, scales, divisor)} "
            f"is worse than max-abs {baseline}"
        )


def test_the_search_actually_shrinks_some_blocks() -> None:
    """A search that never moves is a search that does nothing; it must fire on outlier blocks."""
    values = _outlier_heavy(128, 256, seed=11)
    divisor = FULL_RANGE / values.abs().max().item()
    blocks = (values.float() * divisor).reshape(128, 256 // 16, 16)
    ratios = search_block_ratios(blocks)
    assert float(ratios.min()) < 1.0, "no block was shrunk, so the search is inert"
    assert float(ratios.max()) <= 1.0, "a block was grown, which the candidate set cannot do"


def test_a_block_whose_maximum_is_not_an_outlier_keeps_the_max_abs_scale() -> None:
    """Shrinking only pays where clipping beats coarsening; elsewhere it must not fire."""
    values = torch.ones(128, 64) * 3.0
    divisor = FULL_RANGE / 3.0
    blocks = (values.float() * divisor).reshape(128, 64 // 16, 16)
    assert float(search_block_ratios(blocks).min()) == 1.0


def test_the_searched_path_keeps_the_word_contract() -> None:
    """Every property the max-abs path holds, the searched path must hold too."""
    values = _outlier_heavy(128, 64, seed=3)
    divisor = FULL_RANGE / values.abs().max().item()
    codes, scales = _searched(values, divisor)
    assert codes.shape == (128, 32) and codes.dtype == torch.uint8
    assert scales.shape == (128, 4) and scales.dtype == torch.uint8
    # The clamp is what makes the block scale a valid E4M3 word.
    assert float(_step(scales, divisor).max()) <= 448.0

    step = _step(scales, divisor)
    decoded = _decode(codes, scales, divisor)
    scaled = (values / step).abs()
    magnitude = (decoded / step).abs()
    for lower, upper in zip(E2M1_MAGNITUDES, E2M1_MAGNITUDES[1:]):
        inside = (scaled >= lower) & (scaled <= upper)
        if bool(inside.any()):
            error = (magnitude - scaled).abs()[inside]
            # A shrunk scale clips, so the bound is checked only on the represented side: a value
            # beyond the block's top magnitude is the search's deliberate trade, not a miscode.
            assert float((error / ((upper - lower) / 2)).max()) <= 1.0


def test_the_searched_words_survive_the_registered_layout_round_trip() -> None:
    values = _outlier_heavy(128, 64, seed=5)
    divisor = FULL_RANGE / values.abs().max().item()
    codes, scales = _searched(values, divisor)
    divisor_word = torch.tensor((divisor,), dtype=torch.float32).view(torch.uint8).numpy().tobytes()
    got_codes, got_scales, got_divisor = decode_nvfp4_words(
        encode_nvfp4(codes, scales, divisor_word, (128, 64)), (128, 64))
    assert torch.equal(got_codes, codes)
    assert torch.equal(got_scales, scales)
    assert bytes(got_divisor.numpy().tobytes()) == divisor_word


def test_the_default_path_is_unchanged() -> None:
    """Artifacts built to date used the max-abs scale; omitting `ratios` must still produce it.

    If this ever fails, every artifact converted before the change can no longer be reproduced
    byte-for-byte from the same source, which is the only reason the default is kept.
    """
    values = _outlier_heavy(128, 64, seed=7)
    divisor = FULL_RANGE / values.abs().max().item()
    plain_codes, plain_scales = encode_block(values, divisor)
    units_codes, units_scales = encode_block(values, divisor, torch.ones(128, 4, 1))
    assert torch.equal(plain_codes, units_codes)
    assert torch.equal(plain_scales, units_scales)

# Written from the format definition, not from the encoder, so the decode below is an
# independent oracle: E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6 and their negatives.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1 = torch.tensor((*E2M1_MAGNITUDES, *(-magnitude for magnitude in E2M1_MAGNITUDES)))


def _step(scales: torch.Tensor, divisor: float) -> torch.Tensor:
    """One E4M3FN block scale per 16 columns, in the source's units."""
    return scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1) / divisor


def _decode(codes: torch.Tensor, scales: torch.Tensor, divisor: float) -> torch.Tensor:
    """Decode with an independent E2M1 table, low nibble first."""
    low, high = codes & 0x0F, (codes >> 4) & 0x0F
    quantized = torch.empty(codes.shape[0], codes.shape[1] * 2)
    quantized[:, 0::2] = E2M1[low.long()].float()
    quantized[:, 1::2] = E2M1[high.long()].float()
    return quantized * _step(scales, divisor)


def test_codes_are_within_half_their_local_bracket() -> None:
    """E2M1 spacing is not uniform (.5 to 2, 1.0 to 4, 2.0 to 6), so the bound is local."""
    torch.manual_seed(0)
    values = torch.randn(128, 64) * 3.0
    divisor = FULL_RANGE / values.abs().max().item()
    codes, scales = encode_block(values, divisor)
    assert codes.shape == (128, 32) and codes.dtype == torch.uint8
    assert scales.shape == (128, 4) and scales.dtype == torch.uint8

    step = _step(scales, divisor)
    decoded = _decode(codes, scales, divisor)

    scaled = (values / step).abs()
    magnitude = (decoded / step).abs()
    worst = 0.0
    for lower, upper in zip(E2M1_MAGNITUDES, E2M1_MAGNITUDES[1:]):
        inside = (scaled >= lower) & (scaled <= upper)
        if bool(inside.any()):
            error = (magnitude - scaled).abs()[inside]
            worst = max(worst, (error / ((upper - lower) / 2)).max().item())
    assert worst <= 1.0, f"a code is more than half a bracket from its source: {worst}"


def test_ties_round_to_the_even_code_and_sign_is_preserved() -> None:
    """RNE is on the code index: 0.75 sits between codes 1 and 2, so it takes 2."""
    ties = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    assert e2m1_rne_codes(ties).tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert e2m1_rne_codes(torch.tensor((-0.75, -5.0))).tolist() == [10, 14]


def test_the_global_divisor_is_the_reciprocal_of_the_global_scale() -> None:
    """s_global = amax / (448 * 6), so the stored divisor is 2688 / amax."""
    values = torch.tensor(((3.0, -1.0), (0.5, 2.0)))
    divisor = FULL_RANGE / values.abs().max().item()
    assert divisor == 2688.0 / 3.0
    # The block scale is E4M3FN, so its largest representable word is 448 and the encoder
    # must not exceed it: that clamp is what makes the block scale a valid E4M3 word.
    assert float(_step(torch.tensor(((0x7E,),)), 1.0).max()) == 448.0


def test_words_survive_the_registered_layout_round_trip() -> None:
    """The encoder's words, framed by the codec, decode back to the same words."""
    torch.manual_seed(1)
    values = torch.randn(128, 64) * 2.0
    divisor = FULL_RANGE / values.abs().max().item()
    codes, scales = encode_block(values, divisor)
    divisor_word = torch.tensor((divisor,), dtype=torch.float32).view(torch.uint8).numpy().tobytes()

    payload = encode_nvfp4(codes, scales, divisor_word, (128, 64))
    got_codes, got_scales, got_divisor = decode_nvfp4_words(payload, (128, 64))
    assert torch.equal(got_codes, codes)
    assert torch.equal(got_scales, scales)
    assert bytes(got_divisor.numpy().tobytes()) == divisor_word
