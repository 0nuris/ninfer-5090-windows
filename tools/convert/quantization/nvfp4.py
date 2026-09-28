"""NVFP4_MAXABS_DIVISOR_RNE_V1: the local NVFP4 encoder.

Ported from the fork that produced the all-NVFP4 artifacts, so a checkpoint whose text
projections are FP8 can be re-encoded rather than imported. See
[artifact conventions](../../../docs/maintainer/artifact-conventions.md), section 1.

Two scales, per Transformer Engine's NVFP4 recipe and confirmed against a ModelOpt
checkpoint whose scaled codes saturate the format maximum exactly:

    s_global = global_amax / (448 * 6)      # 448 = E4M3FN max, 6 = E2M1 max
    s_block  = (block_amax / 6) / s_global  # stored in E4M3FN, so its max is 448

The word the artifact stores as its weight divisor is therefore ``1 / s_global``, and
the codes are E2M1 with round-to-nearest-even, packed low nibble first.

The divisor belongs to the complete packed parent, not to each logical slice or
streaming chunk. The *activation* divisor is not measured here: the recipe supplies it
through the A4 auxiliary, because the source checkpoint already carries the activation
amax its own producer calibrated with.
"""

from __future__ import annotations

import struct

import torch

from tools.convert.methods import AuxiliaryValue

FULL_RANGE = 2688.0  # 6 * 448


def e2m1_rne_codes(values: torch.Tensor) -> torch.Tensor:
    """E2M1 codes for magnitudes in [0, 6], ties to even.

    The boundaries are the midpoints between representable magnitudes
    (0, .5, 1, 1.5, 2, 3, 4, 6); ``tie_up`` marks the ones whose upper neighbour is
    even, so an exact tie rounds to the even code.
    """
    magnitude = values.abs()
    codes = torch.zeros_like(values, dtype=torch.uint8)
    for boundary, tie_up in zip((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.),
                                (False, True, False, True, False, True, False)):
        codes += ((magnitude > boundary) | ((magnitude == boundary) & tie_up)).to(torch.uint8)
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


# The E2M1 grid, read from the format rather than derived from ``e2m1_rne_codes``, so the search
# below scores a candidate against the format's own values and not against the encoder's opinion of
# them. Index order matches the low nibble: bit 3 is the sign.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_VALUES = torch.tensor((*E2M1_MAGNITUDES, *(-magnitude for magnitude in E2M1_MAGNITUDES)))


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """The value each 4-bit E2M1 code represents."""
    return E2M1_VALUES.to(codes.device)[codes.long()]


# Candidate shrink factors for the searched block scale, from the format maximum downwards. 1.0 is
# the max-abs scale, so the search can only match or beat what the recipe already does.
SCALE_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)


def search_block_ratios(blocks: torch.Tensor) -> torch.Tensor:
    """Per-block shrink factor minimising squared reconstruction error.

    The max-abs scale maps a block's largest magnitude onto 6.0, the format maximum, which is the
    right choice only if that element deserves the whole range. A distribution with one outlier
    spends the range on it and gives the remaining fifteen values coarse steps, so shrinking the
    scale trades a little clipping for finer resolution on the bulk. The factor is chosen per block,
    because whether a block is outlier-dominated is a property of that block.

    The candidate is cast to E4M3FN *before* it is scored, so the search compares scales the format
    can actually represent rather than ones it would round away.
    """
    amax = blocks.abs().amax(dim=2, keepdim=True)
    best_error = torch.full(amax.shape, float("inf"), device=blocks.device)
    best_ratio = torch.ones_like(amax)
    for ratio in SCALE_RATIOS:
        scale = (amax / 6.0 * ratio).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        reconstructed = e2m1_values(e2m1_rne_codes(blocks / safe)) * safe
        error = (reconstructed - blocks).pow(2).sum(dim=2, keepdim=True)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_ratio = torch.where(better, torch.full_like(best_ratio, ratio), best_ratio)
    return best_ratio


def encode_block(values: torch.Tensor, divisor: float,
                 ratios: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one whole-tile block: E2M1 codes and E4M3FN scales for ``[rows, columns]``.

    ``ratios`` is an optional per-block shrink factor from ``search_block_ratios``; without it the
    scale is the max-abs one, which is what every artifact built to date used.
    """
    rows, columns = values.shape
    blocks = (values.float() * divisor).reshape(rows, columns // 16, 16)
    amax = blocks.abs().amax(dim=2, keepdim=True)
    factor = torch.ones_like(amax) if ratios is None else ratios
    # The clamp is what makes the block scale a valid E4M3FN word: 448 is that format's maximum.
    scales = (amax / 6.0 * factor).clamp(max=448.).to(torch.float8_e4m3fn)
    decoded = scales.float()
    safe = torch.where(decoded > 0, decoded, torch.ones_like(decoded))
    normalized = torch.where(decoded > 0, blocks / safe, torch.zeros_like(blocks))
    codes = e2m1_rne_codes(normalized).reshape(rows, columns // 2, 2)
    return codes[..., 0] | (codes[..., 1] << 4), scales.reshape(rows, columns // 16).view(torch.uint8)


def _encode_nvfp4(request, name: str, search: bool):
    """Shared body of the two local NVFP4 encoders.

    The parent must be complete: NVFP4's block-scale layout is a 128-row by 16-column
    tile, so a partial parent has no defined divisor. The divisor is a two-pass
    quantity, so every chunk is read once for the global amax before anything is written.
    """
    if request.target.format != "nvfp4" or len(request.target.shape) != 2:
        raise ValueError(f"{name} requires an NVFP4 matrix")
    if request.parameters:
        raise ValueError(f"{name} accepts no numerical parameters")
    n, k = request.target.shape
    if n % 128 or k % 16 or request.source_offsets[-1] != n * k:
        raise ValueError("NVFP4 parent requires complete 128-row and 16-column tiles")
    if request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    chunk = max(128, request.rows_per_chunk // 128 * 128)
    auxiliaries = {}
    for item in request.inputs:
        for use in item.uses:
            key = (*use, "activation_input_divisor")
            if request.policies[use] == "AllowA4":
                if key not in request.auxiliary_overrides:
                    raise ValueError(f"{use}: calibrated activation divisor required")
                auxiliaries[key] = request.auxiliary_overrides[key]

    def produce(output):
        maximum = 0.
        for begin in range(0, n, chunk):
            values = request.values(begin * k, min(n, begin + chunk) * k).float()
            if not torch.isfinite(values).all():
                raise ValueError("NVFP4 source contains NaN or infinity")
            maximum = max(maximum, values.abs().max().item())
        raw = struct.pack("<f", FULL_RANGE / maximum if maximum else 1.)
        AuxiliaryValue.activation_divisor(raw)  # same positive finite FP32 contract
        divisor = struct.unpack("<f", raw)[0]
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k).to(request.device)
            ratios = None
            if search:
                blocks = (values.float() * divisor).reshape(end - begin, k // 16, 16)
                ratios = search_block_ratios(blocks)
            codes, scales = encode_block(values, divisor, ratios)
            output.write_codes(begin, codes.cpu(), scales.cpu(), raw)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def nvfp4_maxabs(request):
    """NVFP4_MAXABS_DIVISOR_RNE_V1: scale each block so its largest value reaches 6.0."""
    return _encode_nvfp4(request, "nvfp4_maxabs", search=False)


def nvfp4_mse(request):
    """NVFP4_MSE_DIVISOR_RNE_V1: scale each block by the factor that minimises its own error.

    Same divisor contract, same code selection, and the max-abs scale is in the candidate set, so
    this can only match or beat ``nvfp4_maxabs`` on reconstruction error. What changes is that a
    block dominated by one outlier is allowed to clip that outlier slightly and spend the range on
    the other fifteen values instead. The source checkpoint this port reads for the nvidia recipe
    records ``calibrator=NVFP4MSECalibrator`` on every NVFP4 weight quantizer, so a scale search is
    what produced it and a max-abs scale is a reproduction rather than a match.
    """
    return _encode_nvfp4(request, "nvfp4_mse", search=True)
