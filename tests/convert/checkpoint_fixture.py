"""A minimal real Safetensors checkpoint, so the recipe seam runs without a 51 GiB download.

The seam is `RECIPES[name](model, recipe, sources)` -> `recipe.prepare(device="cpu")`, whose
`.weights` carry each site's format and method. Codegraph reports RECIPES has one caller and no tests
within three hops, and the GDN audit on 2026-09-29 had to read official_recipes.py by hand to learn
which sites get which format.

Stubbing the source was tried first and rejected: the recipes reach through it in four different ways
-- `sources["quantized"].config`, `store.has(...)`, `store.read_flat(...)`, and
`model.source(name, store, format)` -- so the stub grew a method per call and each addition risked
answering something the real source would not. A real file answers them the same way the converter's
does, and the test then depends on the recipe rather than on the stub.

The names come from the recipes, not from this file's imagination (see names_needed.py):

  <prefix><module>.weight          BF16 2-D, the matrix
  <prefix><module>.input_scale    F32 scalar, required by _activation_divisor
  <prefix><module>.weight_scale_2 F32 scalar, whose *presence* marks the site as already NVFP4

and `<prefix>` is "model." or "model.language_model." depending on whether the config carries
`text_config` -- the recipes test for exactly that, so the fixture does too.

The fixture marks the MLP modules as already NVFP4 and leaves attention and GDN as FP8, which is the
real shape of NVIDIA's checkpoint: its model card and the recipe's own docstring both record MLP on
all 64 layers as NVFP4 with 193 `weight_scale_2` sites, and attention and linear-attention as FP8.
That split is what makes the recipe's import-vs-encode branch observable.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

# The module paths the recipes map logical sites onto, plus the MLP pair. Layer 0 only: the recipes
# iterate parameters, so one layer exercises every branch and keeps the fixture small.
LAYER = 0
MODULES: dict[str, str] = {
    "self_attn.q_proj": "5120x5120",
    "self_attn.k_proj": "5120x5120",
    "self_attn.v_proj": "5120x5120",
    "self_attn.o_proj": "5120x5120",
    "linear_attn.in_proj_qkv": "15360x5120",
    "linear_attn.in_proj_z": "5120x5120",
    "linear_attn.out_proj": "5120x5120",
    "mlp.gate_proj": "17408x5120",
    "mlp.up_proj": "17408x5120",
    "mlp.down_proj": "5120x17408",
}

# The sites NVIDIA's checkpoint already stores as NVFP4. Everything else is FP8 in that checkpoint,
# which is what the recipe's docstring records and what makes the two branches distinguishable.
ALREADY_NVFP4 = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

_DTYPE = {"BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4)}

# torch's dtype spelling to the safetensors file spelling, which is what SafetensorsSource's _DTYPE
# is keyed by. Getting this wrong raises KeyError: 'BFLOAT16' rather than anything about the fixture.
_SPELLING = {"BFLOAT16": "BF16", "FLOAT32": "F32"}


def _dims(spec: str) -> tuple[int, int]:
    rows, columns = spec.split("x")
    return int(rows), int(columns)


def write_checkpoint(
    root: Path,
    *,
    nvfp4: bool,
    composite: bool,
    modules: dict[str, str] | None = None,
) -> Path:
    """Write one minimal checkpoint directory and return its path.

    `nvfp4` decides whether the MLP modules carry `weight_scale_2`, which is the whole difference
    between the two branches of _activation_divisor. `composite` adds `text_config`, which is what
    the recipes test to choose the "model.language_model." prefix over "model.".
    """
    root.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vocab_size": 4096,
        "num_hidden_layers": 1,
        "hidden_size": 5120,
    }
    if composite:
        config["text_config"] = {"hidden_size": 5120, "num_hidden_layers": 1}
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    prefix = "model.language_model." if composite else "model."
    chosen = modules if modules is not None else MODULES
    tensors: dict[str, torch.Tensor] = {}
    for module, spec in chosen.items():
        rows, columns = _dims(spec)
        base = f"{prefix}layers.{LAYER}.{module}"
        # Distinct, non-zero values so a test can tell one site's matrix from another's.
        fill = float(rows % 7 + 1) / 8.0
        tensors[f"{base}.weight"] = torch.full((rows, columns), fill, dtype=torch.bfloat16)
        # An FP8 site records amax/448 and an NVFP4 one amax/(6*448); 1/448 is the FP8 form and
        # makes the derived divisor 6/((1/448)) = 2688, the full range. Any positive value works;
        # what matters is that it is a real F32 scalar the recipe can read.
        tensors[f"{base}.input_scale"] = torch.tensor([1.0 / 448.0], dtype=torch.float32)
        if nvfp4 and module in ALREADY_NVFP4:
            tensors[f"{base}.weight_scale_2"] = torch.tensor([1.0 / 2688.0], dtype=torch.float32)

    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        # The safetensors spelling is "BF16", not torch's "BFLOAT16"; the loader's _DTYPE map is
        # keyed by the file's spelling, so the same mapping has to be applied here.
        tag = str(tensor.dtype).upper().replace("TORCH.", "")
        tag = _SPELLING.get(tag, tag)
        flat = tensor.contiguous().reshape(-1)
        # A byte view rather than .numpy(): numpy has no bfloat16, and .numpy() on a BF16 tensor
        # raises "Got unsupported ScalarType BFloat16" rather than doing something lossy.
        blob = flat.view(torch.uint8).numpy().tobytes() if flat.dtype == torch.bfloat16 else (
            flat.numpy().tobytes()
        )
        header[name] = {
            "dtype": tag,
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(blob)],
        }
        payload.extend(blob)
    encoded = json.dumps(header).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(payload)
    )
    return root
