# Third-party notices for the NInfer 512K release zip

The release zip redistributes the following. Full terms are in `licenses\` and `NOTICE`.

| Component | Where | License | Source |
|---|---|---|---|
| NInfer engine | `bin\ninfer*.exe` | Apache-2.0 | This repository; upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) via [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) |
| YaRN context extension | compiled into `bin\ninfer*.exe` | Apache-2.0 | [kido5217/ninfer-yarn PR #14](https://github.com/kido5217/ninfer-yarn/pull/14) |
| Pinned-memory fix | compiled into `bin\ninfer*.exe` | Apache-2.0 | [alphastorm/ninfer](https://github.com/alphastorm/ninfer) ([omp-ninfer#48](https://github.com/alphastorm/omp-ninfer/issues/48)) |
| FFmpeg shared libraries | `bin\av*.dll`, `bin\sw*.dll` | LGPL-3.0-or-later (built with `--enable-version3`); `licenses\FFmpeg-LICENSE.txt` | FFmpeg commit `d85cdd2597` (N-126965), binaries from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) autobuild 2026-09-29, `win64-lgpl-shared` variant. Source: https://github.com/FFmpeg/FFmpeg/tree/d85cdd2597 |
| Libraries statically linked into the engine | `bin\ninfer*.exe` | see `NOTICE` (cpp-httplib MIT, nlohmann/json MIT, spdlog MIT, utf8proc MIT, llama-jinja MIT, libcurl curl, zlib zlib) | `third_party\` in this repository |

The FFmpeg DLLs are unmodified and dynamically linked; you may replace them with any
ABI-compatible FFmpeg build (the engine needs `avcodec-63`, `avformat-63`, `avutil-61`,
`swscale-10` and their dependency `swresample-7`).

The model is **not** included. `install.ps1` downloads
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer)
(Apache-2.0; derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) and
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) and verifies its
SHA-256 against a pinned value.
