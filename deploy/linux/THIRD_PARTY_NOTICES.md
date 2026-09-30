# Third-party notices for the NInfer 512K Linux tarball

| Component | Where | License | Source |
|---|---|---|---|
| NInfer engine | `bin/ninfer*` | Apache-2.0 | This repository; upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) via [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) |
| YaRN context extension | compiled into `bin/ninfer*` | Apache-2.0 | [kido5217/ninfer-yarn PR #14](https://github.com/kido5217/ninfer-yarn/pull/14) |
| NVIDIA CUDA runtime | `bin/libcudart.so.*` | NVIDIA CUDA Toolkit EULA (redistributable component); `licenses/NVIDIA-CUDA-EULA.txt` | NVIDIA CUDA Toolkit 13.3 |
| Libraries statically linked into the engine | `bin/ninfer*` | see `NOTICE` (cpp-httplib MIT, nlohmann/json MIT, spdlog MIT, utf8proc MIT, llama-jinja MIT) | `third_party/` in this repository |

**Not bundled:** FFmpeg (`libavcodec`, `libavformat`, `libavutil`, `libswscale`) and libcurl are
dynamically linked from the distribution's own packages (Ubuntu 24.04: `libavcodec60`,
`libavformat60`, `libavutil58`, `libswscale7`, `libcurl4t64`), under their own licenses.

The Windows-only pinned-memory fix from [alphastorm/ninfer](https://github.com/alphastorm/ninfer)
is compiled only into the Windows build.

The model is **not** included. `install.sh` downloads
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer)
(Apache-2.0; derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) and
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) and verifies its
SHA-256 against a pinned value.
