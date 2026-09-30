#!/usr/bin/env bash
# Setup for NInfer 512K on Linux: prerequisite checks and model download with SHA-256
# verification. Needs no root and changes nothing else on the system: missing system packages
# are reported with the command to install them, and services and firewall rules are left to
# you (see README.md). Safe to re-run. Settings come from ninfer.conf.
#
#   ./install.sh              check prerequisites, download and verify the model
#   ./install.sh --skip-model check prerequisites only (e.g. when copying the model yourself)
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ninfer.conf
. "$ROOT/ninfer.conf"
rooted() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$ROOT/$1" ;; esac; }
EXE_PATH="$(rooted "$EXE")"
MODEL_PATH="$(rooted "$MODEL")"
SKIP_MODEL=0
[ "${1:-}" = "--skip-model" ] && SKIP_MODEL=1

# Pinned artifact: cometkim/Qwen3.8-27B-nvfp4full-NInfer (v3 container, Apache-2.0), the same
# pin as upstream's download_model.py.
MODEL_URL="https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer/resolve/main/qwen3_8_27b_nvfp4full.ninfer"
MODEL_SIZE=19407229188
MODEL_SHA256="ac98cd392c84a04b2a21c2f5c3988dece88d20a697ba1de663fb32d5998b8ee9"
# The engine is built with CUDA 13.3; the driver must support at least that CUDA version.
MIN_DRIVER_CUDA="13.3"

step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   ok: %s\n' "$1"; }
warn() { printf '   WARN: %s\n' "$1"; }
fail() { printf '   ERROR: %s\n' "$1" >&2; exit 1; }

step "Prerequisites"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: install the NVIDIA driver first."
gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)"
case "$gpu" in
    *"RTX 5090"*) ok "$gpu" ;;
    *) fail "found '$gpu'. This build is compiled only for the RTX 5090 (sm_120a) and will not run on other GPUs." ;;
esac
driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1)"
if [ -n "$driver_cuda" ] && [ "$(printf '%s\n%s\n' "$MIN_DRIVER_CUDA" "$driver_cuda" | sort -V | head -n1)" = "$MIN_DRIVER_CUDA" ]; then
    ok "driver supports CUDA $driver_cuda"
else
    warn "driver reports CUDA ${driver_cuda:-unknown}; this build needs a driver supporting CUDA $MIN_DRIVER_CUDA or newer"
fi

[ -x "$EXE_PATH" ] || fail "engine not found or not executable at $EXE_PATH (set EXE in ninfer.conf)"
missing="$(LD_LIBRARY_PATH="$(dirname "$EXE_PATH")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$EXE_PATH" | awk '/not found/ { print $1 }')"
if [ -n "$missing" ]; then
    printf '   ERROR: shared libraries missing: %s\n' "$(echo $missing)" >&2
    printf '   On Ubuntu 24.04 they come from:\n     sudo apt install libavcodec60 libavformat60 libavutil58 libswscale7 libswresample4 libcurl4t64\n' >&2
    exit 1
fi
ok "engine $EXE_PATH (all shared libraries found)"

ram_gib="$(awk '/MemTotal/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo)"
pin_gib="$(awk -v k="$HOST_KV_MIB" -v s="$HOST_STATE_SLOTS" 'BEGIN { printf "%.1f", (k + 150 * s) / 1024 }')"
if awk -v r="$ram_gib" -v p="$pin_gib" 'BEGIN { exit !(p > r / 2) }'; then
    warn "ninfer.conf pins ~$pin_gib GiB of $ram_gib GiB RAM; consider lowering HOST_KV_MIB"
else
    ok "$ram_gib GiB RAM (~$pin_gib GiB will be pinned)"
fi

step "Model"
if [ "$SKIP_MODEL" = 1 ]; then
    echo "   skipped (--skip-model)"
else
    if [ -f "$MODEL_PATH" ] && [ "$(stat -c %s "$MODEL_PATH")" = "$MODEL_SIZE" ]; then
        ok "already present ($MODEL_PATH)"
    else
        mkdir -p "$(dirname "$MODEL_PATH")"
        free_gb="$(df -Pk "$(dirname "$MODEL_PATH")" | awk 'NR == 2 { printf "%d", $4 / 1000000 }')"
        [ "$free_gb" -ge 20 ] || fail "need ~19.4 GB free for the model, have ${free_gb} GB"
        echo "   downloading 19.4 GB from Hugging Face (resumable; re-run to continue)..."
        curl -L --fail -C - --retry 5 -o "$MODEL_PATH.part" "$MODEL_URL" \
            || fail "download failed; re-run to resume"
        mv -f "$MODEL_PATH.part" "$MODEL_PATH"
    fi
    [ "$(stat -c %s "$MODEL_PATH")" = "$MODEL_SIZE" ] || fail "model size mismatch: expected $MODEL_SIZE bytes"
    echo "   verifying SHA-256 (takes a minute)..."
    actual="$(sha256sum "$MODEL_PATH" | awk '{ print $1 }')"
    [ "$actual" = "$MODEL_SHA256" ] \
        || fail "SHA-256 mismatch ($actual). Delete $MODEL_PATH and re-run; if it persists the publisher replaced the file."
    ok "SHA-256 verified"
fi

step "Done"
echo "   start the server: $ROOT/start-ninfer.sh"
