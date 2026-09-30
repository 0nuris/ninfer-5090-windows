#!/usr/bin/env bash
# Maintainer tool: assemble the Linux release tarball from a source build.
#   deploy/linux/stage-release.sh <build dir> <version>      e.g.  build v1.1.0
# Produces dist/ninfer-512k-<version>-linux-x86_64-rtx5090.tar.gz with bin/ (engine + the CUDA
# runtime it links), the deploy scripts, docs, licenses and SHA256SUMS. The model is never
# packaged; FFmpeg and libcurl come from the distribution's packages.
set -eu

BUILD="$(cd "$1" && pwd)"
VERSION="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NAME="ninfer-512k-$VERSION-linux-x86_64-rtx5090"
STAGE="$REPO/dist/$NAME"

rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/licenses" "$STAGE/models" "$STAGE/logs"

for f in ninfer-serve ninfer-perplexity ninfer; do
    [ -x "$BUILD/apps/$f" ] && cp "$BUILD/apps/$f" "$STAGE/bin/"
done
[ -x "$STAGE/bin/ninfer-serve" ] || { echo "no ninfer-serve under $BUILD/apps" >&2; exit 1; }

# Bundle the CUDA runtime the engine links dynamically (NVIDIA CUDA EULA redistributable);
# the driver's libcuda comes from the target system.
ldd "$STAGE/bin/ninfer-serve" | awk '/libcudart\.so/ { print $3 }' | while read -r lib; do
    [ -n "$lib" ] || continue
    cp -L "$lib" "$STAGE/bin/$(basename "$lib")"
done
cuda_eula="$(ls /usr/local/cuda/EULA.txt 2>/dev/null || true)"
[ -n "$cuda_eula" ] && cp "$cuda_eula" "$STAGE/licenses/NVIDIA-CUDA-EULA.txt"

for f in ninfer.conf start-ninfer.sh ensure-ninfer.sh install.sh README.md THIRD_PARTY_NOTICES.md; do
    cp "$HERE/$f" "$STAGE/"
done
chmod +x "$STAGE"/*.sh "$STAGE"/bin/ninfer*
cp "$REPO/LICENSE" "$STAGE/licenses/LICENSE-ninfer-Apache-2.0.txt"
cp "$REPO/NOTICE" "$STAGE/"

( cd "$STAGE" && find . -type f ! -name SHA256SUMS -printf '%P\n' | sort | xargs sha256sum > SHA256SUMS )

mkdir -p "$REPO/dist"
tar -C "$REPO/dist" -czf "$REPO/dist/$NAME.tar.gz" "$NAME"
( cd "$REPO/dist" && sha256sum "$NAME.tar.gz" )
