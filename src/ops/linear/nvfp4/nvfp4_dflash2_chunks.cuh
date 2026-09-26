#pragma once

// Port-only NVFP4 A16 launch selection. Upstream's `nvfp4_launch.cuh` provides the A16 launchers
// and the shape-file schedule selection, but not these two: an exact-launcher table indexed by
// token count, and a chunked walk for extents wider than a single launch's token family. Both
// predate the upstream unification and are needed by the DFlash2 drafter shapes, which upstream
// does not register.
//
// The launchers themselves are upstream's `nvfp4_linear_a16_simt` / `nvfp4_linear_a16_gemv`. The
// port previously carried its own copies of the A16 GEMV and SIMT kernels to reach them; those
// were removed in favour of upstream's `nvfp4_a16_gemv.cuh` and `nvfp4_a16_simt.cuh`, which are
// line-for-line the same kernels under the `Nvfp4A16*` names and are the maintained ones.

#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {

// A table of exact-token launchers, one per token count in [First, Last]. Each entry is
// specialized on its own token count so the kernel's live-column count is a compile-time value.
template <class Geometry, int First, template <int> class Schedule, std::size_t... Indices>
constexpr auto nvfp4_exact_launchers(std::index_sequence<Indices...>) {
    return std::array<Nvfp4Launch, sizeof...(Indices)>{
        &nvfp4_linear_a16_simt<Geometry, First + static_cast<int>(Indices),
                                Schedule<First + static_cast<int>(Indices)>, true>...};
}

// Each shape supplies its measured exact interval and schedule.
template <class Geometry, int First, int Last, template <int> class Schedule>
Nvfp4Launch select_nvfp4_exact(std::int32_t tokens) {
    static constexpr auto launchers = nvfp4_exact_launchers<Geometry, First, Schedule>(
        std::make_index_sequence<Last - First + 1>{});
    return launchers.at(static_cast<std::size_t>(tokens - First));
}

// Walk a wide token extent in fixed-size chunks, dispatching each chunk through the shape's own
// token-count selection. Used for extents above a shape's exact family.
template <int Chunk, Nvfp4Launch (*Select)(std::int32_t)>
void launch_nvfp4_a16_chunks(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const int count = std::min(Chunk, x.ne[1] - offset);
        auto input      = x.slice(1, offset, count);
        auto output     = out.slice(1, offset, count);
        Select(count)(input, weight, output, stream);
    }
}

} // namespace ninfer::ops::detail
