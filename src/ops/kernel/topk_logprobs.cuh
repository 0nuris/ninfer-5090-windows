#pragma once

// Implements: include/ninfer/ops/topk_logprobs.h
// Match: wrapper-validated contiguous BF16 [physical_rows,C], with I32 [C,K] and FP32 [C,K] outputs.
// Algorithm assumptions: one independent CTA per column; two 8-bit radix passes over the
// order-preserving BF16 key locate the k-th largest exactly, then a counted collection fills the K
// slots and a fixed-width bitonic sort orders them. No device allocation; all scratch is static
// shared memory sized for K<=256.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTopkLogprobsBlock = 256;
inline constexpr int kTopkLogprobsBins  = 256;
// The sort's width. Padding K up to this keeps it a power of two, so the compare-exchange needs no
// bounds handling and the shared scratch is a compile-time size.
inline constexpr int kTopkLogprobsWidth = 256;

// Order-preserving map from a BF16 bit pattern to an unsigned key, so comparing keys compares values.
// For IEEE-754 the standard trick complements the word when the sign bit is set and otherwise sets
// it, which turns sign-magnitude into a plain unsigned ordering with -0 below +0. BF16 shares that
// layout in its top 16 bits, so it works unchanged -- and 16 bits is why this needs two 8-bit radix
// passes where an FP32 key would need four.
//
// The key is order-EXACT over finite values, not merely order-preserving. That is load-bearing rather
// than incidental: the key selects the k-th largest, so any disagreement with the decoded value order
// would return the k-th largest by key, which is not the k-th largest by value. Positives keep their
// bit order (setting a bit is monotone), negatives have it reversed (~ is order-reversing and more
// negative means larger magnitude means larger bits), and every negative key is below 0x8000 while
// every positive key is at or above it.
__device__ __forceinline__ std::uint16_t topk_bf16_key(std::uint16_t bits) {
    return static_cast<std::uint16_t>((bits & 0x8000u) != 0u ? ~bits : (bits | 0x8000u));
}

__device__ __forceinline__ float topk_bf16_value(std::uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

// Block-wide maximum of a float, one independent CTA. Fused into the first pass so the logsumexp
// maximum and the first radix histogram cost one read of the column rather than two.
template <int BlockSize>
__device__ __forceinline__ float topk_block_max(float value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int Warps = BlockSize / kWarpSize;
    __shared__ float warp_maxima[Warps];
    __shared__ float result;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < Warps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

// Finds the histogram bin holding the `rank`-th largest entry, 1-based, and reports how many entries
// lie strictly above that bin. `rank` is reduced in place to the number still wanted from within the
// bin, which is what makes the caller's later arithmetic fall out of this. Runs on thread 0 and
// publishes through the caller's shared state.
//
// Serial on purpose. There are 256 bins and this runs twice per column, so it is 512 sequential
// iterations against a column of up to 248,320 elements read in parallel; a parallel scan would add
// barriers to save work the column read already dominates, and this is an offline diagnostic rather
// than a decode-path Op.
__device__ __forceinline__ int topk_threshold_bin(const int* histogram, int& rank, int& above) {
    if (threadIdx.x != 0) { return 0; }
    int cumulative = 0;
    for (int bin = kTopkLogprobsBins - 1; bin >= 0; --bin) {
        const int count = histogram[bin];
        if (cumulative + count >= rank) {
            above  = cumulative;
            rank -= cumulative;
            return bin;
        }
        cumulative += count;
    }
    // Unreachable while rank <= valid_rows, which the wrapper enforces. Reported as a negative bin
    // rather than silently returning 0, because a silent 0 would read as "the smallest bin" and
    // quietly return the k *smallest* entries.
    above = cumulative;
    rank  = 0;
    return -1;
}

// Descending by value, ascending by row on equal values. Two entries can compare equal in both
// components only if they are the same entry, so this is a total order and the sort is deterministic.
__device__ __forceinline__ bool topk_precedes(float a, int ai, float b, int bi, int mutation) {
    if (a != b) { return a > b; }
#if defined(NINFER_OP_TEST_MUTATIONS)
    // Mutation seam. The tie-break is the single load-bearing decision for the "ascending row index"
    // half of the contract and the one a reading of this file cannot check, so it is the seam a
    // perturbation test flips. Compiled only where BUILD_TESTING is on, which is the test tree alone;
    // a shipping binary expands to the original two lines and this costs nothing.
    if ((mutation & 1) != 0) { return ai > bi; }
#endif
    return ai < bi;
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void topk_logprobs_kernel(const __nv_bfloat16* logits, std::int32_t* indices, float* logprobs,
                              std::int32_t valid_rows, std::int32_t physical_rows, std::int32_t k,
                              int mutation) {
    static_assert(BlockSize == kTopkLogprobsBlock, "one bin per thread is assumed");
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base   = static_cast<std::int64_t>(column) * physical_rows;
    const int tid              = static_cast<int>(threadIdx.x);

    __shared__ int s_hist_a[kTopkLogprobsBins];
    __shared__ int s_hist_b[kTopkLogprobsBins];
    __shared__ float s_value[kTopkLogprobsWidth];
    __shared__ int s_index[kTopkLogprobsWidth];
    __shared__ int s_slot;
    __shared__ int s_bin;
    __shared__ int s_remaining;
    __shared__ int s_low_bin;
    __shared__ int s_needed;
    __shared__ int s_equal_base[kTopkLogprobsBlock];
    __shared__ float s_sum;
    __shared__ float s_scratch_f[BlockSize / kWarpSize];

    const __nv_bfloat16* column_logits = logits + base;
    // Rows per thread for the tie-breaking pass, which hands each thread a CONTIGUOUS block. The two
    // histogram passes stay strided because their order does not matter and coalescing does; this one
    // needs each thread's entries to be consecutive in row order, which is what makes a per-thread
    // prefix sum the true global rank. Under strided assignment thread t owns rows t, t+BlockSize,
    // t+2*BlockSize..., so consecutive threads interleave within every round and a per-thread prefix
    // sum ranks them as (t=0,r0), (t=1,r0), (t=0,r1) when the true row order is 0, BlockSize,
    // 2*BlockSize. Contiguity is the fix; the cost is one uncoalesced pass, which is why the two
    // passes that can stay coalesced do.
    const int span = (valid_rows + BlockSize - 1) / BlockSize;

    // Pass 1: column maximum and the high-byte histogram, in one read of the column.
    if (tid < kTopkLogprobsBins) { s_hist_a[tid] = 0; }
    __syncthreads();
    float local_max = -CUDART_INF_F;
    for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
        const std::uint16_t bits =
            *reinterpret_cast<const std::uint16_t*>(column_logits + row);
        local_max = fmaxf(local_max, topk_bf16_value(bits));
        atomicAdd(&s_hist_a[topk_bf16_key(bits) >> 8], 1);
    }
    const float maximum = topk_block_max<BlockSize>(local_max);

    // Pass 2: log-partition numerator and the low-byte histogram, restricted to the high byte the
    // k-th largest lives in. Also one read of the column.
    //
    // topk_threshold_bin runs on thread 0 only, so its results are published through shared state
    // rather than returned per-thread.
    if (tid == 0) {
        int rank  = k;
        int above = 0;
        s_bin     = topk_threshold_bin(s_hist_a, rank, above);
        s_remaining = rank;
    }
    if (tid < kTopkLogprobsBins) { s_hist_b[tid] = 0; }
    __syncthreads();

    float local_sum = 0.0f;
    if (s_bin >= 0) {
        for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
            const std::uint16_t bits =
                *reinterpret_cast<const std::uint16_t*>(column_logits + row);
            local_sum += expf(topk_bf16_value(bits) - maximum);
            const std::uint16_t key = topk_bf16_key(bits);
            if (static_cast<int>(key >> 8) == s_bin) { atomicAdd(&s_hist_b[key & 0xFFu], 1); }
        }
    }
    // block_reduce_sum's cross-warp step is guarded by `if (warp == 0)`, so its result is the true
    // total only inside warp 0. Publishing it through shared state is what makes the log-partition
    // correct in the threads that write the output, which is every thread with j < K.
    const float sum = block_reduce_sum<BlockSize>(local_sum, s_scratch_f);
    if (tid < kWarpSize) { s_sum = sum; }

    if (tid == 0) {
        int rank  = s_remaining;
        int above = 0;
        s_low_bin = topk_threshold_bin(s_hist_b, rank, above);
        s_needed  = rank;
    }
    if (tid == 0) { s_slot = 0; }
    __syncthreads();

    const std::uint16_t threshold =
        s_bin >= 0 ? static_cast<std::uint16_t>((static_cast<unsigned>(s_bin) << 8) |
                                                static_cast<unsigned>(s_low_bin))
                   : static_cast<std::uint16_t>(0);
    // Exactly this many entries are strictly above the threshold, so they occupy the leading slots.
    // The `needed` entries equal to the threshold follow them, in ascending row order.
    const int n_above = k - s_needed;

    // Pass 3: collect everything strictly above the threshold. Their order is irrelevant -- all of
    // them are wanted and the sort below fixes their sequence -- so this stays strided and coalesced.
    for (std::int32_t row = tid; row < valid_rows; row += BlockSize) {
        const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(column_logits + row);
        if (topk_bf16_key(bits) > threshold) {
            const int slot = atomicAdd(&s_slot, 1);
            // The guard is what stops a miscounted threshold from writing past the K slots. With a
            // 151,936-wide BF16 vocabulary equal values are common and a whole column can be one
            // value, so the threshold bin can hold far more than K entries; without the guard the
            // failure is memory corruption rather than a wrong answer.
            if (slot < k) {
                s_value[slot] = topk_bf16_value(bits);
                s_index[slot] = row;
            }
        }
    }

    // Pass 4: the `needed` entries equal to the threshold, in ascending row order. They all carry the
    // same value, so the only thing distinguishing them is the row, and which ones are picked decides
    // the output -- a shared atomic would collect an arbitrary subset and make the result depend on
    // scheduling, which for a measurement instrument is not a rounding difference but a different
    // number. Contiguous per-thread blocks plus a prefix sum make "first `needed` in row order" exact.
    {
        const std::int32_t block_begin = tid * span;
        const std::int32_t block_end   = (tid + 1) * span < valid_rows ? (tid + 1) * span : valid_rows;
        int local_equal                 = 0;
        for (std::int32_t row = block_begin; row < block_end; ++row) {
            if (topk_bf16_key(*reinterpret_cast<const std::uint16_t*>(column_logits + row)) ==
                threshold) {
                ++local_equal;
            }
        }
        s_equal_base[tid] = local_equal;
    }
    __syncthreads();
    if (tid == 0) {
        int running = 0;
        for (int t = 0; t < BlockSize; ++t) {
            const int count = s_equal_base[t];
            s_equal_base[t]  = running;
            running += count;
        }
    }
    __syncthreads();

    if (s_bin >= 0) {
        const std::int32_t block_begin = tid * span;
        const std::int32_t block_end =
            (tid + 1) * span < valid_rows ? (tid + 1) * span : valid_rows;
        int seen = 0;
        for (std::int32_t row = block_begin; row < block_end; ++row) {
            const std::uint16_t bits =
                *reinterpret_cast<const std::uint16_t*>(column_logits + row);
            if (topk_bf16_key(bits) == threshold) {
                // s_equal_base[tid] + seen is this entry's global row-order rank among the equals,
                // because thread t's block entirely precedes every later thread's block.
                const int slot = n_above + (s_equal_base[tid] + seen);
                ++seen;
                if (slot < k) {
                    s_value[slot] = topk_bf16_value(bits);
                    s_index[slot] = row;
                }
            }
        }
    }

    // Pad the K..255 tail the sort reads. No slot in [0,K) is left unwritten: the count of entries
    // strictly above the threshold is exactly K - needed, and the equals supply exactly `needed` of
    // them, because the threshold by construction has at least `needed` entries equal to it.
    //
    // The stride starts at k + tid, not k. Starting at k would pad one slot and leave the rest of the
    // tail as uninitialized shared memory, which the sort below then ranks as if it were data.
    for (int i = k + tid; i < kTopkLogprobsWidth; i += BlockSize) {
        s_value[i] = -CUDART_INF_F;
        s_index[i] = INT_MAX;
    }
    __syncthreads();

    // Bitonic sort over the fixed 256-wide padded array.
    //
    // The direction is inverted relative to the textbook ascending form on purpose: this sorts
    // DESCENDING, which is what putting `(i & size) == 0` on the descending branch achieves. That
    // matters at the last stage, where `i & 256` is 0 for every valid i, so the branch taken there
    // alone decides the final order. Written the other way round the sort is correct-looking and
    // silently returns the K *smallest* entries first, with the -INF padding leading.
    for (int size = 2; size <= kTopkLogprobsWidth; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            for (int i = tid; i < kTopkLogprobsWidth; i += BlockSize) {
                const int partner = i ^ stride;
                if (partner > i) {
#if defined(NINFER_OP_TEST_MUTATIONS)
                    const bool ascending = (mutation & 2) != 0;
                    const bool take = ((i & size) == 0) != ascending
                                          ? topk_precedes(s_value[partner], s_index[partner], s_value[i],
                                                          s_index[i], mutation)
                                          : topk_precedes(s_value[i], s_index[i], s_value[partner],
                                                          s_index[partner], mutation);
#else
                    const bool take = ((i & size) == 0)
                                          ? topk_precedes(s_value[partner], s_index[partner], s_value[i],
                                                          s_index[i], mutation)
                                          : topk_precedes(s_value[i], s_index[i], s_value[partner],
                                                          s_index[partner], mutation);
#endif
                    if (take) {
                        const float tv  = s_value[i];
                        const int ti    = s_index[i];
                        s_value[i]       = s_value[partner];
                        s_index[i]       = s_index[partner];
                        s_value[partner] = tv;
                        s_index[partner] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }

    const float log_partition = maximum + logf(s_sum);
    for (int j = tid; j < k; j += BlockSize) {
        indices[static_cast<std::int64_t>(column) * k + j]  = s_index[j];
        logprobs[static_cast<std::int64_t>(column) * k + j] = s_value[j] - log_partition;
    }
}

} // namespace ninfer::ops
