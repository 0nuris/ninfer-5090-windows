#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/** The largest K this Op accepts, fixed so the per-CTA scratch is a compile-time size. */
inline constexpr std::int32_t kTopkLogprobsMaxK = 256;

/**
 * Op: Top-k log-probabilities per column
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c], and let
 *
 *     Z[c] = log(sum_{r=0..valid_rows-1} exp(l[r,c]))
 *
 *   be the column's log-partition. The k returned entries are exactly the k largest l[.,c] under the
 *   order (value descending, then row index ascending on equal values), written in descending value
 *   order. For j in [0,k):
 *
 *     indices[c,j]  = the row r of the j-th largest entry
 *     logprobs[c,j] = l[r,c] - Z[c]
 *
 *   so the pair is an ordered sample of the column's softmax. Ties on value are broken by ascending
 *   row index, so the result is a deterministic function of the input and does not depend on
 *   reduction order, thread scheduling, or column ordering.
 *
 * Logical shapes:
 *   logits is [physical_rows,C]; indices and logprobs are each [C,K], row-major with C the slower
 *   axis. C>0, 1<=K<=256, and 1<=valid_rows<=physical_rows. Physical rows [valid_rows,physical_rows)
 *   do not participate in Z[c] and can never appear in indices.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, indices is contiguous I32, logprobs is contiguous FP32.
 *   Storage has its dtype's natural alignment. Finite is a real requirement and not a formality: the
 *   selection compares an order-preserving key of the stored bit pattern, which orders NaN
 *   incorrectly.
 *
 * Numeric:
 *   logprobs is the FP32 approximation of the formula above. Reduction association and private
 *   accumulator precision are implementation choices; the independent oracle evaluates the full
 *   formula in FP64 from the represented BF16 inputs. Selection is exact rather than approximate:
 *   the returned set is the true k largest, because the ranking compares the stored BF16 values
 *   themselves rather than computed exponentials.
 *
 * Effects:
 *   Writes every element of indices and logprobs and preserves logits. Neither output may overlap
 *   logits or the other output.
 *
 * Workspace:
 *   None. The Op allocates no device memory; its per-CTA scratch is static shared memory sized for
 *   K<=256.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state. One independent CTA per column, so columns
 *   make no ordering assumption about each other.
 */
void topk_logprobs(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& indices,
                    Tensor& logprobs, cudaStream_t stream);

} // namespace ninfer::ops
