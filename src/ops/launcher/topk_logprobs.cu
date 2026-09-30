// Implements: include/ninfer/ops/topk_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/topk_logprobs.h"

#include "core/device.h"
#include "ops/kernel/topk_logprobs.cuh"

namespace ninfer::ops::detail {

void topk_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k,
                          Tensor& indices, Tensor& logprobs, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    topk_logprobs_kernel<kTopkLogprobsBlock><<<columns, kTopkLogprobsBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(indices.data),
        static_cast<float*>(logprobs.data), valid_rows, logits.ne[0], k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
