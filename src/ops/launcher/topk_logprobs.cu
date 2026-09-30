// Implements: include/ninfer/ops/topk_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/topk_logprobs.h"

#include "core/device.h"
#include "ops/kernel/topk_logprobs.cuh"

#include <cstdlib>

namespace ninfer::ops::detail {
namespace {

// The perturbation seam, read once per launch. It exists only where BUILD_TESTING is on, which is the
// test tree alone -- a shipping binary compiles this to `return 0` and the kernel's mutation branches
// are gone, so nothing here is reachable in a release build.
//
// Why an environment variable rather than a kernel argument threaded from the caller: the mutation
// has to reach inside the sort's comparator, and the only way to get it there without widening the
// public contract is a build-time define plus a test-only input. tools/release/check_test_mutation.py
// is the consumer, and it sets this for exactly one test invocation at a time.
int topk_logprobs_active_mutation() {
#if defined(NINFER_OP_TEST_MUTATIONS)
    const char* const text = std::getenv("NINFER_OP_MUTATION");
    if (text == nullptr || *text == '\0') { return 0; }
    return std::atoi(text);
#else
    return 0;
#endif
}

} // namespace

void topk_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k,
                          Tensor& indices, Tensor& logprobs, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    topk_logprobs_kernel<kTopkLogprobsBlock><<<columns, kTopkLogprobsBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<std::int32_t*>(indices.data),
        static_cast<float*>(logprobs.data), valid_rows, logits.ne[0], k,
        topk_logprobs_active_mutation());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
