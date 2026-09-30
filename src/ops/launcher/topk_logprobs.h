#pragma once

// ninfer::ops::detail - private launch prototype for topk_logprobs.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void topk_logprobs_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k,
                          Tensor& indices, Tensor& logprobs, cudaStream_t stream);

} // namespace ninfer::ops::detail
