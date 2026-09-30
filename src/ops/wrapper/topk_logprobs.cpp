// ninfer::ops - topk_logprobs wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/topk_logprobs.h"

#include "ops/launcher/topk_logprobs.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("topk_logprobs: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_columns_k(const Tensor& tensor, std::int32_t columns, std::int32_t k, const char* label) {
    if (tensor.ne[0] != columns || tensor.ne[1] != k || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("topk_logprobs: ") + label + " must have shape [" +
                                    std::to_string(columns) + "," + std::to_string(k) + "]");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("topk_logprobs: ") + label + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("topk_logprobs: ") + label + " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("topk_logprobs: ") + label +
                                    " is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void topk_logprobs(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& indices,
                    Tensor& logprobs, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("topk_logprobs: logits must be BF16");
    }
    if (indices.dtype != DType::I32) {
        throw std::invalid_argument("topk_logprobs: indices must be I32");
    }
    if (logprobs.dtype != DType::FP32) {
        throw std::invalid_argument("topk_logprobs: logprobs must be FP32");
    }

    require_rank_two(logits, "logits");
    const std::int32_t columns = logits.ne[1];
    require_columns_k(indices, columns, k, "indices");
    require_columns_k(logprobs, columns, k, "logprobs");
    if (k <= 0 || k > kTopkLogprobsMaxK) {
        throw std::invalid_argument("topk_logprobs: k must be in [1," +
                                    std::to_string(kTopkLogprobsMaxK) + "]");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("topk_logprobs: valid_rows must be in [1, physical_rows]");
    }
    // k cannot exceed the number of participating rows: asking for more entries than exist has no
    // answer, and returning padding silently would read as "these were the k largest".
    if (k > valid_rows) {
        throw std::invalid_argument("topk_logprobs: k must not exceed valid_rows");
    }

    (void)logits.bytes();
    (void)indices.bytes();
    (void)logprobs.bytes();
    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(indices, alignof(std::int32_t), "indices");
    require_accessible(logprobs, alignof(float), "logprobs");
    if (overlaps(indices, logits) || overlaps(logprobs, logits) || overlaps(indices, logprobs)) {
        throw std::invalid_argument("topk_logprobs: outputs must not overlap each other or logits");
    }

    detail::topk_logprobs_launch(logits, valid_rows, k, indices, logprobs, stream);
}

} // namespace ninfer::ops
