#include "ninfer/ops/topk_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kTopkLogprobsFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

struct Reference {
    std::vector<std::int32_t> indices;
    std::vector<double> logprobs;
};

// The oracle, per the Op contract: an independent naive evaluation of the whole formula in FP64
// from the represented BF16 inputs, selecting the k largest by (value descending, index ascending)
// and normalising by the full log-partition.
//
// It deliberately does not sort with a radix select, reuse the Op's key trick, or share any of its
// arithmetic. A selection bug in the Op that the oracle reproduced would pass, so the oracle's
// selection is the obvious std::stable_sort over the decoded values -- the slowest thing that is
// obviously right, which is the point of an oracle.
Reference topk_logprobs_oracle(const std::vector<std::uint16_t>& logits, std::int32_t physical_rows,
                               std::int32_t valid_rows, std::int32_t k) {
    const auto columns = static_cast<std::int32_t>(logits.size()) / physical_rows;
    Reference reference;
    reference.indices.assign(static_cast<std::size_t>(columns) * k, -1);
    reference.logprobs.assign(static_cast<std::size_t>(columns) * k, 0.0);

    std::vector<double> values(static_cast<std::size_t>(valid_rows));
    std::vector<std::int32_t> rows(static_cast<std::size_t>(valid_rows));
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            values[static_cast<std::size_t>(row)] =
                static_cast<double>(bf16_to_f32(logits[base + static_cast<std::size_t>(row)]));
            rows[static_cast<std::size_t>(row)] = row;
        }

        double maximum = -std::numeric_limits<double>::infinity();
        for (double value : values) { maximum = std::max(maximum, value); }
        double sum = 0.0;
        for (double value : values) { sum += std::exp(value - maximum); }
        const double log_partition = maximum + std::log(sum);

        // Descending by value, and stable so equal values keep ascending row order.
        std::vector<std::int32_t> order(static_cast<std::size_t>(valid_rows));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::int32_t a, std::int32_t b) {
            return values[static_cast<std::size_t>(a)] > values[static_cast<std::size_t>(b)];
        });

        for (std::int32_t j = 0; j < k; ++j) {
            const std::int32_t row = order[static_cast<std::size_t>(j)];
            const std::size_t slot = static_cast<std::size_t>(column) * k + static_cast<std::size_t>(j);
            reference.indices[slot] = row;
            reference.logprobs[slot] =
                values[static_cast<std::size_t>(row)] - log_partition;
        }
    }
    return reference;
}

// from_device returns the stored dtype; verify_reduction compares in double, so widen here rather
// than letting the criterion read a float buffer through a double span.
std::vector<double> f32_as_double(const void* device, std::size_t count) {
    const auto values = from_device<float>(device, count);
    return std::vector<double>(values.begin(), values.end());
}

std::vector<std::int32_t> i32_from_device(const void* device, std::size_t count) {
    const auto values = from_device<std::int32_t>(device, count);
    return std::vector<std::int32_t>(values.begin(), values.end());
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                       std::int32_t columns, unsigned seed) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + 1) * 1013904223u + seed;
            const float value = -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

// Every valid row shares one value. This is the case the collection pass' guard exists for: the
// threshold bin holds valid_rows entries, far more than k, and an unguarded atomic write would run
// off the end of the k slots. It is also the tie rule's worst case, since all k returned entries
// have equal value and the ascending-index tiebreak is the only thing ordering them.
std::vector<std::uint16_t> make_uniform_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                               std::int32_t columns, float value) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(112.0f);
        }
    }
    return logits;
}

// Few distinct values, so the threshold bin is large but the values above it are not: the case where
// a threshold that is one rank too high silently drops a genuinely larger value.
std::vector<std::uint16_t> make_coarse_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                              std::int32_t columns, int distinct) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const int bucket = (row * 7 + column * 3) % distinct;
            logits[base + static_cast<std::size_t>(row)] =
                f32_to_bf16(static_cast<float>(bucket) * 2.0f - 5.0f);
        }
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(120.0f);
        }
    }
    return logits;
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             std::int32_t columns, std::int32_t k, const std::vector<std::uint16_t>& logits) {
    const auto reference = topk_logprobs_oracle(logits, physical_rows, valid_rows, k);
    const auto slots = static_cast<std::size_t>(columns) * static_cast<std::size_t>(k);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(slots * sizeof(std::int32_t));
    GuardedDeviceBuffer device_logprobs(slots * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_indices.fill(0xcd);
    device_logprobs.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor indices_tensor(device_indices.data(), DType::I32, {columns, k});
    Tensor logprobs_tensor(device_logprobs.data(), DType::FP32, {columns, k});
    ops::topk_logprobs(logits_tensor, valid_rows, k, indices_tensor, logprobs_tensor, nullptr);
    cuda_synchronize();

    int failures = 0;
    const auto got_indices = i32_from_device(device_indices.data(), slots);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (got_indices[slot] != reference.indices[slot]) {
            std::cerr << label << ": slot " << slot << " index " << got_indices[slot]
                      << ", expected " << reference.indices[slot] << '\n';
            ++failures;
            if (failures > 8) { break; }
        }
    }
    failures += verify_reduction(label, f32_as_double(device_logprobs.data(), slots),
                                reference.logprobs, kTopkLogprobsFp32Criterion);
    failures +=
        verify_exact((label + " preserves logits").c_str(),
                     from_device<std::uint16_t>(device_logits.data(), logits.size()), logits);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_indices.verify_guards(label + " index guards");
    failures += device_logprobs.verify_guards(label + " logprob guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer indices_data(3 * 2 * sizeof(std::int32_t));
    DeviceBuffer logprobs_data(3 * 2 * sizeof(float));

    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor indices(indices_data.p, DType::I32, {3, 2});
    Tensor logprobs(logprobs_data.p, DType::FP32, {3, 2});
    Tensor fp32_indices(indices_data.p, DType::FP32, {3, 2});

    int failures = 0;
    failures += expect_invalid("wrong logits dtype", [&] {
        Tensor wrong(logits_data.p, DType::FP32, {8, 3});
        ops::topk_logprobs(wrong, 8, 2, indices, logprobs, nullptr);
    });
    failures += expect_invalid("wrong indices dtype", [&] {
        ops::topk_logprobs(logits, 8, 2, fp32_indices, logprobs, nullptr);
    });
    failures += expect_invalid("k of zero", [&] {
        ops::topk_logprobs(logits, 8, 0, indices, logprobs, nullptr);
    });
    failures += expect_invalid("k above the supported maximum", [&] {
        Tensor big_indices(indices_data.p, DType::I32, {3, ops::kTopkLogprobsMaxK + 1});
        Tensor big_logprobs(logprobs_data.p, DType::FP32, {3, ops::kTopkLogprobsMaxK + 1});
        ops::topk_logprobs(logits, 8, ops::kTopkLogprobsMaxK + 1, big_indices, big_logprobs, nullptr);
    });
    failures += expect_invalid("k above valid_rows", [&] {
        ops::topk_logprobs(logits, 1, 2, indices, logprobs, nullptr);
    });
    failures += expect_invalid("valid_rows above physical_rows", [&] {
        ops::topk_logprobs(logits, 9, 2, indices, logprobs, nullptr);
    });
    failures += expect_invalid("indices shape disagrees with columns", [&] {
        Tensor wrong(indices_data.p, DType::I32, {4, 2});
        ops::topk_logprobs(logits, 8, 2, wrong, logprobs, nullptr);
    });
    failures += expect_invalid("outputs alias logits", [&] {
        Tensor aliased(logits_data.p, DType::I32, {3, 2});
        ops::topk_logprobs(logits, 8, 2, aliased, logprobs, nullptr);
    });
    failures += expect_invalid("outputs alias each other", [&] {
        Tensor shared(logprobs_data.p, DType::I32, {3, 2});
        ops::topk_logprobs(logits, 8, 2, shared, logprobs, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // The real vocabulary, which is the shape this Op exists for: 151,936 rows at 4 bytes of BF16
    // is a 291 MiB read per column, and the collection pass' tie guard has to hold at that width.
    failures += run_case("topk_logprobs full vocabulary C=3", 248320, 248077, 3, 60,
                         make_logits(248320, 248077, 3, 0u));
    failures += run_case("topk_logprobs non-aligned rows C=257", 523, 509, 257, 40,
                         make_logits(523, 509, 257, 7u));
    // k at the documented maximum, which is also k == valid_rows: every entry is selected, so the
    // sort has nothing to exclude and the equals carry the whole column.
    failures += run_case("topk_logprobs k at maximum equals valid_rows", 263, 256, 33, 256,
                         make_logits(263, 256, 33, 11u));
    failures += run_case("topk_logprobs k equals valid_rows", 131, 129, 33, 129,
                         make_logits(131, 129, 33, 11u));
    failures += run_case("topk_logprobs single column k=1", 4099, 4093, 1, 1,
                         make_logits(4099, 4093, 1, 3u));
    failures += run_case("topk_logprobs uniform logits, ties everywhere", 263, 257, 17, 60,
                         make_uniform_logits(263, 257, 17, 3.5f));
    failures += run_case("topk_logprobs uniform logits, k near valid_rows", 263, 257, 9, 256,
                         make_uniform_logits(263, 257, 9, -7.0f));
    failures += run_case("topk_logprobs two distinct values", 263, 257, 23, 40,
                         make_coarse_logits(263, 257, 23, 2));
    failures += run_case("topk_logprobs one distinct value", 263, 257, 23, 40,
                         make_coarse_logits(263, 257, 23, 1));
    failures += run_case("topk_logprobs few distinct values", 1021, 1013, 65, 60,
                         make_coarse_logits(1021, 1013, 65, 3));
    failures += run_validation_cases();

    std::cout << (failures ? "FAIL" : "OK") << " topk_logprobs\n";
    return failures ? 1 : 0;
}
