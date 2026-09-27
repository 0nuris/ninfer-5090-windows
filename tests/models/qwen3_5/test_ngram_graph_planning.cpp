// Adapted from Wallawalla47/ninfer-custom (HEAD 600d8ac), test_ngram_graph_planning.cpp.
//
// Only the profile-coverage half ports. Three blocks of the fork's test exercise code this tree
// does not have, and are deliberately not reproduced rather than approximated:
//
//   * The mixed-width MTP aliasing rule ("mixed-width MTP graphs cannot alias"). The fork widened
//     mtp_graph_profiles to (capacity, width, neural) so one family can alias a neural width and an
//     ngram width. Ours is (capacity, draft_window) and has no such rule, because we are not
//     adopting the mixed-width MTP family.
//   * wide_residual_verification and residual_projection_policy. Both are the fork's own residual
//     precision feature; neither symbol exists here, and neither is part of ngram.
//
// The fork's --real mode compared planned device bytes with and without CUDA graphs against an
// expected family allowance, at a set of ngram widths. That measurement is deferred, not dropped:
// today plan.draft_window is still taken from draft_tokens alone, so ngram_draft_tokens does not
// reach the planner and the measurement would record this tree's inert state rather than the cost
// of the feature. It returns when the plan consumes the ngram width.
//
// The coverage invariant is the one ngram actually stresses: it changes the round width, and every
// width must still yield gap-free profiles spanning the full capacity. MTP is swept to 63, wider
// than the 15-draft product cap, because the cap is enforced when options are validated and this is
// the planning function's own domain. The masked-draft sweep stops at the round's column domain
// instead, because that planner refuses a wider width outright.
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

namespace qwen = ninfer::models::qwen3_5;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

// A profile set is usable only if it tiles [0, capacity) with no gap and no overlap, because the
// round launcher selects by the sequence's frontier and a gap would leave it without a graph.
void require_full_coverage(const std::vector<qwen::GraphExecutionProfile>& profiles,
                           std::uint32_t capacity, const char* what) {
    if (profiles.empty()) { return; }
    std::uint32_t frontier = 0;
    for (const auto& profile : profiles) {
        if (profile.min != frontier || profile.max < profile.min) {
            throw std::runtime_error(std::string(what) + ": profile coverage has a gap or overlap");
        }
        frontier = profile.max + 1U;
    }
    require(frontier == capacity, "profile coverage must reach full capacity");
}

void verify_profiles() {
    for (const std::uint32_t capacity : {2U, 128U, 4090U, 4096U, 8192U, 32768U, 262144U}) {
        for (std::uint32_t width = 1; width <= 63; ++width) {
            const auto mtp = qwen::detail::mtp_graph_profiles(capacity, width);
            require(!mtp.empty(), "a valid MTP width needs graph profiles");
            require_full_coverage(mtp, capacity, "MTP");
        }
        // The masked-draft planner refuses a width above the round's column domain, so its range is
        // the shipped bound rather than the MTP one. This is the third independent place the 15
        // cap is enforced (with kDFlashDecodeMaximumDrafts and the GDN record width); the fork
        // raised its equivalent to 63 to admit wider ngram rounds, and this guard is why that
        // would not fit here without growing the round constants.
        constexpr std::uint32_t kMaskedMaximumDrafts = qwen::kDFlashDecodeMaximumDrafts;
        for (const auto backend :
             {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::DFlash2}) {
            for (std::uint32_t width = 1; width <= kMaskedMaximumDrafts; ++width) {
                const auto profiles =
                    qwen::detail::dflash_graph_profiles(backend, capacity, width, 1);
                require_full_coverage(profiles, capacity, "masked draft");
            }
            bool rejected = false;
            try {
                (void)qwen::detail::dflash_graph_profiles(backend, capacity,
                                                          kMaskedMaximumDrafts + 1U, 1);
            } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "a masked width past the round's column domain must be refused");
        }
    }
}

} // namespace

int main() {
    try {
        verify_profiles();
        std::cout << "Ngram graph profile coverage passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
