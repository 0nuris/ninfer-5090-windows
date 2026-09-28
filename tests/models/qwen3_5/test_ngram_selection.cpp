// The decision core for masked-draft copy selection, tested exhaustively and off any GPU.
//
// This is a control, not coverage. `select_copy_round` decides one thing that is easy to get subtly
// wrong in both directions: it must not drop a copy that would have paid, and it must not claim more
// drafts than the round can verify. Both failure modes are silent in production -- one leaves
// performance on the table, the other reads past buffers sized for the round -- so the cases below pin
// each boundary from both sides and then sweep the whole input space at them.
//
// The rule has no widened round. An earlier version of this file tested a `verify_window` parameter
// and a `kNgramWideRoundMargin` trigger, on the design where a copy round ran wider than a neural one.
// That design was withdrawn: measurement on this product found the neural lane does not want a wider
// round, and no shipped engine widens one either. The surviving boundary is the clamp, and it now
// guards a different thing -- a copy may be longer than the round, and the round verifies exactly
// `draft_window` drafts, so the excess is truncated rather than served.
#include "models/qwen3_5/program/speculative/ngram_selection.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::models::qwen3_5::CopyRoundDecision;
using ninfer::models::qwen3_5::select_copy_round;
using ninfer::models::qwen3_5::selection_source_counts;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class F>
bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

constexpr std::uint32_t kNeural = 7;  // the shipped drafter width, and the copy round's width too
// A second width to exercise the clamp where it is tightest. Deliberately a local constant rather
// than the model's kDFlashDecodeMaximumDrafts: including the round buffers here would pull CUDA
// headers into a policy test that needs no device, and the clamp arithmetic does not care which
// width it is handed.
constexpr std::uint32_t kMax = 15;

// std::span has no initializer_list constructor, so even a one-row batch must be named as a real
// contiguous container. Building it here keeps every call site below reading as the case it is.
std::vector<std::uint32_t> rows(std::initializer_list<std::uint32_t> values) {
    return std::vector<std::uint32_t>(values);
}

void rejects_impossible_rounds() {
    // A zero neural width means there is no round to fall back to and nothing to verify against, so
    // asking is a planning error. An empty batch with a valid width is not an error, and is covered
    // as a fallback below -- the two must not be conflated.
    check(throws_invalid([] { (void)select_copy_round(rows({}), 0, 1); }),
          "a zero neural draft window was accepted");
    check(throws_invalid([] { (void)select_copy_round(rows({12}), 0, 1); }),
          "a zero neural draft window was accepted with rows present");
}

void falls_back_to_the_neural_round() {
    // No row produced a copy at all.
    check(!select_copy_round(rows({}), kNeural, 1).copy,
          "an empty batch was served from the pool");
    // A copy shorter than min_drafts is not worth a round. Both sides of the threshold are checked:
    // a rule that only rejected the far side would pass a test that never tried the edge.
    check(!select_copy_round(rows({2}), kNeural, 3).copy, "a copy below min_drafts was served");
    check(select_copy_round(rows({3}), kNeural, 3).copy, "a copy exactly at min_drafts was dropped");
    // A zero min_drafts would let a one-token copy claim the round.
    check(!select_copy_round(rows({12}), kNeural, 0).copy, "a zero min_drafts was honoured");
    // A row with no copy at all keeps the drafter, however long its neighbours are.
    check(!select_copy_round(rows({12, 0}), kNeural, 1).copy, "a batch with an empty row was served");
    check(!select_copy_round(rows({12, 1, 12}), kNeural, 2).copy,
          "a batch with one row below min_drafts was served");
}

void serves_at_the_round_width() {
    // A copy inside the round is served at its own length.
    const auto exact = select_copy_round(rows({5}), kNeural, 1);
    check(exact.copy && exact.verify_drafts == 5, "a copy inside the round was not served at its length");

    const auto at_width = select_copy_round(rows({kNeural}), kNeural, 1);
    check(at_width.copy && at_width.verify_drafts == kNeural,
          "a copy exactly at the round width was not served");

    // A copy longer than the round is truncated to it. This is the boundary the withdrawal moved: the
    // buffers, the captured graph and the GDN record are all sized to the round's width, and there is
    // no second layout, so a longer copy must not read past them.
    const auto beyond = select_copy_round(rows({63}), kNeural, 1);
    check(beyond.copy && beyond.verify_drafts == kNeural,
          "a copy longer than the round width was not clamped to it");

    // The same clamp at the plan's maximum window, which is the widest a round can ever be.
    const auto widest = select_copy_round(rows({kMax}), kMax, 1);
    check(widest.copy && widest.verify_drafts == kMax, "a copy at the maximum window was not served");

    // Several rows: all must clear min_drafts, and the round takes the widest, truncated to the width.
    const auto mixed = select_copy_round(rows({3, kNeural, 5}), kNeural, 1);
    check(mixed.copy && mixed.verify_drafts == kNeural,
          "a fully covered batch did not take the widest, clamped to the round");
    const auto mixed_beyond = select_copy_round(rows({kMax, 2, 4}), kNeural, 2);
    check(mixed_beyond.copy && mixed_beyond.verify_drafts == kNeural,
          "a batch whose widest copy exceeds the round was not clamped");
}

// The whole input space at both boundaries: every per-row length from 0 past the round width, against
// a min_drafts that also walks its range, with the expected answer derived independently of the rule.
void exhaustive_boundary_sweep() {
    std::uint32_t checked = 0;
    for (std::uint32_t count = 0; count <= kNeural + 2; ++count) {
        const std::vector<std::uint32_t> row{count};
        for (std::uint32_t min_drafts = 1; min_drafts <= kNeural; ++min_drafts) {
            const CopyRoundDecision got = select_copy_round(row, kNeural, min_drafts);
            // Independent expectation: serve only when the row clears min_drafts, and never claim
            // more drafts than the round verifies.
            const bool want_copy = count >= min_drafts;
            const std::uint32_t want_width = want_copy ? std::min(count, kNeural) : 0U;
            if (got.copy != want_copy || got.verify_drafts != want_width) {
                std::cerr << "FAIL: count=" << count << " min=" << min_drafts << " got copy="
                          << got.copy << " width=" << got.verify_drafts << " want copy=" << want_copy
                          << " width=" << want_width << '\n';
                ++failures;
            }
            ++checked;
        }
    }
    // The same sweep at the maximum window, so the clamp is exercised where it is tightest.
    for (std::uint32_t count = 0; count <= kMax + 2; ++count) {
        const std::vector<std::uint32_t> row{count};
        for (std::uint32_t min_drafts = 1; min_drafts <= kMax; ++min_drafts) {
            const CopyRoundDecision got = select_copy_round(row, kMax, min_drafts);
            const bool want_copy = count >= min_drafts;
            const std::uint32_t want_width = want_copy ? std::min(count, kMax) : 0U;
            if (got.copy != want_copy || got.verify_drafts != want_width) {
                std::cerr << "FAIL(max) count=" << count << " min=" << min_drafts << " got copy="
                          << got.copy << " width=" << got.verify_drafts << " want copy=" << want_copy
                          << " width=" << want_width << '\n';
                ++failures;
            }
            ++checked;
        }
    }
    std::cout << "selection boundary cases=" << checked << '\n';
}

void source_attribution() {
    const auto copied = selection_source_counts(15, 12, true);
    check(copied.drafted == 15 && copied.accepted == 12 && copied.from_copy,
          "a copy round misattributed its extent");
    const auto neural = selection_source_counts(7, 6, false);
    check(neural.drafted == 7 && neural.accepted == 6 && !neural.from_copy,
          "a neural round misattributed its extent");
    // Full acceptance and total rejection are the two ends a real round hits.
    check(selection_source_counts(9, 9, true).accepted == 9, "full acceptance misattributed");
    check(selection_source_counts(9, 0, true).accepted == 0, "total rejection misattributed");
    check(throws_invalid([] { (void)selection_source_counts(4, 5, true); }),
          "more accepted than verified was accepted");
    check(throws_invalid([] { (void)selection_source_counts(0, 0, true); }),
          "an empty extent was accepted");
}

} // namespace

int main() {
    rejects_impossible_rounds();
    falls_back_to_the_neural_round();
    serves_at_the_round_width();
    exhaustive_boundary_sweep();
    source_attribution();
    if (failures != 0) {
        std::cerr << failures << " copy-selection check(s) failed\n";
        return 1;
    }
    std::cout << "copy selection: PASS\n";
    return 0;
}
