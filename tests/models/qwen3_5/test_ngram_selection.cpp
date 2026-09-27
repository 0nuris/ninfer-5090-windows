// The decision core for masked-draft copy selection, tested exhaustively and off any GPU.
//
// This is a control, not coverage. `select_copy_round` decides two things that are easy to get
// subtly wrong in opposite directions: it must not widen a round that the copy cannot pay for, and
// it must not drop a copy that would have paid. Both failure modes are silent in production -- one
// pays for width it does not use, the other leaves performance on the table -- so the cases below
// pin the boundary from both sides and then sweep the whole input space at it.
#include "models/qwen3_5/program/speculative/ngram_selection.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::models::qwen3_5::CopyRoundDecision;
using ninfer::models::qwen3_5::kNgramWideRoundMargin;
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

constexpr std::uint32_t kNeural = 7;   // the shipped drafter width
constexpr std::uint32_t kWide   = 15;  // the plan's verify window

// The width at which leaving the neural round starts to pay, by the shared rule.
constexpr std::uint32_t kTrigger = kNeural + kNgramWideRoundMargin;

// std::span has no initializer_list constructor, so even a one-row batch must be named as a real
// contiguous container. Building it here keeps every call site below reading as the case it is.
std::vector<std::uint32_t> rows(std::initializer_list<std::uint32_t> values) {
    return std::vector<std::uint32_t>(values);
}

void rejects_impossible_rounds() {
    // A zero neural width means there is no neural round to fall back to, so asking is a planning
    // error. An empty batch with a valid width is not an error and is covered as a fallback below.
    check(throws_invalid([] { (void)select_copy_round(rows({}), 0, 1, kWide); }),
          "a zero neural draft window was accepted");
    // A verify window that cannot exceed the neural width by the margin can never justify a wide
    // round, so asking is a planning error rather than a decision.
    check(throws_invalid([] { (void)select_copy_round(rows({12}), kNeural, 1, kNeural + 1); }),
          "a verify window below draft_tokens + margin was accepted");
    check(throws_invalid([] { (void)select_copy_round(rows({12}), kNeural, 1, kTrigger - 1); }),
          "a verify window one below the trigger was accepted");
}

void falls_back_to_the_neural_round() {
    // No row produced a copy at all.
    check(!select_copy_round(rows({}), kNeural, 1, kWide).copy,
          "an empty batch was served from the pool");
    // A copy shorter than min_drafts is not worth a round. Both sides of the threshold are checked:
    // a rule that only rejected the far side would pass a test that never tried the edge.
    check(!select_copy_round(rows({kTrigger - 1}), kNeural, kTrigger, kWide).copy,
          "a copy below min_drafts was served");
    check(select_copy_round(rows({kTrigger}), kNeural, kTrigger, kWide).copy,
          "a copy exactly at min_drafts was dropped");
    // A zero min_drafts would let a one-token copy claim a wide round.
    check(!select_copy_round(rows({12}), kNeural, 0, kWide).copy, "a zero min_drafts was honoured");
    // A copy no wider than the neural round must not trigger the switch, even though it is long.
    check(!select_copy_round(rows({kNeural}), kNeural, 1, kWide).copy,
          "a copy at the neural width widened the round");
    check(!select_copy_round(rows({kTrigger - 1}), kNeural, 1, kWide).copy,
          "a copy one below the trigger widened the round");
}

void serves_and_clamps() {
    const auto exact = select_copy_round(rows({kTrigger}), kNeural, 1, kWide);
    check(exact.copy && exact.verify_drafts == kTrigger,
          "a copy at the trigger did not serve at its own width");

    const auto wide = select_copy_round(rows({kWide}), kNeural, 1, kWide);
    check(wide.copy && wide.verify_drafts == kWide, "a maximal copy was not served at full width");

    // Beyond the provisioned window the round is clamped, not widened: the buffers, the graph and
    // the GDN record are all sized to verify_window, and a longer copy must not read past them.
    const auto beyond = select_copy_round(rows({63}), kNeural, 1, kWide);
    check(beyond.copy && beyond.verify_drafts == kWide,
          "a copy longer than the verify window was not clamped");

    // Several rows: all must have a copy, and the round takes the widest.
    const auto mixed = select_copy_round(rows({kTrigger, kWide, kTrigger + 2}), kNeural, 1, kWide);
    check(mixed.copy && mixed.verify_drafts == kWide,
          "a fully covered batch did not take the widest");
    // One row short of min_drafts makes the whole batch keep the drafter, because the draft model
    // runs over the batch and cannot be skipped for part of it.
    check(!select_copy_round(rows({kWide, kTrigger - 1}), kNeural, kTrigger, kWide).copy,
          "a batch with one unusable row still skipped the drafter");
    check(!select_copy_round(rows({kWide, 0}), kNeural, 1, kWide).copy,
          "a batch containing an empty row still skipped the drafter");
}

// The whole input space at the boundary: every per-row length from 0 past the window, against a
// min_drafts that also walks the boundary, with the expected answer derived independently.
void exhaustive_boundary_sweep() {
    std::uint32_t checked = 0;
    for (std::uint32_t count = 0; count <= kWide + 2; ++count) {
        const std::vector<std::uint32_t> row{count};
        for (std::uint32_t min_drafts = 1; min_drafts <= kTrigger; ++min_drafts) {
            for (std::uint32_t window = kTrigger; window <= kWide; ++window) {
                const CopyRoundDecision got = select_copy_round(row, kNeural, min_drafts, window);
                // Independent expectation: serve only when the row clears min_drafts and reaches the
                // trigger, and never past the provisioned window.
                const bool want_copy = count >= min_drafts && count >= kTrigger;
                const std::uint32_t want_width = want_copy ? std::min(count, window) : 0U;
                if (got.copy != want_copy || got.verify_drafts != want_width) {
                    std::cerr << "FAIL: count=" << count << " min=" << min_drafts
                              << " window=" << window << " got copy=" << got.copy
                              << " width=" << got.verify_drafts << " want copy=" << want_copy
                              << " width=" << want_width << '\n';
                    ++failures;
                }
                ++checked;
            }
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
    serves_and_clamps();
    exhaustive_boundary_sweep();
    source_attribution();
    if (failures != 0) {
        std::cerr << failures << " copy-selection check(s) failed\n";
        return 1;
    }
    std::cout << "copy selection: PASS\n";
    return 0;
}
