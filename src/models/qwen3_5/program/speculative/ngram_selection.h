#pragma once

// Copy drafting for the masked-draft (DFlash/DFlash2) route, by selection rather than by chaining.
//
// The source design (JGamboa/ninfer-4090-windows) chains: it appends the pool's continuation to the
// MTP proposal inside one round. That works there because an MTP proposal is built on the host, in
// `SequenceState::mtp_drafts`, so the chain is an ordinary host operation. Our masked-draft route
// cannot use it. `prepare_masked_block` writes the draft block on the device and the draft model
// refines it in place, so the proposal only exists in `frame.draft_tokens` once a kernel has run.
// Appending a copy to it would mean reading those tokens back to the host, chaining there, and
// uploading the extension -- a host/device synchronisation in the middle of a round that costs
// about six milliseconds in total, which is a far worse trade than the feature's upside.
//
// Selection avoids the synchronisation entirely, because the round's input is already on the host:
// the lane's committed ledger. The pool proposes from the ledger before anything is launched, the
// copy is uploaded directly into the draft-token block, and the draft model is skipped for that
// round. `speculative_prepare_verify_inputs` and the target then proceed unchanged, since neither
// knows or cares where the draft tokens came from. The draft model's own state simply catches up on
// a later round, which is why skipping it is safe rather than merely cheap.
//
// This also matches the measurements better. On edit-heavy work vLLM's own numbers have n-gram
// alone at 1.90 ms TPOT against 2.13 for the n-gram + EAGLE combination: keeping the neural drafter
// running for a row that has a copy is a cost the row does not need. See
// docs/research/ngram-drafting-designs.md and docs/research/ngram-outside-github.md.
//
// Cost of a wide round, measured on this product (DFlash2, NVFP4 27B, 8192 context, bf16 KV, CUDA
// graph, optimised head): 5.567 ms per round at draft window 7 against 5.886 ms at 15, so the wide
// window costs +5.7 % and commits +28.7 % tokens per round. Non-copy rounds keep the neural width and
// pay nothing, which is what bounds the downside.
//
// Cost of the second graph family, also measured rather than estimated, and measured *before* the
// round was written so the price was known before the purchase. On the shipped CLI at 8192 context
// with draft window 7:
//
//   ngram off              CUDA Graph allowance 288.0 MiB   planned device total 22.2 GiB
//   ngram chain, max 10    CUDA Graph allowance 576.0 MiB   planned device total 22.5 GiB
//   ngram chain, max 15    CUDA Graph allowance 576.0 MiB   planned device total 22.5 GiB
//
// So the wide family costs +288 MiB, exactly doubling the allowance, and the plan's own slack falls
// from 8.07 to 7.79 GiB. That +288 MiB is the figure an earlier estimate in this work put at ~480
// MiB, and the 17-23 % of free-memory headroom asserted from it was wrong on both counts. Note also
// that max 10 and max 15 cost the same: the width does not change how many topologies a family has,
// so the window should be chosen on benefit, not on allowance.
//
// The draft model still runs on a copy round, and its proposal is then overwritten. Skipping it
// looks like the obvious optimisation and is not: in dflash_decode_batch_body, append_context_impl
// advances the draft model's own context into the destination state slots and runs *before*
// propose_batch_impl, so the append and the proposal are ordered and coupled. Skipping the proposal
// alone would advance that context with features the draft model never consumed, and whether a later
// round can recover the skipped span is not something this design may assume. Running the drafter
// and discarding its tokens wastes its work on the rounds where a copy hits, and is the same cost
// profile the source design measured; it keeps the draft model's state consistent every round, which
// is worth more than the discarded work. Both shapes need the second captured family, so this choice
// costs nothing in graph allowance either way.
#include "models/qwen3_5/program/speculative/ngram_policy.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

struct CopyRoundDecision {
    // True when the round is served from the pool and the draft model is skipped. False means keep
    // the neural proposal at the neural width, which is also the answer for every failure to copy.
    bool copy = false;
    // Drafts to verify. Only meaningful when `copy`; otherwise zero, and the caller uses the
    // neural draft window.
    std::uint32_t verify_drafts = 0;
};

/**
 * Decide whether a round is served from the copy pool.
 *
 * `usable` holds each row's copy length after the budget, context and capacity clamps, and is empty
 * when no row produced a copy. `draft_window` is the neural draft width, `min_drafts` the shortest
 * pool extension worth spending a round on, and `verify_window` the widest window the plan
 * provisioned.
 *
 * The draft model runs over the whole batch, so skipping it is a batch-wide decision: a round is
 * served from the pool only when *every* row has a copy that clears `min_drafts`. At the shipped
 * concurrency of one that is just the single row's answer. A mixed batch keeps the neural proposal,
 * so a copy never costs a row anything when some other row needed the drafter.
 *
 * The round is only widened when the copy is materially wider than the neural depth, by
 * kNgramWideRoundMargin. A copy shorter than that is dropped in favour of the neural round, because
 * the wide window would cost more than the copy returns.
 *
 * Throws std::invalid_argument when asked to decide a round the plan never provisioned for: an empty
 * or zero-width round, or a verify window too narrow to ever justify leaving the neural one.
 */
[[nodiscard]] inline CopyRoundDecision select_copy_round(std::span<const std::uint32_t> usable,
                                                          std::uint32_t draft_window,
                                                          std::uint32_t min_drafts,
                                                          std::uint32_t verify_window) {
    if (draft_window == 0) {
        throw std::invalid_argument("copy selection needs a neural draft window");
    }
    if (verify_window < draft_window + kNgramWideRoundMargin) {
        throw std::invalid_argument("copy selection needs a verify window above the neural width");
    }
    if (usable.empty() || min_drafts == 0) { return {}; }

    std::uint32_t widest = 0;
    for (const std::uint32_t count : usable) {
        if (count < min_drafts) { return {}; }  // a row without a usable copy keeps the drafter
        widest = std::max(widest, count);
    }
    if (widest < draft_window + kNgramWideRoundMargin) { return {}; }
    return {.copy = true, .verify_drafts = std::min(widest, verify_window)};
}

/**
 * Attribute a selection round's verified extent to its source. A selection round is served wholly
 * from one source, so the split is all-copy or all-neural rather than the chain's positional split.
 * `accepted` leading drafts were committed; the rest of `extent` was verified and rejected.
 */
struct SelectionSourceCounts {
    std::uint32_t drafted    = 0;
    std::uint32_t accepted   = 0;
    bool from_copy            = false;
};

[[nodiscard]] inline SelectionSourceCounts selection_source_counts(std::uint32_t extent,
                                                                  std::uint32_t accepted,
                                                                  bool from_copy) {
    if (extent == 0 || accepted > extent) {
        throw std::invalid_argument("accepted drafts exceed the verified extent");
    }
    return {.drafted = extent, .accepted = accepted, .from_copy = from_copy};
}

} // namespace ninfer::models::qwen3_5
