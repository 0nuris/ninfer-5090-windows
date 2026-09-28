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
// the lane's committed ledger. The pool proposes from the ledger before anything is launched, and the
// copy rides the round's existing ingress upload. `speculative_prepare_verify_inputs` substitutes it
// into `verify_ids`, which is the buffer the target consumes, so the target verifies the copy
// without knowing or caring where the draft tokens came from. The draft model still runs and its own
// proposal is discarded for a copied row: it writes into `frame.draft_tokens`, which nothing reads
// once the copy has replaced it in `verify_ids`. Applying the override at the point of consumption
// rather than at the point of production is what makes the change safe without reasoning about the
// draft model's state at all -- see the note on skipping below.
//
// This also matches the measurements better. On edit-heavy work vLLM's own numbers have n-gram
// alone at 1.90 ms TPOT against 2.13 for the n-gram + EAGLE combination: keeping the neural drafter
// running for a row that has a copy is a cost the row does not need. See
// docs/research/ngram-drafting-designs.md and docs/research/ngram-outside-github.md.
//
// Cost of a wide round: NOT YET MEASURED, and a figure that once sat here has been withdrawn.
//
// An earlier version of this comment recorded 5.567 ms per round at draft window 7 against 5.886 ms
// at 15, and concluded the wide window costs +5.7 % while committing +28.7 % tokens per round. Those
// numbers have no artifact behind them. The sweep that was meant to produce them ran eight times and
// wrote eight zero-byte logs and a CSV containing only its header row, after which the summarising
// step failed casting that header as data. A search of the tree and of the session's saved artifacts
// finds the figures in exactly one place: this comment. They were recorded as a measurement and
// repeated into a design simplification and a 288 MiB budget line. They are withdrawn.
//
// What *is* established here is structural rather than measured, and it is what the allowance rests
// on: for DFlash2 the profile planner ignores the draft window entirely, assigning one topology
// class per frontier range (graph_profiles.cpp:98-104). Width therefore does not change how many
// topologies a family has, which is why the two widths cost the same allowance, and why adding the
// copy window doubles it rather than scaling it. That is read from the source, not measured.
//
// Measured on 2026-09-27, and the result argues against the design this file describes.
//
// DFlash2, NVFP4 27B, bf16 KV, 8192 context, 64 generated tokens, optimised proposal head, CUDA
// graph, RTX 5090. `ninfer_bench --spec dflash2 --draft-tokens k -n 64 -r 1 --warmup 1`, two
// interleaved samples per width agreeing to 0.3-0.8 %, with -r 1 so that `spec_rounds` (a sum) and
// `decode_seconds_mean` (a mean) share a convention and need no inference to divide:
//
//   window   round ms   acceptance   tok/round   throughput
//        4      17.816       22.06 %       1.882      105.7 tok/s
//        7      18.487        7.27 %       1.488       80.5 tok/s
//       11      18.767        3.61 %       1.362       72.6 tok/s
//       15      19.702        4.20 %       1.600       81.2 tok/s
//
// Three things follow, and the first two are the opposite of what this design assumed.
//
// Round cost is not flat in width. It rises monotonically, +10.6 % from window 4 to window 15, and
// +6.6 % from the shipped window 7 to 15. The cost of a wide round is therefore a real and
// per-column price, not a bounded one.
//
// A wider window commits fewer tokens per round, not more. Acceptance collapses from 22.1 % at
// window 4 to 3.6-4.2 % at windows 11-15, and tokens per round falls from 1.882 to 1.362 before a
// slight recovery to 1.600 at 15. The wide window is worse on both axes at once: more time per round
// and fewer tokens per round. Throughput at window 4 is 31 % above the shipped window 7.
//
// So the neural lane does not want a wider round, and running wide unconditionally is not a
// simplification available here -- it is a regression.
//
// A separate primary-source review reaches the same conclusion about a copy round from the other
// direction, and takes the width question out of the design entirely. See
// docs/research/ngram-copy-selection-signals.md.
//
//   - No shipped engine widens a copy round. TensorRT-LLM's `SADraftEnhancer`, the only copy-versus-
//     neural selector in production anywhere, uses one `max_draft_len` for both arms and has no
//     width parameter in the file. Baseten lists "dynamic-length speculation" as future work and
//     advertises the gain as arriving "without requiring any changes to configuration parameters
//     such as draft length". The two shipped adaptive-width mechanisms, vLLM's
//     `num_speculative_tokens_per_batch_size` and TRT-LLM's `draft_len_schedule`, are keyed on batch
//     size, not on which drafter won a round.
//   - Because the draft model runs on both arms here, the mixing rate cancels out of the break-even
//     entirely: a copy round is worth taking exactly when its tokens per round exceed the neural
//     round's times the cost factor. How often the policy chooses copy does not affect that at all,
//     so there is no "how aggressive should selection be" tuning problem to solve.
//
// With the cost factor at 1.0 for a same-width copy round, the bar is simply "beats the neural round
// per round", which is a much easier thing for a verbatim repetition to clear than a wide round
// needs to clear while also paying +6.6 % for the width.
//
// DESIGN CONSEQUENCE: the wide copy round is dropped. A copy round runs at the same window as a
// neural round. That removes the second RoundStateLayout, the second captured graph family, the
// width discriminator in `topology_class` that would otherwise collide and silently drop the wide
// family's topologies at graphs.cpp:53-58, and the 288 MiB allowance that was being charged for a
// family this tree never built. What survives is exactly the part that was already right: the pool,
// the length-threshold policy, the copy override in `speculative_prepare_verify_inputs`, the
// counters, and the CLI. The override is width-agnostic, so it is unaffected.
//
// What still has to be measured before the feature earns its place: how many tokens per round a
// copy actually commits on this lane, against the 1.488 the neural lane commits at window 7 and the
// 1.882 it commits at window 4. Note also that the earlier figure of 58 % neural acceptance, which
// the primary-source review used in its arithmetic, does not reconcile with the 7.27 % measured
// above on this artifact; that discrepancy is unresolved and is not resolved by picking whichever
// number suits a conclusion.
//
// One published number does argue for the feature: TensorRT-LLM PR #11434 measures MTP + selection
// at 299.31 tok/s against 192.37 for the neural drafter alone, on DeepSeek-V3.1-NVFP4 across 8xB200
// with a code-edit corpus, raising acceptance from 0.2704 to 0.5513. That is a third-party
// measurement on different hardware, a different drafter and a more repetitive workload, and the
// review notes the selector's value falls as the drafter strengthens. It is a reason to measure, not
// a result for this product.
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
    // True when the round is served from the pool. The draft model still runs either way; this flag
    // selects the proposal, not which model executes. False means keep the neural proposal at the
    // neural width, which is also the answer for every failure to copy.
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
 * The rule is a function of copy *length* alone. It reads no history: not the previous round's
 * outcome, not a running hit rate, not a confidence on the last emitted token. A longer copy is
 * treated as a better bet because it has more columns to be accepted in, which is a proxy for
 * acceptance rather than a measurement of it. If n-gram acceptance turns out to be strongly
 * history-dependent -- bursts of hits separated by long dry stretches -- then a length threshold is
 * the wrong signal and a memory of recent outcomes is the right one. That is an open question, not
 * a settled one, and it is recorded here so the next reader knows which assumption this rests on.
 * See docs/research/ for the note that examines it.
 *
 * A round is served from the pool only when *every* row has a copy that clears `min_drafts`. At the
 * shipped concurrency of one that is just the single row's answer, which is the configuration the
 * policy was designed and measured for.
 *
 * Above concurrency one this rule is close to inert, and that is worth stating rather than leaving
 * to be discovered: with a per-round copy acceptance of a few percent, a batch of eight rows will
 * essentially never all hold a usable copy at once. The rule originally existed because the design
 * skipped the draft model, which made choosing a copy a batch-wide commitment. That is no longer the
 * mechanism -- the draft model runs on every round regardless -- and the Op underneath is per-row, so
 * a mixed batch is representable today. The all-or-nothing rule is therefore a conservative choice
 * carried over from the superseded design, not a constraint the current one imposes. Widening it to
 * per-row is the obvious next step and is deliberately not taken here: it changes a policy that
 * 1,080 exhaustive boundary cases currently pin, and it should be decided against a measurement on a
 * concurrent workload rather than folded into the round integration as a drive-by.
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
