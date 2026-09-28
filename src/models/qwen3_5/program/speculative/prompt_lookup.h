#pragma once

// Prompt-lookup proposals for masked-draft (DFlash/DFlash2) verify trees.
//
// Ported from satellitedown/cinference (https://github.com/satellitedown/cinference, HEAD
// 1d07410), Apache-2.0, which lists this file among its added paths in
// upstream-provenance.json -- that is, it is their own work, not inherited from NInfer. Body
// unchanged. Cinference in turn derives from Neroued/ninfer, which is this project's upstream, so
// the licence and the lineage both hold.
//
// This replaces NgramDraftPool, ported earlier from JGamboa/ninfer-4090-windows. Four things make it
// the better proposer, and each is a defect in the pool rather than a preference:
//
//   - It matches three window lengths (8, 4, 2) and takes the longest that hits, instead of one
//     configured length. A single length is a bet on the workload: 8 misses on prose and 2 matches
//     nothing on a long verbatim block.
//   - It re-checks the matched tokens against the context rather than trusting the hash
//     (`std::equal` on the suffix), so a collision cannot produce a wrong proposal. The pool's
//     `propose` reads the stored token for a hash and cannot tell a collision from a match.
//   - It carries a learned, decayed, per-window acceptance estimate and reports it as a
//     log-probability, so the consumer can *score* a copy against a neural proposal rather than
//     accept or reject it. The pool has no estimate at all; ngram_selection.h had to gate on a
//     bare length threshold, and recorded that as an open question about whether history was needed.
//     This is the answer to that question, and it was already written.
//   - It is per-sequence, so its estimate is per-request and nothing is retained across requests.
//     The pool was one table shared by the whole engine, which is the cross-request retention the
//     JGamboa header argued the measured value was not in. Per-sequence is the scope the evidence
//     pointed at.
//
// It also re-indexes from scratch when the context stops extending what it indexed, which is the
// splice case a retried or edited conversation produces.
//
// NOT YET WIRED. The consumer this was written for is a verify *tree*: candidate_selector_tree grows
// a best-first lattice of the drafter's candidates and grafts the lookup chain in as extra scored
// nodes, so a chain node at depth d scores d * log_probability() and a lattice node under a chain
// parent takes the better of its two scores. This tree has no candidate_selector_tree, no
// tree_parents, and no tree-aware GDN replay or target attention, so nothing calls this yet -- the
// same dead surface the pool had. It is landed ahead of that work deliberately, as the better
// proposer, rather than alongside a second one that would have to be withdrawn with it.

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// A proposal continues the context (whose last token is the round's anchor) with the tokens that
// followed the most recent earlier occurrence of its longest indexed suffix window. Past the end of
// the context the continuation repeats with the occurrence's period, which is what makes a verbatim
// block or an indentation run yield several tokens rather than one.
//
// Each window length keeps its own estimate of the probability that a proposed token is committed
// given that the previous one was, learned from how far earlier proposals matched, decayed so the
// estimate follows the output's current mode.
class PromptLookup {
public:
    // Longest first: a longer match is a stronger signal and is preferred when it hits.
    static constexpr std::array<std::uint32_t, 3> kWindows{8, 4, 2};

    // Writes up to out.size() tokens and returns how many. The context is indexed incrementally; one
    // that no longer extends the indexed tokens is indexed again from the start. A proposal is the
    // full out.size() on any match, so the caller bounds the width by passing the span it wants.
    std::uint32_t propose(std::span<const TokenId> context, std::span<TokenId> out);

    // Scores the last proposal against the round's committed tokens. A hit is the length of the
    // matching prefix, not a flag, which conditions the estimate far better than a binary does.
    void observe(std::span<const TokenId> committed);

    // Log-probability of the last proposal's window, 0 when there is no proposal.
    [[nodiscard]] float log_probability() const noexcept;

    // The window the last proposal came from, or -1 when there was none. Exposed for tests; the
    // consumer only needs log_probability().
    [[nodiscard]] int proposal_window() const noexcept { return proposal_window_; }

private:
    // Open-addressed map from a window's 64-bit hash to its latest end position. `ends` holds end + 1
    // so that 0 marks an empty slot and a position of 0 is representable.
    struct WindowTable {
        std::vector<std::uint64_t> keys;
        std::vector<std::uint32_t> ends;
        std::size_t size = 0;

        void insert(std::uint64_t key, std::uint32_t end);
        [[nodiscard]] std::uint32_t find(std::uint64_t key) const noexcept;
        void clear() noexcept;
    };

    struct Estimate {
        double hits   = 0.0;
        double misses = 0.0;
    };

    void index_until(std::span<const TokenId> context, std::size_t end);

    std::array<WindowTable, kWindows.size()> tables_;
    std::array<Estimate, kWindows.size()> estimates_;
    std::size_t indexed_  = 0;  // window end positions [0, indexed_) are in the tables
    TokenId indexed_last_ = 0;
    std::vector<TokenId> proposal_;
    int proposal_window_ = -1;
};

} // namespace ninfer::models::qwen3_5
