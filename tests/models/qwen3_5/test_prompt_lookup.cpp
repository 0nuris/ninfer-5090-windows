// PromptLookup: the proposer ported from satellitedown/cinference (HEAD 1d07410), Apache-2.0.
// The first four cases are that project's own, kept so the port is checked against its semantics
// rather than only against ours. The rest pin the properties that make it preferable to the
// NgramDraftPool it replaces, which the pool's own test could not have covered.

#include "models/qwen3_5/program/speculative/prompt_lookup.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::PromptLookup;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<TokenId> propose(PromptLookup& lookup, const std::vector<TokenId>& context,
                             std::size_t count) {
    std::vector<TokenId> out(count, -1);
    out.resize(lookup.propose(context, out));
    return out;
}

} // namespace

int main() {
    {
        // The longest window's most recent earlier occurrence wins; the suffix itself never does.
        PromptLookup lookup;
        const std::vector<TokenId> context{3, 4, 7, 9, 1, 3, 4, 8, 5, 6, 3, 4};
        expect(propose(lookup, context, 3) == std::vector<TokenId>({8, 5, 6}),
               "latest earlier occurrence of the 2-token suffix");
        const std::vector<TokenId> longer{10, 11, 12, 13, 14, 15, 16, 17, 18, 50, 13, 14,
                                          15, 16, 17, 60, 10, 11, 12, 13, 14, 15, 16, 17};
        expect(propose(lookup, longer, 4) == std::vector<TokenId>({18, 50, 13, 14}),
               "8-token match over a more recent 4-token one");
    }
    {
        // Past the anchor the continuation repeats with the occurrence's period.
        PromptLookup lookup;
        const std::vector<TokenId> context{5, 6, 5, 6, 5, 6};
        expect(propose(lookup, context, 5) == std::vector<TokenId>({5, 6, 5, 6, 5}),
               "periodic continuation");
    }
    {
        // No earlier occurrence, then a context that no longer extends the indexed one.
        PromptLookup lookup;
        expect(propose(lookup, {1, 2, 3, 4}, 4).empty(), "no occurrence");
        expect(propose(lookup, {1, 2, 3, 4, 1, 2}, 2) == std::vector<TokenId>({3, 4}),
               "incremental extension");
        expect(propose(lookup, {7, 2, 9, 7, 2}, 2) == std::vector<TokenId>({9, 7}),
               "replaced context re-indexed");
    }
    {
        // Matching proposals raise the estimate; a first-token miss lowers it.
        PromptLookup lookup;
        const std::vector<TokenId> context{1, 2, 3, 4, 5, 1, 2};
        propose(lookup, context, 3);
        const float prior = lookup.log_probability();
        lookup.observe(std::vector<TokenId>{3, 4, 5, 9});
        propose(lookup, context, 3);
        const float after_hits = lookup.log_probability();
        lookup.observe(std::vector<TokenId>{8});
        lookup.observe(std::vector<TokenId>{8});
        propose(lookup, context, 3);
        lookup.observe(std::vector<TokenId>{8});
        propose(lookup, context, 3);
        const float after_miss = lookup.log_probability();
        expect(prior < 0.0F && after_hits > prior, "hits raise the estimate");
        expect(after_miss < after_hits, "misses lower the estimate");
    }
    {
        // A match fills the caller's span exactly. The caller bounds the width by passing the span
        // it wants, so a proposal that returned less would silently narrow the round instead of
        // truncating, and a proposal that returned more would overrun it.
        PromptLookup lookup;
        const std::vector<TokenId> context{9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
        for (std::size_t width = 1; width <= 8; ++width) {
            expect(propose(lookup, context, width).size() == width,
                   "a match fills the requested span exactly at width " + std::to_string(width));
        }
        // An empty span proposes nothing rather than reading uninitialised output.
        expect(propose(lookup, context, 0).empty(), "an empty span proposes nothing");
    }
    {
        // The estimate is a function of the window that produced the proposal, and windows are
        // stored apart: a window that has been trained reads differently from one that has not.
        //
        // Both lookups below see the same contexts and the same number of observations, and differ
        // only in whether those observations were hits or misses. That isolates the estimate from
        // the match that produced it.
        //
        // Note what is NOT tested here, having tried to: two windows trained inside one lookup.
        // PromptLookup tracks one forward-growing context per sequence -- propose() resets when
        // indexed_ > length, because a context that got shorter is not an extension of the indexed
        // one -- so alternating two lengths to hit different windows resets the estimates every
        // round and both readings sit at the prior. A test written that way passes for the wrong
        // reason, and did until the readings were printed.
        const std::vector<TokenId> context{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 3, 4, 5, 6, 7, 8, 1, 2};
        std::vector<TokenId> buf(3, -1);
        PromptLookup untrained;
        untrained.propose(context, buf);
        const float prior = untrained.log_probability();

        PromptLookup hits, misses;
        for (int i = 0; i < 6; ++i) {
            propose(hits, context, 3);
            hits.observe(std::vector<TokenId>{3, 4, 5});
            propose(misses, context, 3);
            misses.observe(std::vector<TokenId>{3, 4, 8});
        }
        propose(hits, context, 3);
        const float after_hits = hits.log_probability();
        propose(misses, context, 3);
        const float after_partial = misses.log_probability();
        // A partial hit still counts: observe() records the length of the matching prefix, so three
        // tokens matching and one not is better evidence than a bare miss and worse than a clean hit.
        expect(after_hits > prior, "hits lift the window above the prior");
        expect(after_partial < after_hits, "a partial match scores below a clean one");
        expect(after_partial != prior, "a partial match moves the estimate at all");
    }
    {
        // observe with no proposal pending is a no-op, not a write into a stale window's estimate.
        // The contract resets proposal_window_ after scoring, so a second observe cannot double-count.
        PromptLookup lookup;
        const std::vector<TokenId> context{4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        propose(lookup, context, 3);
        lookup.observe(std::vector<TokenId>{4, 4, 4});
        expect(lookup.proposal_window() == -1, "observe consumes the proposal");
        expect(lookup.log_probability() == 0.0F, "no proposal reports a zero log-probability");
        lookup.observe(std::vector<TokenId>{7, 7, 7});
        propose(lookup, context, 3);
        const float once = lookup.log_probability();
        propose(lookup, context, 3);
        expect(lookup.log_probability() == once,
               "a stray observe between proposals does not move the estimate");
    }
    {
        // A context that diverges from the indexed one re-indexes AND resets the estimates, so a
        // spliced or retried conversation does not carry a stale confidence into unrelated text.
        // "Reset" means the prior, not zero: a fresh estimate is (0+1)/(0+0+1+3) = 0.25, so the
        // reported log-probability is log(0.25) and only an absent proposal reports 0. The check is
        // therefore against a fresh lookup's reading of the same window, not against a constant.
        PromptLookup lookup;
        PromptLookup fresh;
        const std::vector<TokenId> first{6, 7, 8, 9, 6, 7, 8, 6, 7, 8};
        for (int i = 0; i < 6; ++i) {
            propose(lookup, first, 3);
            lookup.observe(std::vector<TokenId>{6, 7, 8});
        }
        propose(lookup, first, 3);
        const float learned = lookup.log_probability();
        expect(learned < 0.0F, "the estimate was learned before the splice");

        const std::vector<TokenId> spliced{1, 2, 3, 1, 2, 3};
        std::vector<TokenId> buf(3, -1);
        lookup.propose(spliced, buf);
        fresh.propose(spliced, buf);
        expect(lookup.log_probability() == fresh.log_probability(),
               "a diverged context resets to the prior rather than carrying a stale estimate");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " prompt_lookup\n";
    return failures == 0 ? 0 : 1;
}
