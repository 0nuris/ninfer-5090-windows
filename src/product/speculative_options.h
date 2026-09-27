#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// A copy-proposal round shares the target's verification window, which the engine provisions to
// kDFlashDecodeMaximumDrafts (15) columns; a configured width binds a prefix of it. Measured:
// planned device total and CUDA Graph allowance are identical for every neural draft width up to
// that bound, because the buffers are sized to the maximum and a width binds a prefix. Widening
// past it would mean growing the round constants and their pinned arrays, which is not done.
inline constexpr std::uint32_t kNgramMaximumDraftTokens = 15;

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    // Copy drafting supplements a neural drafter; it is not a drafter itself. This layer validates
    // the option's own domain. The interaction between the verify window and the neural depth is
    // checked in the model layer, which owns the wide-round margin that decides it.
    if (options.ngram.mode != NgramDraftMode::Off) {
        if (options.ngram.mode != NgramDraftMode::Chain) {
            throw std::invalid_argument("unknown n-gram draft mode");
        }
        if (options.backend != SpeculativeBackend::Mtp &&
            options.backend != SpeculativeBackend::DFlash &&
            options.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument("n-gram copy drafting requires --spec mtp|dflash|dflash2");
        }
        if (options.ngram.max_drafts == 0 || options.ngram.max_drafts > kNgramMaximumDraftTokens) {
            throw std::invalid_argument("n-gram verify window must be in [1,15]");
        }
        if (options.ngram.match_tokens == 0 || options.ngram.match_tokens > 64) {
            throw std::invalid_argument("n-gram match length must be in [1,64]");
        }
        if (options.ngram.min_drafts == 0 || options.ngram.min_drafts > options.ngram.max_drafts) {
            throw std::invalid_argument("n-gram minimum draft must be in [1,verify window]");
        }
        if (options.ngram.pool_bytes < sizeof(std::uint32_t) ||
            options.ngram.pool_bytes > (4ULL << 30U)) {
            throw std::invalid_argument("n-gram pool size must be in [4 B,4 GiB]");
        }
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
