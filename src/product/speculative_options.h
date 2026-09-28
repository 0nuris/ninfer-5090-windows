#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// A copy-proposal round runs at the target's own verification width, which the engine provisions to
// kDFlashDecodeMaximumDrafts (15) columns. A copy round shares that width rather than widening
// it, so a copy proposal longer than the round is truncated rather than served at a wider one.
// Round cost and acceptance both vary with the draft width, and not smoothly: see
// ngram_selection.h, which records the measurements.
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
    // the option's own domain. The model layer owns the policy that decides whether a copy is worth
    // a round, and that policy reads no width this layer has to keep in step with.
    if (options.ngram.mode != NgramDraftMode::Off) {
        if (options.ngram.mode != NgramDraftMode::Chain) {
            throw std::invalid_argument("unknown n-gram draft mode");
        }
        // Only the masked-draft route is wired. The MTP route proposes on the host, so it could
        // chain onto a proposal the way the source design does, but that path is not built here and
        // accepting the flag would advertise a combination that does nothing.
        if (options.backend != SpeculativeBackend::DFlash &&
            options.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument("n-gram copy drafting requires --spec dflash|dflash2");
        }
        if (options.ngram.max_drafts == 0 || options.ngram.max_drafts > kNgramMaximumDraftTokens) {
            throw std::invalid_argument("n-gram copy width must be in [1,15]");
        }
        if (options.ngram.min_drafts == 0 || options.ngram.min_drafts > options.ngram.max_drafts) {
            throw std::invalid_argument("n-gram minimum copy must be in [1,copy width]");
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
