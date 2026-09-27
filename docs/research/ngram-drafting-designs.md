# N-gram / prompt-lookup speculative drafting in production engines

Research note. Written 2026-09-26 against upstream `master`/`main` of each engine. Every factual
claim below is followed by the file and line range that owns it, or by the issue/PR URL. Claims
that come from a third party rather than a maintainer are labelled **[community]** or
**[vendor]**; claims I could not source are marked **[no evidence found]**.

Base URLs used for the `file:line` citations:

| Short name | Repository | Raw base |
|---|---|---|
| `llama.cpp` | `ggml-org/llama.cpp` | `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/` |
| `vllm` | `vllm-project/vllm` | `https://raw.githubusercontent.com/vllm-project/vllm/main/` |
| `sglang` | `sgl-project/sglang` | `https://raw.githubusercontent.com/sgl-project/sglang/main/` |
| `trtllm` | `NVIDIA/TensorRT-LLM` | `https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/` |

## 0. What exists, in one table

Three families of implementation are in production code. They differ in **where the pattern
index lives**, **how long a proposal is**, and **what the verify step runs**.

| | Index | Proposal length | Verify width | Shared across requests |
|---|---|---|---|---|
| llama.cpp `ngram-mod` | flat `int32` hash pool, 2²² slots, n-gram → next token | **variable**, `n_min`..`n_max` (or none) | **variable**: batch is `1 + len(draft)`, unpadded | yes, one 16 MiB pool for all slots |
| vLLM `ngram` / `ngram_gpu` | **none**; a CPU/GPU mirror of the token history, rescanned per step | **fixed** `k = num_speculative_tokens` | fixed `1 + k` rows, remainder masked | no (a global-scope PR is open) |
| SGLang `NGRAM` | C++/CUDA corpus (trie + suffix automaton), capacity-bounded | **fixed node budget** `draft_token_num`, shaped as a *tree* | fixed `draft_token_num` rows + a tree mask | yes, opt-in external corpora |
| TensorRT-LLM `NGram` | `dict[pattern_tuple, OrderedSet[match_tuple]]` | **variable** = stored match length, capped by `max_draft_len` | `batch × (max_draft_len + 1)` | yes, `is_public_pool=True` by default |

Sources for each row are in §1–§3.

Two of the four run the drafter on the host and two keep a persistent structure; that is the axis
that matters for an engine with 2.1–2.9 GiB of free VRAM, because only SGLang's design puts the
index on the device.

---

## 1. llama.cpp `ngram-mod`

### 1.1 The data structure

`common/ngram-mod.h:13-38` declares `common_ngram_mod`: a `std::vector<int32_t> entries`, a
`uint16_t n`, and a `used` counter. `common/ngram-mod.cpp`:

- `idx()` (`:15-25`) folds the `n` tokens with an LCG-style multiplier
  (`res = res*6364136223846793005ULL + tokens[i]`) and reduces modulo the table size.
- `add()` (`:27-35`) stores **only the next token**: `entries[i] = tokens[n]`.
- `get()` (`:37-41`) returns the stored token or `EMPTY` (`-1`).

So the map is *n-gram → next token*, a first-order successor, not a stored continuation. The docs
say the same thing in prose (`docs/speculative.md:203`): "The hash pool is a map from n-gram hash
to the next token (not the next m-gram as in ngram-map)."

There is **no collision check**. A hash collision silently returns a token that never followed
that n-gram. That is a throughput defect, not a correctness one — the token is only a proposal and
the target model verifies it (§6.1).

Size is fixed at construction: `mod(params.ngram_mod.n_match, 4*1024*1024)`
(`common/speculative.cpp:1894`) = 4,194,304 × 4 B = **16 MiB**, matching the docs' "~16 MB"
(`docs/speculative.md:172`). A real server log line in PR #19164 confirms the table size:
`begin: ngram_mod occupancy = 1519/4194304 (0.00)`.

### 1.2 Is the hash pool shared across slots?

Yes, and the code says so explicitly. `common/speculative.cpp:1870-1871`:

```cpp
// shared across all sequences
common_ngram_mod mod;
```

Only three per-sequence values exist (`seq_info`, `:1876-1887`): `i_last` (how far the prompt was
indexed), `n_draft_last`, and `n_low` (the low-acceptance streak). The pool itself is a single
member of the implementation object, and `begin()` (`:1912-1938`) inserts **every** sequence's
whole prompt into it. The docs confirm the intent: "a single hash pool is shared across all server
slots, so different requests can benefit from each other" (`docs/speculative.md:176`), and
TensorRT-LLM's blog reports the same benefit being worth up to 70 % latency reduction on
translation traffic ([vendor], §5.3).

Consequence worth stating: the two reset paths are **global**, not per-request. Occupancy above
0.25 at `begin()` resets the pool for everybody (`:1929-1937`), and five consecutive rounds with
acceptance fraction < 0.25 from *any* sequence reset it too (`:2014-2038`).

### 1.3 Verifying the two doc claims

**"Constant memory and complexity"** (`docs/speculative.md:173`, restated in PR #19164's body) —
**half true, and the distinction matters**:

- *Memory*: yes, literally constant. A fixed 16 MiB table regardless of context length, request
  count, or number of slots.
- *Complexity*: constant **only on the steady-state per-token path**. `draft_one`
  (`:1957-1964`) adds just the n-grams since `sinfo.i_last` in one pass, so indexing is amortised
  O(1) per generated token. But `begin()` (`:1923-1925`) loops over the whole prompt —
  `for (size_t i = 0; i < prompt.size() - n; ++i) mod.add(prompt.data() + i);` — which is
  O(prompt length) **per request**, on the request-admission path. A 100 k-token prompt is 100 k
  hash insertions before the first draft is possible.

**"Can generate variable draft lengths (i.e. m is not fixed)"** (`docs/speculative.md:174`) —
**verified true**. `draft_one` (`:1966-1990`):

```cpp
result.resize(n + params.n_max);
... seed the first n entries with the last n tokens ...
for (int i = 0; i < params.n_max; ++i) {
    const llama_token token = mod.get(result.data() + i);
    if (token == common_ngram_mod::EMPTY) {
        if (i < params.n_min) { result.clear(); return; }   // too short: no draft at all
        result.resize(n + i);
        break;
    }
    result[n + i] = token;
}
```

The chain walk stops at the first empty slot, and a break before `n_min` suppresses the draft
entirely. The emitted length is therefore in `[n_min, n_max]`, or empty. `m` is not fixed; `n`
(the match length) is fixed by `n_match`.

### 1.4 The real defaults

`common/common.h:353-357`:

```cpp
struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;
    int32_t n_max = 64;
    int32_t n_min = 48;
};
```

CLI flags are `--spec-ngram-mod-n-match`, `--spec-ngram-mod-n-min`, `--spec-ngram-mod-n-max`
(`common/arg.cpp:4255-4283`, each validated to 1..1024). `--spec-default` pushes
`ngram-mod` with exactly these three values (`common/arg.cpp`, `--spec-default` handler), carrying
the maintainer's own doubt in a comment: "TODO: not sure if this is a good config - explore more
settings and potentially enable it". `docs/speculative.md:317-321` documents the same default (24)
for `n_match`.

So the shipped default is a **very long** draft: min 48, max 64 tokens, matched on 24 tokens.

### 1.5 Does ngram drafting interact with CUDA graph capture?

llama.cpp does use CUDA graphs, and ngram-mod touches that path only through tensor shapes — but
the shape path is load-bearing.

The graph layer is `ggml/src/ggml-cuda/ggml-cuda.cu`:

- One captured instance per first-node pointer, in an `unordered_map`
  (`ggml/src/ggml-cuda/common.cuh:1461`, accessor `:1465-1486`), with a sweep that evicts entries
  unused for ≥ 10 s (`:1468-1478`).
- A graph is used **only after two consecutive calls with unchanged node properties**
  (`ggml-cuda.cu:4456-4475`). If properties changed, `warmup_complete` is reset and that call
  *executes directly, outside the graph* (`:4467-4471`).
- "Properties" include every node's and every source's `data`, `ne` and `nb`
  (`:2613-2629`). A change in any shape therefore counts as a property change.
- On `cudaGraphExecUpdate` failure the instance is destroyed and re-instantiated
  (`:2646-2656`).

The server's verify batch is **exactly** `1 + spec_draft.size()` tokens per slot, with no padding
(`tools/server/server-context.cpp:526-535`):

```cpp
spec_i_batch.push_back(batch.size());
for (size_t i = 0; i < spec_draft.size(); i++) { spec_i_batch.push_back(batch.size() + i + 1); }
add_ok &= batch.add(id, sampled, pos0++, true, false);
for (auto token : spec_draft) { add_ok &= batch.add(this->id, token, pos0++, true, false); }
```

So a variable-length drafter changes `ne` on every matmul from step to step, which trips the
warmup gate, which drops the step out of the graph. There is no ngram-specific code in the graph
layer; the interaction is entirely indirect, and it runs in the *unfavourable* direction.

Two second-order costs are visible in the code:

- The per-sequence output budget is `1 + n_draft` rows and `n_parallel × (1 + n_draft)` in total
  (`common/speculative.cpp:2604-2613`), consumed by the server at
  `tools/server/server-context.cpp:44-48`. With the shipped `n_max = 64` and 8 slots that is 520
  output rows.
- `n_draft_max` is additionally clamped per request by remaining context
  (`tools/server/server-context.cpp:483-502`), so the width is not even stable across the life of
  one request.

### 1.6 How ngram-mod combines with a neural drafter in llama.cpp

By **priority with fallback**, not by addition. `common_speculative_init` builds the
implementation list in a fixed priority order with all ngram types first
(`common/speculative.cpp:2629-2645`):

```cpp
// this list here defines the priority of the speculators
// the one with highest priority are listed first
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    params.draft.ctx_dft != nullptr);
...
```

and `common_speculative_draft` stops at the first implementation that returns something
(`:2844-2848`):

```cpp
if (dp.drafting && !result.empty()) { dp.drafting = false; ... }
```

`dp.drafting`'s own doc comment states the contract (`common/speculative.h:55-58`): "after the
first successful draft from an implementation, we set it to false to prevent further drafts for
that sequence."

So a sequence that gets an ngram proposal **does not run the neural drafter that round**. The
neural drafter is the fallback for rounds where ngram misses. `docs/speculative.md:207` restates
it: "If a draft model is combined with a draftless decoding the draftless decoding has higher
precedence."

---

## 2. vLLM `ngram`

### 2.1 The implementation

`vllm/v1/spec_decode/ngram_proposer.py` keeps **no index at all**. It keeps a CPU mirror of every
request's token history — `token_ids_cpu` of shape `(batch_size, max_model_len)`, `int32` — and
rescans it on every proposal step.

The scan is a KMP/longest-prefix-suffix pass over the **reversed** token array
(`ngram_proposer.py:209-295`): the longest match whose length lies in `[min_ngram, max_ngram]` is
found, ties resolved toward the *earliest* position in the original array (`:260-262`, `:289-293`),
and the `k` tokens following it are returned (`:293-295`).

`min_n = prompt_lookup_min`, `max_n = prompt_lookup_max`, `k = num_speculative_tokens`
(`:18-25`). So:

- the **match length** is variable within `[prompt_lookup_min, prompt_lookup_max]`;
- the **proposal length is fixed** at `k` (truncated only by the end of the context, `:228`, `:294`);
- no match at or above `min_ngram` ⇒ no proposal (`:285-287`).

The whole batch runs through `@njit(parallel=True)` (`:180-206`), with the numba thread count
capped at 1 by construction: `self.num_numba_thread_available = min(1, (cpu_count // 2))`
(`:48`), carrying the note "TODO(ekagra-ranjan): bump up the cap from 1 to 8". The 8192-token
threshold at `:36`/`:105` gates even that.

A second, GPU implementation exists: `vllm/v1/spec_decode/ngram_proposer_gpu.py`, selected by
`SpeculativeConfig.use_ngram_gpu()` (`vllm/config/speculative.py`, `method == "ngram_gpu"`). It is
a `@support_torch_compile` module using `unfold` + `argmax` over all sequences, and it needs a
device→host copy of the per-request valid draft count
(`vllm/v1/worker/gpu_model_runner.py:905-918`).

### 2.2 `prompt_lookup_min` / `prompt_lookup_max`: the real defaults

The field defaults are `None` (`vllm/config/speculative.py:471`, `:474`), and the docstring for
`prompt_lookup_min` claims "if provided. Defaults to 1" — which is **wrong**. The actual
resolution is at `vllm/config/speculative.py:1191-1214`:

| input | resolved |
|---|---|
| neither set | `min = 5`, `max = 5`, under `# TODO(woosuk): Tune these values. They are arbitrarily chosen.` |
| only `max` | `min = max` |
| only `min` | `max = min` |
| neither or inconsistent | `ValueError` (`min > max` rejected at `:1211-1214`) |

The docstring mismatch is being fixed in open PR #56517
(`[Doc] Fix prompt_lookup_min/max defaults in SpeculativeConfig docstrings`,
https://github.com/vllm-project/vllm/pull/56517). The documented example in
`docs/features/speculative_decoding/n_gram.md:15-19` sets only `prompt_lookup_max: 4`, which under
the code above means `min = max = 4` — not `min = 1`.

### 2.3 Interaction with CUDA graph capture

vLLM captures one graph per size and states the rule in its own config docstring
(`vllm/config/compilation.py`): "cudagraph: a cudagraph captured for a specific size can only be
used for the same size. We need to capture all the sizes we want to use."

The speculative decode step is a *different size* from plain decode, and its size is fixed:

- `uniform_decode_query_len = 1 + num_speculative_tokens`
  (`vllm/v1/cudagraph_dispatcher.py:36`, mirrored in
  `vllm/v1/worker/gpu_model_runner.py:883`).
- At dispatch, `num_reqs = num_tokens_padded // uniform_decode_query_len` with
  `assert num_tokens_padded % uniform_decode_query_len == 0`
  (`vllm/v1/cudagraph_dispatcher.py:138-149`).
- `adjust_cudagraph_sizes_for_spec_decode` rounds every capture size **up** to a multiple of
  `1 + k` (`vllm/config/compilation.py`, same function).

**Answer to "does a wider ngram draft force extra graph variants?" — no, and that is the point of
the design.** The set of captured sizes is not re-expanded; each existing size is rounded up to a
multiple of `1 + k`, so a wider draft makes each captured graph cover proportionally more rows
(more activation memory per variant) and, because the sizes are de-duplicated by `set()`, small
sizes below `1 + k` collapse into one. The *number of tokens actually proposed* never enters into
it: the verify forward always runs `1 + k` rows per request and the per-request valid count masks
the remainder. That is visible in the buffer shapes —
`torch.zeros((max_num_seqs, self.k))` plus a parallel
`valid_ngram_num_drafts` array (`ngram_proposer.py:31-32`), and the GPU path's
`_num_valid_draft_tokens` D2H buffer (`gpu_model_runner.py:905-918`).

Two related constraints from the same source:

- If the attention backend does not advertise `UNIFORM_BATCH` support, full cudagraphs are
  **downgraded to piecewise or none** when spec decode is on
  (`vllm/config/compilation.py`, `uniform_decode_query_len > 1` branch).
- Variable-length verification is a separate, later feature that needs more than a mask:
  `docs/features/speculative_decoding/adaptive_verification.md:37-38` requires device-decided query
  lengths and states "Full cudagraphs are required"; it is supported "today … only … for DSpark
  with a confidence head" (`:15`), and `lilicorr.md` says it "requires … variable-length CUDA
  graph support". In other words: **within vLLM, a drafter that cannot report a per-position
  confidence cannot drive variable-length verification.**

### 2.4 Reported speedups and acceptance rates from vLLM

vLLM's own n-gram documentation reports **no numbers at all**. `n_gram.md` is a 27-line config
example whose only external reference is a tweet
(`https://x.com/joao_gante/status/1747322413006643259`, `n_gram.md:4`). No acceptance rate, no
speedup, no benchmark command.

The numbers that exist in the vLLM tracker are in PR #24344, run by its author — see §4.3 for the
table and for the fact that the PR is still open.

The one maintainer statement about ngram's cost is in issue #22408 (§4.2).

---

## 3. SGLang and TensorRT-LLM

### 3.1 SGLang `NGRAM`

SGLang's implementation is the most elaborate of the four, and the only one that puts the index on
the device.

- The index is a C++/CUDA **corpus** — a trie plus, in the newer work, a suffix automaton —
  reached through `self.ngram_corpus.batch_get(req_ids, batch_tokens, total_lens)`
  (`python/sglang/srt/speculative/ngram_worker.py:300-302`).
- Drafts are **trees, not chains**. `ngram_corpus.batch_get` returns a mask, and the worker
  reconstructs positions and a tree attention mask with
  `reconstruct_indices_from_tree_mask(...)` (`:346-355`) and builds a per-request mask of shape
  `(draft_token_num, seq_len + draft_token_num)` (`:330-337`, `:359-374`).
- The node count is **fixed and asserted**: `assert total_draft_token_num == bs *
  self.draft_token_num` (`:306-308`). The verify forward is therefore always
  `draft_token_num` rows per request (`batch.forward_mode = ForwardMode.TARGET_VERIFY`,
  `batch.input_ids = draft_tokens`, `:376-386`), and the graph shape does not depend on how much of
  the tree matched.
- Matching only looks at the last `max_trie_depth` tokens of input+output
  (`:254-261`, `:293-297`).
- Defaults (`python/sglang/srt/arg_groups/fields/spec.py:230-261`):
  `speculative_ngram_min_bfs_breadth = 1`, `max_bfs_breadth = 10`, `match_type = "BFS"`,
  `max_trie_depth = 18`, `capacity = 10_000_000`, `external_sam_budget = 0`,
  `external_corpus_max_tokens = 10_000_000`.
- **Cross-request corpora** are a first-class feature: `speculative_ngram_external_corpus_path`
  pre-loads a JSONL corpus at startup and `POST /add_external_corpus` adds more at runtime
  (`arg_groups/fields/spec.py:250-252`; worker methods `add_external_corpus`,
  `remove_external_corpus`, `list_external_corpora`, `ngram_worker.py:161-171`).
- Tree depth is node-budgeted, not depth-capped: "NGRAM trees are node-budgeted with no depth cap:
  the corpus BFS only stops on the node budget" (`ngram_info.py:49-52`).
- One in-tree performance claim, from a source comment rather than a benchmark:
  "QLEN_MASK is faster than FULL_MASK … Testing shows about 8% performance improvement (the effect
  is roughly proportional to batch size)" (`ngram_worker.py:357-358`).

SGLang ships exactly one speculation algorithm at a time: `speculative_algorithm` is an enum of
`EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO`
(`arg_groups/fields/spec.py:30-32`). A hybrid is an open PR (§4.4).

### 3.2 TensorRT-LLM `NGram`

`tensorrt_llm/_torch/speculative/ngram.py`, class `NGramPoolManager`:

- The pool is `dict[pattern_tuple, OrderedSet[match_tuple]]` — **multi-candidate**: one pattern
  maps to several stored continuations (`ngram.py:43-45`, `:16-24`). `is_keep_all` controls
  whether more than one is retained (`:34-35`).
- `is_use_oldest` chooses the oldest or newest match (`:36-37`, `:132`).
- `is_public_pool` chooses one pool for all requests or one per request (`:39-40`, `:72-74`).
  The public pool is never pruned: `update_resources` returns immediately with
  `# TODO: Here should be an strategy to update the pool in public pool mode.` (`:72-74`).
- The pool is updated **incrementally**: `start_index[request_id]` advances so only
  `max_total_draft_tokens + max_matching_ngram_size - 1` trailing tokens are re-indexed
  (`:103`, `:137-139`).
- Pattern sizes are tried from `max_matching_ngram_size` down to 1, first hit wins
  (`:127-134`); the proposal is the stored match, truncated to the remaining context
  (`:91-93`, `:133`). So the **proposal length is variable**, bounded by
  `max_total_draft_tokens = tokens_per_gen_step - 1` (`:53`).
- `NGramDrafter` asserts `"NGram only supports linear tree."` (`:180`) — chains only, no tree.

Configuration surface and defaults are in `examples/ngram/README.md:15-26`:
`max_draft_len` default 4 ("usually from 4 to 10"), `max_matching_ngram_size` default 2, with the
trade-off spelled out: a larger value "indicat[es] higher acceptance rate, but the higher
probability of miss-match and higher overhead appear, which fall back to normal generation".

`decoding_type` is a single choice — `MTP`, `Eagle3`, `NGram`, `DraftTarget`, `PARD`, `DFlash`,
`SA` (`docs/source/features/speculative-decoding.md:228-236`). `SA` (suffix automaton,
`tensorrt_llm/_torch/speculative/sa_worker.py`) is a separate option from `NGram`, not a
combination with it.

**TensorRT-LLM's auto-enable heuristic is the most useful single piece of guidance here**, and it
is code, not prose (`tensorrt_llm/_torch/speculative/auto_heuristic.py:1-17`):

```python
def suggest_spec_config(max_batch_size: int) -> "DecodingBaseConfig":
    """Suggests a reasonable draft model free speculation scheme.
    Used when the user specifies spec_mode == AUTO.
    For now, we always use an ngram scheme that gets disabled at BS>=32.
    """
    return NGramDecodingConfig(
        max_draft_len=5 if max_batch_size <= 4 else 3,
        max_matching_ngram_size=3 if max_batch_size <= 4 else 5,
        max_concurrency=32, is_keep_all=True, is_use_oldest=True, is_public_pool=True,
    )
```

A vendor shipping ngram as the *default* draft-model-free scheme still switches it **off at
batch size 32**, and halves the draft width above batch size 4.

---

## 4. The combination question: ngram drafting **on top of** a neural drafter

This is the part with the most measurements and the least consensus. Three independent attempts
exist; **none has shipped**.

### 4.1 llama.cpp issue #23184 — the claim is unverified, and its premise is wrong

https://github.com/ggml-org/llama.cpp/issues/23184, "Feature Request: Pipeline speculative
decoding strategies (draft-mtp → ngram-mod)", by **ElSnacko**, opened 2026-05-17, closed
2026-07-12 as stale.

The claim under test, quoted: "`draft-mtp` alone achieves ~78% acceptance rate with `n_max=3`
(tested on Qwen3.6-35B-A3B-MTP, Vulkan 780M iGPU). Adding `ngram-mod` independently on top
provides **no speedup**, only verification overhead."

Findings:

1. **Not maintainer-verified.** The thread has exactly one human reply, from llama.cpp maintainer
   **feffy380**: "Would it kill you to proofread your slop before posting it?" The second comment is
   the stale-bot. No measurement was attached by the reporter and none was requested and produced.
2. **The premise contradicts the code, and did so at the time.** The issue asserts "the two
   speculative decoding strategies run independently and generate separate draft token streams.
   They do not share context". The chaining and the ngram-first priority order were already in
   `common/speculative.cpp` before the issue was filed. I fetched the file at the last commit
   touching it before 2026-05-17 — `255582687b8dd211fdbc582e43ab842491554e94`, "llama + spec: MTP
   Support (#22673)", 2026-05-16 — and the priority list there is
   `ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, draft-simple, draft-eagle3,
   draft-mtp` (lines 1238-1275), with the first-successor-wins `dp.drafting = false` chaining at
   lines 1450 and 1487. Requesting a pipeline in which "ngram-mod takes those MTP-predicted tokens
   as additional context and extends the draft sequence further" is therefore asking for
   *concatenation*, not for the chaining that already exists — but the issue's stated diagnosis
   ("run independently") does not describe the shipped behaviour.
3. **The 78 % acceptance figure is a single unpublished run** on a Vulkan iGPU with a Qwen3.6-35B
   MTP build. It is not reproducible from the thread.
4. **No counter-result.** I searched the llama.cpp tracker for `ngram-mod` (40 most recent hits)
   and read the threads that touch the combination. Nothing measures ngram-mod stacked on MTP or
   EAGLE and reports a different outcome. The only related crash report is
   [#23154](https://github.com/ggml-org/llama.cpp/issues/23154) ("Eval bug: CUDA ERROR crash when
   using MTP ngram-mod", `-mt 3`, `--spec-type ngram-mod,draft-mtp`, RTX 4090, Qwen3.6-27B-MTP
   Q4_K_XL), where the reporter shows GPU memory climbing from 79 % to 100 % during generation
   with `ngram-mod` on and flat at 76 % with it off, and the crash lands in
   `cudaGraphInstantiate(&graph->instance, ...)` inside `ggml_cuda_graph_update_executable`. That
   is the re-instantiation branch at `ggml-cuda.cu:2646-2656`, i.e. the shape-change path from
   §1.5. The reporter's own diagnosis ("It looks like there is a memory leak on GPU side") is
   **[community]**; a maintainer asked for memory numbers, the reporter supplied screenshots, and
   the issue was then closed as stale with no diagnosis. Two related combination reports are
   [#23929](https://github.com/ggml-org/llama.cpp/issues/23929) (`-sm tensor + MTP + ngram-mod =
   crash`) and [#27839](https://github.com/ggml-org/llama.cpp/issues/27839) (combined
   `draft-dflash,draft-mtp,ngram-mod` with `-md` fails at init), both open, neither diagnosed.

**Verdict on #23184:** the "no speedup, only verification overhead" statement is an unreplicated
community claim resting on a premise the code contradicts. It should not be cited as a finding.
What the tracker *does* contain is an open, undiagnosed OOM crash in the combination.

### 4.2 vLLM issue #22408 — the 5-8 % figure is real and it was acted on

https://github.com/vllm-project/vllm/issues/22408, "[Performance]: n-gram speculative decoding
drafting optimizations", by **Jialin**, opened 2025-08-06, auto-closed as stale 2025-12-20.

The figure, quoted: "n-gram speculative decoding drafting consumes 5-8% critical path time for
small models (e.g. opt-125M), and 0.9% for large models (e.g. Llama4 Scout 17B). And the cost
would increase linearly when we increase the ngram range".

- **Verified as a maintainer-side measurement** in the sense that it was filed by a vLLM team
  member against their own profiling and shipped as PR
  [#22390](https://github.com/vllm-project/vllm/pull/22390) ("could reduce n-gram speculative
  decoding drafting cost by 38+%").
- **Acted on, and the result is in the code I read.** The two changes named in the issue are both
  present in current `main`: the O(ngram_range × total_tokens) scan became a single O(total_tokens)
  KMP/LPS pass (`vllm/v1/spec_decode/ngram_proposer.py:209-295`), and the batch drafting moved
  into numba (`:180-206`). So the 5-8 % figure describes the *pre*-PR implementation.
- The issue's own follow-up ("replace KMP with Rabin-Karp or SuffixAutomaton, which should
  preprocess … and overlap with prefill (so it's not on critical path)") is **not** done. A
  non-affiliated commenter (Thisish3, 2026-09-20) reports a Counting-Bloom-Filter incremental
  index beating repeated full KMP rescans by "~9-10x" cumulative over a 131 072-token synthetic
  stream, while conceding it loses on a single cold-start call, that replicating the
  longest-match-across-a-range property "would eat into the measured gain", and that it needs
  "~1MB of persistent per-request state". **[community]**, self-described as "not a vLLM PR", with
  code at a third-party repository.

### 4.3 vLLM PR #24344 — the one head-to-head measurement of ngram + EAGLE

https://github.com/vllm-project/vllm/pull/24344, "[Spec Decode][Hybrid] Add ngram-eagle SD
method", by **ekagra-ranjan**, opened 2025-09-05, **still open** as of 2026-09-26 (last activity
2026-04-09, a mergify conflict notice). Its motivating RFC,
[#18633](https://github.com/vllm-project/vllm/issues/18633), was closed as *not planned* (stale).

Design: **selection, not concatenation.** From the RFC body:

> For any given timestep and for a given seq in a batch: find if there any ngram match — If yes,
> then we let these be the draft token and skip EAGLE for this seq; If no, then we put them to a
> batch to be processed by EAGLE.

The maintainer objection and the answer are both on the record. **benchislett**:
"Won't this require some special management of the EAGLE kv-cache for sequences that use Eagle, but
skip it for some steps where ngram is used instead?" **ekagra-ranjan** replies by keeping the
drafter's prefill running for all sequences and only gating its decode, so "the total overhead of
`ngram-eagle` will be close to `eagle`". **renjie0** asked the useful question — "Why are they
exclusive? Why not combine the proposed sequences from both ngram and eagle and verify in the same
[pass]?" — and got three reasons:

> 1. combining proposal from both techniques requires tree attention which is not available
> 2. combining them will increase the number of draft which need to verify at each step
> 3. a hypothesis that seq that have ngram matches will benefit more from ngram rather than eagle

Point 1 is the load-bearing one for an engine design decision, and it is corroborated
independently by SGLang, which is the one engine that *does* have the tree attention vLLM lacked
(§3.1).

Measured results, from the PR body. Model `meta-llama/Llama-3.1-8B-Instruct`, TP 1,
`--max-concurrency 1`, median TPOT in ms (lower is better):

| workload | vanilla | eagle (K=3) | ngram (K=5, lookup 2..5) | ngram-eagle (ngram 5, eagle 3, min 5) |
|---|---:|---:|---:|---:|
| MTBench (chat) | 7.00 | **4.19** | 6.18 | 4.30 |
| Blazedit (edit, ≤25 % of output changed) | 6.95 | 3.96 | **1.90** | 2.13 |
| InstructCoder | 6.98 | 3.41 | 3.32 | **2.96** |

Acceptance lengths from the same PR, on the offline harness, using its "sequence-normalised" metric
(defined in the PR body as `total_tokens / (num_drafts + tokens_generated_without_sd)`, intended to
predict end-to-end speedup rather than per-draft precision):

| workload | eagle | ngram | ngram-eagle |
|---|---:|---:|---:|
| Blazedit (edit distance 0.1–0.25) | 2.46 | 2.96 | **4.04** |
| MTBench | **2.29** | 1.18 | 2.22 |
| InstructCoder | 2.84 | 1.95 | **3.30** |

The honest reading of that table, which the PR's own analysis supports:

- On **edit/copy-heavy** work, combining is *worse than ngram alone* on TPOT (2.13 vs 1.90) even
  though its acceptance length is higher (4.04 vs 2.96). The drafter still has to run.
- On **code** work, combining wins (2.96 vs 3.32 ngram, 3.41 eagle) — this is the case the design
  predicts, where the two drafters cover complementary regions.
- On **chat**, ngram alone is nearly worthless (6.18 against 7.00 vanilla, ~12 %) and combining
  merely returns to eagle's number (4.30 vs 4.19).
- The PR's mechanism claim is consistent: "when input is equal to output: it uses ngram draft …
  when input is not same as output: it uses eagle draft".

Concurrency: **PramuPerera** (vLLM) benchmarked the branch on H200 and reported "performance gain
of ngram-eagle drops with higher batch size (concurrency)", suspecting ngram's own weakening at
batch size and linking [#26595](https://github.com/vllm-project/vllm/issues/26595).
**ekagra-ranjan**: "SD in general doesnt hold good for very high concurrency and is not a
byproduct of this method."

### 4.4 SGLang PR #37237 — batch-level routing, with negative results at some batch sizes

https://github.com/sgl-project/sglang/pull/37237, "[Feature] Batch-level hybrid speculative
decoding", by **Jin-Chuan**, opened 2026-08-31, **open**, all three CI states red.

Design: a `HYBRID` algorithm with a controller that routes **per decode batch** between a retrieval
worker and a neural worker. From the PR body:

- Routing is a two-level threshold. Batch level: retrieval is chosen only if qualifying requests are
  at least `bs * min_matching_ratio`. Request level: a request qualifies when its retrieval draft
  quality score reaches `min_continuation_ratio` of the retrieval draft, where the score weights
  the actual draft length by the suffix-match length and normalises by the draft width.
- **Both workers' CUDA graphs and attention backends are captured at init and switched on routing
  change**: "CUDA graph runners / attention backends of both workers are captured at init and
  switched on routing change". Both states are kept live: "The suffix tree and the EAGLE draft KV
  are updated with accepted tokens every step, so either worker can be picked in the next step."
- The ngram worker is extended so the prompt is inserted into the trie (drafts can match the input)
  and so it returns the suffix-match length the router consumes.

Measured, H20-141GB, Qwen3.5-35B-A3B, output tok/s with mean acceptance length in parentheses.
Retrieval = NGRAM with 6 draft tokens, breadth 1, `match_type=PROB`; neural = EAGLE3 with 4 draft
tokens, topk 1, 3 steps; `min_continuation_ratio: 1`, `min_matching_ratio: 0.5`:

| workload | bs | hybrid | eagle3 | ngram | autoregressive | hybrid vs best single |
|---|---:|---:|---:|---:|---:|---:|
| AdaLEval-textsort | 1 | 562.01 (4.74) | 481.99 (3.48) | 561.03 (4.70) | 254.09 | +0.2 % |
| | 2 | 852.10 (5.03) | 716.22 (3.48) | 812.43 (4.50) | 422.78 | **+4.9 %** |
| | 4 | 1163.54 (4.43) | 1057.47 (3.49) | 1217.95 (4.75) | 667.24 | **−4.5 %** |
| | 8 | 1538.45 (4.46) | 1401.33 (3.48) | 1412.64 (4.20) | 912.50 | **+8.9 %** |
| InstructCoder | 1 | 394.95 (2.96) | 412.55 (2.74) | 250.30 (1.87) | 261.41 | −4.3 % |
| | 2 | 584.86 (2.78) | 636.40 (2.73) | 393.89 (1.86) | 441.42 | −8.1 % |
| | 4 | 915.52 (2.77) | 957.97 (2.73) | 613.72 (1.88) | 708.32 | −4.4 % |
| | 8 | 1261.14 (2.76) | 1301.22 (2.74) | 798.15 (1.83) | 934.58 | −3.1 % |
| GSM8K | 1 | 387.25 (2.89) | 462.72 (3.07) | 235.06 (1.80) | 263.79 | −16.3 % |
| | 2 | 569.89 (2.72) | 702.71 (3.05) | 368.37 (1.79) | 444.06 | −18.9 % |
| | 4 | 941.57 (2.87) | 1064.67 (3.06) | 578.10 (1.80) | 714.61 | −11.6 % |
| | 8 | 1412.35 (2.98) | 1484.08 (3.06) | 774.10 (1.77) | 965.44 | −4.8 % |

The pattern across both combination attempts is the same and is the most transferable finding here:
**stacking retrieval drafting on a neural drafter is at best a small win on repetition-heavy work,
and a measurable loss everywhere else**, because the neural drafter's cost is still paid and the
router is imperfect. On this PR's own numbers, hybrid loses to EAGLE3 alone by 3–19 % on code and
maths, and one of four repetition-heavy points is also negative.

One unexplained item in that PR, reported as-is: "On GSM8K … hybrid reaches 86.0% accuracy vs 87.0 %
for autoregressive decoding". Speculative decoding is supposed to be output-identical to
autoregressive decoding, so a one-point accuracy difference is not explained by the PR.

### 4.5 Maintainer consensus on the underlying trade-off

Three independent maintainer statements, from three projects, all saying the same thing:

- **benchislett** (vLLM), closing [#26595](https://github.com/vllm-project/vllm/issues/26595)
  ("ngram performance drops with higher batch size"): "This is a normal, well-documented
  consequence of using speculative decoding. Increasing the batch size also increases the relative
  cost of verifying the speculated tokens."
- **ekagra-ranjan** (vLLM), on #24344: "SD in general doesnt hold good for very high concurrency
  and is not a byproduct of this method."
- **ggerganov** (llama.cpp), on PR #19164: "Larger ngram size and larger drafts increase the
  chances that we will draft only when the LLM is repeating an existing text. Basically, we are
  trying to detect long repeating blocks without doing exhaustive searches. So unless your use case
  involves such repeating blocks of text, this method won't help."
- vLLM's own docs make the same point quantitatively
  (`docs/features/speculative_decoding/dynamic_speculative_decoding.md:5`): "As BS increases, the
  effective BS becomes BS\*K which increases the compute requirement during verification. When this
  BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT)."

### 4.6 The structural answer to "why not just concatenate?"

Concatenation (ngram extends the neural draft) is rejected in the one place it was seriously
proposed, and the stated reason is architectural, not empirical: combining both proposals into one
verify pass "requires tree attention which is not available" and "will increase the number of draft
which need to verify at each step" (vLLM #18633, ekagra-ranjan). SGLang is the counterexample that
tree attention is buildable — its ngram path already verifies a tree with a
`(draft_token_num, seq_len + draft_token_num)` mask — but SGLang has not combined it with EAGLE in
shipped code; the hybrid PR routes between workers instead of merging their proposals.

---

## 5. Measured acceptance and speedup by workload

Every row states its source and configuration. Where the number came from a blog or an unmerged
PR, it is labelled; the maintenance status of the *claim* is not the same as the maintenance status
of the *number*.

### 5.1 Chat / general conversation

| number | configuration | source |
|---|---|---|
| AL 1.37 turn 1, 1.66 turn 2; 10–60 % E2E first-turn speed-up, 30–90 % second-turn; 96.13 % at BS 1, 63.99 % at BS 4, 33.06 % at BS 32 | Llama-4-Scout-17B-16E FP8, 8× B200, TP 8; NGram `k=3, v=5`; Magpie-Llama-3.1-Pro-MT-300K-Filtered, 3000 conversations | **[vendor]** NVIDIA tech blog 07, `docs/source/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.md:8-9`, `:119-124`, `:144` |
| "Average accepted length (AL) is ~1.3 in generic chat (MT-Bench, Magpie with the first round of conversation)" | same | **[vendor]** same file, `:31` |
| seq-normalised AL: eagle 2.29, ngram **1.18**, ngram-eagle 2.22; median TPOT 7.00 / 4.19 / 6.18 / 4.30 ms | Llama-3.1-8B-Instruct, TP 1, concurrency 1, MTBench 80 prompts; ngram `K=5, prompt_lookup 2..5` | vLLM PR #24344 body (unmerged, author-run) |
| ngram 235–774 tok/s vs autoregressive 264–965 tok/s (AL 1.77–1.80) | Qwen3.5-35B-A3B, H20, NGRAM 6 draft tokens | SGLang PR #37237 (unmerged) |

The consistent finding: on generic chat, ngram drafting buys roughly 10–15 % and no more. Its
acceptance length there is close to the 1.0 floor, because ordinary conversation has little exact
long-range repetition.

### 5.2 Edit / copy-heavy and agentic

| number | configuration | source |
|---|---|---|
| median TPOT: vanilla 6.95, eagle 3.96, ngram **1.90**, ngram-eagle 2.13 ms; seq-normalised AL ngram 2.96, ngram-eagle **4.04** | Blazedit, ≤25 % normalised edit distance, 90 prompts × 2048 output tokens, Llama-3.1-8B-Instruct, TP 1 | vLLM PR #24344 body |
| ngram AL **4.75** at BS 4 (vs eagle3 3.49, autoregressive 667 tok/s) on a repetition-heavy sorting task | AdaLEval-textsort, Qwen3.5-35B-A3B, H20, NGRAM 6 tokens breadth 1 `PROB` | SGLang PR #37237 |
| acceptance 0.25 / 0.047 / 0.37 / 0.047 / 0.63 across five requests at 114–167 tok/s; `ngram_mod` self-time 2.4–6.1 ms over 6912 calls; 181.60 tok/s with acceptance 0.00000 in one degenerate case | gpt-oss-120b, 200-line source file repeated verbatim, `--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 48 --draft-max 64` | **[community]** bfroemel, logs pasted in llama.cpp PR #19164 |
| ngram AL 1.83–1.88 vs eagle3 2.73–2.74 on code editing | InstructCoder-style, Qwen3.5-35B-A3B, H20 | SGLang PR #37237 |
| InstructCoder median TPOT: vanilla 6.98, eagle 3.41, ngram 3.32, ngram-eagle **2.96** ms | Llama-3.1-8B-Instruct, 1000 prompts | vLLM PR #24344 body |
| "It works pretty well in OpenCode (GLM 4.7 Flash with thinking enabled), but I'm not sure if it's real or placebo. I assume that a draft acceptance rate above 0.1 indicates some speedup. (I see also >0.5)" | GLM 4.7 Flash, thinking on | **[community]** jacekpoplawski, llama.cpp PR #19164 |
| SGLang docs: Suffix Decoding "can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g. self-reflection, self-consistency), and RL rollouts" — **no numbers** | — | `docs/features/speculative_decoding/suffix.md:7` (vLLM docs) |

llama.cpp's own doc lists the intended applications as "Iterating over a block of text/code (e.g.
in llama.vim)", "Reasoning models (when they have to repeat their thinking in the final answer)" and
"Summarization" (`docs/speculative.md:189-194`), with no numbers attached.

### 5.3 Summarisation and translation

| number | configuration | source |
|---|---|---|
| AL **3.44 → 4.73** as `max_draft_len` grows 7 → 23 (k = 3 or 5); "New requests can benefit from cache generated by previous requests with similar tasks and reduce latency by up to 70%" | 4000 translation requests, Llama-4-Scout-17B-16E, 8× B200, TP 8 | **[vendor]** NVIDIA blog 07, `:10`, `:162-168` |
| llama.cpp names summarisation as a target application, with a demo video in PR #19164, no figures | — | `docs/speculative.md:189-197` |

**[no evidence found]** I did not find a primary-source measurement of ngram drafting on a
summarisation benchmark in any of the four engines' repositories or trackers. The claim rests on
the TRT-LLM translation result and llama.cpp's assertion.

### 5.4 Structured output (JSON / grammar-constrained)

| number | configuration | source |
|---|---|---|
| ngram output throughput 215.02 / 599.20 / 587.80 tok/s at qps 1 / 4 / 8 on a `json` dataset, against eagle3 247.19 / 969.96 / 1790.25 and mtp 123.37 / 494.72 / 1013.82 on the same rows | 8× B200; Kimi-K2.6 (eagle3), Qwen3.6-35B-A3B (mtp, ngram) | vLLM PR #41897 body (unmerged; these are the *control* rows of an unrelated A/B) |
| ngram 261.90 / 408.51 / 449.65 tok/s on `xgrammar_bench` at qps 1 / 4 / 8, vs mtp 301.51 / 1191.61 / 2332.09 | same | same |

SGLang's ngram path is explicitly grammar-aware: it stages the tree mask and draft tokens for the
grammar bitmask — `self.grammar_tree_host = (mask, req_drafts) if batch.has_grammar else None`
(`ngram_worker.py:343`) — and splits into `batch.has_grammar` / `grammar_needs_sync()` cases
(`:250`, `:311-315`). So structured output is supported there by construction, not by accident.

**[no evidence found]** no engine publishes an acceptance rate for ngram drafting under a JSON or
XML grammar, as distinct from a throughput number on a JSON dataset. The vLLM numbers above are
the closest thing and are from an unrelated benchmark's control arm.

### 5.5 The two ends of the range, from the same vendor

NVIDIA's blog gives both extremes with the same engine: **AL 1.37** on generic chat and **AL 4.73**
on translation, with speed-ups from ~10 % to ~96 % depending on batch size and turn. That 3.4×
spread in acceptance length across workloads, from one implementation with one set of
hyperparameters, is the single most useful calibration in this note: the workload, not the
algorithm, decides whether ngram drafting pays.

---

## 6. Known correctness and performance pitfalls

### 6.1 Stale state carried across requests (the worst one, and it is measured)

llama.cpp [#27852](https://github.com/ggml-org/llama.cpp/issues/27852), "ngram-cache speculative
decoding keeps per-slot context cache across requests (begin() is a no-op) — acceptance 86% → 11%,
slower than no speculation" (open). `common_speculative_impl_ngram_cache::begin()` was a no-op, so
a slot's ingested-prompt watermark survived into the next request; a shorter follow-up prompt was
never ingested and drafts were proposed from the *previous* request's n-grams.

Measured on an RTX A4500, Qwen3.8-Flash-Next UD-Q4_K_XL, one slot, same binary and prompts:

| sequence on one slot | accepted / proposed | decode |
|---|---|---|
| fresh server → copy request | 903 / 1052 (86 %) | 41.5 t/s |
| fresh server → thinking request → copy request | 73 / 660 (11 %) | **19.4 t/s** |
| `--spec-type none`, copy request | – | 25.9 t/s |

That middle row is the number to keep: **speculation that is 25 % slower than no speculation at
all**, purely from stale proposal state. Maintainer **mndodd** reproduced the same failure pattern
on `draft-mtp` (0/197 accepted, 16.00 t/s, recovering to 87.36 t/s after the fix, identical output
md5 in every case) and noted the ordering problem: "begin() runs at DONE_PROMPT, after prefill, so
while the fix you've got works, an explicit reset() is what draft-mtp and others will need."
**LeoBorcherding** checked whether `ngram-mod` shares the defect and it does not — its `begin()`
zeroes `i_last` and re-ingests the new prompt (`common/speculative.cpp:1912-1927`).

vLLM's ngram proposer has no equivalent state to go stale (it is stateless, §2.1), which is a
second argument for the stateless design. A **global** (cross-request) scope is an open proposal
there, off by default, with a bounded LRU of 100 000 entries
([#44597](https://github.com/vllm-project/vllm/pull/44597)).

### 6.2 Proposals that are correct but unwanted

The only primary report I found is in llama.cpp PR #19164, from **EndeavoringOrb**: when the pasted
file has CRLF line endings and the model prefers LF, the copied n-grams stop matching even though
the task is verbatim repetition. Same model, same prompt, same settings
(`--spec-type ngram-mod --spec-ngram-size-n 24 --draft-min 32 --draft-max 48`, Devstral-2-123B
UD-Q5_K_XL, aarch64):

| input | acceptance | decode |
|---|---|---|
| CRLF file, "Repeat verbatim." | 0.11458 (11/96) | 2.32 t/s |
| LF file, "Repeat verbatim." | 0.83013 (518/624) | 29.79 t/s |

A 12.8× throughput difference from invisible byte-level divergence. **[community]**, but the
mechanism is generic to any token-exact copy proposal.

### 6.3 The reset criteria destroying useful state

Both of `ngram-mod`'s resets throw away the whole pool, including n-grams harvested from the
prompt. From PR #19164:

- **bfroemel**: "I think I just observed the effects of an early low acceptance streak (3) and the
  triggered reset clears the actually still very useful ngrams from prompt processing."
- **ggerganov** on the streak threshold: "Yes, the `3` is currently hardcoded." (the shipped code
  now uses `f_acc < 0.25` and `n_low >= 5`, `common/speculative.cpp:2024-2034`), and on the
  occupancy threshold: "The hash container can store a lot of ngram hashes (hundred thousands with
  the current size) before collisions start to occur" — which is why the 0.25 occupancy reset at
  `begin()` is conservative.
- **[community]** treo, on the same PR: "When using ngram-mod, one should define `--draft-min` to be
  more than 0, which is the default." — a usability trap, because `n_min = 0` lets a single-token
  match through and manufactures rejections.

### 6.4 Hash collisions produce wrong proposals (by design, and harmless)

`common_ngram_mod::get` returns whatever occupies the slot with no verification
(`ngram-mod.cpp:37-41`). Two distinct 24-grams sharing a slot silently yield a bogus successor
token. The target model rejects it, so output correctness is unaffected — the cost is a wasted
verify row. This is the direct-mapped-hash trade-off: constant memory and O(1) lookup bought with
a collision rate that grows with occupancy, which is exactly why the 0.25 occupancy reset exists.

### 6.5 Acceptance that collapses to zero on later requests

[#19231](https://github.com/ggml-org/llama.cpp/issues/19231), "Speculative decoding only works once
with /v1/chat/completions" (closed as completed after ggerganov replied "It works, just sometimes
does not trigger. Read #19164 for more information."). The reporter's follow-up: "It never works
except for the first request to the server … only the first request is accelerated", with a
regenerate-in-chat reproduction confirmed by a second user (**ddh0**) in PR #19164. Whatever the
cause, the practical lesson is that acceptance on a repeat request is not guaranteed and has to be
measured per workload.

### 6.6 Grammar / structured output versus draft trees

vLLM only gained grammar traversal of draft trees in 2026, and it is still experimental:
[#41897](https://github.com/vllm-project/vllm/pull/41897) "Add support for `traverse_draft_trees`
for structured outputs + specdec" (open), wrapping XGrammar's `traverse_draft_trees`. A drafter that
proposes a *tree* (SGLang, EAGLE, Medusa) needs this to stay inside a grammar; a chain drafter does
not. vLLM separately gates spec decode against tool calling
([#53638](https://github.com/vllm-project/vllm/pull/53638)).

### 6.7 Recurrent / hybrid state corruption with ngram spec decode

Three independent reports, all open, all on the state-update path rather than the proposal path —
and directly relevant to a GDN/recurrent target:

- vLLM [#40738](https://github.com/vllm-project/vllm/pull/40738) "Fix GDN conv + SSM state
  corruption with ngram spec decode"
- vLLM [#52942](https://github.com/vllm-project/vllm/pull/52942) "Fix stale mamba/GDN states with
  ngram spec decode on hybrid models"
- SGLang [#37794](https://github.com/sgl-project/sglang/pull/37794) "fix(spec): commit PLE state
  after ReplaySSM verify, NGRAM on Qwen4-Exp"

Plus vLLM's `ngram_gpu` producing a different acceptance rate from the CPU proposer
([#44054](https://github.com/vllm-project/vllm/pull/44054),
[#44056](https://github.com/vllm-project/vllm/pull/44056)) and drafting past `max_model_len`
([#43049](https://github.com/vllm-project/vllm/pull/43049)).

### 6.8 Repetition loops

The one primary report I found runs the *opposite* way from the worry. **characharm**, in llama.cpp
PR #19164: "in the case of GPT-OSS in high mode, when the model falls into a reasoning loop, it can
now recover much faster. Token generation jumps to around 200, and the model even produces a
meaningful result." **ggerganov** confirms he saw the same and adds "Overall, I think this
speculator can become enabled by default in `llama-server`." **[community]**, but corroborated by
the author.

The related repetition bug in the tracker is not about ngram: [#23577](https://github.com/ggml-org/llama.cpp/issues/23577)
"MTP with Qwen3.6 27B outputs repeated `////` after long session". On the vLLM side the inverse
problem is being solved in the sampler — [#57540](https://github.com/vllm-project/vllm/pull/57540)
"Add native GPU output no-repeat n-gram masking".

### 6.9 Safety filters being bypassed by copied content

**[no evidence found].** I searched the vLLM, llama.cpp and SGLang trackers for ngram/prompt-lookup
combined with guardrail, safety, moderation or filter terms and found nothing. The structural
reason a report is unlikely: a copied token is only a *proposal*, and acceptance is decided by the
target model's own logits and sampling, so any filter implemented target-side still runs. vLLM
states the contract in its own docs: "Accepted tokens are exactly those the target model would have
produced under the same sampling configuration; rejected draft tokens are discarded and regenerated
by the target model" (`docs/features/speculative_decoding/speculators.md:24`). That is an argument
from the speculative-decoding contract, not an observation from an engine issue — treat it as
reasoning, not as a finding. A filter applied *outside* the sampler (an output string check, a
`logprobs`-based policy, a client-side validator) does see the copied text, and no engine documents
that case either.

### 6.10 A note on measuring this at all

llama.cpp's per-slot statistics block prints draft acceptance directly
(`tools/server/server-context.cpp:658-678`: `draft acceptance rate`, mean acceptance length, and
per-position acceptance rates), and the ngram-specific counters appear in the same line as
`statistics ngram_mod: #calls = …, #gen drafts = …, #acc drafts = …` (see the pasted logs in PR
#19164). vLLM ships an acceptance-metrics doc
(`docs/features/speculative_decoding/acceptance_metrics.md`) — and PR #24344's main methodological
contribution is that vLLM's built-in AL is a *per-draft precision*, not an end-to-end predictor,
which is why ngram's 4.59 "AL" corresponded to a seq-normalised 2.96 and to a TPOT that lost to
EAGLE.

---

## 7. What the sources support for a startup-captured-graph engine

Constraints as given: decode graphs captured at startup, startup-fixed concurrency 1..8, one
compact decode batch per round, 2.1–2.9 GiB free VRAM at the shipped configuration. The three
designs below are the three shapes that actually exist in production code. I am reporting what the
sources show about each, not recommending one.

### 7.1 The three designs and their known cost

**A. Bounded host-side hash pool, variable-length proposal** (llama.cpp `ngram-mod`; closest to
what this port already has).

- *Cost, measured or read directly:* the graph layer's warmup gate compares every source tensor's
  `ne`/`nb` and refuses the graph for a call whose shapes changed
  (`ggml-cuda.cu:2613-2629`, `:4456-4475`), while the server submits an unpadded
  `1 + len(draft)` batch (`server-context.cpp:526-535`). A variable-length drafter therefore
  oscillates the graph out of use unless the width is pinned or padded. Fixed memory: 16 MiB
  regardless of load. O(prompt) hashing per request at `begin()`, amortised O(1) per token after.
  Global reset paths mean one unlucky sequence can flush everyone's index.

**B. Stateless full-context rescan, fixed-length proposal** (vLLM `ngram` / `ngram_gpu`).

- *Cost, measured:* drafting was 5–8 % of critical path for small models and 0.9 % for a 17B model
  *before* the KMP+numba rewrite that PR #22390 shipped, which cut it 38 %+
  (vLLM #22408, maintainer-filed). It remains O(total context) **per step**, which is what the
  open follow-up and the third-party Bloom-filter benchmark are both attacking. Host memory is
  `max_num_seqs × max_model_len × 4` bytes (the `token_ids_cpu` mirror,
  `ngram_proposer.py:83-85`) — at 256 sequences and 128 k context that is 128 MiB of host RAM.
  *Benefit that matters here:* the proposal length is fixed at `k`, so the verify width is fixed
  and the graph shape is fixed; and being stateless, it cannot go stale across requests (§6.1).

**C. Device-side corpus with a fixed node budget and a tree mask** (SGLang `NGRAM`; the
multi-candidate variant of TensorRT-LLM's pool is the tree-free cousin).

- *Cost, read from source:* a per-request tree attention mask of
  `(draft_token_num, seq_len + draft_token_num)` (`ngram_worker.py:330-337`, `:359-374`), a
  persistent corpus whose default capacity is 10 M entries plus an optional external corpus of up
  to 10 M tokens (`arg_groups/fields/spec.py:246-261`), and a breadth-first search whose
  per-call cost is not characterised in any comment or benchmark I found. *Benefit:* the verify
  width is a fixed node budget by assertion (`:306-308`), so graph shapes are stable, and the tree
  buys multiple candidates per position — which is also the prerequisite for ever *concatenating*
  with a neural drafter (§4.6).

### 7.2 The graph question, answered narrowly

The three engines resolve the width problem three different ways, and all three resolve it by
**not letting the proposal length change the graph**:

- llama.cpp lets it change and pays in graph warmth (`ggml-cuda.cu:4456-4475`).
- vLLM pads to `1 + k` and masks the remainder; capture sizes are rounded up to a multiple of
  `1 + k`, so a wider draft costs *memory per variant*, not extra variants
  (`cudagraph_dispatcher.py:36`, `compilation.py::adjust_cudagraph_sizes_for_spec_decode`).
- SGLang fixes the node budget and puts the variability inside the tree
  (`ngram_worker.py:306-308`).

For an engine that captures graphs once at startup and has 2.1–2.9 GiB free, the sourced cost of
each escape hatch is specific: re-capture churn (llama.cpp), a proportional activation-memory
increase in every captured variant (vLLM), or a tree mask and a device-resident corpus (SGLang).
Only vLLM's route is expressed in the currency of "startup-fixed capture", because only vLLM
publishes the capture-size arithmetic.

### 7.3 On adding ngram on top of MTP/EAGLE

The evidence is: the two PRs below are unmerged, but they are not the whole field, and the shape of
the result depends on hardware class more than on the design. On consumer-class silicon the
combination is worth low single digits at best and has been measured negative; in the datacenter it
has been measured both strongly positive and strongly negative, on the same PR family. See
[ngram-outside-github](ngram-outside-github.md) for the full table, and section 7.4 item 1 for the
correction to the "nothing shipped" reading.

- vLLM PR #24344 (unmerged): on edit-heavy work the combination is *slower than ngram alone*
  (2.13 vs 1.90 ms TPOT); on code it wins (2.96 vs 3.32); on chat it only matches EAGLE
  (4.30 vs 4.19). The design is selection-with-fallback, and the stated reason concatenation is
  refused is that it needs tree attention and a wider verify.
- SGLang PR #37237 (unmerged, CI red): batch-level routing, +0.2 % to +8.9 % on a
  repetition-heavy task with one −4.5 % outlier, and −3 % to −19 % on code and maths.
- llama.cpp: ships the fallback chain but has published **no** measurement of the combination; its
  only combination report is an open, undiagnosed OOM crash (#23154), and the one widely-cited
  negative claim (#23184) is unreplicated and premised on behaviour the code does not have.

Both surviving designs keep both drafters' state live, and both pay the neural drafter's cost
whenever they route to retrieval (SGLang keeps the EAGLE prefill for every sequence and gates only
its decode; vLLM's author makes the same trade in the RFC). Neither has a published measurement of
what that costs in VRAM, which is the binding constraint here.

### 7.4 Where the evidence is thin — explicitly

1. ~~**No shipped engine combines a neural drafter with ngram drafting.**~~ **Corrected
   2026-09-27; this was wrong.** It rested on vLLM and SGLang being open PRs, which said nothing
   about the engines that already ship the combination. Selection has shipped in production with
   published numbers -- TensorRT-LLM added `use_sa_spec` / `sa_spec_threshold` to its MTP path
   (PR #11434; the deprecated standalone `NGram` flags were removed in #12130), and vLLM ships
   `method: "ngram"`. Concatenation has shipped in exactly one engine, FastDeploy's
   `mtp_strategy: with_ngram` (2 MTP + 3 ngram into a single verify, kernels ported), and it has
   published **no** measurements of the hybrid at all. See
   [ngram-outside-github](ngram-outside-github.md).
   The thin spot is narrower and sharper than the one originally stated: **no published
   measurement of the combination on a masked-draft drafter (DFlash/DFlash2), and none on a
   consumer discrete GPU at RTX 5090 class.** That is precisely this product's configuration, so
   the combination's effect here is unmeasured, not absent.
2. **No measurement of a variable-width drafter under a startup-fixed capture regime.** Every
   production design either pads, re-captures, or fixes the width. The specific cost of
   *not* padding — how often a graph leaves and re-enters the warm path, and what that costs in
   tok/s — is not published anywhere I looked.
3. **Index memory is barely quantified.** The only figures in primary sources are llama.cpp's flat
   16 MiB, SGLang's default 10 M-entry capacity, and a third-party "~1 MB of persistent
   per-request state". vLLM's host mirror is computable from the code but nobody has published it
   as a sizing figure. Nothing states the VRAM cost of SGLang's device corpus.
4. **Recurrent-state interaction is reported as bugs, never as a cost.** The GDN/SSM corruption
   reports (§6.7) are all open with no published root cause or measurement. For a GDN target this
   is the least-understood risk in the whole area.
5. **No acceptance-rate figures for structured output**, and none for summarisation from an
   engine's own benchmark. The translation result is the only high-acceptance non-chat number, and
   it is from a vendor blog.
6. **The safety-filter question is unanswered**, not answered negatively (§6.9).
7. **llama.cpp's two headline claims are only half true**, and the false half is the one that
   matters under a fixed capture regime: memory is constant, but the *complexity* claim excludes
   the O(prompt) per-request index build, and the variable-length property is in direct tension
   with the engine's own CUDA-graph warmup gate.

---

## Sources

**llama.cpp** — `common/ngram-mod.{h,cpp}`, `common/speculative.{h,cpp}`, `common/common.h`,
`common/arg.cpp`, `docs/speculative.md`, `tools/server/server-context.cpp`,
`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/common.cuh` (all `master`, fetched
2026-09-26); PR [#19164](https://github.com/ggml-org/llama.cpp/pull/19164) (ngram-mod, by
ggerganov, merged 2026-01-30); issues
[#23184](https://github.com/ggml-org/llama.cpp/issues/23184),
[#23154](https://github.com/ggml-org/llama.cpp/issues/23154),
[#19231](https://github.com/ggml-org/llama.cpp/issues/19231),
[#27852](https://github.com/ggml-org/llama.cpp/issues/27852),
[#28860](https://github.com/ggml-org/llama.cpp/issues/28860),
[#23929](https://github.com/ggml-org/llama.cpp/issues/23929),
[#27839](https://github.com/ggml-org/llama.cpp/issues/27839),
[#23577](https://github.com/ggml-org/llama.cpp/issues/23577);
historical file state at `255582687b8dd211fdbc582e43ab842491554e94` (2026-05-16).

**vLLM** — `vllm/v1/spec_decode/ngram_proposer.py`, `ngram_proposer_gpu.py`,
`vllm/config/speculative.py`, `vllm/config/compilation.py`, `vllm/v1/cudagraph_dispatcher.py`,
`vllm/v1/worker/gpu_model_runner.py`, `docs/features/speculative_decoding/{n_gram,speculators,
dynamic_speculative_decoding,adaptive_verification,lilicorr,suffix,acceptance_metrics}.md` (all
`main`, fetched 2026-09-26); issues
[#22408](https://github.com/vllm-project/vllm/issues/22408),
[#18633](https://github.com/vllm-project/vllm/issues/18633),
[#26595](https://github.com/vllm-project/vllm/issues/26595); PRs
[#22390](https://github.com/vllm-project/vllm/pull/22390),
[#24344](https://github.com/vllm-project/vllm/pull/24344) (open),
[#41897](https://github.com/vllm-project/vllm/pull/41897) (open),
[#44597](https://github.com/vllm-project/vllm/pull/44597) (open),
[#56517](https://github.com/vllm-project/vllm/pull/56517) (open),
[#40738](https://github.com/vllm-project/vllm/pull/40738),
[#52942](https://github.com/vllm-project/vllm/pull/52942),
[#44054](https://github.com/vllm-project/vllm/pull/44054),
[#44056](https://github.com/vllm-project/vllm/pull/44056),
[#43049](https://github.com/vllm-project/vllm/pull/43049),
[#53638](https://github.com/vllm-project/vllm/pull/53638),
[#57540](https://github.com/vllm-project/vllm/pull/57540).

**SGLang** — `python/sglang/srt/speculative/ngram_worker.py`, `ngram_info.py`,
`cpp_ngram/ngram_corpus.py`, `python/sglang/srt/arg_groups/fields/spec.py` (all `main`, fetched
2026-09-26); PR [#37237](https://github.com/sgl-project/sglang/pull/37237) (open),
[#37794](https://github.com/sgl-project/sglang/pull/37794),
[#23629](https://github.com/sgl-project/sglang/pull/23629),
[#35102](https://github.com/sgl-project/sglang/pull/35102),
[#36978](https://github.com/sgl-project/sglang/pull/36978),
[#22569](https://github.com/sgl-project/sglang/pull/22569).

**TensorRT-LLM** — `tensorrt_llm/_torch/speculative/ngram.py`,
`tensorrt_llm/_torch/speculative/auto_heuristic.py`, `examples/ngram/README.md`,
`docs/source/features/speculative-decoding.md`,
`docs/source/blogs/tech_blog/blog07_NGram_performance_Analysis_And_Auto_Enablement.md` **[vendor
blog: NVIDIA-authored, first-party for their own product, not independently reviewed]**.

**Method note.** No blog post or secondary summary was used as the source of a factual claim about
an implementation. The one exception is explicitly marked **[vendor]**: NVIDIA's own tech blog,
used only for the measured numbers in §5.1/§5.3 and the AUTO-heuristic rationale, and labelled
wherever it appears. Line numbers are from the files as fetched on 2026-09-26 and will drift.
