# MLOps Assignment — Report

## Phase 1 — Serving Configuration

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80GB, served with vLLM 0.10.2 (bf16).

**Workload shape that drove the config:** 1.5–3K-token prompts (rendered schema +
question), short structured SQL outputs, and 2–3 dependent LLM calls per agent
request. Target SLO: P95 end-to-end < 5s at 10+ RPS.

**Key insight — this model is memory-bound, not compute-bound.**
Qwen3-30B-A3B is a Mixture-of-Experts model (boot log confirms
`Resolved architecture: Qwen3MoeForCausalLM`): 30B total parameters but only
~3B active per token. Compute per token is cheap (3B-class), but all 30B weights
(~60GB in bf16) must sit in VRAM. So the binding constraint is fitting the weights
and then maximizing the KV cache left over for concurrency — every concurrency
decision is really a memory decision.

### Flags and rationale

| Flag | Value | Justification |
|---|---|---|
| `--max-model-len` | `8192` | Prompts are ~3K and outputs short. The model's native 256K context would reserve enormous KV cache per request slot. Capping at 8192 frees KV cache for more concurrent requests → higher achievable RPS. |
| `--gpu-memory-utilization` | `0.90` | ~60GB of weights leave ~20GB free; raising utilization to 0.90 hands maximum headroom to the KV cache for batching, without tipping into OOM. |
| `--max-num-seqs` | `64` | Bounds concurrent sequences to keep KV-cache pressure in check while still batching enough to reach the 10+ RPS target. |
| `--enable-chunked-prefill` | on | With ~3K-token prompts, prefill is expensive; chunking interleaves prefill with ongoing decode so one long prompt doesn't stall everyone else's generation — improves tail latency under load. |

**Also observed:** vLLM auto-enabled prefix caching (`enable_prefix_caching=True`
in the boot log). This is a real win for this agent: `generate_sql` / `verify` /
`revise` all share the same schema prefix, so after the first call the schema is
served from cache — visible as high `input_cache_read` token counts in the
Langfuse traces.

**Sanity check:** a manual query against `/v1/chat/completions` returned correct
SQL (`SELECT COUNT(*) ... WHERE department = 'Sales'`), and 4 eval-set questions
fired through the agent returned sensible SQL — including one
(`formula_1` coordinates) that triggered a verify→revise loop to add `DISTINCT`.
See `screenshots/vllm_manual_query.png`.

*This is the baseline configuration; it is revisited in Phase 6 under load. FP8
weight quantization is the obvious next lever (halves weight footprint → roughly
doubles KV-cache room → more concurrency), traded against a small quality risk.*
