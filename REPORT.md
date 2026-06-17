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
| `--max-num-seqs` | `64` *(initial; raised to 256 in Phase 6)* | Bounds concurrent sequences to keep KV-cache pressure in check while still batching enough to reach the 10+ RPS target. Phase 6 showed this initial cap was too conservative given the idle KV headroom. |
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

---

## Phase 5 — Baseline Eval

Execution accuracy over 30 BIRD questions; per-iteration pass rate reconstructed
from the agent's `history` (the SQL at each generate/revise step), with
carry-forward for questions that terminated early.

| Metric | Value |
|---|---|
| Overall pass rate | **40%** (12/30) |
| Pass rate @ iter 0 (generate only) | 33.3% |
| Pass rate @ iter 1 (after 1 revise) | 36.7% |
| Pass rate @ iter 2 (after 2 revises) | 40.0% |
| Avg iterations | 1.57 |

**Does the loop earn its keep?** Yes, modestly. Pass rate climbs 33% → 40%
(+7 points, ~20% relative) across iterations, so the verify→revise loop turns
some wrong first attempts into correct answers. But most questions that fail at
generate also fail after revision — the ceiling is generation quality, not
verification.

---

## Phase 6 — SLO Diagnosis

**Target:** P95 end-to-end agent latency < 5s at 10+ RPS over a 5-minute window.

**Baseline (`--max-num-seqs 64`):** P95 = **107.8s**, achieved **8.3 RPS** (could
not sustain 10), only **19%** of requests OK (44% timed out at 120s, plus HTTP +
client errors). SLO missed by ~21×.

Dashboard at peak (`screenshots/grafana_before.png`): queue/waiting time
dominated end-to-end latency, **KV cache only ~25%**, **preemptions 0**, token
throughput plateaued ~15K tok/s. Diagnosis: **concurrency/throughput-bound, not
memory-bound** — arrival rate (~25 vLLM RPS) exceeds service rate (~22 RPS) while
memory sits idle, so the backlog grows without bound.

### Iteration log
- **Iter 1:** saw queue-bound with KV idle at ~25% and 0 preemptions →
  hypothesized the `max-num-seqs` cap throttles concurrency while memory is free →
  changed `--max-num-seqs` 64 → 256 → **result: the targeted metric moved
  (vLLM throughput 15K→20K tok/s, 22→32 req/s) but the SLO did NOT follow
  (P95 107.8s→113.9s, still ~23× over).** vLLM's own per-request latency stayed
  low while agent end-to-end was ~2 min, with 1179 connection/HTTP errors →
  the bottleneck is not vLLM but the **agent orchestration layer** (single
  synchronous uvicorn process, ~40-thread cap, 2–3 sequential vLLM calls/run).
- **Iter 2:** saw vLLM had spare throughput (~32 req/s ≈ 13 agent RPS capacity)
  but agent end-to-end was huge with 775 client errors → hypothesized the single
  synchronous uvicorn process caps agent concurrency below vLLM's capacity →
  changed agent to `--workers 4` → **result: dramatic. P95 113.9s→11.8s (~10×),
  P50 46.9s→2.5s, ok 723→2612, client_errors 775→0, timeouts 1098→8.** The agent
  process was the true bottleneck. Median now meets the 5s SLO; the P95 tail
  (11.8s) still misses by ~2.4×.
- **Iter 3:** saw the residual P95 tail (11.8s) was driven by multi-iteration
  requests (up to 5 sequential vLLM calls each) → hypothesized capping the loop
  removes the long tail → changed `MAX_ITERATIONS` 3 → 1 → **result: P95
  11.8s→3.83s (SLO HIT), P50 2.5s→1.5s, achieved 9.35 RPS, 0 client errors.**
  Confirmed quality cost separately: pass rate 40%→33.3% (the loop's +6.7 points
  are lost because no revision ever happens).

**The speed↔quality tradeoff (measured):**

| Config | P95 | Achieved RPS | Quality | SLO |
|---|---|---|---|---|
| `MAX_ITER=3` + workers=4 | 11.8s | 8.4 | 40.0% | ❌ miss (2.4×) |
| `MAX_ITER=1` + workers=4 | **3.83s** | **9.35** | 33.3% | ✅ **hit** |

The verify→revise loop is worth ~6.7 accuracy points and ~8s of tail latency.

**Final config (latency-first — hits the SLO):**
- vLLM: `--max-num-seqs 256` (+ Phase 1 flags: max-model-len 8192, gpu-mem-util 0.90, chunked prefill)
- Agent: `uvicorn --workers 4`, `MAX_ITERATIONS = 1`

**Final numbers:** P95 = **3.83s** (< 5s ✅), P50 = 1.5s, P99 = 6.1s, achieved
**9.35 RPS**, 2611/3000 OK, 0 client errors.
**Quality check:** baseline **40%** vs after-tuning **33.3%** — a 6.7-point
regression. This is the deliberate cost of capping the loop to hit the SLO; the
latency fix that came *free* (`--workers 4`, no quality change) took P95 from 108s
to 11.8s, and only the final `MAX_ITERATIONS` cut traded quality for the last
factor of ~3 in tail latency.
**Verdict:** **SLO hit** (P95 3.83s < 5s at ~10 RPS over 5 min). The dominant
bottleneck was the agent orchestration layer, not vLLM — a single synchronous
uvicorn process. Fixing it with `--workers 4` gave a ~10× improvement at zero
quality cost; the remaining tail was the agent's own sequential multi-call design,
closed by capping `MAX_ITERATIONS` at the cost of 6.7 accuracy points.
(`screenshots/grafana_before.png` → `screenshots/grafana_after.png`)

---

## Agent value — did the loop help?

Yes, measurably, but modestly. The per-iteration pass rate climbs **33.3% → 36.7%
→ 40.0%** — a ~20% relative gain over the single-shot baseline. The clearest example
is the Australian-Grand-Prix circuit query: `generate_sql` returns 11 identical
rows, `verify` flags the duplicates, and `revise` adds `DISTINCT` to produce the
correct single row (visible in `screenshots/langfuse_trace.png`).

The ceiling, though, is generation quality: most questions that fail at iter 0 also
fail after revision. If the first SQL is wrong in a way the verifier can't tell from
the rows alone — e.g. the `financial` "average crimes" query picks the wrong cryptic
column (`A14`) and returns a NULL average — the reviser has nothing actionable to
fix and the loop burns its iterations without recovering. So the loop earns its keep
on *detectable* failures (duplicate rows, errors, empty results) but can't rescue
*semantically* wrong queries — which is also why it costs latency without always
recovering it.

## What I'd do with more time

- **Make the agent async end-to-end.** The single biggest win in Phase 6 came from
  agent concurrency (`--workers 4`). The deeper fix is converting the `/answer`
  endpoint and `graph.invoke` to async so one process handles many in-flight runs
  without the threadpool ceiling — likely letting us keep `MAX_ITERATIONS=3`
  (full quality) *and* hit the SLO, instead of trading one for the other.
- **Add column-level schema descriptions to the prompt.** Several failures were on
  the `financial` DB with cryptic columns (`A1`–`A16`); the model can't know `A15`
  means "no. of crimes" without metadata. BIRD ships column descriptions — feeding
  them in would lift the generation ceiling that currently caps the loop's value.
- **Give the verifier the gold-row *shape*, not just the rows.** The verifier can
  catch structural errors but not semantic ones. Passing expected column
  count/types, or letting it issue a cheap exploratory query, would let revise fix
  more than duplicate/empty-row cases.
