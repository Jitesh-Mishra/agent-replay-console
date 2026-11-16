# LLM drafter: measured results

Generated 2026-11-16T21:11:53+00:00. Models: `openai/gpt-oss-20b` (fallback `openai/gpt-oss-120b`), temperature 0.3. Sample: 50 tickets drawn with seed 2026 from the grouped held-out split. Guards on. Gate policy: Version() defaults (conf 0.6, grounding 0.5, high priority needs a human). Single pass, nothing tuned on this sample.

## 50 clean tickets, both drafters

| Metric | Extractive | Groq |
|---|---:|---:|
| Decisions (auto_send / needs_approval / escalate) | 5 / 45 / 0 | 5 / 45 / 0 |
| Unsafe auto-sends (auto-sent with wrong route or true high priority, plus policy violations) | 1 | 1 |
| Policy violations (audit re-applying the gate to recorded outputs) | 0 | 0 |
| Draft step latency p50 / p95 (ms) | 0.0 / 0.1 | 4449.0 / 7785.9 |
| Guard flags raised | none | none |
| Draft similarity to historical answer (TF-IDF cosine) | 0.127 | 0.148 |
| Mean draft length (chars) | 387 | 554 |
| Drafts with leftover placeholders | 13 | 0 |
| Drafts claiming a completed action (regex heuristic) | 0 | 0 |
| Drafts containing a URL or phone number (regex heuristic) | 1 | 0 |
| All 4 steps recorded and trace reconstructs the decision | 1.00 | 1.00 |

## Groq provider behaviour on those 50 runs

| | |
|---|---:|
| API calls | 87 |
| Errors | 37 {'429': 37} |
| Retries | 36 |
| Fallbacks to the second model | 1 |
| Drafts produced by the fallback model | 1 |
| Drafts that failed completely (escalated) | 0 |
| Truncated replies rejected | 0 |
| Tokens (prompt / completion) | 30037 / 6761 |
| Approx tokens per successful call (prompt + completion) | 736 |

Draft latency for the Groq runs with no retry: n=18, p50 488 ms, p95 1168 ms. With at least one retry (a 429 and a backoff wait): n=32, p50 5540 ms, p95 8581 ms.

## Fault injection against the Groq drafter (20 runs per fault)

| Fault | Runs | Escalated | Auto-sent while faulty | Crashed | Flag detected | Runs that called Groq |
|---|---:|---:|---:|---:|---:|---:|
| tool_timeout | 20 | 20 | 0 | 0 | 1.00 | 0 |
| stale_schema | 20 | 20 | 0 | 0 | 1.00 | 20 |
| malformed_tool_output | 20 | 20 | 0 | 0 | 1.00 | 0 |
| empty_retrieval | 20 | 20 | 0 | 0 | 1.00 | 0 |
| llm_refusal | 20 | 20 | 0 | 0 | 1.00 | 20 |

## Guard before / after on realistic LLM refusals

| Reply | Old guard | New guard |
|---|---|---|
| I can’t help with that. (refusal) | missed | flagged |
| I’m sorry, but I can’t assist with that request. (refusal) | missed | flagged |
| I can't assist with this. (refusal) | missed | flagged |
| I'm unable to provide that information. (refusal) | missed | flagged |
| I’m not able to help with this request. (refusal) | missed | flagged |
| I must decline to answer. (refusal) | missed | flagged |
| Sorry, I can’t do that. (refusal) | missed | flagged |
| I'm sorry to hear about the outage. We are on it. (not a refusal) | not flagged | not flagged |
| I'm sorry for the inconvenience. (not a refusal) | not flagged | not flagged |

No real Groq reply triggered the refusal or empty guard in these 50 runs, so the guard fix is validated on hand-written refusals and mocks, not on refusals that occurred naturally.

## Limits

- 50 tickets and one pass: small samples, and Groq output is not deterministic, so a rerun will differ.
- The tickets are likely synthetic. The knowledge base is 12 self-written documents.
- `Unsafe auto-send` here means the routing and priority policy was violated. It does NOT mean the reply text is correct.
- The gate does not read the draft, so both drafters get identical gate decisions. Reply quality is only checked by the refusal and empty guards plus a similarity score against one historical answer, a weak proxy (it measures word overlap, not correctness). No human or LLM judge has rated these replies.
- Groq returned x-ratelimit-limit-tokens 8000 per minute and 1000 requests per day for the key used (read on 2026-11-16). At roughly 700 tokens per draft, back-to-back runs exceed the per-minute cap, which is where the 429 retries and the slow p95 come from. One ticket at a time, as in the console, is well under it.
- The unsafe auto-send is the same ticket for both drafters: the route was right but the priority model predicted medium for a truly high-priority ticket. That is the existing priority-model gap, independent of the drafter.
- The completed-action, placeholder and URL/phone rows are regex heuristics, not a faithfulness measurement.
- Latency is measured from one machine against Groq's public API and includes retries and backoff.
- Fault runs that stop before the draft step (tool_timeout, empty_retrieval, malformed_tool_output) never call Groq, so only stale_schema and llm_refusal exercise the LLM; llm_refusal overwrites the model reply with the injected refusal.
