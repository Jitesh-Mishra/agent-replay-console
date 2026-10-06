# Probe prep (answers come only from what this run did)

Run `pytest -q`, `cd gateway && npm test`, `cd console && npm test`, then start the three services and click through one clean run and one faulted run before an interview.

## How it works, in 5 lines
1. `arc/components.py` loads the tickets and clusters paraphrase families (TF-IDF cosine ≥ 0.45, connected components) for a grouped split. It trains TF-IDF + logistic regression for queue and priority (C tuned by grouped CV) and builds Chroma collections over past tickets and the KB with offline LSA embeddings.
2. `arc/agent.py` is a LangGraph graph `classify → retrieve → draft → gate`. Every node is wrapped by a tracer that writes input, output, error and latency to SQLite, then emits a live event. Validators flag faults. The gate is a pure function, so the trace alone can rebuild the decision.
3. `arc/api.py` (FastAPI) starts runs in a worker thread and streams their steps over SSE. Approvals are written back to the trace.
4. `gateway/` (TypeScript, graphql-yoga) turns that SSE stream into a GraphQL `runSteps` subscription. `console/` (React) renders the live trace and the approval panel.
5. `arc/evaluate.py` measures routing, retrieval, the gate on 1,000 tickets, 5 injected faults with guards on and off, and replay of 9 candidate regressions plus A/A.

## Probes

**Q: Your router is at 0.30 macro-F1. Isn't that bad?**
- 15s: It's honest. On a random split it scored 0.645, but 53% of test tickets had a paraphrase in train. When I split by paraphrase family it dropped to 0.30, and grouped CV inside train predicted that. So the agent is designed around it: it auto-sends only the 2.5% of tickets it's confident on (96% correct) and sends the rest to a human.

**Q: How do you know a reply never goes out when something breaks?**
- I injected 5 faults (timeout, stale schema, malformed tool output, empty retrieval, refusal) into 300 tickets each. With guards on, 1,500 of 1,500 escalated and none auto-sent. With guards off, 22 bad replies went out (11 to a dead queue, 11 refusals) and 300 runs crashed. An independent audit re-applies the v1 policy to every recorded trace and found 0 violations.

**Q: What's a trace good for?**
- Debugging and audit. Every step's input and output are stored before streaming, and the gate is a pure function, so `reconstruct()` rebuilds each run's decision from the steps alone. That worked for 100% of clean and faulty runs.

**Q: How does replay work, and how do you avoid crying wolf?**
- I record 200 runs and re-run them on the candidate build with writes mocked, then use paired tests against the baseline (McNemar on routing and on bad auto-sends, paired t on draft similarity) with Bonferroni, plus a hard rule that no auto-send v1 would block is allowed. A/A replays (same build, new sampling seeds) gave 0% false alarms over 100 runs. It catches safety-gate and routing breaks every time.

**Q: It missed 5 of 9 regressions.**
- They're quality drifts of 0.004 to 0.008 similarity, too small to see at 200 runs. I measured the power: see `reports/replay_power.csv` for runs needed per regression. In practice: always run the cheap invariant checks, size the replay set for the smallest regression you care about, and add an LLM judge for quality.

**Q: Biggest weakness?**
- The gate trusts a priority model with 49% recall on high priority, so 10 of the 25 auto-sent replies were truly high-priority tickets. Next I'd auto-send only when the model is confident the ticket is low priority, and measure the trade-off.

**Q: Why an extractive drafter?**
- No LLM API is reachable from the build environment, and I wanted the runtime, gate and eval measured independently of the model. The LLM sits behind a `Drafter` interface; the Groq version is there and gets measured next with an LLM judge.

**Q: Why a Node gateway in front of Python?**
- It keeps the client contract (typed GraphQL schema, subscriptions) separate from the agent service, which can change or scale on its own. Vitest contract tests pin the mapping, including enum-validated fault names, which are rejected before reaching the API.
