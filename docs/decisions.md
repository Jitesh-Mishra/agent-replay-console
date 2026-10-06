# Decisions

**D1 Data: the official CSV from a public GitHub mirror**
Kaggle and HuggingFace are blocked from the workspace and from Jitesh's Mac sandbox. The publisher's file `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (28,587 rows) was taken from a public GitHub repo that commits it, and matches the size listed on HuggingFace. The English rows with an answer (16,335) are committed in `data/`. License: CC BY-NC 4.0 (attribution, non-commercial). The publisher sells a synthetic ticket generator, and the tickets read as generated (uniform style, `<name>` placeholders), so they are treated as likely synthetic, never as real support data.

**D2 Grouped split, because the random split leaks**
The data contains families of paraphrased tickets. Under a random 80/20 split, 53% of test tickets have a train neighbour with TF-IDF cosine ≥ 0.6, and copying that neighbour's queue is 98.8% accurate on them. Near-duplicates were clustered (connected components at cosine ≥ 0.45: 8,389 families, the largest 110) and split by family with StratifiedGroupKFold. Routing macro-F1 drops from 0.645 (random split, same model) to 0.300 (grouped). The grouped number is the headline. Hyperparameters are tuned with grouped CV for the same reason.

**D3 Offline embeddings and an extractive drafter in the proof**
Default Chroma and HuggingFace embedding weights can't be downloaded here, and Groq is unreachable. Chroma stores TF-IDF → 256-d LSA vectors instead. The drafter samples one of the top-3 retrieved historical answers (softmax over similarity, temperature 0.3). `GroqDrafter` is wired in behind the same interface, but no LLM numbers are claimed until Jitesh runs it.

**D4 Gate thresholds set before evaluation**
Auto-send requires route confidence ≥ 0.6, predicted priority not high, retrieval similarity ≥ 0.5, and no fault flags. At this router quality that auto-sends a small slice and sends the rest to a human, which is the intended conservative behaviour.

**D5 Refusal-detector false positive, found and fixed**
The first refusal regex matched any "I'm sorry", including empathetic openers ("I'm sorry to hear…") that appear in 0.6% of historical answers. It escalated 8 of 1,000 clean runs. The regex now excludes empathy phrases, and a test covers both directions. All reported numbers come from the fixed build.

**D6 Replay detection rule**
Each candidate re-runs the same 200 recorded tickets with writes mocked. It is compared with the recorded baseline using three paired tests: McNemar on routing correctness, a paired t-test on draft similarity to the historical answer, and McNemar on bad auto-sends (wrong queue or truly high priority). A Bonferroni α of 0.05/3 applies across them, plus a hard invariant: any auto-send the v1 policy would not allow. A/A (same build, new sampling seeds) measures false alarms.

**D7 Memoised deterministic calls**
Router and retrieval results are cached per component set, so replay can run thousands of candidate runs. The stochastic drafter is never cached.
