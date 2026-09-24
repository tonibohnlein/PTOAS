# FrontierSynch Phase A parity inventory

Paper baseline: `4905f8a` (v0.44), with the operative Phase A rules inherited
unchanged from `a13650a` (v0.43). Implementation baseline: `6220bb6c` on
`codex/handoff-foundation`. This inventory distinguishes specified rules from
open research problems and prevents an implemented interface from being
mistaken for a qualified answer.

| Review gate | Specified draft contract | Current implementation | Acceptance evidence |
| --- | --- | --- | --- |
| 1. Factored provenance | Section 3.4 and Appendix I.1: shared `Both`/`Choose` and transfer/application expressions retain one original guard interpretation for writer, readers, demands and continuation. Reads query old writer; definite writes including RMW query old writers/readers before strong update. Partial writes retain conservative origins. Backward next-use is dual. Incoming/no-producer cases remain explicit. | `OriginalLifetimes` has four marginal may bit matrices and eager per-source requirements. `OriginalReadQueries` has a distinct guarded reader DAG. No joined provenance/demand DAG. Every imported write has `definiteWrite=false`. | Conditional writer and optional readers; RMW; independent guards with linear DAG formation; full/partial exact-cell effects; incoming path; backward next use. Actual imported full-write qualification must be demonstrated before declaring end-to-end parity. |
| 2. Directional boundaries | Appendix I.3: source and target exact/covering answers are independent. On a qualified original cut interval, source trims no-hit suffix and target trims no-hit prefix; choice/finite-loop boundaries preserve multiplicity, guards, extra work and optional no-access executions. Unknown effects obstruct exclusion. | `ProgramAnalysis::boundary` returns Exact/NoHit/Unknown. No covering answer or cut-interval rule. | `A; if(g) B; U` source cut excludes unrelated suffix `U`; target dual; zero-reader and unknown-effect cases; transparent wrapper retains semantic cut. |
| 3. Exact occurrence and frontiers | Section 3.6 D1–D4 and Appendix I.2: guarded alternatives and incoming cases, qualified physical permutation and initial visits, first/last readers per generation, counted invariant readers, original owner/re-entry transport, and the restricted max/min interval rule with safe empty/arithmetic/availability cases. Unsupported cases remain Unknown. | FixedVisit rejects cycles/intervening accesses; PeriodicSameRole is narrow; structural reader expressions exist; interval rule and general re-entry transport absent. | Vector-add prologue/tail, GEMM nested loops and early L1 last read, FA repeated Q use and shortened batches, plus independent small semantic fixtures for each admitted rule. Do not require the draft's open general symbolic matching. |
| 4. Look-ahead preparation | Section 3.5 and Section 4.1: immutable original obligation IDs, distinct request groups and three predeclared descriptor slots; original source/deadline adjacency; all referenced source cuts/qualified roles subscribed before traversal with finite support closure; unresolved retains conservative hooks; same-occurrence two-link consequences are opportunities and grant no credit. | Indexed requirements and direct source subscriptions exist; descriptor identities/closure/two-link service absent. | Verify every declared source can be captured when crossed, source/target independence, stable IDs under descriptor alternatives, same-occurrence consequence joins, and rejection of cross-branch or cross-iteration false joins. |

Cross-cutting existing services: shared `SyncInput` effects, exact and conservative
physical cells, original control, `All`, `MayAfter`, typed scalar prerequisites,
and original source cuts. Their may results must not be promoted to exact
occurrence or selected completion facts. The full-cell overwrite qualifier must
derive from shared effects and storage coverage, never from an instruction
whitelist or an additional supported-op audit.

Research gaps outside these acceptance gates: arbitrary symbolic access matching
through resets/re-entry, general arithmetic and endpoint observation solving,
general selected conditional proof contexts, and arbitrary recurring protocol
construction. When an admitted query cannot be qualified, preserve its
underlying requirement and report the obstruction.
