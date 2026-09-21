# Bounded relay selection using receiver and intermediate effects

2026-09-21, based on `bec7dfe37`. This implements the local correction following
the [two linked review witnesses](oahs-return-query-relay-review.md). It changes
ordinary qualified FIFO relay selection, with no new option or pass mode.
Physical slot qualification, source boundaries, binding and staged validation
remain unchanged. Frozen device tasks retain their original commits.

## The decision now compares both ends

For each eligible intermediate engine, construction retains the existing
earliest forwarding opportunity after intervening FIFO sends. It compares two
sets rather than one incidental-prefix score:

1. **Forwarded incidental history:** completion in the intermediate prefix
   beyond the selected source, excluding completion already acquired or still
   required at the final receiver's current deadline. Exclusion requires the
   same access occurrence along the qualified straight corridor.
2. **New intermediate prerequisites:** source-history prerequisites absent from
   actual gates of crossed middle-engine payloads and selected command
   endpoints. Entries name the access class and the payload cut or stable
   endpoint ID. Outward SETs use their actual prefix, not the payload gate or
   the end of the whole word. Existing acquisitions and barriers are inspected
   too. A later wait cannot justify an earlier publication in the same word.

A candidate replaces the incumbent only when neither set grows and at least
one shrinks. Equal sets retain the later-gap tie. Incomparable sets retain the
first deterministic candidate; fewer forwarded histories alone no longer win.
Required-completion information guides ranking only: it is not inserted into
the causal state. Actual selected transfers still supply every receipt.

The first version of this comparison exposed a further interaction in the
second witness. After choosing M for slot 0, the next relay could prepend the
slot-1 receipt to the word forwarding slot 0. That put the later writer's
completion into the already selected earlier receipt. The final correction
also checks crossed publications from the candidate middle to the same final
observer. If the new source history would broaden one of those receipts, that
middle uses the current consumer deadline instead of the early gap. Unknown
or regenerated history does not justify early forwarding. The late word's
legality, coverage and physical key state are checked normally.

This is a bounded policy, **not a general contextual ordering certificate**.
The views concern represented access histories and selected endpoints through
the current deadline. They do not enumerate every payload relation, predict
future insertions, or establish equivalence of all open event interfaces.
The same-observer receipt safeguard is narrower than the still-open generic
outward-interface certificate. The incomparable example is not advertised as
a strict improvement over the old selected plan.

## Existing analyses and bounded work

The rule reads the existing shared causal frontier, current residual, occurrence
freshness index, ledger endpoint identities and replay checkpoints. It adds no
independent completion analysis, hypothetical receipt or persistent cache.
All comparisons use the current ledger version; candidates do not mutate it.

One ordered straight-corridor view is shared by the candidate scans. Only the
chosen proposal undergoes the existing full staged protocol solve. The new
structural visits are charged to `relayPreparationSites`; history comparisons
also contribute work and are not a constant-time guarantee. No all-pairs
transitive closure or completed-plan deletion search enters production.

Physical binding is still checked after ranking, and the second leg still
requires preexisting source-time rearming credit. Positive binding before
ranking and private application of the first receipt for second-key reuse
remain separate extensions.

## Discriminating linked tests

`test/oahs/selected_relay_test.cpp` constructs the plans with recurring omission
and helper trials disabled, cold-checks them, then compares complete payload
relation sets with the independent ordinary-prefix graph oracle.

| Case | Result |
| --- | --- |
| Original independent-middle witness | Retains the deterministic late M route. Both alternatives have two pairs / 15 relations; four removed and four added distinguish their incomparable orders. |
| Original receiver-required witness | Selects the useful M route and preserves the earlier receipt. Five pairs remain; 34→30 relations, four removed and none added. |
| Force the later receipt into the earlier word | Safe protocol, but 30→34 relations with four additions. The selected policy avoids it. |
| Required receiver completion, equal late gaps | Selects the second enumerated middle because its prefix is useful; 19→15 relations against the supplied worse route, three→two pairs. Only one construction decision occurs: no earlier transfer supplies this credit. |
| Already-acquired receiver completion | The same strict four-relation reduction, using actual receiver credit. |
| Middle payload already gated | Keeps useful early forwarding; 20→16 relations against a supplied late control. |
| Outward SET before the credit-providing WAIT | Retains the late route; the early control would newly gate the outward receiver. |
| Outward SET after that WAIT | Early forwarding remains available; 24→20 relations against the late control. |

Twenty-one complete-leg deletion checks isolate lost memory support while
remaining balanced, acyclic and rearming-valid. These finite fixtures use
explicit portable FIFO contracts and virgin keys. Repeated native attention
cases and missing-rearming mutations are covered by the separate native
regression; the small witnesses alone do not establish recurring reuse.

## Native and corpus validation

Final results are recorded under `../relay-selection-work/`. The frozen
`baseline-selected-test` precedes this change; the corpus baseline uses the
archived `b2d8482fc` outputs (the intervening `bec7dfe37` changed only docs).

- All 24 portable suites and the final expanded relay suite pass. Both native
  test executables pass. `relay-final.log` records the linked comparisons.
- Both pinned attention inputs pass six finite cases each, including repeated
  entries, phase boundaries and missing-memory/rearming negatives. The original
  one-entry reduction remains 80 full payload relations with zero additions,
  and two entries retain 160 removed / zero added. Evidence is in
  `attention-native/`; these are host graph checks, not device execution.
- All 88 corpus inputs / 97 functions construct and reconstruct successfully.
  Every plan is byte-identical to the baseline, including both attention
  variants and the newer joint-reader projection plans. Aggregate replay stays
  3,262,799 evaluations. `evaluate.py` and `corpus-final/results.json` retain
  commands, source/input/output identities and work counters.
- Per affected AIC, the two staged relay analyses still cost 1,318 site
  evaluations and replay stays 22,772. Sharing the corridor reduces recorded
  relay-preparation site visits from 1,434 to 1,020, despite the additional
  checks. History/set comparisons also cost work: no compiler wall-time
  improvement is inferred from this counter.

Reproduction with existing build trees:

```sh
cmake --build ../oahs-m1-core-build --parallel 2
ctest --test-dir ../oahs-m1-core-build --parallel 2 --output-on-failure
cmake --build ../oahs-m1-native-build --parallel 2 \
  --target pto-oahs-selected-test pto-oahs-native-test pto-test-opt
../oahs-m1-native-build/tools/pto-test-opt/pto-oahs-native-test
../oahs-m1-native-build/tools/pto-test-opt/pto-oahs-selected-test
python3 test/benchmarks/attention_relay/run.py \
  --build ../oahs-m1-native-build --out ../relay-selection-work/attention-native \
  PATH_TO_PINNED_ROW48/prepared.pto PATH_TO_PINNED_ROW49/prepared.pto
```

The input hashes and native regression requirements are in the committed
[attention README](../../test/benchmarks/attention_relay/README.md).

No new device speedup is claimed. A changed synthetic plan is not sufficient
reason to restart a device campaign whose native binaries are unchanged.
